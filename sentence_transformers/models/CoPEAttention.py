import torch
from torch import Tensor
from torch import nn
from typing import Union, Tuple, List, Iterable, Dict
import os
import json
import math


class CoPE (nn . Module):
    def __init__(self, npos_max, head_dim):
        super().__init__()
        self.npos_max = npos_max
        self.pos_emb = nn.parameter.Parameter(
            torch.zeros(1, head_dim, npos_max))

    def forward(self, query, attn_logits):
        # compute positions
        gates = torch.sigmoid(attn_logits)
        pos = gates.flip(-1).cumsum(dim=-1).flip(-1)
        pos = pos.clamp(max=self.npos_max - 1)
        # interpolate from integer positions
        pos_ceil = pos.ceil().long()
        pos_floor = pos.floor().long()
        logits_int = torch.matmul(query, self.pos_emb)
        logits_ceil = logits_int.gather(-1, pos_ceil)
        logits_floor = logits_int.gather(-1, pos_floor)
        w = pos - pos_floor
        return logits_ceil * w + logits_floor * (1 - w)


class CoPEAttention(nn.Module):
    """Performs pooling (max or mean) on the token embeddings.

    Using pooling, it generates from a variable sized sentence a fixed sized sentence embedding. This layer also allows to use the CLS token if it is returned by the underlying word embedding model.
    You can concatenate multiple poolings together.

    :param word_embedding_dimension: Dimensions for the word embeddings
    :param pooling_mode: Can be a string: mean/max/cls. If set, overwrites the other pooling_mode_* settings
    :param pooling_mode_cls_token: Use the first token (CLS token) as text representations
    :param pooling_mode_max_tokens: Use max in each dimension over all tokens.
    :param pooling_mode_mean_tokens: Perform mean-pooling
    :param pooling_mode_mean_sqrt_len_tokens: Perform mean-pooling, but devide by sqrt(input_length).
    :param pooling_mode_weightedmean_tokens: Perform (position) weighted mean pooling, see https://arxiv.org/abs/2202.08904
    :param pooling_mode_lasttoken: Perform last token pooling, see https://arxiv.org/abs/2202.08904 & https://arxiv.org/abs/2201.10005
    """

    def __init__(self,
                 word_embedding_dimension: int,  # transtorm的输出维度
                 pooling_mode: str = None,
                 pooling_mode_cls_token: bool = True,
                 pooling_mode_max_tokens: bool = False,
                 pooling_mode_mean_tokens: bool = False,
                 pooling_mode_mean_sqrt_len_tokens: bool = False,
                 pooling_mode_weightedmean_tokens: bool = False,  # 加权平均池化
                 pooling_mode_lasttoken: bool = False,  # 最后一个标记池化
                 ):
        super(CoPEAttention, self).__init__()

        self.config_keys = ['word_embedding_dimension', 'pooling_mode_cls_token', 'pooling_mode_mean_tokens',
                            'pooling_mode_max_tokens',
                            'pooling_mode_mean_sqrt_len_tokens', 'pooling_mode_weightedmean_tokens',
                            'pooling_mode_lasttoken']

        if pooling_mode is not None:  # Set pooling mode by string
            pooling_mode = pooling_mode.lower()
            assert pooling_mode in ['mean', 'max', 'cls', 'weightedmean', 'lasttoken']
            pooling_mode_cls_token = (pooling_mode == 'cls')
            pooling_mode_max_tokens = (pooling_mode == 'max')
            pooling_mode_mean_tokens = (pooling_mode == 'mean')
            pooling_mode_weightedmean_tokens = (pooling_mode == 'weightedmean')
            pooling_mode_lasttoken = (pooling_mode == 'lasttoken')

        self.word_embedding_dimension = word_embedding_dimension

        self.pooling_mode_cls_token = pooling_mode_cls_token
        self.pooling_mode_mean_tokens = pooling_mode_mean_tokens
        self.pooling_mode_max_tokens = pooling_mode_max_tokens
        self.pooling_mode_mean_sqrt_len_tokens = pooling_mode_mean_sqrt_len_tokens
        self.pooling_mode_weightedmean_tokens = pooling_mode_weightedmean_tokens
        self.pooling_mode_lasttoken = pooling_mode_lasttoken

        pooling_mode_multiplier = sum([pooling_mode_cls_token, pooling_mode_max_tokens, pooling_mode_mean_tokens,
                                       pooling_mode_mean_sqrt_len_tokens, pooling_mode_weightedmean_tokens,
                                       pooling_mode_lasttoken])
        self.pooling_output_dimension = (pooling_mode_multiplier * word_embedding_dimension)

        self.query_layer = nn.Linear(768, 768)
        self.key_layer = nn.Linear(768, 768)
        self.value_layer = nn.Linear(768, 768)

        self.cope = CoPE(512, 768)  # 512是序列长，768是维度/注意力头数=768/1,1个头
        self.head_dim = 768

    def __repr__(self):
        return "Pooling({})".format(self.get_config_dict())

    def get_pooling_mode_str(self) -> str:
        """
        Returns the pooling mode as string
        用于获取当前设置的池化模式并将其作为字符串返回
        """
        modes = []
        if self.pooling_mode_cls_token:
            modes.append('cls')
        if self.pooling_mode_mean_tokens:
            modes.append('mean')
        if self.pooling_mode_max_tokens:
            modes.append('max')
        if self.pooling_mode_mean_sqrt_len_tokens:
            modes.append('mean_sqrt_len_tokens')
        if self.pooling_mode_weightedmean_tokens:
            modes.append('weightedmean')
        if self.pooling_mode_lasttoken:
            modes.append('lasttoken')

        return "+".join(modes)

    def forward(self, features: Dict[str, Tensor]):
        token_embeddings = features['token_embeddings']
        attention_mask = features['attention_mask']

        # Pooling strategy
        output_vectors = []
        # cls_token = features.get('cls_token_embeddings', token_embeddings[:, 0])  # Take first token by default

        query = self.query_layer(token_embeddings)
        key = self.key_layer(token_embeddings)
        val = self.value_layer(token_embeddings)
        # mask = attention_mask  # [bs,len]

        # q , k , v have dimensions batch x seq_len x head_dim
        attn_logits = torch.bmm(query, key.transpose(-1, -2))
        attn_logits = attn_logits / math.sqrt(self.head_dim)

        # mask = mask.unsqueeze(1)  # [bs,1,len],在广播时自动扩展为[bs,len,len]
        mask = attention_mask.unsqueeze(1).expand(-1, token_embeddings.size(1), -1)  # [bs,len,len]

        # attn_logits += mask.log()
        attn_logits = attn_logits.masked_fill(mask == 0, float('-inf'))

        attn_logits += self.cope(query, attn_logits)
        attn = torch.softmax(attn_logits, dim=-1)
        out = torch.bmm(attn, val)

        cls_out = out[:, 0, :]
        output_vectors.append(cls_out)

        output_vector = torch.cat(output_vectors, 1)
        features.update({'sentence_embedding': output_vector})
        return features

    def get_sentence_embedding_dimension(self):
        return self.pooling_output_dimension

    def get_config_dict(self):
        return {key: self.__dict__[key] for key in self.config_keys}

    def save(self, output_path):
        with open(os.path.join(output_path, 'config.json'), 'w') as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)

    @staticmethod
    def load(input_path):
        with open(os.path.join(input_path, 'config.json')) as fIn:
            config = json.load(fIn)

        return CoPEAttention(**config)
