import torch
from torch import Tensor
from torch import nn
from typing import Union, Tuple, List, Iterable, Dict
import os
import json
import logging
import time
import numpy as np


# from .weight_learner import weight_learner
from .config import parser


import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.autograd import Variable

# import torch.autograd
# torch.autograd.set_detect_anomaly(True)

logger = logging.getLogger(__name__)


def lr_setter(optimizer, epoch, args, bl=False):
    lr = args.lrbl * (0.1 ** (epoch // (args.epochw * 0.5)))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def cov(x, w):
    '''
    :param x: [bs,len,768]
    :param w: [bs,len]
    :return:
    '''

    a = w.unsqueeze(2) * x
    cov = torch.matmul(a, a.transpose(1, 2))
    e = torch.mean(a, dim=2).unsqueeze(2)
    e2 = torch.sum(a, dim=2).unsqueeze(2)
    res = cov - torch.matmul(e, e2.transpose(1, 2))
    return res  # [bs,512,512]


def lossb_expect(featurecs, weight):
    '''
    :param featurecs: [bs,len,768]
    :param weight: [bs,len]
    :return:
    '''
    loss = Variable(torch.FloatTensor([0]).cuda())
    cov1 = cov(featurecs, weight)
    cov_matrix = cov1 * cov1
    loss += torch.sum(cov_matrix) - torch.sum(torch.diagonal(cov_matrix, dim1=-2, dim2=-1))
    return loss


def weight_learner(last_hidden_state, mask, args):
    softmax = nn.Softmax(1).cuda()

    weight = torch.ones(last_hidden_state.size()[0], last_hidden_state.size()[1]).cuda()
    weight = torch.where(mask == 0, float('-inf'), weight).detach().requires_grad_(True)

    def mask_hook(grad):
        return grad * mask
    weight.register_hook(mask_hook)

    featurec = torch.FloatTensor(last_hidden_state.size()).cuda()
    featurec.data.copy_(last_hidden_state.data)

    optimizerbl = torch.optim.SGD([weight], lr=args.lrbl, momentum=0.8)

    for epoch in range(args.epochw):
        lr_setter(optimizerbl, epoch+1, args, bl=True)

        optimizerbl.zero_grad()

        lossb = lossb_expect(featurec, softmax(weight))
        lossp = softmax(weight).pow(args.decay_pow).sum()
        lossg = lossb / args.lambdap + lossp

        if lossg.requires_grad is not True:
            lossg.requires_grad = True

        lossg.backward(retain_graph=True)

        # print("lossb:", lossb)
        # print("lossp:", lossp)
        # print(epoch, " lossg:", lossg)

        # print(lossg)
        torch.nn.utils.clip_grad_norm_(weight, 1)
        optimizerbl.step()

    softmax_weight = softmax(weight)

    return softmax_weight


class Pooling(nn.Module):
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
                 word_embedding_dimension: int,
                 pooling_mode: str = None,
                 pooling_mode_cls_token: bool = True,
                 pooling_mode_max_tokens: bool = False,
                 pooling_mode_mean_tokens: bool = False,
                 pooling_mode_mean_sqrt_len_tokens: bool = False,
                 pooling_mode_weightedmean_tokens: bool = False,
                 pooling_mode_lasttoken: bool = False,
                 pooling_mode_weigth: bool = False,
                 ):
        super(Pooling, self).__init__()

        self.config_keys = ['word_embedding_dimension', 'pooling_mode_cls_token', 'pooling_mode_mean_tokens', 'pooling_mode_max_tokens',
                            'pooling_mode_mean_sqrt_len_tokens', 'pooling_mode_weightedmean_tokens', 'pooling_mode_lasttoken', 'pooling_mode_weigth']

        if pooling_mode is not None:        # Set pooling mode by string
            pooling_mode = pooling_mode.lower()
            assert pooling_mode in ['mean', 'max', 'cls', 'weightedmean', 'lasttoken', 'weigth']
            pooling_mode_cls_token = (pooling_mode == 'cls')
            pooling_mode_max_tokens = (pooling_mode == 'max')
            pooling_mode_mean_tokens = (pooling_mode == 'mean')
            pooling_mode_weightedmean_tokens = (pooling_mode == 'weightedmean')
            pooling_mode_lasttoken = (pooling_mode == 'lasttoken')
            pooling_mode_weigth = (pooling_mode == 'weigth')

        self.word_embedding_dimension = word_embedding_dimension

        self.pooling_mode_cls_token = pooling_mode_cls_token
        self.pooling_mode_mean_tokens = pooling_mode_mean_tokens
        self.pooling_mode_max_tokens = pooling_mode_max_tokens
        self.pooling_mode_mean_sqrt_len_tokens = pooling_mode_mean_sqrt_len_tokens
        self.pooling_mode_weightedmean_tokens = pooling_mode_weightedmean_tokens
        self.pooling_mode_lasttoken = pooling_mode_lasttoken
        self.pooling_mode_weigth = pooling_mode_weigth

        pooling_mode_multiplier = sum([pooling_mode_cls_token, pooling_mode_max_tokens, pooling_mode_mean_tokens, 
            pooling_mode_mean_sqrt_len_tokens, pooling_mode_weightedmean_tokens, pooling_mode_lasttoken, pooling_mode_weigth])
        self.pooling_output_dimension = (pooling_mode_multiplier * word_embedding_dimension)

    def __repr__(self):
        return "Pooling({})".format(self.get_config_dict())

    def get_pooling_mode_str(self) -> str:
        """
        Returns the pooling mode as string
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
        if self.pooling_mode_weigth:
            modes.append('weigth')

        return "+".join(modes)

    def forward(self, features: Dict[str, Tensor]):
        token_embeddings = features['token_embeddings']
        attention_mask = features['attention_mask']

        # Pooling strategy
        output_vectors = []

        if self.pooling_mode_cls_token:
            cls_token = features.get('cls_token_embeddings', token_embeddings[:, 0])  # Take first token by default
            output_vectors.append(cls_token)

        if self.pooling_mode_max_tokens:
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            token_embeddings[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
            max_over_time = torch.max(token_embeddings, 1)[0]
            output_vectors.append(max_over_time)
        if self.pooling_mode_mean_tokens or self.pooling_mode_mean_sqrt_len_tokens:
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)

            # If tokens are weighted (by WordWeights layer), feature 'token_weights_sum' will be present
            if 'token_weights_sum' in features:
                sum_mask = features['token_weights_sum'].unsqueeze(-1).expand(sum_embeddings.size())
            else:
                sum_mask = input_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)

            if self.pooling_mode_mean_tokens:
                output_vectors.append(sum_embeddings / sum_mask)
            if self.pooling_mode_mean_sqrt_len_tokens:
                output_vectors.append(sum_embeddings / torch.sqrt(sum_mask))
        if self.pooling_mode_weightedmean_tokens:
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            # token_embeddings shape: bs, seq, hidden_dim
            weights = (
                    torch.arange(start=1, end=token_embeddings.shape[1] + 1)
                    .unsqueeze(0)
                    .unsqueeze(-1)
                    .expand(token_embeddings.size())
                    .float().to(token_embeddings.device)
                )
            assert weights.shape == token_embeddings.shape == input_mask_expanded.shape
            input_mask_expanded = input_mask_expanded * weights

            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)

            #If tokens are weighted (by WordWeights layer), feature 'token_weights_sum' will be present
            if 'token_weights_sum' in features:
                sum_mask = features['token_weights_sum'].unsqueeze(-1).expand(sum_embeddings.size())
            else:
                sum_mask = input_mask_expanded.sum(1)

            sum_mask = torch.clamp(sum_mask, min=1e-9)
            output_vectors.append(sum_embeddings / sum_mask)
        if self.pooling_mode_lasttoken:
            bs, seq_len, hidden_dim = token_embeddings.shape
            # attention_mask shape: (bs, seq_len)
            # Get shape [bs] indices of the last token (i.e. the last token for each batch item)
            # argmin gives us the index of the first 0 in the attention mask; We get the last 1 index by subtracting 1
            gather_indices = torch.argmin(attention_mask, 1, keepdim=False) - 1 # Shape [bs]

            # There are empty sequences, where the index would become -1 which will crash
            gather_indices = torch.clamp(gather_indices, min=0)

            # Turn indices from shape [bs] --> [bs, 1, hidden_dim]
            gather_indices = gather_indices.unsqueeze(-1).repeat(1, hidden_dim)
            gather_indices = gather_indices.unsqueeze(1)
            assert gather_indices.shape == (bs, 1, hidden_dim)

            # Gather along the 1st dim (seq_len) (bs, seq_len, hidden_dim -> bs, hidden_dim)
            # Actually no need for the attention mask as we gather the last token where attn_mask = 1
            # but as we set some indices (which shouldn't be attended to) to 0 with clamp, we
            # use the attention mask to ignore them again
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            embedding = torch.gather(token_embeddings * input_mask_expanded, 1, gather_indices).squeeze(dim=1)
            output_vectors.append(embedding)
        if self.pooling_mode_weigth:
            # args = parser.parse_args()
            args, unknown = parser.parse_known_args()

            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            mask_embeddings = token_embeddings * input_mask_expanded

            weight = weight_learner(mask_embeddings, attention_mask, args)
            weight = weight.unsqueeze(2)

            weighted_token_embeddings = weight * mask_embeddings

            weighted_embeddings = torch.sum(weighted_token_embeddings, dim=1)
            output_vectors.append(weighted_embeddings)

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

        return Pooling(**config)
