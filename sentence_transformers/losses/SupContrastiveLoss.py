from enum import Enum
from typing import Iterable, Dict
from torch import nn, Tensor
from sentence_transformers.SentenceTransformer import SentenceTransformer
import torch.nn as nn
import torch.nn.functional as F
from pytorch_metric_learning import losses as loss_fun
from pytorch_metric_learning.regularizers import LpRegularizer


class SupContrastiveLoss(nn.Module):

    def __init__(self, model: SentenceTransformer):
        super(SupContrastiveLoss, self).__init__()
        self.model = model

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        rep_a = reps[0]

        query_embed = F.normalize(rep_a, dim=1)

        loss_function = loss_fun.SupConLoss(temperature=0.1, embedding_regularizer=LpRegularizer())

        return loss_function(query_embed, labels)
