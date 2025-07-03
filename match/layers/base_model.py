# -*- coding:utf-8 -*-

"""
@date: 2024/8/22 下午7:33
@summary:
"""
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Dropout

class DNN(nn.Module):
    def __init__(self, inputs_dim, hidden_units, activation='relu', dropout=0., **kwargs):
        """
        DNN层
        :param inputs_dim:
        :param hidden_units:
        :param activation:
        :param dnn_dropout:
        :param kwargs:
        """
        super(DNN, self).__init__()
        self.hidden_units = [inputs_dim] + list(hidden_units)
        self.activation = self.activation_layer(activation)
        self.dropout = Dropout(dropout)

        self.fc = nn.ModuleList(
            [nn.Linear(self.hidden_units[i], self.hidden_units[i+1]) for i in range(len(self.hidden_units)-1)]
        )

    def activation_layer(self, act_name):
        """"""
        if isinstance(act_name, str):
            if act_name.lower() == 'sigmoid':
                act_layer = F.sigmoid
            elif act_name.lower() == 'relu':
                act_layer = F.relu
            elif act_name.lower() == 'prelu':
                act_layer = F.prelu
            else:
                raise NotImplementedError

            return act_layer

    def forward(self, x):
        """"""
        for fc in self.fc:
            x = self.activation(fc(x))
            x = self.dropout(x)
        return x


class BaseTower(nn.Module):
    def __init__(self, user_sparse_feature_columns, item_sparse_feature_columns, user_dense_feature_columns=(),
                 item_dense_feature_columns=()):
        """
        双塔模型基础模型
        :param user_sparse_feature_columns:
        :param item_sparse_feature_columns:
        :param user_dense_feature_columns:
        :param item_dense_feature_columns:
        :param user_dnn_hidden_units:
        :param item_dnn_hidden_units:
        """
        super(BaseTower, self).__init__()
        self.user_sparse_feature_columns = user_sparse_feature_columns
        self.user_dense_feature_columns = user_dense_feature_columns
        self.item_sparse_feature_columns = item_sparse_feature_columns
        self.item_dense_feature_columns = item_dense_feature_columns

        self.user_embed_layers = {
            'embed_' + str(feat['feat']): nn.Embedding(num_embeddings=feat['feat_num'], embedding_dim=feat['embed_dim'])
            for feat in self.user_sparse_feature_columns
        }

        self.item_embed_layers = {
            'embed_' + str(feat['feat']): nn.Embedding(num_embeddings=feat['feat_num'], embedding_dim=feat['embed_dim'],
                                                       padding_idx=0)
            for feat in self.item_sparse_feature_columns
        }

    def compute_input_dim(self, feature_columns):
        input_dim = 0
        for feat in feature_columns:
            input_dim = input_dim + feat['embed_dim']
        return input_dim


    def forward(self, x):
        """"""

