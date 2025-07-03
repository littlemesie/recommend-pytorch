# -*- coding:utf-8 -*-

"""
@date: 2024/10/19 下午3:50
@summary:
"""
import torch
import torch.nn.functional as F
from match.layers.base_model import DNN
from match.layers.base_model import BaseTower

class DSSM(BaseTower):
    def __init__(self, user_sparse_feature_columns, item_sparse_feature_columns, user_dense_feature_columns=(),
                 item_dense_feature_columns=(), user_dnn_hidden_units=(64, 32), item_dnn_hidden_units=(64, 32),
                 activation='relu', dropout=0.):
        """
        双塔模型基础模型
        :param user_sparse_feature_columns:
        :param item_sparse_feature_columns:
        :param user_dense_feature_columns:
        :param item_dense_feature_columns:
        :param user_dnn_hidden_units:
        :param item_dnn_hidden_units:
        :param activation:
        :param dropout:
        """
        super(DSSM, self).__init__(user_sparse_feature_columns, item_sparse_feature_columns, user_dense_feature_columns,
                 item_dense_feature_columns)
        self.user_dnn = DNN(self.compute_input_dim(user_sparse_feature_columns) + len(user_dense_feature_columns),
                            user_dnn_hidden_units, activation, dropout)
        self.item_dnn = DNN(self.compute_input_dim(item_sparse_feature_columns) + len(user_dense_feature_columns),
                            item_dnn_hidden_units, activation, dropout)

    def forward(self, inputs):
        """"""
        # 只有sparse特征，需要dense特征可以自行加上
        user_sparse_inputs, item_sparse_inputs = inputs
        if user_sparse_inputs and item_sparse_inputs:
            user_sparse_embed = torch.concat([self.user_embed_layers['embed_{}'.format(k)](v)
                                           for k, v in user_sparse_inputs.items()], -1)
            user_dnn_out = self.user_dnn(user_sparse_embed)
            print(user_dnn_out.shape)
            item_sparse_embed = torch.concat([self.item_embed_layers['embed_{}'.format(k)](v)
                                              for k, v in item_sparse_inputs.items()], -1)
            item_dnn_out = self.item_dnn(item_sparse_embed)
            cosine = F.cosine_similarity(user_dnn_out, item_dnn_out, dim=2)
            return cosine
        elif user_sparse_inputs:
            user_sparse_embed = torch.concat([self.user_embed_layers['embed_{}'.format(k)](v)
                                              for k, v in user_sparse_inputs.items()], -1)
            user_dnn_out = self.user_dnn(user_sparse_embed)
            return user_dnn_out
        elif item_sparse_inputs:
            item_sparse_embed = torch.concat([self.item_embed_layers['embed_{}'.format(k)](v)
                                              for k, v in item_sparse_inputs.items()], -1)
            item_dnn_out = self.item_dnn(item_sparse_embed)
            return item_dnn_out
        else:
            raise 'Input Error!'



if __name__ == '__main__':
    """"""
    # user_features = [{'feat': 'user_id', 'feat_num': 100, 'embed_dim': 8}]
    # item_features = [{'feat': 'item_id', 'feat_num': 100, 'embed_dim': 8}]
    # model = DSSM(user_features, item_features)
    # user_sparse_inputs = {'user_id': torch.LongTensor([[1, 3, 32, 50]])}
    # item_sparse_inputs = {'item_id': torch.LongTensor([[11, 13, 2, 5]])}
    # model((user_sparse_inputs, item_sparse_inputs))