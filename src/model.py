import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from Encoder import Encoder, EncoderLayer, MultiHeadedAttention, PositionwiseFeedForward, ConcatFeedForward
import copy
import logging
import pdb
from base_model import BASE
logger = logging.getLogger(__name__)

class SAHR(BASE):

    def __init__(self,
                 n_users,
                 n_items,
                 n_feats,
                 n_user_feats,
                 n_item_feats,
                 std=0.01,
                 embed_dim=32,
                 att_dim=16,
                 N=3,
                 d_model=32,
                 h=4,
                 d_ff=64,
                 d_ff2=[500,64],
                 dropout=0.1,
                 max_count=13,
                 self_attention=True,
                 l2_reg=0,
                 K=5,
                 model='SAHR'
                 ):
        BASE.__init__(self, n_users=n_users, n_items=n_items, n_feats=n_feats,
                      n_user_feats=n_user_feats, n_item_feats=n_item_feats,
                      std=std, embed_dim=embed_dim, att_dim=att_dim, N=N,
                      d_model=d_model, h=h, d_ff=d_ff, d_ff2=d_ff2, dropout=dropout,
                      max_count=max_count, self_attention=self_attention, l2_reg=l2_reg, K=K,
                      model=model)
        #self.user_att = nn.Linear(self.embed_dim, self.n_item_feats)
        #self.mlp = ConcatFeedForward((self.n_user_feats + self.n_item_feats + 2) * self.d_model, 200, 1, dropout)
    def get_feats(self, users, feats):
        user_att = self.user_att(users)
        user_att = F.softmax(user_att, dim=-1).unsqueeze(-1)
        feats = feats * user_att
        return feats

    def get_cb_vector(self, users, items, user_feats, user_feats_mask, item_feats, item_feats_mask, valid):
        if self.n_user_feats > 0:
            mask = torch.cat((user_feats_mask, item_feats_mask), 2)
            #mask = None
            feats = self.encoder(torch.cat((user_feats, item_feats), 1), mask, valid)
            user_feats = feats[:,:self.n_user_feats,:]
            user_feats = self.user_mlp(user_feats)
        else:
            feats = self.encoder(item_feats, item_feats_mask, valid)
            user_feats = None
        item_feats = feats[:,self.n_user_feats:,:]
        item_feats = self.item_mlp(item_feats)
        return user_feats, item_feats

    def get_cb_score(self, users, items, user_feats, item_feats):
        feats = self.encoder(torch.cat((users.unsqueeze(1), items.unsqueeze(1), user_feats, item_feats), 1))
        scores = self.mlp(feats)
        #item_feats = self.item_mlp(item_feats)
        return scores

    def fm_score(self, users, items, user_feats, item_feats):
        #feats = torch.cat((users.unsqueeze(1), items.unsqueeze(1), item_feats), 1)
        feats = torch.cat((users.unsqueeze(1), items.unsqueeze(1), user_feats, item_feats), 1)
        term_1 = torch.sum(feats, 1) ** 2
        term_2 = torch.sum(feats ** 2, 1)
        #fm_vec = 0.5 * (term_1 - term_2)
        #score = self.nfm(fm_vec)
        score = 0.5 * torch.sum(term_1 - term_2, 1, keepdim=True)
        return score

    def forward(self, users, items, user_feats, user_feats_mask, item_feats, item_feats_mask, targets, valid=False):
        """
         users : (N, 1)
         feats : (N, K)
         mask : (N, K)
         Let
         N = batch size,
         K = maximum number of features
        :return: the RMSE loss
        """
        item_bias = self.item_bias(items)
        items = self.item_embeddings(items)
        user_bias = self.user_bias(users)
        users = self.user_embeddings(users)
        if user_feats.shape[1] > 0:
            user_feats = self.feat_embeddings(user_feats)
        item_feats = self.feat_embeddings(item_feats)
        # bias
        bias_score = user_bias + item_bias + self.global_bias.weight
        # cf
        cf_score = torch.sum(users * items, 1).unsqueeze(-1)
        # cb
        cb_users, cb_items = self.get_cb_vector(users, items, user_feats, user_feats_mask, item_feats, item_feats_mask, valid)
        if cb_users is not None:
            cb_score = torch.sum(cb_users * cb_items, 1).unsqueeze(-1)
        else:
            cb_score = torch.sum(users * cb_items, 1).unsqueeze(-1)
        # attention score
        if cb_users is not None:
            cf_users, cb_users = self.get_att_vector(users, cb_users)
            users = cf_users + cb_users
        cf_items, cb_items = self.get_att_vector(items, cb_items)
        items = cf_items + cb_items

        score = torch.sum(users * items, 1).unsqueeze(-1)
        # total loss
        loss = self.loss(cf_score + bias_score, target=targets, loss_type='rmse')
        loss += self.loss(cb_score + bias_score, target=targets, loss_type='rmse')
        loss += self.loss(score + bias_score, target=targets, loss_type='rmse')
        if self.l2_reg > 0:
            l2_reg = self.l2_reg * torch.sum(self.feat_embeddings.weight ** 2)
            l2_reg += self.l2_reg * torch.sum(self.user_embeddings.weight ** 2)
            l2_reg += self.l2_reg * torch.sum(self.item_embeddings.weight ** 2)
            loss += l2_reg

        return loss

    def predict(self, users, items, user_feats, user_feats_mask, item_feats, item_feats_mask, valid=True):
        """
         users : (N, 1)
         pos_feats : (N, K)
         pos_mask : (N, K)
         neg_feats : (N, K)
         neg_mask : (N, K)
         prev_feats : (N, K)
         prev_mask : (N, K)
         Let
         N = batch size,
         K = maximum number of features
        :return: the BPR loss
        """

        with torch.no_grad():
            item_bias = self.item_bias(items)
            items = self.item_embeddings(items)
            user_bias = self.user_bias(users)
            users = self.user_embeddings(users)
            if user_feats.shape[1] > 0:
                user_feats = self.feat_embeddings(user_feats)
            item_feats = self.feat_embeddings(item_feats)
            bias_score = user_bias + item_bias + self.global_bias.weight
            # cb
            cb_users, cb_items = self.get_cb_vector(users, items, user_feats, user_feats_mask, item_feats, item_feats_mask, valid)
            if cb_users is not None:
                cf_users, cb_users = self.get_att_vector(users, cb_users)
                users = cf_users + cb_users
            cf_items, cb_items = self.get_att_vector(items, cb_items)
            items = cf_items + cb_items
            score = torch.sum(users * items, 1).unsqueeze(-1)
            # total score
            total_score = score + bias_score
        return total_score



