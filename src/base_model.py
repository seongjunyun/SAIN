import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from Encoder import Encoder, EncoderLayer, MultiHeadedAttention, PositionwiseFeedForward, ConcatFeedForward
import copy
import logging
import pdb
logger = logging.getLogger(__name__)

class BASE(nn.Module):

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
        super(BASE, self).__init__()
        # list of attribute's number
        self.embed_dim = embed_dim
        self.att_dim = att_dim
        self.n_users = n_users
        self.n_items = n_items
        self.n_feats = n_feats
        self.n_user_feats = n_user_feats
        self.n_item_feats = n_item_feats
        self.std = std
        self.N = N
        self.d_model = d_model
        self.h = h
        self.d_ff = d_ff
        self.dropout = dropout
        self.max_count = max_count
        self.model = model
        # SAHR
        if model == 'SAHR':
            self.user_bias = nn.Embedding(self.n_users, 1)
            self.item_bias = nn.Embedding(self.n_items, 1)
            self.global_bias = nn.Linear(1, 1, bias=False)
            self.user_embeddings = nn.Embedding(self.n_users, self.embed_dim)
            self.user_feat_embeddings = nn.Embedding(self.n_users, self.embed_dim)
            self.item_embeddings = nn.Embedding(self.n_items, self.embed_dim)
            self.feat_embeddings = nn.Embedding(self.n_feats, self.d_model)
        # FM
        elif model == 'FM':
            self.linear_embeddings = nn.Embedding(self.n_feats, 1)
            self.feat_embeddings = nn.Embedding(self.n_feats, self.embed_dim)
        
        self.encoder = self.build_self_attention_network(N, d_model, h, d_ff, K, dropout)
        if self.n_user_feats > 0:
            self.user_mlp = ConcatFeedForward((self.n_user_feats) * self.d_model, d_ff2[0], d_ff2[1], dropout)
        if self.n_item_feats > 0:
            self.item_mlp = ConcatFeedForward((self.n_item_feats) * self.d_model, d_ff2[0], d_ff2[1], dropout)
        self.self_attention = self_attention
        self.l2_reg = l2_reg
        # attention layer
        self.att_w = nn.Linear(self.embed_dim, self.att_dim)
        self.att_h = nn.Linear(self.att_dim, 1, bias=False)
        #self.att_h = nn.Linear(self.embed_dim, 1, bias=False)
        # user attention layer
        self.user_att = nn.Linear(self.embed_dim, self.max_count-1) 

        self.counter = 0
        self.init_weight()

    def init_weight(self):
        if self.model == 'SAHR':
            nn.init.zeros_(self.user_bias.weight)
            nn.init.zeros_(self.item_bias.weight)
            nn.init.constant_(self.global_bias.weight, 0.1)
            nn.init.normal_(self.user_embeddings.weight, std=self.std)
            nn.init.normal_(self.item_embeddings.weight, std=self.std)
            nn.init.normal_(self.feat_embeddings.weight, std=self.std)
            nn.init.normal_(self.att_w.weight, std=self.std)
            nn.init.normal_(self.att_h.weight, std=self.std)
            nn.init.normal_(self.encoder.f_weight, std=self.std)
            if self.n_user_feats > 0:
                nn.init.normal_(self.user_mlp.w_3.weight, std=self.std)
            nn.init.normal_(self.item_mlp.w_3.weight, std=self.std)

    def build_self_attention_network(self, N, d_model, h, d_ff, K, dropout):
        c = copy.deepcopy
        attn = MultiHeadedAttention(h, d_model, self.embed_dim, K, dropout)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        encoder = Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N)
        return encoder

    def cf_score(self, users, items):
        score = torch.sum(users * items, 1)
        return score.unsqueeze(-1)

    def get_cb_vector(self, users, items, user_feats, item_feats):
        user_feats = self.user_mlp(user_feats)
        item_feats = self.item_mlp(item_feats)
        return user_feats, item_feats

    def get_att_vector(self, cf, cb):
        #att_cf = torch.exp(self.att_h(cf))
        #att_cf = torch.exp(self.att_h(F.relu(self.att_w(cf))))
        att_cf = torch.exp(self.att_h(self.att_w(cf)))
        #att_cb = torch.exp(self.att_h(cb))
        #att_cb = torch.exp(self.att_h(F.relu(self.att_w(cb))))
        att_cb = torch.exp(self.att_h(self.att_w(cb)))
        att_cf = att_cf / (att_cf + att_cb)
        att_cb = 1 - att_cf
        self.counter += 1
        #if self.counter % 1240 == 0:
        #    pdb.set_trace()
        return att_cf * cf, att_cb * cb

    def loss(self, pos_score, neg_score=None, target=None, loss_type='rmse'):
        if loss_type == 'rmse':
            target = target.unsqueeze(-1).type(torch.cuda.FloatTensor)
            loss = torch.mean((pos_score - target) ** 2)
            loss = loss ** 0.5
        return loss

    def forward(self, users, items, user_feats, item_feats, targets):
        """
         users : (N, 1)
         feats : (N, K)
         mask : (N, K)
         Let
         N = batch size,
         K = maximum number of features
        :return: the MSE loss
        """
        item_bias = self.item_bias(items)
        items = self.item_embeddings(items)
        user_bias = self.user_bias(users)
        users = self.user_embeddings(users)
        user_feats = self.feat_embeddings(user_feats)
        item_feats = self.feat_embeddings(item_feats)
        # cf
        cf_score = torch.sum(users * items, 1).unsqueeze(-1)
        # cb
        cb_users, cb_items = self.get_cb_vector(users, items, user_feats, item_feats)
        cb_score = torch.sum(cb_users * cb_items, 1).unsqueeze(-1)
        # bias
        bias_score = user_bias + item_bias + self.global_bias.weight
        # attention score
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

    def predict(self, users, items, user_feats, item_feats):
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
            user_feats = self.feat_embeddings(user_feats)
            item_feats = self.feat_embeddings(item_feats)
            # cb
            cb_users, cb_items = self.get_cb_vector(users, items, user_feats, item_feats)
            # bias
            bias_score = user_bias + item_bias + self.global_bias.weight
            # attention score
            cf_users, cb_users = self.get_att_vector(users, cb_users)
            users = cf_users + cb_users
            cf_items, cb_items = self.get_att_vector(items, cb_items)
            items = cf_items + cb_items
            score = torch.sum(users * items, 1).unsqueeze(-1)
            # total score
            total_score = score + bias_score

        return total_score



