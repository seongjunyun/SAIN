import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
from allennlp.nn.util import masked_softmax
import pdb
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def attention(query, key, value, K, mask=None, dropout=None, f_weight=None, valid=False):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
        #scores = masked_softmax(scores, mask)#scores.masked_fill(mask == 0, -1e9)
    #if scores.shape[2] == 3:
    #f2 = f_weight * 1
    #for i in range(f2.shape[-1]):
    #    f2[:,:,i] = f2[:,:,i].triu() + f2[:,:,i].triu().t()  
    #f2 = f2.unsqueeze(0).unsqueeze(0)
    #f_weight = f_weight.unsqueeze(0).unsqueeze(0)
    #temp = torch.matmul(query.unsqueeze(-2), f2.transpose(-2, -1)).squeeze(3)
    #scores += temp \
    #        / math.sqrt(d_k)
    if scores.shape[2] > K and valid is False:
        topk_mask = torch.zeros_like(scores)
        topk_mask = topk_mask.scatter_(-1, torch.topk(scores, K)[1], 1)
        scores = scores.masked_fill(topk_mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    # p_attn = p_attn * mask.permute(0,1,3,2)
    if dropout is not None:
        p_attn = dropout(p_attn) 

    return torch.matmul(p_attn, value), p_attn

class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        self.f_weight = nn.Parameter(torch.ones((23, 23, 8)))
    def forward(self, x, mask=None, valid=False):
        "Pass the input (and mask) through each layer in turn."
        for i, layer in enumerate(self.layers):
            x = layer(x, mask, self.f_weight, valid, idx=i)
        # return self.norm(x)
        return x
    
class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return self.norm(F.relu(x + self.dropout(sublayer(x))))
        #return F.relu(x + self.dropout(sublayer(x)))

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size
    def forward(self, x, mask, f_weight, valid, idx=0):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask, idx, f_weight, valid))
        #return self.sublayer[1](x, self.feed_forward)
        return x


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, embed_dim, K, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU()), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.K = K
        
    def forward(self, query, key, value, mask=None, idx=0, f_weight=None, valid=False):
        "Implements Figure 2"
        
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, self.K, mask=mask, 
                                 dropout=self.dropout, f_weight=f_weight, valid=valid)

        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)

        
        #return self.linears[-1](x)
        return x
    
class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1, h=2):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
    
class ConcatFeedForward(nn.Module):

    def __init__(self, d_model, d_ff1, d_ff2, dropout=0.1):
        super(ConcatFeedForward, self).__init__()
        self.d_model = d_model
        self.w_1 = nn.Linear(d_model, d_ff1)
        self.w_2 = nn.Linear(d_ff1, d_ff2)
        self.w_3 = nn.Linear(d_model, d_ff2)
        #self.w_3 = nn.Linear(d_ff2, 1)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        x = x.view(-1, self.d_model)
        return self.dropout(F.relu(self.w_3(x)))
        #return self.dropout(self.w_2(self.dropout(F.relu(self.w_1(x)))))
        #return self.w_3(self.dropout(F.relu(self.w_2(self.dropout(F.relu(self.w_1(x)))))))
