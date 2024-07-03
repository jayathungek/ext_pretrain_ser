import copy, math
import torch
from torch import nn as nn
import torch.nn.functional as F

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Encoder(nn.Module):
    def __init__(self, size, h, feed_forward, dropout, N):
        super(Encoder, self).__init__()
        layer = EncoderLayer(size, h, feed_forward, dropout)
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class EncoderLayer(nn.Module):
    def __init__(self, size, h, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadedAttention(h, size)
        self.feed_forward = PositionwiseFeedForward(size, feed_forward, dropout)
        self.size = size

        self.norm1 = LayerNorm(size)
        self.norm2 = LayerNorm(size)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x1 = self.norm1(x)
        x = x + self.drop1(self.self_attn(x1, x1, x1, mask))
        x2 = self.norm2(x)
        x = x + self.drop2(self.feed_forward(x2))
        return x

def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1).unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class Decoder(nn.Module):
    def __init__(self, size, h, feed_forward, dropout):
        super(Decoder, self).__init__()
        self.size = size
        self.audio_attn = MultiHeadedAttention(h, size)
        self.video_attn = MultiHeadedAttention(h, size)
        self.fused_attn = MultiHeadedAttention(h, size*2)
        self.feed_forward = PositionwiseFeedForward(size*2, feed_forward, dropout)
        self.norm1 = LayerNorm(size)
        self.norm2 = LayerNorm(size)
        self.norm3 = LayerNorm(size*2)
        self.norm4 = LayerNorm(size*2)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.drop3 = nn.Dropout(dropout)
        self.drop4 = nn.Dropout(dropout)
 
    def forward(self, a, v, mask=None):
        a1 = a + self.drop1(self.audio_attn(self.norm1(a), v, v, mask))
        v1 = v + self.drop2(self.video_attn(self.norm2(v), a, a, mask))
        f = torch.cat((a1, v1), dim=2)
        f1 = self.norm3(f)
        f = f + self.drop3(self.fused_attn(f1, f1, f1, mask))
        f2 = self.norm4(f)
        f = f + self.drop4(self.feed_forward(f2))
        return f

def test():
    num_heads = 1
    num_layers = 1
    dim_ff = 1024
    dropout = 0.2
    dim_feature = 128

    batch_size = 2
    sequence_leng = 100

    lengths = torch.tensor([100,50]).long()
    mask = torch.arange(max(lengths))[None, :] < lengths[:, None]
    mask = mask.long()

    audio_encoder = Encoder(dim_feature, num_heads, dim_ff, dropout, num_layers)
    video_encoder = Encoder(dim_feature, num_heads, dim_ff, dropout, num_layers)
    fused_decoder = Decoder(dim_feature, num_heads, dim_ff, dropout)

    a = torch.rand(batch_size, sequence_leng, dim_feature)
    v = torch.rand(batch_size, sequence_leng, dim_feature)

    a = audio_encoder(a, mask)
    v = video_encoder(v, mask)
    y = fused_decoder(a, v, mask)
    print(y.shape)

if __name__=="__main__":
    test()