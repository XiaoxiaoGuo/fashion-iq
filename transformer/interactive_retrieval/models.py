import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_
from torch.autograd import Variable
import copy
import numpy as np


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len=200, dropout=None):
        super().__init__()
        self.d_model = d_model
        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
        # create constant 'pe' matrix with values dependant on
        # pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                    math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = \
                    math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        # add constant to embedding
        seq_len = x.size(1)
        pe = Variable(self.pe[:, :seq_len], requires_grad=False)
        if x.is_cuda:
            pe.cuda()
        x = x + pe
        if self.dropout:
            return self.dropout(x)
        return x


class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6, calibrate=True):
        super().__init__()

        self.size = d_model
        self.calibrate = calibrate
        # create two learnable parameters to calibrate normalisation
        if self.calibrate:
            self.alpha = nn.Parameter(torch.ones(self.size))
            self.bias = nn.Parameter(torch.zeros(self.size))

        self.eps = eps

    def forward(self, x):
        if self.calibrate:
            norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
               / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        else:
            norm = (x - x.mean(dim=-1, keepdim=True)) \
                   / (x.std(dim=-1, keepdim=True) + self.eps)
        return norm


def attention(q, k, v, d_k, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)

    scores = F.softmax(scores, dim=-1)

    if dropout is not None:
        scores = dropout(scores)

    output = torch.matmul(scores, v)
    return output


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)

        # perform linear operation and split into N heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * N * sl * d_model
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, mask, self.dropout)
        # concatenate heads and put through final linear layer
        concat = (scores.transpose(1, 2).contiguous()
                  .view(bs, -1, self.d_model))
        output = self.out(concat)

        return output


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=1024):
        super().__init__()
        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        # self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = F.relu(self.linear_1(x))
        x = self.linear_2(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=None):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model)

    def forward(self, x, mask=None):
        x2 = self.norm_1(x)
        x = x + self.attn(x2, x2, x2, mask)
        x2 = self.norm_2(x)
        x = x + self.ff(x2)
        return x


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class Encoder(nn.Module):
    def __init__(self, d_model, N_layers, heads=8, dropout=None):
        super().__init__()
        self.N_layers = N_layers
        self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = get_clones(
            EncoderLayer(d_model, heads, dropout), N_layers)
        self.norm = Norm(d_model)

    def forward(self, x):
        x = self.pe(x)
        for i in range(self.N_layers):
            x = self.layers[i](x)
        return self.norm(x)


class RetrieverTransformer(nn.Module):
    def __init__(self, vocab_size, word_mat, img_dim, hist_dim,
                 layer_num, attribute_num):
        super().__init__()

        # text part
        glove_dim = 300
        # glove_dim = len(word_mat[0])

        self.word_emb = nn.Embedding(vocab_size+1, glove_dim)
        print('[INFO] Load glove embedding ({})'.format(glove_dim))
        self.word_emb.weight.data.copy_(
            torch.from_numpy(np.asarray(word_mat)))
        self.word_emb.weight.requires_grad = False
        self.fix_word_emb = True

        self.text_linear = nn.Linear(glove_dim, hist_dim)
        self.text_norm = Norm(hist_dim)

        # image part
        self.img_emb = nn.Linear(img_dim, hist_dim, bias=False)
        self.img_norm = Norm(hist_dim)

        # attribute part
        self.attr_emb = nn.Embedding(attribute_num, hist_dim)
        self.attr_emb.scale_grad_by_freq = True
        self.attr_norm = Norm(hist_dim)

        # response encoder
        self.tran = Encoder(
            d_model=hist_dim, N_layers=layer_num)
        self.layer_num = layer_num

        # output part
        self.out_linear = nn.Linear(hist_dim, hist_dim, bias=True)

        self.vocab_size = vocab_size+1
        self.hist_vectors = []

        self.sp_token = nn.Parameter(
            torch.zeros(size=(1, hist_dim)), requires_grad=False)

        self.hist_dim = hist_dim
        self.init_parameters()

    def init_parameters(self):
        return

    def init_hist(self):
        self.hist_vectors.clear()
        return

    def encode_image(self, images):
        return self.img_norm(self.img_emb(images))

    def get_sp_emb(self, batch_size):

        with torch.no_grad():
            sp_emb = self.sp_token.expand(
                size=(batch_size, 1, self.hist_dim))

        sp_emb = self.text_norm(sp_emb)
        return sp_emb

    # input:
    #   text: B x L x V
    #   image: B x Hi
    #   hist: B x Hh
    def forward(self, text, image, attribute):
        # special token
        sp_emb = self.get_sp_emb(text.size(0))
        self.hist_vectors.append(sp_emb)

        # text part
        # B x L x H
        with torch.no_grad():
            text_emb = self.word_emb(text)

        text_emb = self.text_linear(text_emb)
        text_emb = self.text_norm(text_emb)
        self.hist_vectors.append(text_emb)

        # attribute part
        attr_emb = self.attr_emb(attribute)
        attr_emb = self.attr_norm(attr_emb)
        self.hist_vectors.append(attr_emb)

        # image part
        # B x 1 x H
        img_emb = self.encode_image(image).unsqueeze(dim=1)
        self.hist_vectors.append(img_emb)

        full_input = torch.cat(self.hist_vectors, dim=1)
        outs = self.tran(full_input)
        outs = self.out_linear(F.relu(outs.mean(dim=1)))

        return outs

    def convert_onehot(self, text):
        B, L = text.size(0), text.size(1)
        onehot = torch.zeros(B * L, self.vocab_size).to(text.device)
        onehot.scatter_(1, text.view(-1, 1), 1)
        onehot = onehot.view(B, L, self.vocab_size)
        return onehot

