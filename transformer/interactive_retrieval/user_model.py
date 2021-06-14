import torch
import torch.nn as nn
import math
import torch.nn.functional as F

import copy
from torch.autograd import Variable


class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embed(x)


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len=200, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
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
        return self.dropout(x)


class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()

        self.size = d_model

        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))

        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
               / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
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
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
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
        concat = scores.transpose(1, 2).contiguous() \
            .view(bs, -1, self.d_model)
        output = self.out(concat)

        return output


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super().__init__()

        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn(x2, x2, x2, mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x


# build a decoder layer with two multi-head attention layers and
# one feed-forward layer
class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.norm_3 = Norm(d_model)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)

        self.attn_1 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.attn_2 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)

    def forward(self, x, e_outputs, src_mask=None, trg_mask=None):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn_1(x2, x2, x2, trg_mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(
            self.attn_2(x2, e_outputs, e_outputs, src_mask))
        x2 = self.norm_3(x)
        x = x + self.dropout_3(self.ff(x2))
        return x


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class Encoder(nn.Module):
    def __init__(self, d_model, N_layers, heads, dropout):
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


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, N_layers, heads, dropout):
        super().__init__()
        self.N_layers = N_layers
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = get_clones(
            DecoderLayer(d_model, heads, dropout), N_layers)
        self.norm = Norm(d_model)

    def forward(self, trg, e_outputs, trg_mask):
        x = self.embed(trg)
        x = self.pe(x)
        for i in range(self.N_layers):
            x = self.layers[i](x, e_outputs, src_mask=None, trg_mask=trg_mask)
        return self.norm(x)


class CNN_Embedding(nn.Module):
    def __init__(self, d_model, model_name, pretrained_model=None):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super().__init__()

        self.d_model = d_model
        self.model_name = model_name
        print("cnn model name: ", model_name)
        if model_name[:6] == 'resnet':
            if model_name == "resnet101":
                in_features = 2048
            elif model_name == "resnet18":
                in_features = 512
        elif model_name[:12] == "efficientnet":
            if model_name == "efficientnet-b7":
                in_features = 2560
            elif model_name == "efficientnet-b4":
                in_features = 1792

        self.linear = nn.Linear(in_features, d_model)
        self.bn = nn.BatchNorm1d(in_features, momentum=0.01)

    def forward(self, img_ft):
        # (batch_size, d, d, f) -> (batch_size, d^2, f)
        img_ft = self.linear(
            self.bn(img_ft.squeeze(1))).unsqueeze(1)
        return img_ft


class Joint_Encoding:
    def __init__(self, joint_encoding_function):
        if joint_encoding_function == 'addition':
            self.joint_encoding_function = lambda x1, x2 : x1 + x2
        elif joint_encoding_function == 'deduction':
            self.joint_encoding_function = lambda x1, x2 : x1 - x2
        elif joint_encoding_function == 'max':
            self.joint_encoding_function = lambda x1, x2 : torch.max(x1,x2)
        elif joint_encoding_function == 'element_multiplication':
            self.joint_encoding_function = lambda x1, x2 : x1 * x2

    def __call__(self,E1, E2):
        return self.joint_encoding_function(E1, E2)


class Attribute_Embedding(nn.Module):
    def __init__(self, d_model, attribute_vocab_size):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super().__init__()
        self.embed = Embedder(attribute_vocab_size, d_model)

    def forward(self, attribute):
        attribute = self.embed(attribute)
        return attribute


class Transformer(nn.Module):
    def __init__(self, trg_vocab, d_model, N, heads, dropout, cnn_model_name,
                 joint_encoding_function, attribute_vocab_size=1000,
                 cnn_pretrained_model=None, add_attribute=False):
        super().__init__()
        self.add_attribute = add_attribute
        self.cnn1 = CNN_Embedding(d_model, cnn_model_name, cnn_pretrained_model)
        self.cnn2 = CNN_Embedding(d_model, cnn_model_name, cnn_pretrained_model)

        if self.add_attribute:
            self.attribute_embedding1 = Attribute_Embedding(
                d_model, attribute_vocab_size)
            self.attribute_embedding2 = Attribute_Embedding(
                d_model, attribute_vocab_size)
        self.joint_encoding = Joint_Encoding(joint_encoding_function)
        self.encoder = Encoder(d_model, N, heads, dropout)
        self.decoder = Decoder(trg_vocab, d_model, N, heads, dropout)
        self.out = nn.Linear(d_model, trg_vocab)

    def forward(self, image0, image1, trg, trg_mask,
                image0_attribute, image1_attribute):

        image0 = self.cnn1(image0)
        image1 = self.cnn2(image1)

        if self.add_attribute:
            image0_attribute = self.attribute_embedding1(image0_attribute)
            image1_attribute = self.attribute_embedding2(image1_attribute)
            joint_encoding = self.joint_encoding(image0, image1)
            joint_encoding = torch.cat((joint_encoding, image0_attribute), 1)
            joint_encoding = torch.cat((joint_encoding, image1_attribute), 1)
        else:
            joint_encoding = self.joint_encoding(image0, image1)

        joint_encoding = self.encoder(joint_encoding)
        output = self.decoder(trg, joint_encoding, trg_mask)
        output = self.out(output)

        return output


def load_trained_model(model_name):
    checkpoint = torch.load(model_name, map_location='cpu')
    model_opt = checkpoint['settings']

    model = Transformer(model_opt.vocab_size, model_opt.d_model,
                        model_opt.n_layers, model_opt.n_heads,
                        model_opt.dropout, model_opt.cnn_name,
                        model_opt.joint_enc_func,
                        model_opt.attribute_vocab_size,
                        model_opt.cnn_pretrained_model,
                        model_opt.add_attribute)

    model.load_state_dict(checkpoint['model'])
    print('[Info] Trained model state loaded from: ', model_name)
    return model


def create_model(opt):
    assert opt.d_model % opt.n_heads == 0
    assert opt.dropout < 1
    model = Transformer(opt.vocab_size, opt.d_model, opt.n_layers,
                        opt.n_heads, opt.dropout, opt.cnn_name,
                        opt.joint_enc_func, opt.attribute_vocab_size,
                        opt.cnn_pretrained_model, opt.add_attribute)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model
