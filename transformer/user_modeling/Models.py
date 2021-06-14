import torch
import torch.nn as nn 
from Layers import EncoderLayer, DecoderLayer
from Embed import Embedder, PositionalEncoder
from Sublayers import FeedForward, MultiHeadAttention, Norm
import copy
import torchvision.models as models
import numpy as np
from torch.autograd import Variable
from efficientnet_pytorch import EfficientNet

Constants_PAD = 0

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class Encoder(nn.Module):
    def __init__(self, d_model, N_layers, heads, dropout):
        super().__init__()
        self.N_layers = N_layers
        # self.embed = Embedder(vocab_size, d_model)
        # self.pe = PositionalEncoder(d_model, dropout=dropout)
        # self.attn = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.layers = get_clones(EncoderLayer(d_model, heads, dropout), N_layers)
        self.norm = Norm(d_model)
        # self.norm_1 = Norm(d_model)
        # self.norm_2 = Norm(d_model)
        # self.dropout= nn.Dropout(dropout)
    def forward(self, x):
        # x = self.embed(src)
        # x = self.pe(x)
        # x = src
        for i in range(self.N_layers):
            x = self.layers[i](x)
        return self.norm(x)

    # def forward(self, image1, image2):
    #     image1 = self.norm_1(image1)
    #     image2 = self.norm_2(image2)
    #     x = self.dropout(self.attn(image1,image2,image2))
    #     for i in range(self.N_layers):
    #         x = self.layers[i](x)
    #     return self.norm(x)
    
class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, N_layers, heads, dropout):
        super().__init__()
        self.N_layers = N_layers
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = get_clones(DecoderLayer(d_model, heads, dropout), N_layers)
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

        if model_name[:6] == 'resnet':
            print("cnn model name: ", model_name)
            if model_name == "resnet101":

                model = models.resnet101(pretrained=True)
            elif model_name == "resnet18":
                model = models.resnet18(pretrained=True)

            if pretrained_model:
                print("cnn initialed from pretrained_model")
                ckpt = torch.load(pretrained_model, map_location='cpu')
                if "model_state" in ckpt:
                    model.load_state_dict(ckpt["model_state"])
                else:
                    model.load_state_dict(ckpt)

            modules = list(model.children())[:-1]  # delete the last fc layer.
            self.model = nn.Sequential(*modules)

            for param in self.model.parameters():
                param.requires_grad = False
            self.linear = nn.Linear(model.fc.in_features, d_model)
            self.bn = nn.BatchNorm1d(model.fc.in_features, momentum=0.01)


        elif model_name[:12] == "efficientnet":
            self.model = EfficientNet.from_pretrained(model_name)
            if pretrained_model:
                ckpt = torch.load(pretrained_model, map_location='cpu')
                if "model_state" in ckpt:
                    self.model.load_state_dict(ckpt["model_state"])
                else:
                    self.model.load_state_dict(ckpt)

            for param in self.model.parameters():
                param.requires_grad = False

            self.linear = nn.Linear(self.model._fc.in_features, d_model)
            self.bn = nn.BatchNorm1d(self.model._fc.in_features, momentum=0.01)


    def get_trainable_parameters(self):
        return list(self.linear.parameters()) + list(self.bn.parameters()) 

   

    def forward(self, image):
        with torch.no_grad():
            if self.model_name[:12] == "efficientnet":
                img_ft = self.model.extract_features(image)
                img_ft = self.model._avg_pooling(img_ft)
                img_ft = img_ft.flatten(start_dim=1)
                img_ft = self.model._dropout(img_ft)
            else:
                img_ft = self.model(image)

        img_ft = self.linear(self.bn(img_ft.reshape(img_ft.size(0), img_ft.size(1), -1)).transpose(1,2)) # (batch_size, d, d, f) -> (batch_size, d^2, f)
                                                                                                       #(batch_size, f, 1, 1) -> (batch_size, 1, f)
        return img_ft#.transpose(0,1)#(1, batch_size,f)

class Joint_Encoding:
    def __init__(self, joint_encoding_function):
        # super().__init__()
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
        self.embed = nn.Linear(attribute_vocab_size, d_model)#Embedder(attribute_vocab_size, d_model)
        self.norm = nn.BatchNorm1d(attribute_vocab_size, momentum=0.01)

    def forward(self, attribute):
        attribute = self.norm(attribute)
        attribute = self.embed(attribute)
        return attribute

class Transformer(nn.Module):
    def __init__(self, trg_vocab, d_model, N, heads, dropout, cnn_model_name, \
        joint_encoding_function, attribute_vocab_size=1000, cnn_pretrained_model=None, add_attribute=False):
        super().__init__()
        self.add_attribute = add_attribute
        self.cnn1 = CNN_Embedding(d_model, cnn_model_name, cnn_pretrained_model)
        self.cnn2 = CNN_Embedding(d_model, cnn_model_name, cnn_pretrained_model)
        # self.bn = nn.BatchNorm1d(d_model, momentum=0.01)
        if self.add_attribute:
            self.attribute_embedding = Attribute_Embedding(d_model, attribute_vocab_size)
            # self.attribute_embedding2 = Attribute_Embedding(d_model, attribute_vocab_size)
        self.joint_encoding = Joint_Encoding(joint_encoding_function)
        self.encoder = Encoder(d_model, N, heads, dropout)
        self.decoder = Decoder(trg_vocab, d_model, N, heads, dropout)
        self.out = nn.Linear(d_model, trg_vocab)

    def get_trainable_parameters(self):
        if not self.add_attribute:
            return self.cnn1.get_trainable_parameters() \
                + self.cnn2.get_trainable_parameters() \
                +  list(self.encoder.parameters()) \
                + list(self.decoder.parameters()) \
                + list(self.out.parameters())
        else:
            return self.cnn1.get_trainable_parameters() \
                    + self.cnn2.get_trainable_parameters() \
                    +  list(self.encoder.parameters()) \
                    + list(self.decoder.parameters()) \
                    + list(self.out.parameters()) \
                    + list(self.attribute_embedding.parameters())
                    # + list(self.bn.parameters()) \
                    # + list(self.attribute_embedding1.parameters()) \
                    # + list(self.attribute_embedding2.parameters())

    # def get_parameters_to_initial(self):
    #     return list(self.encoder.parameters()) \
    #             + list(self.decoder.parameters()) \
    #             + list(self.out.parameters()) \
    #             + list(self.attribute_embedding1.parameters()) \
    #             + list(self.attribute_embedding2.parameters())


    def forward(self, image0, image1, trg, trg_mask, image0_attribute, image1_attribute):
        #image1, image2 = image2, image1

        image0 = self.cnn1(image0)

        image1 = self.cnn2(image1)

        if self.add_attribute:
            attribute = self.attribute_embedding(image0_attribute - image1_attribute).unsqueeze(1)
            # attribute = self.norm(attribute)

            # image0_attribute = self.attribute_embedding1(image0_attribute)

            # image1_attribute = self.attribute_embedding2(image1_attribute)

            # image0 = torch.cat((image0, image0_attribute), 1)
            # image1 = torch.cat((image1, image1_attribute), 1)

            #joint_encoding = self.joint_encoding(torch.cat((image0, image0_attribute),1), torch.cat((image1,image1_attribute),1))
            joint_encoding = self.joint_encoding(image0, image1)
            joint_encoding = torch.cat((joint_encoding, attribute), 1)
            # joint_encoding = self.bn(joint_encoding.transpose(1,2)).transpose(1,2)
            # joint_encoding = torch.cat((joint_encoding, image0_attribute), 1)

            # joint_encoding = torch.cat((joint_encoding, image1_attribute), 1)
        else:
            joint_encoding = self.joint_encoding(image0, image1)

        joint_encoding = self.encoder(joint_encoding)
        #print("DECODER")
        output = self.decoder(trg, joint_encoding, trg_mask)

        output = self.out(output)

        return output

def get_model(opt, load_weights=False):
    
    
       
    if load_weights:

        device = torch.device('cuda' if opt.cuda else 'cpu')

        checkpoint = torch.load(opt.pretrained_model + '.chkpt')

        model_opt = checkpoint['settings']

        model = Transformer(model_opt.vocab_size, model_opt.d_model, \
            model_opt.n_layers, model_opt.n_heads, model_opt.dropout, \
            model_opt.cnn_name, model_opt.joint_enc_func, \
            model_opt.attribute_vocab_size, model_opt.cnn_pretrained_model, model_opt.add_attribute,
            )

        model.load_state_dict(checkpoint['model'])
        
        print('[Info] Trained model state loaded from: ', opt.pretrained_model)

        model = model.to(device)

        
    else:
        assert opt.d_model % opt.n_heads == 0

        assert opt.dropout < 1

        model = Transformer(opt.vocab_size, opt.d_model, opt.n_layers, opt.n_heads, opt.dropout, \
            opt.cnn_name, opt.joint_enc_func, opt.attribute_vocab_size, opt.cnn_pretrained_model, \
            opt.add_attribute)

        for p in model.get_trainable_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p) 
    
        model.to(opt.device)
    
    return model

def nopeak_mask(size):
    np_mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8')
    np_mask = Variable(torch.from_numpy(np_mask) == 0)

    return np_mask

def create_masks(trg):
    # src_mask = (src != Constants_PAD.unsqueeze(-2)

    if trg is not None:
        trg_mask = (trg != Constants_PAD).unsqueeze(-2)
        size = trg.size(1) # get seq_len for matrix
        np_mask = nopeak_mask(size).to(trg_mask.device)

        trg_mask = trg_mask & np_mask
        
    else:
        trg_mask = None

    return trg_mask



    
