import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class DummyImageEncoder(nn.Module):
    def __init__(self, embed_size):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(DummyImageEncoder, self).__init__()
        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]  # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(resnet.fc.in_features, momentum=0.01)

    def get_trainable_parameters(self):
        return list(self.bn.parameters()) + list(self.linear.parameters())

    def load_resnet(self, resnet=None):
        if resnet is None:
            resnet = models.resnet152(pretrained=True)
            modules = list(resnet.children())[:-1]  # delete the last fc layer.
            self.resnet = nn.Sequential(*modules)
            self.resnet_in_features = resnet.fc.in_features
        else:
            self.resnet = resnet
        return

    def delete_resnet(self):
        resnet = self.resnet
        self.resnet = None
        return resnet

    def forward(self, image):
        with torch.no_grad():
            img_ft = self.resnet(image)

        out = self.linear(self.bn(img_ft.reshape(img_ft.size(0), -1)))
        return out


class DummyCaptionEncoder(nn.Module):
    def __init__(self, vocab_size, vocab_embed_size, embed_size):
        super(DummyCaptionEncoder, self).__init__()
        self.out_linear = nn.Linear(embed_size, embed_size, bias=False)
        self.rnn = nn.GRU(vocab_embed_size, embed_size)
        self.embed = nn.Embedding(vocab_size, vocab_embed_size)

    def forward(self, input, lengths):
        input = self.embed(input)
        lengths = torch.LongTensor(lengths)
        [_, sort_ids] = torch.sort(lengths, descending=True)
        sorted_input = input[sort_ids]
        sorted_length = lengths[sort_ids]
        reverse_sort_ids = sort_ids.clone()
        for i in range(sort_ids.size(0)):
            reverse_sort_ids[sort_ids[i]] = i
        packed = pack_padded_sequence(sorted_input, sorted_length, batch_first=True)
        output, _ = self.rnn(packed)
        padded, output_length = torch.nn.utils.rnn.pad_packed_sequence(output)
        output = [padded[output_length[i]-1, i, :] for i in range(len(output_length))]
        output = torch.stack([output[reverse_sort_ids[i]] for i in range(len(output))], dim=0)
        output = self.out_linear(output)
        return output

    def get_trainable_parameters(self):
        return list(self.parameters())

#
# model = DummyCaptionEncoder(100, 64, 10)
#
# x1 = [
#     [45, 4, 7, 9, 2, 0, 0],
#     [11, 2, 3, 4, 5, 6, 7],
#     [99, 98, 97, 96, 7, 8, 0],
#     [89, 87, 86, 2, 0, 0, 0]
#     ]
# len1 = [5, 2, 3, 2]
# x1 = torch.tensor(x1)
# y1 = model(x1, len1)
#
# x2 = [
#     [56, 56, 3, 0, 0, 0, 0],
#     [89, 87, 86, 1, 0, 0, 0],
#     [1, 36, 4, 7, 8, 4, 0],
#     [99, 98, 97, 96, 4, 0, 0]
#     ]
# len2 = [2, 2, 5, 3]
# x2 = torch.tensor(x2)
# y2 = model(x2, len2)
#
# print('max dif 1', (y1[3,:] - y2[1,:]).max(), (y1[3,:] - y2[1,:]).min())
# print('max dif 2', (y1[2,:] - y2[3,:]).max(), (y1[2,:] - y2[3,:]).min())