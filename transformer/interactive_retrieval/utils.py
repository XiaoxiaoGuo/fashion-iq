import os
import json
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
import torch
import tqdm
import math
import numpy as np
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_image_batch(fts, batch_ids):
    image = fts['image'][batch_ids]
    return image


def get_attribute_batch(fts, batch_ids):
    attribute = fts['attribute'][batch_ids]
    return attribute


def extract_features(data_loader, attr_file, attr2idx_file, device, image_model,
                     attribute_topk=8, batch_size=128):
    model = EfficientNet.from_pretrained('efficientnet-b7')
    ckpt = torch.load(image_model, map_location='cpu')
    print('[INFO] Loading weights from {}'.format(image_model))
    if "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
    else:
        model.load_state_dict(ckpt)
    model = model.to(device)
    model = model.eval()

    with open(attr_file, 'r') as f:
        predicted_attr = json.load(f)

    with open(attr2idx_file, 'r') as f:
        attr2idx = json.load(f)

    num_data = len(data_loader)
    num_batch = math.floor(num_data / batch_size)
    asins = []
    image_ft = []
    attributes = []

    def compute_features(data):
        with torch.no_grad():
            outs = model.extract_features(data)
            outs = model._avg_pooling(outs)
            outs = outs.flatten(start_dim=1)
            image_ft_batch = model._dropout(outs)
        return image_ft_batch

    def compute_attribute_idx(asin_batch):
        labels = [predicted_attr[asin[0]]['predict'][:attribute_topk]
                            for asin in asin_batch]
        attribute_idx = [[attr2idx[attr] for attr in label]
                         for label in labels]
        return attribute_idx

    def append_batch(first, last):
        batch_ids = torch.tensor(
            [j for j in range(first, last)],
            dtype=torch.long, device=device)
        [data, meta_info] = data_loader.get_items(batch_ids)
        data = data.to(device)
        image_ft_batch = compute_features(data)

        image_ft.append(image_ft_batch)
        asins.extend(meta_info)
        attribute_idx = compute_attribute_idx(meta_info)
        attributes.extend(attribute_idx)

    for i in tqdm.tqdm(range(num_batch), ascii=True):
        append_batch(i * batch_size, (i + 1) * batch_size)

    if num_batch * batch_size < num_data:
        append_batch(num_batch * batch_size, num_data)

    image_ft = torch.cat(image_ft, dim=0).to('cpu')
    attributes = torch.from_numpy(np.asarray(attributes, dtype=int))
    features = {'asins': asins, 'image': image_ft, 'attribute': attributes}

    return features
















#
# #
# def extract_features(caption_model, data_loader, save_name):
#     # a list of {'asin':[X], 'img_ft': Tensor, 'si_1': Tensor}
#     # keep each separate
#     # load side info models
#     model_path = os.path.join(SI_MODEL.format(args.data_set, 'image'))
#     si_image_encoder = ImageEncoder(1024).to(device)
#     resnet = si_image_encoder.delete_resnet()
#     si_image_encoder.load_state_dict(torch.load(model_path, map_location=device))
#     si_image_encoder.load_resnet(resnet)
#
#     model_path = os.path.join(SI_MODEL.format(args.data_set, 'text'))
#     si_text_decoder = MultiColumnPredictor(1024).to(device)
#     si_text_decoder.load_state_dict(torch.load(model_path, map_location=device))
#     si_image_encoder.eval()
#     si_text_decoder.eval()
#
#     batch_size = 128
#
#     num_data = len(data_loader)
#     num_batch = math.floor(num_data / batch_size)
#     asins = []
#     image_ft = []
#     texture_ft = []
#     fabric_ft = []
#     shape_ft = []
#     part_ft = []
#     style_ft = []
#     for iter in tqdm(range(num_batch), ascii=True):
#         # for i in range(2):
#         batch_ids = torch.LongTensor([i for i in range(iter * batch_size, (iter + 1) * batch_size)])
#         [data, meta_info] = data_loader.get_items(batch_ids)
#         data = data.to(device)
#         with torch.no_grad():
#             image_ft_batch = caption_model['image_encoder'](data)
#             si_ft_batch = si_image_encoder(data)
#             si_ft_batch = si_text_decoder(si_ft_batch)
#
#         image_ft.append(image_ft_batch)
#         texture_ft.append(si_ft_batch['texture'])
#         fabric_ft.append(si_ft_batch['fabric'])
#         shape_ft.append(si_ft_batch['shape'])
#         part_ft.append(si_ft_batch['part'])
#         style_ft.append(si_ft_batch['style'])
#         asins.extend(meta_info)
#
#     if num_batch * batch_size < num_data:
#         batch_ids = torch.LongTensor([i for i in range(num_batch * batch_size, num_data)])
#         [data, meta_info] = data_loader.get_items(batch_ids)
#         data = data.to(device)
#         with torch.no_grad():
#             image_ft_batch = caption_model['image_encoder'](data)
#             si_ft_batch = si_image_encoder(data)
#             si_ft_batch = si_text_decoder(si_ft_batch)
#
#         image_ft.append(image_ft_batch)
#         texture_ft.append(si_ft_batch['texture'])
#         fabric_ft.append(si_ft_batch['fabric'])
#         shape_ft.append(si_ft_batch['shape'])
#         part_ft.append(si_ft_batch['part'])
#         style_ft.append(si_ft_batch['style'])
#         asins.extend(meta_info)
#
#     image_ft = torch.cat(image_ft, dim=0)
#     texture_ft = torch.cat(texture_ft, dim=0)
#     fabric_ft = torch.cat(fabric_ft, dim=0)
#     shape_ft = torch.cat(shape_ft, dim=0)
#     part_ft = torch.cat(part_ft, dim=0)
#     style_ft = torch.cat(style_ft, dim=0)
#     features = {'asins':asins, 'image':image_ft, 'texture':texture_ft,
#                 'fabric':fabric_ft, 'part':part_ft, 'shape':shape_ft,
#                 'style':style_ft}
#
#     torch.save(features, save_name)
#
#     return features