import os
import torch
import shutil
import json
import math
from PIL import Image
from joblib import Parallel, delayed

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.mkdir(path)
    print('Experiment dir : {}'.format(path))
    if scripts_to_save is not None:
        if not os.path.exists(os.path.join(path, 'scripts')):
            os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)
    return


class Ranker():
    def __init__(self, root, image_split_file, transform=None, num_workers=16):
        self.num_workers = num_workers
        self.root = root
        with open(image_split_file, 'r') as f:
            data = json.load(f)
        self.data = data
        self.ids = range(len(self.data))
        self.transform = transform
        return

    def get_item(self, index):
        data = self.data
        id = self.ids[index]
        img_name = data[id] + '.jpg'
        image = Image.open(os.path.join(self.root, img_name)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, data[id]

    def get_items(self, indexes):
        items = Parallel(n_jobs=self.num_workers)(
            delayed(self.get_item)(
                i) for i in indexes)
        images, meta_info = zip(*items)
        images = torch.stack(images, dim=0)
        return images, meta_info

    def update_emb(self, image_encoder, batch_size=64):
        data_emb = []
        data_asin = []
        num_data = len(self.data)
        num_batch = math.floor(num_data / batch_size)
        print('updating emb')
        for i in range(num_batch):
            batch_ids = torch.LongTensor([i for i in range(i * batch_size, (i + 1) * batch_size)])
            images, asins = self.get_items(batch_ids)
            images = images.to(device)
            with torch.no_grad():
                feat = image_encoder(images)
            data_emb.append(feat)
            data_asin.extend(asins)

        if num_batch * batch_size < num_data:
            batch_ids = torch.LongTensor([i for i in range(num_batch * batch_size, num_data)])
            images, asins = self.get_items(batch_ids)
            images = images.to(device)
            with torch.no_grad():
                feat = image_encoder(images)
            data_emb.append(feat)
            data_asin.extend(asins)

        self.data_emb = torch.cat(data_emb, dim=0)
        self.data_asin = data_asin
        print('emb updated')
        return

    def compute_rank(self, inputs, target_ids):
        rankings = []
        for i in range(inputs.size(0)):
            distances = (self.data_emb - inputs[i]).pow(2).sum(dim=1)
            ranking = (distances < distances[self.data_asin.index(target_ids[i])]).sum(dim=0)
            rankings.append(ranking)
        return torch.FloatTensor(rankings).to(device)

    def get_nearest_neighbors(self, inputs, topK=50):
        neighbors = []
        for i in range(inputs.size(0)):
            [_, neighbor] = (self.data_emb - inputs[i]).pow(2).sum(dim=1).topk(dim=0, k=topK, largest=False, sorted=True)
            neighbors.append(neighbor)
        return torch.stack(neighbors, dim=0).to(device)