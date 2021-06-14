import math
import torch
import utils


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Ranker:
    def __init__(self, device):
        self.data_emb = None
        self.device = device
        return

    def update_emb(self, img_fts, batch_size, model):

        self.data_emb = []
        num_data = len(img_fts['asins'])
        num_batch = math.floor(num_data / batch_size)

        def append_emb(first, last):
            batch_ids = torch.tensor(
                [j for j in range(first, last)], dtype=torch.long)

            feat = utils.get_image_batch(img_fts, batch_ids)

            feat = feat.to(device)
            with torch.no_grad():
                feat = model.encode_image(feat)
            self.data_emb.append(feat)

        for i in range(num_batch):
            append_emb(i*batch_size, (i+1)*batch_size)

        if num_batch * batch_size < num_data:
            append_emb(num_batch * batch_size, num_data)

        self.data_emb = torch.cat(self.data_emb, dim=0)

        return

    def nearest_neighbors(self, inputs):
        neighbors = []
        for i in range(inputs.size(0)):
            [_, neighbor] = ((self.data_emb - inputs[i]).pow(2)
                             .sum(dim=1).min(dim=0))

            neighbors.append(neighbor)
        return torch.tensor(neighbors).to(
            device=self.device, dtype=torch.long)

    def compute_rank(self, inputs, target_ids):
        rankings = []
        for i in range(inputs.size(0)):
            distances = (self.data_emb - inputs[i]).pow(2).sum(dim=1)
            ranking = (distances < distances[target_ids[i]]).float().sum(dim=0)
            rankings.append(ranking)
        return torch.tensor(rankings).to(device=self.device, dtype=torch.float)

