import torch
import os
from PIL import Image
from joblib import Parallel, delayed
import json


class Dataset():
    def __init__(self, root, data_file_name, transform=None, num_workers=4):
        """Set the path for images, captions and vocabulary wrapper.
        
        Args:
            root: image directory.
            data: index file name
            vocab: vocabulary wrapper.
            transform: image transformer.
        """
        self.num_workers = num_workers
        self.root = root
        with open(data_file_name, 'r') as f:
            data = json.load(f)
        self.data = data
        self.ids = range(len(self.data))
        self.transform = transform

    def get_item(self, index):
        """Returns one data pair (image and caption)."""
        data = self.data
        id = self.ids[index]

        img_name = data[id] + '.jpg'

        image = Image.open(os.path.join(self.root, img_name)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        return image, [data[id]]

    def get_items(self, indexes):
        items = Parallel(n_jobs=self.num_workers)(
            delayed(self.get_item)(
                i) for i in indexes)

        return collate_fn(items)

    def __len__(self):
        return len(self.ids)


def collate_fn(data):
    images, meta_info = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, dim=0)

    return images, meta_info

