import torch
import torch.utils.data as data
import os
import json
from PIL import Image
import multiprocessing


class Dataset(data.Dataset):
    def __init__(self, root, data_file_name, class_file, transform=None):
        """Set the path for images, captions and vocabulary wrapper.

        Args:
            root: image directory.
            data_file_name: asin --> [tag]
            transform: image transformer.
        """
        self.root = root

        with open(data_file_name, 'r') as f:
            self.data = json.load(f)
        self.ids = range(len(self.data))

        self.asin = []
        for key, _ in self.data.items():
            self.asin.append(key)

        with open(class_file, 'r') as f:
            self.cls = json.load(f)

        self.transform = transform
        return

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        id = self.ids[index]
        asin = self.asin[id]
        img_name = self.asin[id] + '.jpg'

        image = Image.open(os.path.join(self.root, img_name)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        attribute_labels = self.data[asin]
        # convert text words to idx, flattened
        label = torch.zeros(len(self.cls))
        for sublist in attribute_labels[1:]:
            for tag in sublist:
                label[self.cls[tag]] = 1

        return image, label, asin

    def __len__(self):
        return len(self.ids)


def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, tags).
    """
    images, labels, asins = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, dim=0)
    labels = torch.stack(labels, dim=0)

    return images, labels, asins


def get_loader(root, data_file, class_file, transform, batch_size, shuffle):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    cpu_num = multiprocessing.cpu_count()
    num_workers = cpu_num - 2 if cpu_num > 2 else 1

    dataset = Dataset(root=root,
                      data_file_name=data_file,
                      class_file=class_file,
                      transform=transform)

    data_loader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, collate_fn=collate_fn, pin_memory=True)

    return data_loader

