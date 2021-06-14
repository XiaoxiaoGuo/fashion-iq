import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import nltk
from PIL import Image
import json
import numpy as np

SEQ_LEN = None
class Dataset(data.Dataset):

    def __init__(self, data_file_name, vocab, transform=None, max_seq_len=64, label_len=5):
        """Set the path for images, captions and vocabulary wrapper.
        
        Args:
            root: image directory.
            data: index file name.
            transform: image transformer.
            vocab: pre-processed vocabulary.
        """
        # self.root = root
        with open(data_file_name, 'r') as f:
            self.data = json.load(f)
        self.ids = range(len(self.data))
        self.vocab = vocab
        self.transform = transform
        # self.return_target = return_target
        self.seq_len = max_seq_len
        SEQ_LEN = max_seq_len
        self.label_len = label_len

    def __getitem__(self, index):
        """Returns one data pair (image and concatenated captions)."""
        data = self.data
        vocab = self.vocab
        id = self.ids[index]

        image0 = data[id]['image0']
        image0 = Image.open(os.path.join(image0)).convert('RGB')
        if self.transform is not None:
            image0 = self.transform(image0)

        image1 = data[id]['image1']
        image1 = Image.open(os.path.join(image1)).convert('RGB')
        if self.transform is not None:
            image1 = self.transform(image1)
   
        caption = []
        caption_texts = data[id]['captions']
        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(str(caption_texts).lower()) 

        if len(tokens) >= self.seq_len:
            tokens = tokens[:self.seq_len]
                
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        caption = torch.Tensor(caption)

        # image0_label = torch.Tensor(data[id]['image0_label'][:self.label_len]).long()
        # image1_label = torch.Tensor(data[id]['image1_label'][:self.label_len]).long()
        image0_label = torch.Tensor(data[id]['image0_full_score']).float()
        image1_label = torch.Tensor(data[id]['image1_full_score']).float()

        return image0, image1, caption, image0_label, image1_label

    def __len__(self):
        return len(self.ids)


class Dataset_fastrcnn(data.Dataset):

    def __init__(self, data_file_name, vocab, transform=None, max_seq_len=64):
        """Set the path for images, captions and vocabulary wrapper.
        
        Args:
            root: image directory.
            data: index file name.
            transform: image transformer.
            vocab: pre-processed vocabulary.
        """
        # self.root = root
        with open(data_file_name, 'r') as f:
            self.data = json.load(f)
        self.ids = range(len(self.data))
        self.vocab = vocab
        self.transform = transform
        # self.return_target = return_target
        self.seq_len = max_seq_len
        SEQ_LEN = max_seq_len

    def __getitem__(self, index):
        """Returns one data pair (image and concatenated captions)."""
        data = self.data
        vocab = self.vocab
        id = self.ids[index]

        image_0 = data[id]['image0']
        # image_0 = Image.open(os.path.join(image_0)).convert('RGB')
        image_0 = np.load(image_0, allow_pickle=True)
        # if self.transform is not None:
        #     image_0 = self.transform(image_0)

        image_1 = data[id]['image1']
        image_1 = np.load(image_1, allow_pickle=True)
        # image_1 = Image.open(os.path.join(image_1)).convert('RGB')
        # if self.transform is not None:
        #     image_1 = self.transform(image_1)
   
        caption = []
        caption_texts = data[id]['captions']
        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(str(caption_texts).lower()) 

        if len(tokens) >= self.seq_len:
            tokens = tokens[:self.seq_len]
                
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        caption = torch.Tensor(caption)

        return image_0, image_1, caption

    def __len__(self):
        return len(self.ids)





def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).

    Args:
        data: list of tuple (image, caption). 
            - image: torch tensor of shape
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of images.
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    image0, image1, captions, image0_label, image1_label = zip(*data)


    # Merge images (from tuple of 3D tensor to 4D tensor).
    image0 = torch.stack(image0, 0)
    image1 = torch.stack(image1, 0)
    
    image0_label = torch.stack(image0_label, 0)
    image1_label = torch.stack(image1_label, 0)
    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]

    captions_src = torch.zeros(len(captions), max(lengths)).long()
    captions_tgt = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        captions_src[i, :end-1] = cap[:end-1]
        captions_tgt[i, :end-1] = cap[1:end]
    # caption_padding_mask = (captions_src != 0)
    return image0, image1, captions_src, captions_tgt, image0_label, image1_label


def collate_fn_test(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).
    Args:
        data: list of tuple (image, caption). 
            - image: torch tensor of shape
            - caption: torch tensor of shape (?); variable length.
    Returns:
        images: torch tensor of images.
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    image0, image1, _, image0_label, image1_label = zip(*data)
    # Merge images (from tuple of 3D tensor to 4D tensor).
    image0 = torch.stack(image0, 0)
    image1 = torch.stack(image1, 0)

    image0_label = torch.stack(image0_label, 0)
    image1_label = torch.stack(image1_label, 0)
    # # Merge captions (from tuple of 1D tensor to 2D tensor).
    # lengths = [len(cap) for cap in captions]
    # captions_src = torch.zeros(len(captions), max(lengths)).long()
    # captions_tgt = torch.zeros(len(captions), max(lengths)).long()
    # for i, cap in enumerate(captions):
    #     end = lengths[i]
    #     captions_src[i, :end-1] = cap[:end-1]
    #     captions_tgt[i, :end-1] = cap[1:end]
    # # caption_padding_mask = (captions_src != 0)
    # return target_images, candidate_images, captions_src, captions_tgt
    return image0, image1, image0_label, image1_label


def get_loader(data_file_path, vocab, transform, batch_size, shuffle, num_workers=1,max_seq_len=64, attribute_len=5):
    """Returns torch.utils.data.DataLoader for custom dataset."""

    # relative caption dataset
    print('Reading data from',data_file_path)
    dataset = Dataset(
                      data_file_name=data_file_path,
                      vocab=vocab,
                      transform=transform,
                      max_seq_len=max_seq_len,
                      label_len=attribute_len
                      )
    print('data size',len(dataset))
    # Data loader for the dataset
    # This will return (images, captions, lengths) for each iteration.
    # images: a tensor of shape (batch_size, 3, 224, 224).
    # captions: a tensor of shape (batch_size, padded_length).
    # lengths: a list indicating valid length for each caption. length is (batch_size)
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn,
                                              timeout=60)

    return data_loader

def get_loader_test(data_file_path, vocab, transform, batch_size, shuffle, num_workers=1,max_seq_len=64,attribute_len=5):
    """Returns torch.utils.data.DataLoader for custom dataset."""
    # relative caption dataset
    print('Reading data from',data_file_path)
    dataset = Dataset(
                      data_file_name=data_file_path,
                      vocab=vocab,
                      transform=transform,
                      max_seq_len=max_seq_len,
                      label_len=attribute_len
                      )
    print('data size',len(dataset))
    # Data loader for the dataset
    # This will return (images, captions, lengths) for each iteration.
    # images: a tensor of shape (batch_size, 3, 224, 224).
    # captions: a tensor of shape (batch_size, padded_length).
    # lengths: a list indicating valid length for each caption. length is (batch_size)
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=1,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn_test,
                                              timeout=60)
    return data_loader

def load_ori_token_data(data_file_name):
    test_data_captions = []
    with open(data_file_name, 'r') as f:
        data = json.load(f)

        for line in data:
            caption_texts = line['captions']
            temp = []
            for c in caption_texts:
                # tokens = nltk.tokenize.word_tokenize(str(c).lower())
                temp.append(c)
            test_data_captions.append(temp)

    
    return test_data_captions


def load_ori_token_data_new(data_file_name):
    test_data_captions = {}
    with open(data_file_name, 'r') as f:
        data = json.load(f)
        count = 0
        for line in data:
            caption_texts = line['captions']
            caption_texts = ["it " + x for x in caption_texts]
            # temp = []
            # for c in caption_texts:
            #     # tokens = nltk.tokenize.word_tokenize(str(c).lower())
            #     temp.append(c)
            test_data_captions[count] = caption_texts
            count += 1
    
    return test_data_captions
