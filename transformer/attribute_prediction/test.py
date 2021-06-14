import argparse
import glob
import torchvision
from torchvision import transforms
import torch
import torch.utils.data as data
import json
from PIL import Image
import tqdm
from efficientnet_pytorch import EfficientNet


class Dataset(data.Dataset):
    def __init__(self, root, transform=None):
        """Set the path for images, captions and vocabulary wrapper.

        Args:
            root: image directory.
            data_file_name: asin --> [tag]
            transform: image transformer.
        """
        self.root = root
        self.image_list = glob.glob(self.root + "/*")
        # print('image list', self.image_list)

        self.transform = transform
        return

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""

        img_name = self.image_list[index]
        asin = img_name.split('/')[-1]
        asin = asin.split('.')[0]
        image = Image.open(img_name).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        return image, asin

    def __len__(self):
        return len(self.image_list)


def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, tags).
    """
    images, asins = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, dim=0)

    return images, asins


def get_loader(root, transform, batch_size, shuffle, num_workers=4):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    # cpu_num = multiprocessing.cpu_count()
    # num_workers = cpu_num - 2 if cpu_num > 2 else 1

    dataset = Dataset(root=root,
                      transform=transform)

    data_loader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, collate_fn=collate_fn, pin_memory=True)

    return data_loader


def evaluate_model(data_loader, model, idx2attr, device, topk=5):
    model.eval()
    test_num = 0
    prediction = {}
    for images, asins in tqdm.tqdm(data_loader):
        # Set mini-batch dataset
        with torch.no_grad():
            images = images.to(device)
            outs = model(images)
            top_scores, top_outs = outs.topk(dim=1, k=topk*2, largest=True)

            for j in range(images.size(0)):
                top_out_tags = [idx2attr[top_outs[j, m].item()]
                                for m in range(topk*2)]

                prediction[asins[j]] = {
                    'predict': top_out_tags,
                    'pred_score': top_scores[j].cpu().numpy().tolist(),
                }

    return prediction


def evaluate_attributes(args):
    device = 'cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu'
    # Build data loader
    # --Image preprocessing, normalization for the pretrained resnet
    transform = transforms.Compose([
        transforms.Resize(args.crop_size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    image_folder = args.image_folder.format(args.data_set)

    with open(args.label_file, 'r') as f:
        attr2idx = json.load(f)

    idx2attr = {}
    for key, val in attr2idx.items():
        idx2attr[val] = key

    model = EfficientNet.from_pretrained('efficientnet-b7')
    ckpt = torch.load(args.pretrained_model, map_location='cpu')
    if "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
    else:
        model.load_state_dict(ckpt)
    model.to(device)

    def logging(s, print_=True):
        if print_:
            print(s)

    data_loader = get_loader(
        root=image_folder,
        transform=transform,
        batch_size=args.batch_size,
        shuffle=False)

    model.eval()
    logging('-' * 87)
    with torch.no_grad():
        prediction = evaluate_model(data_loader, model, idx2attr, device)

    with open('fashion_iq_{}.json'.format(args.data_set), 'w') as f:
        json.dump(prediction, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument('--image_folder', type=str,
                        default='../resized_images/{}/')
    parser.add_argument('--label_file', type=str,
                        default='data/attribute2idx.json')
    parser.add_argument('--crop_size', type=int, default=224,
                        help='size for randomly cropping images')
    parser.add_argument('--data_set', type=str, default='dress',
                        help='dress / shirt / toptee')

    # model
    parser.add_argument('--pretrained_model', type=str,
                        default='dfattributes_efficientnet_b7ns.pth')

    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--batch_size', type=int, default=2)

    args = parser.parse_args()
    print(args)
    evaluate_attributes(args)
