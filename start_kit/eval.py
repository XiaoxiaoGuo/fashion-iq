import argparse
import torch
import json
import os
from torchvision import transforms
from data_loader import get_loader
from build_vocab import Vocabulary
from models import DummyImageEncoder, DummyCaptionEncoder
from utils import Ranker

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Paths to data
IMAGE_ROOT = 'data/resized_images/{}/'
CAPT = 'data/captions/cap.{}.{}.json'
DICT = 'data/captions/dict.{}.json'
SPLIT = 'data/image_splits/split.{}.{}.json'


def evaluate(args):
    # Image pre-processing, normalization for the pre-trained resnet
    transform_test = transforms.Compose([
        transforms.CenterCrop(args.crop_size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])
    vocab = Vocabulary()
    vocab.load(DICT.format(args.data_set))
    # Build data loader
    data_loader_test = get_loader(IMAGE_ROOT.format(args.data_set),
                                 CAPT.format(args.data_set, args.data_split),
                                 vocab, transform_test,
                                 args.batch_size, shuffle=False, return_target=False, num_workers=args.num_workers)
    ranker = Ranker(root=IMAGE_ROOT.format(args.data_set), image_split_file=SPLIT.format(args.data_set, args.data_split),
                    transform=transform_test, num_workers=args.num_workers)

    # Build the dummy models
    image_encoder = DummyImageEncoder(args.embed_size).to(device)
    caption_encoder = DummyCaptionEncoder(vocab_size=len(vocab), vocab_embed_size=args.embed_size * 2,
                                          embed_size=args.embed_size).to(device)
    # load trained models
    image_model = os.path.join(args.model_folder,
                               "image-{}.th".format(args.embed_size))
    resnet = image_encoder.delete_resnet()
    image_encoder.load_state_dict(torch.load(image_model, map_location=device))
    image_encoder.load_resnet(resnet)

    cap_model = os.path.join(args.model_folder,
                               "cap-{}.th".format(args.embed_size))
    caption_encoder.load_state_dict(torch.load(cap_model, map_location=device))

    ranker.update_emb(image_encoder)
    image_encoder.eval()
    caption_encoder.eval()

    output = json.load(open(CAPT.format(args.data_set, args.data_split)))

    index = 0
    for _, candidate_images, captions, lengths, meta_info in data_loader_test:
        with torch.no_grad():
            candidate_images = candidate_images.to(device)
            candidate_ft = image_encoder.forward(candidate_images)
            captions = captions.to(device)
            caption_ft = caption_encoder(captions, lengths)
            rankings = ranker.get_nearest_neighbors(candidate_ft + caption_ft)
            # print(rankings)
            for j in range(rankings.size(0)):
                output[index]['ranking'] = [ranker.data_asin[rankings[j, m].item()] for m in range(rankings.size(1))]
                index += 1

    json.dump(output, open("{}.{}.pred.json".format(args.data_set, args.data_split), 'w'), indent=4)
    print('eval completed. Output file: {}'.format("{}.{}.pred.json".format(args.data_set, args.data_split)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_folder', type=str, default='models/dress-20190612-112918/',
                        help='path for trained models')
    parser.add_argument('--crop_size', type=int, default=224,
                        help='size for randomly cropping images')
    parser.add_argument('--data_set', type=str, default='dress')
    parser.add_argument('--data_split', type=str, default='test')
    # Model parameters
    parser.add_argument('--embed_size', type=int, default=512,
                        help='dimension of word embedding vectors')
    # Learning parameters
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=16)
    args = parser.parse_args()
    evaluate(args)
