import argparse
import time
import os
import torch
import torch.nn as nn
from torchvision import transforms
from data_loader import get_loader
from build_vocab import Vocabulary
from models import DummyImageEncoder, DummyCaptionEncoder
from utils import create_exp_dir, Ranker

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths to data
IMAGE_ROOT = 'data/resized_images/{}/'
CAPT = 'data/captions/cap.{}.{}.json'
DICT = 'data/captions/dict.{}.json'
SPLIT = 'data/image_splits/split.{}.{}.json'

# Loss function
triplet_avg = nn.TripletMarginLoss(reduction='elementwise_mean', margin=1)


def eval_batch(data_loader, image_encoder, caption_encoder, ranker):
    ranker.update_emb(image_encoder)
    rankings = []
    loss = []
    for i, (target_images, candidate_images, captions, lengths, meta_info) in enumerate(data_loader):
        with torch.no_grad():
            target_images = target_images.to(device)
            target_ft = image_encoder.forward(target_images)
            candidate_images = candidate_images.to(device)
            candidate_ft = image_encoder.forward(candidate_images)
            captions = captions.to(device)
            caption_ft = caption_encoder(captions, lengths)
            target_asins = [ meta_info[m]['target'] for m in range(len(meta_info)) ]
            rankings.append(ranker.compute_rank(candidate_ft + caption_ft, target_asins))
            m = target_images.size(0)
            random_index = [m - 1 - n for n in range(m)]
            random_index = torch.LongTensor(random_index)
            negative_ft = target_ft[random_index]
            loss.append(triplet_avg(anchor=(candidate_ft + caption_ft),
                               positive=target_ft, negative=negative_ft))

    metrics = {}
    rankings = torch.cat(rankings, dim=0)
    metrics['score'] = 1 - rankings.mean().item() / ranker.data_emb.size(0)
    metrics['loss'] = torch.stack(loss, dim=0).mean().item()
    return metrics


def train(args):
    # Image preprocessing, normalization for the pretrained resnet
    transform = transforms.Compose([ 
        transforms.RandomCrop(args.crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])

    transform_dev = transforms.Compose([
        transforms.CenterCrop(args.crop_size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    vocab = Vocabulary()
    vocab.load(DICT.format(args.data_set))

    # Build data loader
    data_loader = get_loader(IMAGE_ROOT.format(args.data_set),
                             CAPT.format(args.data_set, 'train'),
                             vocab, transform,
                             args.batch_size, shuffle=True, return_target=True, num_workers=args.num_workers)

    data_loader_dev = get_loader(IMAGE_ROOT.format(args.data_set),
                                 CAPT.format(args.data_set, 'val'),
                                 vocab, transform_dev,
                                 args.batch_size, shuffle=False, return_target=True, num_workers=args.num_workers)

    ranker = Ranker(root=IMAGE_ROOT.format(args.data_set), image_split_file=SPLIT.format(args.data_set, 'val'),
                    transform=transform_dev, num_workers=args.num_workers)

    save_folder = '{}/{}-{}'.format(args.save, args.data_set, time.strftime("%Y%m%d-%H%M%S"))
    create_exp_dir(save_folder, scripts_to_save=['models.py', 'data_loader.py', 'train.py', 'build_vocab.py', 'utils.py'])

    def logging(s, print_=True, log_=True):
        if print_:
            print(s)
        if log_:
            with open(os.path.join(save_folder, 'log.txt'), 'a+') as f_log:
                f_log.write(s + '\n')

    logging(str(args))
    # Build the dummy models
    image_encoder = DummyImageEncoder(args.embed_size).to(device)
    caption_encoder = DummyCaptionEncoder(vocab_size=len(vocab), vocab_embed_size=args.embed_size * 2,
                                          embed_size=args.embed_size).to(device)

    image_encoder.train()
    caption_encoder.train()
    params = image_encoder.get_trainable_parameters() + caption_encoder.get_trainable_parameters()

    current_lr = args.learning_rate
    optimizer = torch.optim.Adam(params, lr=current_lr)

    cur_patient = 0
    best_score = float('-inf')
    stop_train = False
    total_step = len(data_loader)
    # epoch = 1 for dummy setting
    for epoch in range(1):

        for i, (target_images, candidate_images, captions, lengths, meta_info) in enumerate(data_loader):

            target_images = target_images.to(device)
            target_ft = image_encoder.forward(target_images)

            candidate_images = candidate_images.to(device)
            candidate_ft = image_encoder.forward(candidate_images)

            captions = captions.to(device)
            caption_ft = caption_encoder(captions, lengths)

            # random select negative examples
            m = target_images.size(0)
            random_index = [m - 1 - n for n in range(m)]
            random_index = torch.LongTensor(random_index)
            negative_ft = target_ft[random_index]

            loss = triplet_avg(anchor=(candidate_ft + caption_ft),
                               positive=target_ft, negative=negative_ft)

            caption_encoder.zero_grad()
            image_encoder.zero_grad()
            loss.backward()
            optimizer.step()

            if i % args.log_step == 0:
                logging(
                    '| epoch {:3d} | step {:6d}/{:6d} | lr {:06.6f} | train loss {:8.3f}'.format(epoch, i, total_step,
                                                                                                 current_lr,
                                                                                                 loss.item()))

        image_encoder.eval()
        caption_encoder.eval()
        logging('-' * 77)
        metrics = eval_batch(data_loader_dev, image_encoder, caption_encoder, ranker)
        logging('| eval loss: {:8.3f} | score {:8.5f} / {:8.5f} '.format(
            metrics['loss'], metrics['score'], best_score))
        logging('-' * 77)

        image_encoder.train()
        caption_encoder.train()

        dev_score = metrics['score']
        if dev_score > best_score:
            best_score = dev_score
            # save best model
            resnet = image_encoder.delete_resnet()
            torch.save(image_encoder.state_dict(), os.path.join(
                save_folder,
                'image-{}.th'.format(args.embed_size)))
            image_encoder.load_resnet(resnet)

            torch.save(caption_encoder.state_dict(), os.path.join(
                save_folder,
                'cap-{}.th'.format(args.embed_size)))

            cur_patient = 0
        else:
            cur_patient += 1
            if cur_patient >= args.patient:
                current_lr /= 2
                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_lr
                if current_lr < args.learning_rate * 1e-3:
                    stop_train = True
                    break

        if stop_train:
            break
    logging('best_dev_score: {}'.format(best_score))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', type=str, default='models',
                        help='path for saving trained models')
    parser.add_argument('--crop_size', type=int, default=224,
                        help='size for randomly cropping images')

    parser.add_argument('--data_set', type=str, default='dress')
    parser.add_argument('--log_step', type=int, default=3,
                        help='step size for printing log info')
    parser.add_argument('--patient', type=int, default=3,
                        help='patient for reducing learning rate')

    # Model parameters
    parser.add_argument('--embed_size', type=int , default=512,
                        help='dimension of word embedding vectors')
    # Learning parameters
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=0.001)

    args = parser.parse_args()

    train(args)

