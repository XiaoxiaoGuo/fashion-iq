import argparse
import math
import shutil
import time
import json
import os
import tqdm
import torch
import pickle
import torch.nn as nn
from torchvision import transforms
from models import RetrieverTransformer
from data_loader import Dataset
import utils
import Ranker
import UserModel

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


def load_image_features(args):
    # Image preprocessing, normalization for the pretrained b7
    transform = transforms.Compose([
        transforms.Resize(args.crop_size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    # train split
    fts_file = os.path.join(args.save, 'b7_v2.{}.{}.th'.format(
        args.data_set, 'train'))
    if os.path.isfile(fts_file):
        print('[INFO] loading image features: {}'.format(fts_file))
        fts = torch.load(fts_file, map_location='cpu')
    else:
        print('[INFO] computing image features: {}'.format(fts_file))
        data_loader = Dataset(
            args.image_folder.format(args.data_set),
            args.data_split_file.format(args.data_set, 'train'),
            transform, num_workers=args.num_workers)
        attr_file = args.attribute_file.format(args.data_set)

        fts = utils.extract_features(data_loader, attr_file,
                                     args.attr2idx_file, device,
                                     args.image_model)
        torch.save(fts, fts_file)

    # dev split
    fts_file_dev = os.path.join(args.save, 'b7_v2.{}.{}.th'.format(
        args.data_set, 'val'))
    if os.path.isfile(fts_file_dev):
        print('[INFO] loading image features: {}'.format(fts_file_dev))
        fts_dev = torch.load(fts_file_dev, map_location='cpu')
    else:
        print('[INFO] computing image features: {}'.format(fts_file_dev))
        data_loader_dev = Dataset(
            args.image_folder.format(args.data_set),
            args.data_split_file.format(args.data_set, 'val'),
            transform, num_workers=args.num_workers)
        attr_file_dev = args.attribute_file.format(args.data_set)
        fts_dev = utils.extract_features(data_loader_dev, attr_file_dev,
                                         args.attr2idx_file, device,
                                         args.image_model)
        torch.save(fts_dev, fts_file_dev)

    return fts, fts_dev


def eval_batch(fts, captioner, retriever, args,
               train_mode=False,
               optimizer=None):
    criterion = nn.TripletMarginLoss(
        reduction='mean', margin=args.margin).to(device)
    # generate a mapping for dev, to ensure sampling bias is reduced
    num_target = len(fts['asins'])

    batch_size = args.batch_size
    ranker = Ranker.Ranker(device)
    total_step = math.floor(num_target / batch_size)

    ranking_tracker = [0] * args.num_dialog_turns
    loss_tracker = [0] * args.num_dialog_turns

    with open('data/shuffled.{}.{}.json'.format(
            args.data_set, 'val')) as f:
        first_candidate_set = json.load(f)

    with torch.no_grad():
        retriever.eval()
        ranker.update_emb(fts, args.batch_size, retriever)

    if train_mode:
        retriever.train()
    else:
        retriever.eval()

    for step in tqdm.tqdm(range(total_step)):
        # sample target
        if train_mode:
            target_ids = torch.tensor(
                [0]*args.batch_size, device=device, dtype=torch.long)
            target_ids.random_(0, num_target)
        else:
            target_ids = torch.tensor(
                [i for i in
                 range(step * batch_size, (step + 1) * batch_size)]).to(
                device=device, dtype=torch.long)

        # sample first batch of candidates
        if train_mode:
            candidate_ids = torch.tensor(
                [0]*args.batch_size, device=device, dtype=torch.long)
            candidate_ids.random_(0, num_target)
        else:
            candidate_ids = torch.tensor(
                [first_candidate_set[i]
                 for i in range(step * batch_size, (step + 1) * batch_size)],
                device=device, dtype=torch.long)

        # target_ids.random_(0, num_target)
        target_img_ft = utils.get_image_batch(fts, target_ids)
        target_img_ft = target_img_ft.to(device)
        target_img_emb = retriever.encode_image(target_img_ft)

        target_attr = utils.get_attribute_batch(fts, target_ids)
        target_attr = target_attr.to(device)

        # clean up dialog history tracker
        retriever.init_hist()
        # history_hidden = history_hidden.expand_as(target_img_emb)
        # history_hidden = None
        loss = 0

        for d_turn in range(args.num_dialog_turns):
            # get candidate image features
            candidate_img_ft = utils.get_image_batch(fts, candidate_ids)
            candidate_img_ft = candidate_img_ft.to(device)

            candidate_attr = utils.get_attribute_batch(fts, candidate_ids)
            candidate_attr = candidate_attr.to(device)
            # generate captions from model
            with torch.no_grad():
                sentence_ids = captioner.get_caption(
                    target_img_ft, candidate_img_ft,
                    target_attr, candidate_attr)
            sentence_ids = sentence_ids.to(device)

            candidate_img_ft = candidate_img_ft.to(device)

            history_hidden = retriever.forward(
                text=sentence_ids, image=candidate_img_ft,
                attribute=candidate_attr)

            # sample negatives, update tracker's output to
            # match targets via triplet loss
            negative_ids = torch.tensor(
                [0]*args.batch_size, device=device, dtype=torch.long)
            negative_ids.random_(0, num_target)

            negative_img_ft = utils.get_image_batch(fts, negative_ids)
            negative_img_ft = negative_img_ft.to(device)
            negative_img_emb = retriever.encode_image(negative_img_ft)

            # accumulate loss
            loss_tmp = criterion(history_hidden, target_img_emb,
                                 negative_img_emb)
            loss += loss_tmp
            loss_tracker[d_turn] += loss_tmp.item()

            # generate new candidates, compute ranking information
            with torch.no_grad():
                candidate_ids = ranker.nearest_neighbors(history_hidden)
                ranking = ranker.compute_rank(history_hidden, target_ids)
            ranking_tracker[d_turn] += (ranking.mean().item() /
                                        (num_target * 1.0))

        # update weights
        if train_mode:
            optimizer.zero_grad()
            # loss = loss / args.num_dialog_turns
            loss.backward()
            # clip_grad_norm_(retriever.parameters(), args.clip_norm)
            optimizer.step()
            with torch.no_grad():
                retriever.eval()
                ranker.update_emb(fts, args.batch_size, retriever)
                retriever.train()

    loss = loss.item() / total_step
    for i in range(args.num_dialog_turns):
        ranking_tracker[i] /= total_step
        loss_tracker[i] /= total_step

    metrics = {'loss': loss, 'score': 5 - sum(ranking_tracker),
               'loss_tracker': loss_tracker,
               'ranking_tracker': ranking_tracker}
    return metrics


def train(args):
    save_folder = '{}/{}'.format(
        args.save, args.data_set)
    create_exp_dir(save_folder)

    def logging(s, print_=True, log_=True):
        if print_:
            print(s)
        if log_:
            with open(os.path.join(save_folder, 'log.txt'), 'a+') as f_log:
                f_log.write(s + '\n')
        return

    logging(str(args))

    # load image features (or compute and store them if necessary)
    fts, fts_dev = load_image_features(args)

    # user model: captioner
    captioner = UserModel.UserModel(args)
    captioner.to(device)

    # ranker
    img_dim = fts['image'].size(dim=1)

    # response encoder
    glove_emb_file = args.glove_emb_file.format(args.data_set)
    with open(glove_emb_file, 'rb') as f:
        glove_emb = pickle.load(f)

    retriever = RetrieverTransformer(
        captioner.get_vocab_size(), glove_emb, img_dim,
        args.hidden_unit_num, args.layer_num,
        args.attribute_num).to(device)

    # Loss and optimizer
    params = retriever.parameters()
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)

    current_lr = args.learning_rate
    cur_patient = 0
    best_score = float('-inf')
    best_train_score = float('-inf')

    for epoch in range(100):
        retriever.train()
        metrics = eval_batch(fts, captioner, retriever, args,
                             train_mode=True, optimizer=optimizer)
        res = metrics['ranking_tracker']
        logging(
            '| ({}) train loss: {:8.5f} | lr: {:8.7f} | '
            'score {:8.5f} / {:8.5f} | '
            'rank {:5.3f}/{:5.3f}/{:5.3f}/{:5.3f}/{:5.3f}'.format(
                epoch, metrics['loss'], current_lr,
                metrics['score'], best_train_score,
                1 - res[0], 1 - res[1], 1 - res[2], 1 - res[3], 1 - res[4]))
        logging('-' * 77)

        if metrics['score'] < best_train_score + 1e-3:
            cur_patient += 1
            if cur_patient >= args.patient:
                current_lr *= 0.5
                if current_lr < args.learning_rate * 1e-1:
                    break
                params = retriever.parameters()
                optimizer = torch.optim.Adam(params, lr=current_lr)

        best_train_score = max(metrics['score'], best_train_score)

        # eval on validation split
        if epoch % args.checkpoint == 0:
            retriever.eval()
            logging('-' * 77)

            with torch.no_grad():
                metrics = eval_batch(fts_dev, captioner, retriever, args)
            res = metrics['ranking_tracker']
            logging(
                '| ({}) eval loss: {:8.3f} | score {:8.5f} / {:8.5f} | '
                'rank {:5.3f}/{:5.3f}/{:5.3f}/{:5.3f}/{:5.3f}'.format(
                    epoch, metrics['loss'], metrics['score'], best_score,
                    1-res[0], 1-res[1], 1-res[2], 1-res[3], 1-res[4]))
            logging('-' * 77)
            dev_score = metrics['score']

            if dev_score > best_score:
                best_score = dev_score
                # save best model
                checkpoint_model = {'args': args,
                                    'retrieval_model': retriever}
                torch.save(checkpoint_model, os.path.join(
                    save_folder, 'checkpoint_model.th'))
                cur_patient = 0

    logging('best_dev_score: {}'.format(best_score))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_folder', type=str,
                        default='../resized_images/{}/')
    parser.add_argument('--data_split_file', type=str,
                        default='data/split.{}.{}.json')
    parser.add_argument('--attribute_file', type=str,
                        default='../attribute_prediction/prediction/'
                                'predict_{}_b7_ft.json')
    parser.add_argument('--attr2idx_file', type=str,
                        default='../attribute_prediction/data/'
                                'attribute2idx.json')
    parser.add_argument('--image_model', type=str,
                        default='../attribute_prediction/deepfashion_models/'
                                'dfattributes_efficientnet_b7ns.pth')

    parser.add_argument('--save', type=str, default='models/',
                        help='path for saving ranker models')
    parser.add_argument('--crop_size', type=int, default=224,
                        help='size for randomly cropping images')
    parser.add_argument('--data_set', type=str, default='dress',
                        help='dress / toptee / shirt')

    parser.add_argument('--checkpoint', type=int, default=3,
                        help='step size for saving models')
    parser.add_argument('--patient', type=int, default=2,
                        help='patient for reducing learning rate')
    parser.add_argument('--exp_folder', type=str, default='v1')

    # User model parameters
    parser.add_argument('--user_model_file', type=str,
                        default='../user_modeling/models/'
                                '{}-efficient-b7-finetune.chkpt')
    parser.add_argument('--user_vocab_file', type=str,
                        default='../user_modeling/data/{}_vocab.json')
    parser.add_argument('--max_seq_len', type=int, default=8,
                        help='maximum caption length')
    parser.add_argument('--beam_size', type=int, default=1,
                        help='beam search branch size')
    parser.add_argument('--glove_emb_file', type=str,
                        default='data/{}_emb.pt')
    parser.add_argument('--attribute_num', type=int,
                        default=1000)

    # Model parameters
    parser.add_argument('--hidden_unit_num', type=int, default=256)
    parser.add_argument('--layer_num', type=int, default=6)

    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--num_dialog_turns', type=int, default=5)
    parser.add_argument('--margin', type=float, default=10)
    parser.add_argument('--clip_norm', type=float, default=10)

    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=0.001)

    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    train(args)

