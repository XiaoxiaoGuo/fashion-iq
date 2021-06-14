'''
This script handling the training process.
'''


import argparse
import math
import time
import os
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
# import transformer.Constants as Constants
from dataset import get_loader, load_ori_token_data, get_loader_test, load_ori_token_data_new
from Beam import beam_search
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
import numpy as np
from build_vocab import Vocabulary
# from model import Neural_Naturalist
import torchvision.transforms as transforms
from nltk.translate.bleu_score import sentence_bleu

from torch.optim.lr_scheduler import StepLR
from Optim import NoamOpt, get_std_opt

from Models import get_model, create_masks
from torch.autograd import Variable
from pytorchtools import EarlyStopping

# from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

# # from pycocoevalcap.meteor.meteor import Meteor
# from pycocoevalcap.rouge.rouge import Rouge
# from pycocoevalcap.ciderd.ciderD import CiderD
# from pycocoevalcap.cider.cider import Cider
from test import test

# from dataset import TranslationDataset, paired_collate_fn
# from transformer.Models import Transformer
# from transformer.Optim import ScheduledOptim

"""
key_padding_mask should be a ByteTensor where True values are positions
that should be masked with float('-inf') and False values will be unchanged

"""

Constants_PAD = 0

# def nopeak_mask(size):
#     np_mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8')
#     np_mask = Variable(torch.from_numpy(np_mask) == 0)

#     return np_mask

# def create_masks(trg):
#     # src_mask = (src != Constants_PAD.unsqueeze(-2)

#     if trg is not None:
#         trg_mask = (trg != Constants_PAD).unsqueeze(-2)
#         size = trg.size(1) # get seq_len for matrix
#         np_mask = nopeak_mask(size).to(trg_mask.device)

#         trg_mask = trg_mask & np_mask
        
#     else:
#         trg_mask = None

#     return trg_mask

def cal_performance(pred, gold, smoothing=False):
    ''' Apply label smoothing if needed 
        pred:(batch_size, sequence_len, vocab_size) 
        gold:(batch_size, seq_len) (indices)
    '''

    loss = cal_loss(pred, gold, smoothing)

    #greedy decoding
    pred = pred.max(2)[1]# torch.max() return (values, indices) 
                         #shape:(batch_size, sequence_len)

    # gold = gold.transpose(0,1) #shape:(sequence_len, batch_size)

    non_pad_mask = gold.ne(Constants_PAD)#Compute input!=other element-wise
    n_correct = pred.eq(gold)
    n_correct = n_correct.masked_select(non_pad_mask).sum().item()

    return loss, n_correct
    # return loss

def calculate_bleu(tgt, logits, vocab):
    """
    reference = [['this', 'is', 'small', 'test']]
    candidate = ['this', 'is', 'a', 'test']
    bleu_4 = sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25))
 
    """

    # TODO: Batched Beam Search
    # Therefore, do not use a batch_size greater than 1 - IMPORTANT!

    # Lists to store references (true captions), and hypothesis (prediction) for each image
    # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
    # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]

    word_map = vocab.word2idx

    pred = logits.max(2)[1]

    references = list()
    hypotheses = list()

    img_caps = tgt.tolist()
    img_captions = list(
        map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}],
            img_caps))  # remove <start> and pads
    references.append(img_captions)

    # Hypotheses
    hypotheses.append([w for w in seq if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}])

    bleu4 = sentence_bleu(references, hypotheses)

    return bleu4



def cal_loss(pred, gold, smoothing):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''
    #why the cross_entropy loss is sum, because we want to calculator the total loss devided by the total num of tokens
    #input (Tensor) – (N, C)(N,C) where C = number of classes or (N, C, H, W)
    #target (Tensor) – (N)(N) where each value is 0 <= targets} <= C-1,  0≤targets[i]≤C−1 , or (N, d_1, d_2, ..., d_K)
    #if our pred shape is (batch_size, sequence_len, vocab_size), we either transpose(1,2) or flatten the matrix

    gold = gold.contiguous().view(-1)
    pred = pred.contiguous().view(-1,pred.size(-1))

    if smoothing:
        eps = 0.1
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        non_pad_mask = gold.ne(Constants_PAD)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum()  # average later
    else:
        loss = F.cross_entropy(pred, gold, ignore_index=Constants_PAD, reduction='sum')#This criterion combines log_softmax and nll_loss in a single function

    return loss

def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''

    sz_b, len_s = seq.size()
    mask = (torch.triu(torch.ones(len_s, len_s)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    # subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls

    return mask


def train_epoch(model, training_data, optimizer, device, smoothing=False):
    ''' Epoch operation in training phase'''

    model.train()

    total_loss = 0
    n_word_total = 0
    n_word_correct = 0

    for batch in tqdm(
            training_data, mininterval=2,
            desc='  - (Training)   ', leave=False):

        # prepare data
        image0, image1, captions, gold, image0_attribute, image1_attribute  = map(lambda x: x.to(device), batch)

        """[src/tgt/memory]_key_padding_mask should be a ByteTensor where True values are positions
                                    that should be masked with float('-inf') and False values will be unchanged.
                                    This mask ensures that no information will be taken from position i if
                                    it is masked, and has a separate mask for each sequence in a batch."""

        # caption_padding_mask = captions.eq(Constants_PAD)

        # look_ahead_mask = get_subsequent_mask(captions).to(device)

        # trg_input = caption[:, :-1]
        trg_mask = create_masks(captions).to(device)

        # ys = trg[:, 1:].contiguous().view(-1)

        # forward
        optimizer.optimizer.zero_grad()

        logits = model(image0, image1, captions, trg_mask, image0_attribute, image1_attribute)
        # logits = model(image1, image2, captions, look_ahead_mask, caption_padding_mask)#(batch_size, sequence_len, vocab_size) 

        # backward
        loss, n_correct = cal_performance(logits, gold, smoothing=smoothing)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.get_trainable_parameters(), max_norm=5)
        # update parameters
        optimizer.step()

        # note keeping
        total_loss += loss.item()

        non_pad_mask = gold.ne(Constants_PAD) #pad = 0
        n_word = non_pad_mask.sum().item()
        n_word_total += n_word
        n_word_correct += n_correct

    loss_per_word = total_loss/n_word_total
    accuracy = n_word_correct/n_word_total
    return loss_per_word, accuracy

def eval_epoch(model, validation_data, device, vocab):
    ''' Epoch operation in evaluation phase '''

    model.eval()

    total_loss = 0
    n_word_total = 0
    n_word_correct = 0

    with torch.no_grad():
        for batch in tqdm(
                validation_data, mininterval=2,
                desc='  - (Validation) ', leave=False):

            # prepare data
            image0, image1, captions, gold, image0_attribute, image1_attribute = map(lambda x: x.to(device), batch)

            """[src/tgt/memory]_key_padding_mask should be a ByteTensor where True values are positions
                                    that should be masked with float('-inf') and False values will be unchanged.
                                    This mask ensures that no information will be taken from position i if
                                    it is masked, and has a separate mask for each sequence in a batch."""
            # caption_padding_mask = captions.eq(Constants_PAD)

            # look_ahead_mask = get_subsequent_mask(captions).to(device)

            # # forward
            # logits = model(image1, image2, captions, look_ahead_mask, caption_padding_mask)

            trg_mask = create_masks(captions).to(device)

            logits = model(image0, image1, captions, trg_mask, image0_attribute, image1_attribute)
            
            loss, n_correct = cal_performance(logits, gold, smoothing=False)

            # bleu_4 = calculate_bleu(captions, logits, vocab)

            # note keeping
            total_loss += loss.item()

            non_pad_mask = gold.ne(Constants_PAD)
            n_word = non_pad_mask.sum().item()
            n_word_total += n_word
            n_word_correct += n_correct

    loss_per_word = total_loss/n_word_total
    accuracy = n_word_correct/n_word_total
    return loss_per_word, accuracy


def eval_epoch_bleu(model, validation_data, device, vocab, list_of_refs_dev, args):
    ''' Epoch operation in evaluation phase '''

    model.eval()

    total_loss = 0
    n_word_total = 0
    n_word_correct = 0

    hypotheses = {}
    count = 0

    with torch.no_grad():
        for batch in tqdm(
                validation_data, mininterval=2,
                desc='  - (Validation) ', leave=False):

            # prepare data
            image0, image1, image0_attribute, image1_attribute = map(lambda x: x.to(device), batch)

            """[src/tgt/memory]_key_padding_mask should be a ByteTensor where True values are positions
                                    that should be masked with float('-inf') and False values will be unchanged.
                                    This mask ensures that no information will be taken from position i if
                                    it is masked, and has a separate mask for each sequence in a batch."""

            hyp = beam_search(image0, image1, model, args, vocab, image0_attribute, image1_attribute)

            hyp = hyp.split("<end>")[0].strip()

            hypotheses[count] = [hyp]

            count += 1

        scorer = Bleu(4)

        score, _ = scorer.compute_score(list_of_refs_dev, hypotheses)

    return score

def train(model, training_data, validation_data, optimizer, args, vocab, list_of_refs_dev, validation_data_combined):
    ''' Start training '''

    early_stopping_with_saving = EarlyStopping(patience=args.patience, verbose=True, args=args)

    log_train_file = None
    log_valid_file = None

    if args.log:
        log_train_file = args.log + '.train.log'
        log_valid_file = args.log + '.valid.log'
        log_valid_bleu_file = args.log + '.valid.bleu.log'

        print('[Info] Training performance will be written to file: {} and {}'.format(
            log_train_file, log_valid_file))

        with open(log_train_file, 'w') as log_tf, open(log_valid_file, 'w') as log_vf:
            log_tf.write('epoch,loss,ppl,accuracy\n')
            log_vf.write('epoch,loss,ppl,accuracy\n')

        with open(log_valid_bleu_file, 'w') as log_vf_bleu:
            log_vf_bleu.write('epoch,bleu1,bleu2,bleu3,bleu4\n')

    valid_accus = []

    best_valid_score = float('-inf')

    for epoch_i in range(args.epoch):

        if early_stopping_with_saving.early_stop:
            print("Early stopping")
            break

        print('Epoch {}, lr {}'.format(epoch_i, optimizer.optimizer.param_groups[0]['lr']))

        start = time.time()

        train_loss, train_accu = train_epoch(
            model, training_data, optimizer, args.device)

        print('  - (Training)   ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, '\
              'elapse: {elapse:3.3f} min'.format(
                  ppl=math.exp(min(train_loss, 100)), accu=100*train_accu,
                  elapse=(time.time()-start)/60))

        start = time.time()

        valid_loss, valid_accu = eval_epoch(model, validation_data, args.device, vocab)

        print('  - (Validation) ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, '\
                'elapse: {elapse:3.3f} min'.format(
                    ppl=math.exp(min(valid_loss, 100)), accu=100*valid_accu,
                    elapse=(time.time()-start)/60))

        if epoch_i != 0 and epoch_i % args.bleu_valid_every_n == 0:

            bleu_valid = eval_epoch_bleu(model, validation_data_combined, args.device, vocab, list_of_refs_dev, args)

            print('  - (Validation) bleu-1: {bleu1: 8.5f}, bleu-2: {bleu2: 8.5f}, bleu-3: {bleu3: 8.5f}, bleu-4: {bleu4: 8.5f}'.format(\
                bleu1=bleu_valid[0], bleu2=bleu_valid[1], bleu3=bleu_valid[2], bleu4=bleu_valid[3]))

            # if args.save_model:
            #     if args.save_mode == 'all':
            #         model_name = args.save_model + '_accu_{accu:3.3f}.chkpt'.format(accu=100*valid_accu)
            #         torch.save(checkpoint, model_name)
            #     elif args.save_mode == 'best':
            early_stopping_with_saving(bleu_valid[2], model, epoch_i)

            with open(log_valid_bleu_file, 'a') as log_vf_bleu:
                log_vf_bleu.write('{epoch},{bleu1: 8.5f},{bleu2: 8.5f},{bleu3: 8.5f},{bleu4: 8.5f}\n'.format( \
                    epoch=epoch_i, bleu1=bleu_valid[0], bleu2=bleu_valid[1], bleu3=bleu_valid[2], bleu4=bleu_valid[3]))
                    # model_name = args.save_model + '.chkpt'
                    # if bleu4_valid >= best_valid_score:
                    #     best_valid_score = bleu4_valid
                    #     torch.save(checkpoint, model_name)
                    #     print('    - [Info] The checkpoint file has been updated.')


        # checkpoint = {
        #     'model': model.state_dict(),
        #     'settings': args,
        #     'epoch': epoch_i}
        # torch.save(checkpoint, args.save_model + '.latest.chkpt')

        

        if log_train_file and log_valid_file:
            with open(log_train_file, 'a') as log_tf, open(log_valid_file, 'a') as log_vf:
                log_tf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                    epoch=epoch_i, loss=train_loss,
                    ppl=math.exp(min(train_loss, 100)), accu=100*train_accu))
                log_vf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                    epoch=epoch_i, loss=valid_loss,
                    ppl=math.exp(min(valid_loss, 100)), accu=100*valid_accu))



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



def main():
    ''' Main function '''
    parser = argparse.ArgumentParser()

    parser.add_argument('-data_train', type=str, default="")
    parser.add_argument('-data_dev', required=True)
    parser.add_argument('-data_test', type=str, default="")
    parser.add_argument('-vocab', required=True)

    parser.add_argument('-epoch', type=int, default=10000)
    parser.add_argument('-batch_size', type=int, default=64)

    #parser.add_argument('-d_word_vec', type=int, default=512)
    parser.add_argument('-d_model', type=int, default=512)
    # parser.add_argument('-d_inner_hid', type=int, default=2048)
    # parser.add_argument('-d_k', type=int, default=64)
    # parser.add_argument('-d_v', type=int, default=64)

    parser.add_argument('-n_heads', type=int, default=8)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-n_warmup_steps', type=int, default=4000)

    parser.add_argument('-dropout', type=float, default=0.1)
    # parser.add_argument('-embs_share_weight', action='store_true')
    # parser.add_argument('-proj_share_weight', action='store_true')

    parser.add_argument('-log', default=None)
    parser.add_argument('-save_model', default=None)
    parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='best')

    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-label_smoothing', action='store_true')
    parser.add_argument('-num_workers', type=int, default=1)

    parser.add_argument('-cnn_name', type=str, default="resnet101")
    parser.add_argument('-cnn_pretrained_model', type=str, default="")
    parser.add_argument('-joint_enc_func', type=str, default="element_multiplication")
    # parser.add_argument('-comparative_module_name', type=str, default="transformer_encoder")
    parser.add_argument('-lr', type=float, default=0.01)
    # parser.add_argument('-step_size', type=int, default=1000)
    # parser.add_argument('-gamma', type=float, default=0.9)
    parser.add_argument('-crop_size', type=int, default=224)
    parser.add_argument('-max_seq_len', type=int, default=64)
    parser.add_argument('-attribute_len', type=int, default=5)

    parser.add_argument('-pretrained_model', type=str, default="")

    parser.add_argument('-rank_alpha', type=float, default=1.0)
    parser.add_argument('-patience', type=int, default=7)
    parser.add_argument('-bleu_valid_every_n', type=int, default=5)
    parser.add_argument('-data_dev_combined', required=True)
    parser.add_argument('-beam_size', type=int, default=5)
    parser.add_argument('-seed', type=int, default=0)
    parser.add_argument('-attribute_vocab_size', type=int, default=1000)
    parser.add_argument('-add_attribute', action='store_true')

    

    args = parser.parse_args()
    args.cuda = not args.no_cuda
    args.d_word_vec = args.d_model

    args.load_weights = False
    if args.pretrained_model:
        args.load_weights = True

    np.random.seed(0)
    torch.manual_seed(0)
    args.device = torch.device('cuda' if args.cuda else 'cpu')

    log_path = args.log.split("/")
    log_path = "/".join(log_path[:-1])
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    model_path = args.save_model.split("/")
    model_path = "/".join(model_path[:-1])
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    print(args)

    if args.data_train:
        print("======================================start training======================================")
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
        
        vocab.load(args.vocab)

        args.vocab_size = len(vocab)

        # Build data loader
        data_loader_training = get_loader(args.data_train,
                                             vocab, transform,
                                             args.batch_size, shuffle=True, num_workers=args.num_workers, \
                                             max_seq_len=args.max_seq_len,\
                                             attribute_len=args.attribute_len
                                         )

        data_loader_dev = get_loader(args.data_dev,
                                     vocab, transform_dev,
                                     args.batch_size, shuffle=False, num_workers=args.num_workers, \
                                     max_seq_len=args.max_seq_len,\
                                     attribute_len=args.attribute_len
                                     )

        data_loader_bleu = get_loader_test(args.data_dev_combined,
                                     vocab, transform_dev,
                                     1, shuffle=False,
                                    attribute_len=args.attribute_len
                                     )

        list_of_refs_dev = load_ori_token_data_new(args.data_dev_combined)

        model = get_model(args, load_weights=False)


        print(count_parameters(model))

        # print(model.get_trainable_parameters())
        # init_lr = np.power(args.d_model, -0.5)

        # optimizer = torch.optim.Adam(model.get_trainable_parameters(), lr=init_lr)
        optimizer = get_std_opt(model, args)
        
        train( model, data_loader_training, data_loader_dev, optimizer ,args, vocab, list_of_refs_dev, data_loader_bleu)

    if args.data_test:
        print("======================================start testing==============================")
        args.pretrained_model = args.save_model 
        test(args)




if __name__ == '__main__':
    main()
