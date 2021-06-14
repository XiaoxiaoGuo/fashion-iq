''' Translate input text with trained model. '''

import torch
import torch.utils.data
import argparse
from tqdm import tqdm
import torchvision.transforms as transforms
from nltk.translate.bleu_score import corpus_bleu
from dataset import get_loader_test, load_ori_token_data_new
from build_vocab import Vocabulary
from Models import get_model, create_masks
from Beam import beam_search
from torch.autograd import Variable
import numpy as np

from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
# from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.ciderd.ciderD import CiderD
from pycocoevalcap.cider.cider import Cider
# from pycocoevalcap.spice.spice import Spice


# from dataset import collate_fn, TranslationDataset
# from transformer.Translator import Translator
# from preprocess import read_instances_from_file, convert_instance_to_idx_seq



def main():
    '''Main Function'''

    parser = argparse.ArgumentParser(description='test.py')

    parser.add_argument('-pretrained_model', required=True,
                        help='Path to model .pt file')
    parser.add_argument('-data_test', required=True,
                        help='Path to input file')
    parser.add_argument('-vocab', required=True,
                        help='Path to vocab file')
    parser.add_argument('-output', default='pred.txt',
                        help="""Path to output the predictions (each line will
                        be the decoded sequence""")
    parser.add_argument('-beam_size', type=int, default=5,
                        help='Beam size')
    parser.add_argument('-batch_size', type=int, default=1,
                        help='Batch size must be 1')
    parser.add_argument('-n_best', type=int, default=1,
                        help="""If verbose is set, will output the n_best
                        decoded sentences""")
    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-crop_size', type=int, default=224, help="""crop size""")
    parser.add_argument('-max_seq_len', type=int, default=64, help="""seq length""")
    parser.add_argument('-attribute_len', type=int, default=5, help="""attribute length""")

    opt = parser.parse_args()
    if args.batch_size != 1:
        print("batch size must be 1")
        exit()

    opt.cuda = not opt.no_cuda
    
    opt.device = torch.device('cuda' if opt.cuda else 'cpu')

    # print(args)
    test(opt)

def test(opt):

    transform = transforms.Compose([
        transforms.CenterCrop(opt.crop_size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    vocab = Vocabulary()

    vocab.load(opt.vocab)
  
    data_loader = get_loader_test(opt.data_test,
                                 vocab, transform,
                                 opt.batch_size, shuffle=False, attribute_len=opt.attribute_len)

    list_of_refs = load_ori_token_data_new(opt.data_test)

    model = get_model(opt, load_weights=True)

    count = 0

    hypotheses = {}

    model.eval()

    for batch in tqdm(data_loader, mininterval=2, desc='  - (Test)', leave=False):
        
        image0, image1, image0_attribute, image1_attribute = map(lambda x: x.to(opt.device), batch)

        hyp = beam_search(image0, image1, model, opt, vocab, image0_attribute, image1_attribute)
#         hyp = greedy_search(image1.to(device), image2.to(device), model, opt, vocab)

        hyp = hyp.split("<end>")[0].strip()

        hypotheses[count] = ["it " + hyp]

        count += 1
    
    # =================================================
    # Set up scorers
    # =================================================
    print('setting up scorers...')
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        # (Meteor(),"METEOR"),
        (Rouge(), "ROUGE_L"),
        # (Cider(), "CIDEr"),
        (Cider(), "CIDEr"),
        (CiderD(), "CIDEr-D")
        # (Spice(), "SPICE")
    ]

    for scorer, method in scorers:
        print('computing %s score...'%(scorer.method()))
        score, scores = scorer.compute_score(list_of_refs, hypotheses)
        if type(method) == list:
            for sc, scs, m in zip(score, scores, method):
                # self.setEval(sc, m)
                # self.setImgToEvalImgs(scs, gts.keys(), m)
                print("%s: %0.3f"%(m, sc))
        else:
            # self.setEval(score, method)
            # self.setImgToEvalImgs(scores, gts.keys(), method)
            print("%s: %0.3f"%(method, score))

    for i in range(len(hypotheses)):
        ref = {i:list_of_refs[i]}
        hyp = {i:hypotheses[i]}
        print(ref)
        print(hyp)
        for scorer, method in scorers:
            print('computing %s score...'%(scorer.method()))
            score, scores = scorer.compute_score(ref, hyp)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                 # self.setEval(sc, m)
                 # self.setImgToEvalImgs(scs, gts.keys(), m)
                    print("%s: %0.3f"%(m, sc))
            else:
                 # self.setEval(score, method)
                 # self.setImgToEvalImgs(scores, gts.keys(), method)
                print("%s: %0.3f"%(method, score))
      

def greedy_search(image1, image2, model, opt, vocab):

    # Autoregressive inference
    embedding_1 = model.cnn(image1)#(1, batch_size, embed_size)

    embedding_2 = model.cnn(image2)#(1, batch_size, embed_size)

    joint_embedding = model.joint_encoding(embedding_1, embedding_2)#(1, batch_size, embed_size)

    e_output = model.encoder(joint_embedding)

    preds_t = torch.LongTensor(np.zeros((image1.size(0), opt.max_seq_len), np.int32)).cuda()
    
    init_tok = vocab.word2idx['<start>']
    
    preds_t[:,0] = init_tok

    for j in range(opt.max_seq_len):

        # _, _preds, _ = model(x_, preds)

        trg_mask = create_masks(preds_t).to(opt.device)

        hidden = model.decoder(preds_t, e_output, trg_mask)#(seq_len, batch_size, hidden)

        logits = model.out(hidden)#(batch_size, seq_len, vocab_size)

        _preds = logits.max(2)[1] #(batch_size, seq_len)

        preds_t[:, j] = _preds.data[:, j]

    preds = preds_t.cpu().numpy()

    return ' '.join([vocab.idx2word[str(tok)] for tok in preds[0]])

if __name__ == "__main__":
    main()


