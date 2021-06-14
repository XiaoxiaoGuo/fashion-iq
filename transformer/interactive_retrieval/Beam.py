import torch
import torch.nn.functional as F
import math
import numpy as np
from torch.autograd import Variable


Constants_PAD = 0


def nopeak_mask(size):
    np_mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8')
    np_mask = Variable(torch.from_numpy(np_mask) == 0)

    return np_mask


def create_masks(trg):
    # src_mask = (src != Constants_PAD.unsqueeze(-2)

    if trg is not None:
        trg_mask = (trg != Constants_PAD).unsqueeze(-2)
        size = trg.size(1)  # get seq_len for matrix
        np_mask = nopeak_mask(size).to(trg_mask.device)

        trg_mask = trg_mask & np_mask

    else:
        trg_mask = None

    return trg_mask


def init_vars(image0, image1, model, opt, vocab, image0_attribute,
              image1_attribute):
    init_tok = vocab.word2idx['<start>']
    image0 = model.cnn1(image0)
    image1 = model.cnn2(image1)

    if model.add_attribute:
        image0_attribute = model.attribute_embedding1(image0_attribute)
        image1_attribute = model.attribute_embedding2(image1_attribute)
        joint_encoding = model.joint_encoding(image0, image1)
        joint_encoding = torch.cat((joint_encoding, image0_attribute), 1)
        joint_encoding = torch.cat((joint_encoding, image1_attribute), 1)
    else:
        joint_encoding = model.joint_encoding(image0, image1)

    e_output = model.encoder(joint_encoding)
    outputs = torch.LongTensor([[init_tok]]).to(opt.device)

    trg_mask = nopeak_mask(1).to(opt.device)
    out = model.out(model.decoder(
        outputs, e_output, trg_mask))

    out = F.softmax(out, dim=-1)
    probs, ix = out[:, -1].data.topk(opt.beam_size)
    log_scores = torch.Tensor(
        [math.log(prob) for prob in probs.data[0]]).unsqueeze(0)

    outputs = torch.zeros(opt.beam_size, opt.max_seq_len).long().to(opt.device)
    outputs[:, 0] = init_tok
    outputs[:, 1] = ix[0]

    e_outputs = torch.zeros(
        opt.beam_size, e_output.size(-2), e_output.size(-1)).to(opt.device)
    e_outputs[:, :] = e_output[0]

    return outputs, e_outputs, log_scores


def k_best_outputs(outputs, out, log_scores, i, k):
    probs, ix = out[:, -1].data.topk(k)

    log_probs = (torch.Tensor(
        [math.log(p) for p in probs.data.view(-1)]).view(k,-1) +
                 log_scores.transpose(0, 1))

    k_probs, k_ix = log_probs.view(-1).topk(k)
    row = k_ix // k
    col = k_ix % k
    outputs[:, :i] = outputs[row, :i]
    outputs[:, i] = ix[row, col]
    log_scores = k_probs.unsqueeze(0)
    return outputs, log_scores


def beam_search(image0, image1, model, opt, vocab, image0_label, image1_label):
    outputs, e_outputs, log_scores = init_vars(
        image0, image1, model, opt, vocab, image0_label, image1_label)
    eos_tok = vocab.word2idx['<end>']
    ind = None

    for i in range(2, opt.max_seq_len):
        trg_mask = nopeak_mask(i).to(opt.device)
        out = model.out(model.decoder(outputs[:, :i], e_outputs, trg_mask))
        out = F.softmax(out, dim=-1)
        outputs, log_scores = k_best_outputs(
            outputs, out, log_scores, i, opt.beam_size)
        # Occurrences of end symbols for all input sentences.
        ones = (outputs == eos_tok).nonzero()

        sentence_lengths = torch.zeros(len(outputs), dtype=torch.long).cuda()
        for vec in ones:
            i = vec[0]
            if sentence_lengths[i] == 0:
                # First end symbol has not been found yet
                sentence_lengths[i] = vec[1]  # Position of first end symbol

        num_finished_sentences = len([s for s in sentence_lengths if s > 0])

        if num_finished_sentences == opt.beam_size:
            alpha = 0.7
            div = 1 / (sentence_lengths.type_as(log_scores) ** alpha)
            _, ind = torch.max(log_scores * div, 1)
            ind = ind.data[0]
            break

    if ind is None:
        out_str = ' '.join([vocab.idx2word[str(tok.item())]
                            for tok in outputs[0][1:]])
        out_idx = [tok.item() for tok in outputs[0][1:]]
        return out_idx, out_str
    else:
        length = (outputs[ind] == eos_tok).nonzero()[0]

        out_str = ' '.join([vocab.idx2word[str(tok.item())]
                            for tok in outputs[ind][1:length]])
        out_idx = [tok.item()
                   for tok in outputs[ind][1:length]]
        return out_idx, out_str


def greedy_search(image0, image1, model, opt, vocab,
                  image0_label, image1_label):
    image0 = model.cnn1(image0)
    image1 = model.cnn2(image1)

    if model.add_attribute:
        image0_attribute = model.attribute_embedding1(image0_label)
        image1_attribute = model.attribute_embedding2(image1_label)
        joint_encoding = model.joint_encoding(image0, image1)
        joint_encoding = torch.cat((joint_encoding, image0_attribute), 1)
        joint_encoding = torch.cat((joint_encoding, image1_attribute), 1)
    else:
        joint_encoding = model.joint_encoding(image0, image1)

    e_outputs = model.encoder(joint_encoding)

    outputs = torch.from_numpy(
        np.zeros((image1.size(0), opt.max_seq_len))).to(
        dtype=torch.long, device=opt.device)

    init_tok = vocab.word2idx['<start>']

    outputs[:, 0] = init_tok
    for i in range(1, opt.max_seq_len):
        trg_mask = nopeak_mask(i).to(opt.device)
        out = model.out(model.decoder(outputs[:, :i], e_outputs, trg_mask))
        probs, ix = out.max(dim=2)
        outputs[:, i] = ix[:, -1]

    end_tok = vocab.word2idx['<end>']
    mask = (outputs == end_tok).to(dtype=torch.float).cumsum(dim=1)
    # print('mask', mask)
    outputs = (outputs * ((mask == 0).to(dtype=torch.int)) +
               end_tok * ((mask > 0).to(dtype=torch.int)))
    out_str = [' '.join([vocab.idx2word[str(tok.item())]
                         for tok in outputs[j][0:]])
               for j in range(outputs.size(0))]

    return outputs, out_str
