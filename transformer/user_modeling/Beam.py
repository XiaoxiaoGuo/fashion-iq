import torch
import torch.nn.functional as F
import math
from Models import nopeak_mask, create_masks

def init_vars(image0, image1, model, opt, vocab, image0_attribute, image1_attribute):
    
    init_tok = vocab.word2idx['<start>']
    # src_mask = (src != SRC.vocab.stoi['<pad>']).unsqueeze(-2)
    image0 = model.cnn1(image0)

    image1 = model.cnn2(image1)

    if model.add_attribute:

        attribute = model.attribute_embedding(image0_attribute - image1_attribute).unsqueeze(1)
            # attribute = self.norm(attribute)

            # image0_attribute = self.attribute_embedding1(image0_attribute)

            # image1_attribute = self.attribute_embedding2(image1_attribute)

            # image0 = torch.cat((image0, image0_attribute), 1)
            # image1 = torch.cat((image1, image1_attribute), 1)

            #joint_encoding = self.joint_encoding(torch.cat((image0, image0_attribute),1), torch.cat((image1,image1_attribute),1))
        joint_encoding = model.joint_encoding(image0, image1)
        joint_encoding = torch.cat((joint_encoding, attribute), 1)
        # joint_encoding = model.bn(joint_encoding.transpose(1,2)).transpose(1,2)

    else:
        joint_encoding = model.joint_encoding(image0, image1)

    e_output = model.encoder(joint_encoding)
    
    outputs = torch.LongTensor([[init_tok]]).to(opt.device)
    
    trg_mask = nopeak_mask(1).to(opt.device)
    
    out = model.out(model.decoder(outputs, e_output, trg_mask))# (batch_size, seq_len, vocab_size)

    out = F.softmax(out, dim=-1)
    
    probs, ix = out[:, -1].data.topk(opt.beam_size)

    log_scores = torch.Tensor([math.log(prob) for prob in probs.data[0]]).unsqueeze(0)
    
    outputs = torch.zeros(opt.beam_size, opt.max_seq_len).long().to(opt.device)

    outputs[:, 0] = init_tok

    outputs[:, 1] = ix[0]
    
    e_outputs = torch.zeros(opt.beam_size, e_output.size(-2),e_output.size(-1)).to(opt.device)
    
    e_outputs[:, :] = e_output[0]
    
    return outputs, e_outputs, log_scores

def k_best_outputs(outputs, out, log_scores, i, k):
    
    probs, ix = out[:, -1].data.topk(k)

    log_probs = torch.Tensor([math.log(p) for p in probs.data.view(-1)]).view(k, -1) + log_scores.transpose(0,1)

    k_probs, k_ix = log_probs.view(-1).topk(k)
    
    row = k_ix // k
    col = k_ix % k

    outputs[:, :i] = outputs[row, :i]
    outputs[:, i] = ix[row, col]

    log_scores = k_probs.unsqueeze(0)
    
    return outputs, log_scores

def beam_search(image0, image1, model, opt, vocab, image0_attribute, image1_attribute):
    

    outputs, e_outputs, log_scores = init_vars(image0, image1, model, opt, vocab, image0_attribute, image1_attribute)
    eos_tok = vocab.word2idx['<end>']
    ind = None
    for i in range(2, opt.max_seq_len):
    
        trg_mask = nopeak_mask(i).to(opt.device)

        out = model.out(model.decoder(outputs[:,:i], e_outputs, trg_mask))

        out = F.softmax(out, dim=-1)
    
        outputs, log_scores = k_best_outputs(outputs, out, log_scores, i, opt.beam_size)
        
        ones = (outputs==eos_tok).nonzero() # Occurrences of end symbols for all input sentences.

        sentence_lengths = torch.zeros(len(outputs), dtype=torch.long).cuda()
        for vec in ones:
            i = vec[0]
            if sentence_lengths[i]==0: # First end symbol has not been found yet
                sentence_lengths[i] = vec[1] # Position of first end symbol

        num_finished_sentences = len([s for s in sentence_lengths if s > 0])

        if num_finished_sentences == opt.beam_size:
            alpha = 0.7
            div = 1/(sentence_lengths.type_as(log_scores)**alpha)
            _, ind = torch.max(log_scores * div, 1)
            ind = ind.data[0]
            break
    
    if ind is None:
        # length = (outputs[0]==eos_tok).nonzero()[0]
        # return ' '.join([vocab.idx2word[str(tok.item())] for tok in outputs[0][1:length]])
        return ' '.join([vocab.idx2word[str(tok.item())] for tok in outputs[0][1:]])
    
    else:
        length = (outputs[ind]==eos_tok).nonzero()[0]
        return ' '.join([vocab.idx2word[str(tok.item())] for tok in outputs[ind][1:length]])
