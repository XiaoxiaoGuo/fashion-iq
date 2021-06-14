import argparse
import numpy as np
import pickle
import json


def extract_glove_embedding():
    embs = []
    word2id = {'<PAD>': 0,
               '<OOV>': 1}

    glove_dim = 300
    embs.append(np.asarray([0.0] * glove_dim, 'float32'))
    embs.append(np.asarray([0.0] * glove_dim, 'float32'))

    emb_sum = 0
    count = 0
    with open('glove.6B.{}d.txt'.format(glove_dim), 'r') as f:
        for i, line in enumerate(f):
            array = line.strip().split(' ')
            word = array[0]
            word2id[word] = len(word2id)
            e = np.asarray(array[1:], 'float32')
            # print(len(e))
            emb_sum += e
            count += 1
            embs.append(e)
    emb_sum /= count
    embs[word2id['<OOV>']] = emb_sum

    # special token <ACT>
    word2id['<ACT>'] = len(word2id)
    embs.append(-emb_sum)

    save_data = {'word2id': word2id, 'embs': embs}
    with open('dict_x.pt', 'wb') as f:
        pickle.dump(save_data, f)


def extract_vocab_embedding(args):
    with open(args.glove_file, 'rb') as f:
        glove = pickle.load(f)

    vocab_file = args.user_vocab_file.format(args.data_set)
    with open(vocab_file, 'r') as f:
        vocab = json.load(f)

    glove_word2idx = glove['word2id']
    glove_embs = glove['embs']

    vocab_emb = []
    for idx in range(len(vocab['idx2word'])):
        word = vocab['idx2word'][str(idx)]
        if word in glove_word2idx:
            glove_word = word
        else:
            if word in ['<start>', '<end>']:
                glove_word = '<PAD>'
            elif word == '<pad>':
                glove_word = '<PAD>'
            else:
                glove_word = '<OOV>'
        glove_word_idx = glove_word2idx[glove_word]
        vocab_emb.append(glove_embs[glove_word_idx])
        print('idx:', idx, '\tword:', word, '\tglove_word:', glove_word)
    vocab_emb.append(glove_embs[glove_word2idx['<ACT>']])

    with open(args.save.format(args.data_set), 'wb') as f:
        pickle.dump(vocab_emb, f)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--glove_file', type=str,
                        default='dict_x.pt')
    parser.add_argument('--data_set', type=str,
                        default='shirt')
    parser.add_argument('--user_vocab_file', type=str,
                        default='../user_modeling/data/{}_vocab.json')
    parser.add_argument('--save', type=str,
                        default='data/{}_emb.pt')
    args = parser.parse_args()

    extract_glove_embedding()
    extract_vocab_embedding(args)

