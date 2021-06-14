import json


class Vocabulary(object):
    """Simple vocabulary wrapper."""

    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

    def init_vocab(self):
        self.add_word('<pad>')
        self.add_word('<start>')
        self.add_word('<end>')
        self.add_word('<and>')
        self.add_word('<unk>')

    def save(self, file_name):
        data = {'word2idx': self.word2idx, 'idx2word': self.idx2word,
                'idx': self.idx}
        with open(file_name, 'w') as f:
            json.dump(data, f, indent=4)
        return

    def load(self, file_name):
        with open(file_name, 'r') as f:
            data = json.load(f)
        self.word2idx = data['word2idx']
        self.idx2word = data['idx2word']
        self.idx = data['idx']
        return
