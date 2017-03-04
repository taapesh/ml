import itertools


class Vocab:
    def __init__(self):
        self.vocab = {}
        self.labeled_vocab = {}


    # Add vocab dict to overall vocab
    def add_to_vocab(self, vocab, label=None):
        if label is not None and label not in self.labeled_vocab:
            self.labeled_vocab[label] = {}     

        for word, count in vocab.items():
            if word not in self.vocab:
                self.vocab[word] = 0.0

            if label is not None and word not in self.labeled_vocab[label]:
                self.labeled_vocab[label][word] = 0.0

            self.vocab[word] += count

            if label is not None:
                self.labeled_vocab[label][word] += count


    # Given a list of sentences, return the vocab
    @staticmethod
    def build_vocab(sentences):
        words = list(itertools.chain.from_iterable(sentences))
        wc = {}
        for word in words:
            wc[word] = wc.get(word, 0.0) + 1.0
        return wc


    def get_labeled_vocab(self, label=None):
        if label is None:
            return self.labeled_vocab
        else:
            return self.labeled_vocab[label]


    def vocab_word_count(self, label=None):
        if label is not None:
            return int(sum(self.labeled_vocab[label].values()))
        else:
            return int(sum(self.vocab.values()))