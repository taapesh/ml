import math
from decimal import Decimal
from vocab import Vocab


class NaiveBayes:
    def __init__(self):
        self.priors = {}
        self.labels = {}
        self.vocab = Vocab()


    def train(self, sentences, label):
        # Build vocab
        vocab = Vocab.build_vocab(sentences)

        # Increment label frequency
        self.labels[label] = self.labels.get(label, 0.0) + 1.0

        # Add to label vocab
        self.vocab.add_to_vocab(vocab, label=label)

        # Recalculate prior probabilities
        self.compute_priors()
    

    def compute_priors(self):
        total_labels = sum(self.labels.values())
        
        for label, count in self.labels.items():
            self.priors[label] = count / total_labels


    def classify(self, sentences):
        vocab = Vocab.build_vocab(sentences)
        total_words = self.vocab.vocab_word_count()
        labeled_vocab = self.vocab.get_labeled_vocab()
        log_probabilties = {}


        for label in self.labels:
            log_probabilties[label] = 0.0

        for word, count in vocab.items():
            if word not in self.vocab.vocab:
                continue
            
            p_word = self.vocab.vocab[word] / total_words
            p_word_given_label = {}

            for label in self.labels:
                word_frequency_label = labeled_vocab[label].get(word, 0.0)
                word_count_label = self.vocab.vocab_word_count(label=label)
                p_word_given_label[label] = word_frequency_label / word_count_label


            for label, probability in p_word_given_label.items():
                if probability > 0:
                    log_probabilties[label] += math.log(count * probability / p_word)

        # Calculate scores for each label
        scores = {}
        for label, probability in log_probabilties.items():
            scores[label] = Decimal(math.exp(probability + math.log(self.priors[label])))
        
        total_score = sum(scores.values())

        max_score = -1.0
        max_label = None

        # Determine label with highest score
        for label, score in scores.items():
            if score > max_score:
                max_score = score
                max_label = label


        confidence = max_score / total_score

        return {
            "classification": max_label,
            "confidence": confidence,
            "scores": scores,
            "max_score": max_score
        }