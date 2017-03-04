import string
import nltk
from nltk.corpus import stopwords as stop
from nltk.tokenize import word_tokenize, sent_tokenize
from functools import reduce


class TextProcessor:
    stopwords = set(stop.words("english"))
    punctuation = list(string.punctuation)


    # Converts plain text into list of tokenized sentences
    @staticmethod
    def tokenize(text):
        sentences = sent_tokenize(text)
        return [word_tokenize(sentence) for sentence in sentences]


    # Removes punctuation from a list of sentences
    @staticmethod
    def remove_punctuation(sentences):
        return [[w for w in sentence if w not in TextProcessor.punctuation] for sentence in sentences]


    # Removes stopwords from a list of sentences
    @staticmethod
    def remove_stopwords(sentences):
        return [[w for w in sentence if w.lower() not in TextProcessor.stopwords] for sentence in sentences]


    # Return the word count given a list of sentences
    @staticmethod
    def word_count(sentences):
        return reduce(lambda count, sentence: count + len(sentence), sentences, 0)


    # Tokenize and clean text
    @staticmethod
    def process_text(text):
        sentences = TextProcessor.tokenize(text)
        sentences = TextProcessor.remove_stopwords(sentences)
        sentences = TextProcessor.remove_punctuation(sentences)
        return sentences
