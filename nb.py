import math
import string
import re
from decimal import *
from sh import find

vocab = {}

word_counts = {
    "cryptids": {},
    "dinos": {}
}

def remove_punctuation(s):
    translator = str.maketrans('', '', string.punctuation)
    return s.translate(translator)

def tokenize(text):
    text = remove_punctuation(text)
    text = text.lower()
    return re.split("\W+", text)

def count_words(words):
    wc = {}
    for word in words:
        wc[word] = wc.get(word, 0.0) + 1.0
    return wc

def get_priors():
    priors = {
        "crypto": 0.0,
        "dino": 0.0
    }

    count_dinos = len(find("training/dinos"))
    count_cryptids = len(find("training/cryptids"))

    total = count_dinos + count_cryptids
    priors["crypto"] = count_dinos / total
    priors["dino"] = count_cryptids / total
    return priors

def train():
    for f in find("training"):
        f = f.strip()
        if not f.endswith(".txt"):
            continue

        category = "cryptids" if "cryptid" in f else "dinos"

        text = open(f).read()
        words = tokenize(text)
        counts = count_words(words)

        for word, count in counts.items():
            if word not in vocab:
                vocab[word] = 0.0
            if word not in word_counts[category]:
                word_counts[category][word] = 0.0

            vocab[word] += count
            word_counts[category][word] += count

def classify(f):
    text = open(f).read()
    words = tokenize(text)
    counts = count_words(words)
    priors = get_priors()
    prior_dino = priors["dino"]
    prior_crypto = priors["crypto"]

    log_prob_crypto = 0.0
    log_prob_dino = 0.0

    total_words = sum(vocab.values())
    total_words_dino = sum(word_counts["dinos"].values())
    total_words_crypto = sum(word_counts["cryptids"].values())

    for word, count in counts.items():
        if word not in vocab or len(word) <= 3:
            continue

        p_word = vocab[word] / total_words
        p_word_given_dino = word_counts["dinos"].get(word, 0.0) / total_words_dino
        p_word_given_crypto = word_counts["cryptids"].get(word, 0.0) / total_words_crypto

        if p_word_given_dino > 0:
            log_prob_dino += math.log(count * p_word_given_dino / p_word)
        if p_word_given_crypto > 0:
            log_prob_crypto += math.log(count * p_word_given_crypto / p_word)
    
    score_dino = Decimal(math.exp(log_prob_dino + math.log(prior_dino)))
    score_crypto = Decimal(math.exp(log_prob_crypto + math.log(prior_crypto)))
    classification = "dino" if score_dino > score_crypto else "crypto"
    
    # Determine confidence
    if score_dino >= score_crypto:
        confidence = score_dino / (score_dino + score_crypto)
    else:
        confidence = score_crypto / (score_dino + score_crypto)
    confidence = str(confidence)[:8]
    
    print("Classifying", f)
    print("Score dino:", score_dino)
    print("Score crypto:", score_crypto)
    print("Classification:", classification)
    print("Confidence:", confidence)

train()

for f in find("testing"):
    f = f.strip()
    if not f.endswith(".txt"):
        continue
    classify(f)
    print()