import pprint
from sh import find
from text_processor import TextProcessor as nlp
from vocab import Vocab
from nb_classifier import NaiveBayes


if __name__=="__main__":
    pp = pprint.PrettyPrinter(indent=4, depth=2)

    # Initialize classifier
    classifier = NaiveBayes()

    # Train
    for f in find("data/1/training"):
        f = f.strip()
        if not f.endswith(".txt"):
            continue

        with open(f) as doc:
            text = doc.read()

        sentences = nlp.process_text(text)

        label = "movie" if "movie" in f else "play"

        classifier.train(sentences, label=label)

    # Test
    for f in find("data/1/testing"):
        f = f.strip()
        if not f.endswith(".txt"):
            continue

        with open(f) as doc:
            text = doc.read()

        sentences = nlp.process_text(text)

        print("Classifying:", f.split("/")[-1])
        prediction = classifier.classify(sentences)
        pp.pprint(prediction)
        print()

