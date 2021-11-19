import argparse
import itertools
import re
import os
import json
import pickle
from typing import List

import numpy as np
import spacy
from spacy import displacy
from spacy.language import Language
from spacy.pipeline import Pipe
from pathlib import Path
from tqdm import tqdm

from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn import metrics
from sklearn import preprocessing

### Globals strings ###
ONTONOTES_LABELS = [
    'CARDINAL', 'DATE', 'EVENT', 'FAC', 'GPE', 'LANGUAGE', 'LAW', 'LOC', 'MONEY', 'NORP', 'ORDINAL',
    'ORG', 'PERCENT', 'PERSON', 'PRODUCT', 'QUANTITY', 'TIME', 'WORK_OF_ART']

ontonotes_json = "data/ontonotes5_en_16percent.json"

example_text = "On March 8, 2021, a group of hackers including Kottmann and calling themselves " \
       "'APT - 69420 Arson Cats' gained 'super admin' rights in the network of Verkada, a " \
       "cloud-based security camera company, using credentials they found on the public " \
       "internet. They had access to the network for 36 hours. The group collected about 5 " \
       "gigabytes of data, including live security camera footage and recordings from more" \
       " than 150,000 cameras in places like a Tesla factory, a jail in Alabama, a Halifax " \
       "Health hospital, and residential homes. The group also accessed a list of Verkada " \
       "customers and the company's private financial information, and gained superuser " \
       "access to the corporate networks of Cloudflare and Okta through their Verkada cameras."


def setup_argparse():
    p = argparse.ArgumentParser()
    p.add_argument('--part', choices=['1','2','3','4'], required=True)
    p.add_argument('--ents', choices=ONTONOTES_LABELS, nargs='+')
    p.add_argument('--viz_output', default='entity_viz_example.html',
                   help='Name of output file for the visualisation')
    p.add_argument('--corpus', help='name of corpus file to load in')
    p.add_argument('--tokenization', choices=['standard', 'subword'], default='standard',
                   help='for part 2, whether to print out the tokenized document in standard'
                        'tokenization (whitespace), or showing subwords (BPE)')
    p.add_argument('--classifier_path', help='name for path to save classifier')
    p.add_argument('--baseline', action='store_true', help='use a simple baseline classifier')
    return p.parse_args()

#####

def part_1(args, nlp):
    doc = nlp(example_text)
    ent_list, output_file = args.ents, args.viz_output
    options = {"ents": ent_list} if ent_list else {"ents": ONTONOTES_LABELS}

    html = displacy.render(doc, style="ent", options=options)

    output_path = Path(output_file)
    output_path.open("w",  encoding="utf-8").write(html) #str(html.encode("utf-8")))


def part_2(args, nlp):
    # These are special characters used by the tokenizer, ignore them
    special_chars = re.compile("Ġ|<pad>|<s>|</s>|â|Ģ|ī")
    doc = nlp(example_text)

    print("List of Entities:")
    print(doc.ents)

    if args.tokenization == 'standard':
        print("\nStandard Tokenization:")
        print(" ".join([tok.text for tok in doc]))

    elif args.tokenization == 'subword':
        print("\nSubword Tokenization:")
        subword_string = " ".join([tok for tok in itertools.chain(*doc._.trf_data.wordpieces.strings)])
        cleaned_subword_string = special_chars.sub("", subword_string).strip()

        print(cleaned_subword_string)

### This is for Part 3 ###
class ContextualVectors(Pipe):
    def __init__(self, nlp):
        self._nlp = nlp
        self.combination_function = "average"

    def __call__(self, doc):
        if type(doc) == str:
            doc = self._nlp(doc)
        self.lengths = doc._.trf_data.align.lengths
        self.tensors = doc._.trf_data.tensors
        self.input_texts = doc._.trf_data.tokens['input_texts'][0]
        doc.user_token_hooks["vector"] = self.vector
        return doc

    ### HERE is where vectors are set
    def vector(self, token):

        token_start_idx = 1 + sum([self.lengths[ii] for ii in range(token.i)])
        token_end_idx = token_start_idx + self.lengths[token.i]
        trf_vector = self.tensors[0][0][token_start_idx:token_end_idx]
        
        if len(trf_vector) == 0: # this happens due to token alignment issues
            # print('len(trf_vector) = 0!')
            # print(token_start_idx, token_end_idx)
            # print(len(self.tensors[0][0]))
            # print('token.i:', token.i, token.text)
            # print('token_idx:', token_start_idx, token_end_idx)
            # print('input_texts', self.input_texts[token_start_idx:token_end_idx])
            return []
            
        return self.combine_vectors(trf_vector)

    def combine_vectors(self, trf_vector):
        return np.average(trf_vector, axis=0)



@Language.factory("trf_vector_hook", assigns=["doc.user_token_hooks"])
def create_contextual_hook(nlp, name):
    return ContextualVectors(nlp)


def part_3(args, nlp):

    nlp.add_pipe("trf_vector_hook", last=True)
    max_tok = 145  # max tokens per chunk based on the spacy striding behaviour. I can change this if I want
    def chunks(tokens, n):
        for i in range(0, len(tokens), n):
            yield tokens[i:i+n]


    with open(ontonotes_json) as fin:
        f = json.load(fin)
    # process all the data
    corpus = dict.fromkeys(f.keys())
    for key in f.keys():
        #print("loading {}".format(key))
        embeddings, labels = [], []
        corpus_split = f[key]
        for entry in tqdm(corpus_split, desc=f"Processing {key}"):
            if not entry.get("entities"):
                continue
            this_string = entry["text"]
            # BERT max is 512 wordpiece tokens at once, and there is one sample that exceeeds it
            if len(this_string.split()) > max_tok:
                text_chunks = chunks(this_string, max_tok)
            else:
                text_chunks = [this_string]
            for c in text_chunks:
                this_doc = nlp("".join(c))
                # for silver labels:
                for ent in this_doc.ents:
                    
                    try:
                        if not ent.vector.any(): 
                            continue
                    except:
                        # print(f"Error on entity '{ent}' in document: {this_doc}")
                        # print('ent_idx:', ent.start, ent.end)
                        continue
                    # validation check for nans
                    if np.isnan(ent.vector).any() or np.isinf(ent.vector.any()):
                        print(f"Skipping entry, found nan or inf in vector for entity '{ent}' "
                              f"in document: {this_doc}")
                        continue
                    embeddings.append(ent.vector)
                    labels.append(ent.label_)
        # save processed split of corpus, with matrix of number_samples x features, list of labels
        corpus[key] = [np.vstack(embeddings), labels]

    # print number of entities found in each section for information
    for key in corpus.keys():
        print("{}: {} entities".format(key, len(corpus[key][0])))

    save_file = "data/corpus_average.pkl"
    with open(save_file, "wb") as fout:
        pickle.dump(corpus, fout)

    print(f"Saved full processed corpus to {save_file}")

### Errors
# Token indices sequence length is longer than the specified maximum sequence length for this model (720 > 512). Running this sequence through the model will result in indexing errors


def print_classifier_stats(predictions: List[str], labels: List[str], classes: List[str]):
    # TODO check if this works with NER confusion matrix and if it does make a higher and use twice
    accuracy = np.mean(predictions == labels)
    # matrix_labels = (ONTONOTES_LABELS
    #     [label.name for label in Label] + [] if not conf_thresh else [label.name for label in
    #                                                                   Label] + ["below thresh"]
    # )
    print("Classifier Accuracy: {}".format(accuracy))
    print("-" * 89)
    print("Classification Report:")
    print(metrics.classification_report(labels, predictions, target_names=classes, zero_division=0))
    # TODO currently get a broadcast error, fix ValueError: shape mismatch: objects cannot be broadcast to a single shape
    # print("Confusion Matrix:")
    # print(metrics.confusion_matrix(test_labels_, predictions_, labels=[label_encoder.classes_]))


def part_4(args, nlp):
    # this involves reading in ontonotes data, getting embeddings for the entities,
    # then training a classifier with the paired embeddings and labels.
    classifier = LogisticRegression(
        multi_class="multinomial",
        #class_weight="balanced",
        max_iter=500
    )

    # This loads a dict of TESTING, TRAINING, VALIDATION keys and values as a nested list of
    # 0 as embeddings and 1 as labels (co-indexed, equal length)
    with open(args.corpus, "rb") as fin:
        corpus = pickle.load(fin)

    # process data
    label_encoder = preprocessing.LabelEncoder()  # labels need to be ints not strings
    all_labels = list(itertools.chain(*[corpus[split][1] for split in corpus.keys()]))
    label_encoder.fit(all_labels)

    train_data, train_labels_ = corpus["TRAINING"]  # the _ is the spacy convention for the string representation (rather than int/float)
    test_data, test_labels_ = corpus["TESTING"]

    train_labels = label_encoder.transform(train_labels_)  # transform strings to ints

    if args.baseline:
        for strat in ["most_frequent", "uniform", "stratified"]:
            dummy_classifier = DummyClassifier(strategy=strat)
            dummy_classifier.fit(train_data, train_labels)
            dummy_predictions = dummy_classifier.predict(test_data)
            dummy_predictions_ = label_encoder.inverse_transform(dummy_predictions)

            print(f"Stats for Baseline Classifier: {strat} on Test Set")
            print_classifier_stats(dummy_predictions_, test_labels_, label_encoder.classes_)

    else:
        print("Training classifier with params:")
        print(classifier.get_params())
        
        if 'cupy' in str(type(train_data)):
            train_data = train_data.get()
            test_data = test_data.get()

        classifier.fit(train_data, train_labels)

        print("Saving classifier to {}".format(args.classifier_path))
        with open(args.classifier_path, "wb") as fout:
            pickle.dump(classifier, fout)

        predictions = classifier.predict(test_data)
        predictions_ = label_encoder.inverse_transform(predictions)  # inverse transform to strings for printing

        print("Stats for Logistic Regression Classifier on Test Set")
        print_classifier_stats(predictions_, test_labels_, label_encoder.classes_)



def main(args, nlp):
    dict2func = {
        "1": part_1,
        "2": part_2,
        "3": part_3,
        "4": part_4,
    }

    dict2func[args.part](args, nlp)


if __name__ == "__main__":
    args = setup_argparse()

    gpu = spacy.prefer_gpu()
    print('GPU:', gpu)

    # validation checks
    # that model is downloaded
    spacy_model_name = 'en_core_web_trf'
    if not spacy.util.is_package(spacy_model_name):
        spacy.cli.download(spacy_model_name)
    # that relevant directories exist
    for d in ["models", "data"]:
        if not os.path.exists(d):
            os.makedirs(d)

    # load spacy model
    nlp = spacy.load('en_core_web_trf')

    main(args, nlp)
