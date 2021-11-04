from itertools import product
import numpy as np
import string
import re
from collections import defaultdict
import random

def perplexity(model, lines):
    if isinstance(lines, str):
        lines = lines.split('\n')
    probs = np.array([])
    for line in lines:
        if line =='':
            continue
        sequence = preprocess_line(line)
        probs = np.append(probs,
                np.array([model[sequence[i:i+3]] for i in range(len(sequence)-2)]))
    cross_entropy = -np.mean(np.log2(probs))
    pp = 2**cross_entropy
    return pp

def preprocess_line(line):
    new_line = re.sub('[^a-zA-Z0-9 .]', '', line)
    new_line = re.sub('[0-9]', '0', new_line)
    new_line = new_line.lower()
    new_line = '##'+new_line+'#'
    return new_line

# We added ## at the beginning and # at the end of the lines

def load_model(infile):
    """ loading the model into the dictionary """
    model = dict()
    with open(infile) as f:
        for line in f:
            model[line[:3]] = float(line.split('\t')[-1])
    return model

def load_data(infile, division = [0.8, 0.10, 0.10]):
    with open(infile) as f:
        lines = f.readlines()
    random.Random(3).shuffle(lines)
    n = len(lines)
    train = lines[:int(division[0]*n)]
    val = lines[int(division[0]*n): int((division[0]+division[1])*n)]
    test = lines[int((1-division[2])*n):]
    return train, val, test

def trigram_model(lines, lambdas=(0.05, 0.15 ,0.8), alphas=(0.1,0.1),
                  save = False, model_name = 'best-model.en'):
    """ Using maximum likelihood estimation with alpha smoothing and interpolation"""
    if (sum(lambdas)!=1):
        print('Lambdas must sum to 1!')
        return
    tri_counts = defaultdict(int)
    bi_counts = defaultdict(int)
    uni_counts = defaultdict(int)

    for line in lines:
        line = preprocess_line(line)
        for j in range(len(line) - 2):
            unigram = line[j]
            bigram = line[j:j + 2]
            trigram = line[j:j + 3]
            uni_counts[unigram] += 1
            bi_counts[bigram] += 1
            tri_counts[trigram] += 1

    unigrams_len = sum(uni_counts.values())
    probs = dict()
    alphabet = list(string.ascii_lowercase + ' #0.')
    all_trigrams = product(alphabet, repeat=3)
    for comb in all_trigrams:
        trigram = ''.join(comb)
        p1 = (uni_counts[trigram[2]]) / unigrams_len
        p2 = (bi_counts[trigram[1:]] + alphas[0])/ \
             (uni_counts[trigram[1]]  + alphas[0] * len(alphabet))
        p3 = (tri_counts[trigram] + alphas[1]) / \
             (bi_counts[trigram[0:2]] + alphas[1] * len(alphabet))
        probs[trigram] = lambdas[0] * p1 + lambdas[1] * p2 + lambdas[2] * p3
    if save:
        probs_string = [f'{trigram}\t{str(probs[trigram])}\n' for trigram in probs]
        model_file = open(f'assignment1-data/{model_name}', "w")
        model_file.writelines(probs_string)
        model_file.close()
    return probs


def generate_from_LM(model, n=300):
    trigrams = list(model.keys())
    sequence = '##'
    for i in range(2,n+2):
        poss_trigrams = [t for t in trigrams if t[:2] == sequence[-2:]]
        probs = [model[t] for t in poss_trigrams]
        probs /= np.sum(probs)
        # making sure probabilites are normalized,
        # they should sum to 1 before, but there might have been roundoff errors
        if poss_trigrams != []:
            next_trigram = str(np.random.choice(poss_trigrams, 1,
                                                replace=True, p=probs)[0])
            sequence += next_trigram[-1]
        else:
            poss_trigrams = [t for t in trigrams if t[0] == sequence[-1]]
            probs = [model[t] for t in poss_trigrams]
            probs /= np.sum(probs)
            next_trigram = str(np.random.choice(poss_trigrams, 1,
                                                replace=True, p=probs)[0])
            sequence += next_trigram[1:]
    sequence = sequence.replace('##','')
    sequence = sequence.replace('#',' ')
    sequence = sequence[:n]
    return sequence

model = {'##a': 0.2,  '##b': 0.8,  '###': 0,
         '#aa': 0.2,  '#ab': 0.7,  '#a#': 0.1,
         '#ba': 0.15, '#bb': 0.75, '#b#': 0.1,
         'aaa': 0.4,  'aab': 0.5,  'aa#': 0.1,
         'aba': 0.6,  'abb': 0.3,  'ab#': 0.1,
         'baa': 0.25, 'bab': 0.65, 'ba#': 0.1,
         'bba': 0.5,  'bbb': 0.4,  'bb#': 0.1 }
"""Perplexity stuff"""

#sequence = '##abaab#'
#pp = perplexity(model, sequence)
#print(f'Perplexity of {sequence} is {pp}')


""" Implementing a model and generating from models"""

modelbr_path = './assignment1-data/model-br.en'
modelbr = load_model(modelbr_path)
br_sequence = generate_from_LM(modelbr)

m_trigram = 1
trigram = ''
for i in string.ascii_lowercase + ' #0.':
    for j in string.ascii_lowercase + ' #0.':
        if i == '.':
            i = '\.'
        if j == '.':
            j = '\.'
        grams = [modelbr[x] for x in list(filter(re.compile(f'{i}{j}.').match, list(modelbr.keys())))]
        if len(set(grams))==2 and max(grams)<m_trigram and max(grams) > 0.2333:
            m_trigram = max(grams)
            trigram = f'{i}{j}'
            print(f'{i}{j}', max(grams))
print('Best', trigram)
print([f'{x}: {modelbr[x]}' for x in list(filter(re.compile(f'{trigram[0]}{trigram[1]}.').match, list(modelbr.keys())))])
# en_path = './assignment1-data/training.en'
# de_path = './assignment1-data/training.de'
# es_path = './assignment1-data/training.es'
#
# en_train, en_val, en_test = load_data(en_path, division = [0.85, 0.10, 0.05])
# de_train, de_val, de_test = load_data(de_path, division = [0.85, 0.10, 0.05])
# es_train, es_val, es_test = load_data(es_path, division = [0.85, 0.10, 0.05])
#
# # performing simple grid search, evaluation based on perplexity
# alphas = [0.01, 0.04, 0.05, 0.06, 0.75, 0.1, 0.2]
# alphas = list(product(alphas, alphas))
# lambdas = [[0,0,1], [0.1,0,0.9], [0, 0.1, 0.90], [0.025, 0.075, 0.90],
#            [0.05, 0.10 ,0.85],[0.10, 0.15 ,0.75], [0.1, 0.2, 0.7],
#            [0.15, 0.25, 0.6], [0,0.01,0.99],[0,0.02,0.98], [0.01,0.01,0.98]]
#
# all_combinations = list(product(alphas, lambdas))
# en_perplexity, de_perplexity, es_perplexity = [], [], []
# for parameters in all_combinations:
#     en_model = trigram_model(en_train, alphas = parameters[0], lambdas=parameters[1])
#     de_model = trigram_model(de_train, alphas = parameters[0], lambdas=parameters[1])
#     es_model = trigram_model(es_train, alphas = parameters[0], lambdas=parameters[1])
#     en_perplexity.append(perplexity(en_model, en_val))
#     de_perplexity.append(perplexity(de_model, de_val))
#     es_perplexity.append(perplexity(es_model, es_val))
#
# en_best_params = all_combinations[en_perplexity.index(min(en_perplexity))]
# de_best_params = all_combinations[de_perplexity.index(min(de_perplexity))]
# es_best_params = all_combinations[es_perplexity.index(min(es_perplexity))]
#
# en_bestmodel = trigram_model(en_train, alphas = en_best_params[0],
#                     lambdas=en_best_params[1], save=True, model_name='best_model.en')
# de_bestmodel = trigram_model(de_train, alphas = de_best_params[0],
#                     lambdas=de_best_params[1], save=True, model_name='best_model.de')
# es_bestmodel = trigram_model(es_train, alphas = es_best_params[0],
#                     lambdas=es_best_params[1], save=True, model_name='best_model.es')
#
# en_testperplexity = perplexity(en_bestmodel, en_test)
# de_testperplexity = perplexity(de_bestmodel, de_test)
# es_testperplexity = perplexity(es_bestmodel, es_test)
#
# print(f'Best model parameters for en data: {en_best_params} with val perplexity: '
#       f'{min(en_perplexity)} and test perplexity: {en_testperplexity}')
# print(f'Best model parameters for de data: {de_best_params} with val perplexity: '
#       f'{min(de_perplexity)} and test perplexity: {de_testperplexity}')
# print(f'Best model parameters for es data: {es_best_params} with val perplexity: '
#       f'{min(es_perplexity)} and test perplexity: {es_testperplexity}\n')
#
# """ History n-grams """
# # ng_grams = [f'{x}: {en_bestmodel[x]}' for x in list(filter(re.compile('ng.').match,
# #                                                         list(en_bestmodel.keys())))]
# # for ng in ng_grams:
# #  print(ng)
#
# en_sequence = generate_from_LM(en_bestmodel)
# de_sequence = generate_from_LM(de_bestmodel)
# es_sequence = generate_from_LM(es_bestmodel)
#
# print(f'En model sequence: {en_sequence}\n'
#       f'De model sequence: {de_sequence}\n'
#       f'Es model sequence: {es_sequence}\n'
#       f'Model-br sequence: {br_sequence}\n')
#
# """ Computing perplexity """
# test_path = './assignment1-data/test'
# test_text = open(test_path).read()
# pp_en = perplexity(en_bestmodel, test_text)
# pp_de = perplexity(de_bestmodel, test_text)
# pp_es = perplexity(es_bestmodel, test_text)
#
# print(f'En model perplexity: {pp_en}\n'
#        f'De model perplexity: {pp_de}\n'
#        f'Es model perplexity: {pp_es}\n')