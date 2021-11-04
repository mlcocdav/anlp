'''
Author: Sharon Goldwater 
Date: 2014-09-01, updated 2017-09-30
Copyright: This work is licensed under a Creative Commons
Attribution-NonCommercial 4.0 International License
(http://creativecommons.org/licenses/by-nc/4.0/): You may re-use,
redistribute, or modify this work for non-commercial purposes provided
you retain attribution to any previous author(s).
'''
#from __future__ import division
import sys
from math import log, isclose
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np #numpy provides useful maths and vector operations
from numpy.random import random_sample


#This function isn't actually used in the lab, but included to show
#a simpler example of a bar chart than plot_distributions below.
def plot_histogram(values, counts):
    '''plot_histogram is a very general function that takes a list of
    values and a list of counts (one for each value, counts need no be
    integers), and makes a bar plot showing the count for each value.
    '''
    if len(values) != len(counts):
        print('ERROR: plot_histogram requires two list arguments of the same length')
        sys.exit(1)
    plt.clf()
    #arange() is like range() but returns a numpy array instead of a list
    x_pos = np.arange(len(counts)) 
    #first arg (x_pos) is the position of left hand side of bar
    #second arg (counts) is the height of bar
    plt.bar(x_pos,counts)
    #default bar width is .8, so put labels at left side + .4 (middle of bar)
    plt.xticks(x_pos+.4, values)
    plt.ylim([0,sum(counts)])
    plt.show()

def plot_distributions(true_distr, est_distr):
    ''' plot_distributions takes two distributions, represented as
    dictionaries, with key-value pairs being each outcome and its
    probability.  The first argument is assumed to be the true
    distribution and the second one an estimated distribution. It
    plots these next to each other in a bar plot.  
    '''
    #When we access the items in a dictionary, they are in no
    #particular order, so to make sure the true and estimated
    #probabilities correspond to the same outcome, we need to sort the
    #items in each dictionary, and we also checked to make sure that
    #the keys in the dictionaries match (i.e., they are distributions
    #over the same set of outcomes)
    plt.clf()
    sorted_true = sorted(true_distr.items()) #get sorted list of (key,val) pairs
    #next lines use list comrehension, a concise way to replace for loops
    true_labels = [item[0] for item in sorted_true] #list of keys, still sorted
    true_probs = [item[1] for item in sorted_true] #list of values, still sorted
    sorted_est = sorted(est_distr.items())
    est_labels = [item[0] for item in sorted_est]
    est_probs = [item[1] for item in sorted_est]
    if (true_labels != est_labels):
        print('ERROR: plot_distributions requires two distributions over the same set of outcomes')
        sys.exit(1)
    #the x_pos array will be the position of left side of each bar
    #arange() is like range() except creates an array instead of list
    x_pos = np.arange(len(true_labels))
    bar_width = .4 
    #bar() requires the position and height of each bar;
    #we also add a the width and color
    bars1 = plt.bar(x_pos, true_probs, bar_width, color='r')
    #second distribution has bars shifted to the right by .4
    bars2 = plt.bar(x_pos+.4, est_probs, bar_width, color='b')
    plt.xticks(x_pos+.4, true_labels)
    plt.legend( (bars1[0], bars2[0]), ('True', 'Est'))
    plt.ylim([0,sum(true_probs)]) #set max value of y axis
    plt.xlim([0,len(true_labels)])  #set max value of x axis
    plt.show()

def generate_random_sequence(distribution, N):
    ''' generate_random_sequence takes a distribution (represented as a
    dictionary of outcome-probability pairs) and a number of samples N
    and returns a list of N samples from the distribution.  
    This is a modified version of a sequence generator by fraxel on
    StackOverflow:
    http://stackoverflow.com/questions/11373192/generating-discrete-random-variables-with-specified-weights-using-scipy-or-numpy
    '''
    #As noted elsewhere, the ordering of keys and values accessed from
    #a dictionary is arbitrary. However we are guaranteed that keys()
    #and values() will use the *same* ordering, as long as we have not
    #modified the dictionary in between calling them.
    outcomes = np.array(list(distribution.keys()))
    probs = np.array(list(distribution.values()))
    #make an array with the cumulative sum of probabilities at each
    #index (ie prob. mass func)
    bins = np.cumsum(probs)
    #create N random #s from 0-1
    #digitize tells us which bin they fall into.
    #return the sequence of outcomes associated with that sequence of bins
    #(we convert it from array back to list first)
    return list(outcomes[np.digitize(random_sample(N), bins)])

def normalize_counts(counts):
    ''' normalize_counts takes a dictionary of counts as an argument and
    returns a corresponding dictionary of probabilities by normalizing
    the counts to sum to 1.
    '''
    values = np.array(list(counts.values()))/sum(counts.values())
    normalized_counts = dict(zip(counts.keys(), values))
    ## students need to fill in correct function
    if not (isclose(sum(values), 1.0)):
        print('ERROR: Probability distribution does not sum to 1')
        return None
    return normalized_counts

def compute_likelihood(data, model):
    '''compute_likelihood takes a model (ie distribution, represented
    as a dictionary of outcome-probability pairs) and a list of
    outcomes (the data) and computes the likelihood P(data | model)
    '''
    P = np.prod(np.array([model[letter] for letter in data]))
    return P
        
def compute_log_likelihood(data, model):        
    '''compute_likelihood takes a model (ie distribution, represented
    as a dictionary of outcome-probability pairs) and a list of
    outcomes (the data) and computes the log (base 10                                       ) of the
    likelihood
    '''
    P = sum(np.array([np.log10(model[letter]) for letter in data]))
    return P

def good_touring(data, model):
    estimate = 0
    return estimate
## Main body of code ##

#Create a dictionary that stores a probability distribution
distribution = dict([('a', 0.2),
                    ('b', 0.5),
                    ('c', 0.17),
                    ('d', 0.02),
                    ('e', 0.02),
                    ('f', 0.01),
                    ('g', 0.01),
                    ('h', 0.01),
                    ('i', 0.01),
                    ('j', 0.01),
                    ('k', 0.01),
                    ('l', 0.01),
                    ('m', 0.01),
                    ('n', 0.01)])
if not (isclose(sum(list(distribution.values())), 1.0)):
    print('ERROR: Probability distribution does not sum to 1')
    sys.exit(1)

#Generate a sequence of 50 samples from the distribution.
str_list = generate_random_sequence(distribution, 1000)
#str_list = generate_random_sequence(distribution, 500) #or do 500
print(str_list)

#count how many times each outcome occurred in the sequence and store
#that in a dictionary
#Uses a list comprehension but it could be done with a for loop
str_counts = dict([(s, str_list.count(s)) for s in distribution.keys()])
print('counts:')
#print(sorted(str_counts.items()))
print(str_counts.items())

#normalize to get an estimate of the original distribution
str_probs = normalize_counts(str_counts)
print('est probs:')
#print(sorted(str_probs.items()))
print(str_probs.items())

#compare the true and estimated distributions visually
plot_distributions(distribution, str_probs)
#plot_histogram(np.arange(len(str_counts.values())), str_counts.values())

#for later parts of lab - these functions currently just return 0
L1 = compute_likelihood(str_list,distribution)
L2 = compute_likelihood(str_list,str_probs)
LL1 = compute_log_likelihood(str_list,distribution)
LL2 = compute_log_likelihood(str_list,str_probs)
print('likelihoods: ', L1, L2)
print('log likelihoods: ', LL1, LL2)
