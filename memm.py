#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2016 ayush <ayush@ayush-vm>
#
# Distributed under terms of the MIT license.

import os
import collections
import json
import string
import re
import matplotlib.pyplot as plt
import random

from collections import Counter, defaultdict, OrderedDict
from nltk import FreqDist, untag, pos_tag 
import numpy as np
from numpy import empty, zeros, ones, log, exp, sqrt, add, int32, linalg as la
import csv
from sys import argv
from ast import literal_eval

from fuzzy import DMetaphone as DM

from data_handler import read_data


class MEMMTagger(object):
    
    def __init__(self):
            self.hyphen = re.compile("-")
            self.number = re.compile("\d")
            self.uppercase = re.compile('[A-Z]')
            self.atmention = re.compile("@")
            self.hashtag = re.compile("#")
            self.uri = re.compile("http(s)*")
            self.punctuation = re.compile("[$,\'\"().!?\\-]*i")
            self.emoji = re.compile("(\:\w+\:|\<[\/\\]?3|[\(\)\\\D|\*\$][\-\^]?[\:\;\=]|[\:\;\=B8][\-\^]?[3DOPp\@\$\*\\\)\(\/\|])(?=\s|[\!\.\?]|$)")

            self.W = None 	# weight vector - will be assigned in training
            self.K = None 	# number of classes
            self.M = 0	# number of features
            self.tags = defaultdict(int)
            self.X = None   # the training samples
            self.featurenum = None
            self.featuresets = None
            self.dmeta = DM()

    def innerProd(self, active):
            prod = 0.0
            for i in active:
                    prod += self.W[i]
            return prod

    def phi(self, feat, y):
            features = []
            for f, val in feat.items():
                    features.append( ((f, val), y) )
            return features
    
    def gen_word_freqdist(self, trainsents):
            freqdist = defaultdict(int)
            for i, sent in trainsents.items():
                    for (word, tag) in sent:
                            freqdist[word] += 1
            return freqdist

    def getactivef(self, features):
            active = []
            for f in features:
                if f in self.featurenum:
                    active.append(self.featurenum[f])
            return active
    
    def extract_feat(self, sentence, i, history, rare_word_cutoff=5):
            
            features = {}
            if i == 0:
                    features.update({"w-1": "<START>", "t-1": "<START>",
                            "w-2": "<START>"})
            elif i == 1:
                    features.update({"w-1": sentence[i - 1], "t-1": history,
                            "w-2": "<START>"})
            else:
                    features.update({"w-1": sentence[i - 1], "t-1": history,
                            "w-2": sentence[i - 2]})
            # if i == 0:
                    # features.update({"w-1": "<START>", "t-1": "<START>",
                            # "w-2": "<START>", "t-2 t-1": "<START> <START>"})
            # elif i == 1:
                    # features.update({"w-1": sentence[i - 1], "t-1": history[i - 1],
                            # "w-2": "<START>", "t-2 t-1": "<START> %s" % (history[i - 1])})
            # else:
                    # features.update({"w-1": sentence[i - 1], "t-1": history[i - 1],
                            # "w-2": sentence[i - 2], "t-2 t-1": "%s %s" % (history[i - 2], history[i - 1])})

            features["word"] = sentence[i]
            features["len"] = len(sentence[i])
            if self.word_freqdist[sentence[i]] < rare_word_cutoff:
                    features.update({"suffix(1)": sentence[i][-1:]})
                    features.update({"suffix(2)": sentence[i][-2:]})
                    features.update({"suffix(3)": sentence[i][-3:]})
                    features.update({"prefix(1)": sentence[i][:1]})
                    features.update({"prefix(2)": sentence[i][:2]})
                    features.update({"prefix(3)": sentence[i][:3]})
             
            if self.hyphen.search(sentence[i]) != None:
                    features["contains-hyphen"] = True;
            if self.uppercase.search(sentence[i]) != None:
                    features["contains-uppercase"] = True;
            if self.punctuation.search(sentence[i]) != None:
                    features["contains-punctuation"] = True;
            if self.number.search(sentence[i]) != None:
                    features["contains-digits"] = True
            if self.atmention.search(sentence[i]) != None:
                    features["contains-@"] = True
            if self.uri.search(sentence[i]) != None:
                    features["contains-uri"] = True
            if self.emoji.search(sentence[i]) != None:
                    features["contains-emoji"] = True
            
            # Metaphone features
            metaph = self.dmeta(sentence[i])
            num = 0
            for m in metaph:
                if m != None:
                    features['metaph %d' % num] = m
                    num += 1

            # NLTK Tags
            # features['nltk_tag'] = pos_tag( [sentence[i]] )[0][1]

            return features

    def expectation(self, x):
            """
            find the expectation of feature values given x under current model
            """
            enum = 0.0
            eden = 0.0
            expect = defaultdict(float)
            for tag, active in x['f'].iteritems():
                     
                    enum = exp(self.innerProd(active))
                    eden += enum
                    for k in active:
                            expect[k] += enum

            for k in expect.iterkeys():
                    expect[k] /= eden

            return expect

    def gen_feats(self, trainsents, rare_word_cutoff, rare_feat_cutoff=5):
            features = defaultdict(int)
            self.X = []

            self.word_freqdist = self.gen_word_freqdist(trainsents)
            for row, sent in trainsents.items():
                    history = None
                    untagged = untag(sent)
                    # nltktags = pos_tag(untagged)
                    # print nltktags
                    for (i, (word, tag)) in enumerate(sent):
                            x = dict()
                            feature = self.extract_feat(untagged, i, history, rare_word_cutoff)
                            # feature['nltk_tag'] = nltktags[i]
                            x['features'] = feature
                            x['tag'] = tag
                            x['target_feat'] = self.phi(feature, tag)
                                                       
                            self.X.append(x) 
                            history = tag
                            
                            for f in x['target_feat']:
                                    features[f] += 1
                            
            self.featuresets = OrderedDict(sorted(features.items(), key=lambda t: t[1], reverse = True))

            #cutoff rare features
            # self.featuresets = OrderedDict( (key, val) for (key, val) in self.featuresets.iteritems() if val > rare_feat_cutoff) 
            self.featurenum = dict()

            # for f, count in self.featuresets.items():
                    # print f, count

            for (i, ((f, val), tag)) in enumerate(self.featuresets.iterkeys()):
                
                self.featurenum[((f, val), tag)] = i	
                self.tags[tag] += 1

            # print self.tags

            for x in self.X:
                x['f'] = dict()
                for tag in self.tags.iterkeys():
                    x['f'][tag] = self.getactivef( self.phi(x['features'], tag) )

                x['target_feat'] = x['f'][x['tag']]
                # print x['features'], x['target_feat']

            print '#Features', len(self.featuresets)


    def sgd(self, iterations=10, a0=1, alpha=1):
    
            self.M = len(self.featurenum) 
            self.W = zeros(self.M)
            W = self.W

            for i in xrange(iterations):
                    
                    rate = a0 / (sqrt(i) + 1)
                    prevnorm = la.norm(W)
                    # log-likhood gradients
                    for x in self.X:
                            
                        for (j, v) in self.expectation(x).iteritems():
                                W[j] -= rate*v
                        
                        for j in x['target_feat']: # target feature indexes
                                W[j] += rate	
                    
                    # regularizer gradient
                    for k in xrange(len(W)):
                            W[k] -= rate*alpha*W[k]

                    # print W[: 100] 
                    curnorm = la.norm(W)
					# print 'SGD: iter', i
					# print 'SGD: iter ', i, ' prevnorm: ', prevnorm, ' curnorm: ', curnorm, ' del: ', abs(curnorm-prevnorm)

    def condProb(self, x):
            """
            find the log conditional probability of sample x to be assigned labels y under given model
            x = sent
            """
            enum = 0.0
            eden = 0.0
            condprob = defaultdict(float)
            for tag in self.tags.iterkeys():
                    active = self.getactivef( self.phi(x, tag) )
                    enum = exp(self.innerProd(active))
                    eden += enum
                    condprob[tag] = enum

            maxval = -2.0; label = "None"
            for k in condprob.iterkeys():
                    condprob[k] /= eden
                    condprob[k] = condprob[k]
                    if condprob[k] > maxval:
                        maxval, label = condprob[k], k

            return label, condprob
    
    """
    Find the most likely assignment to labels given the model using Viterbi algorithm
    """
    def argmax(self, sentence, rare_word_cutoff=5):

        N = len(sentence)
        K = len(self.tags)

        mat = dict()
        back = dict()
        for i in xrange(N):
            mat[i] = defaultdict(float)
            back[i] = dict()
            for tag in self.tags.iterkeys():
                mat[i][tag] = -1.0
                back[i][tag] = "START"
        # nltktags = pos_tag(sentence) 
        for i in xrange(N):
            if i == 0:
                 features = self.extract_feat(sentence, i, None, rare_word_cutoff)
                 # features['nltk_tag'] = nltktags[i]
                 k, prob = self.condProb(features)
                 mat[i] = prob
            else:
                for tag in self.tags.iterkeys():
                    features = self.extract_feat(sentence, i, tag, rare_word_cutoff)
                    # features['nltk_tag'] = nltktags[i]

                    _, prob = self.condProb(features) 
                    
                    for t in self.tags.iterkeys():
                        if mat[i][t] < (prob[t] * mat[i - 1][tag]):
                            mat[i][t] = prob[t] * mat[i - 1][tag]
                            back[i][t] = tag

        trace = []
        maxval = -1.0
        y = None
        for tag in self.tags.iterkeys():
            if mat[N - 1][tag] > maxval:
                maxval = mat[N - 1][tag]
                y = tag
        
        for i in reversed(xrange(N)):
            trace.append(y)
            y = back[i][y]

        trace.reverse()
        return trace

    def naive_tagsent(self, sentence, rare_word_cutoff=5):

        trace = []
        history = None
        for i in xrange(len(sentence)):
            features = self.extract_feat(sentence, i, history, rare_word_cutoff)
            tag, condprob = self.condProb(features)    
            history = tag
            trace.append(tag)

        return trace

    def test(self, testsents, clftype='argmax'):
        num = 0 
        numsent = corsent = numword = corword = 0.0

        for row, sent in testsents.items():
            # print '#', row    
            untagged = untag(sent)
           
            if clftype == 'naive':
                history = self.naive_tagsent(untagged)
            else: 
                history = self.argmax(untagged)
            
            mistake = False

            numsent += 1
            for (i, (word, tag)) in enumerate(sent):
                # print word, ' tag: ', tag, ' tagged: ', history[i]
                numword += 1
                if tag == history[i]:
                    corword += 1
                else:
                    mistake = True

            if mistake == False:
                corsent += 1

            # num += 1
            # if num > 20: break

              
        tokenacc =  (corword / numword) * 100
        tweetacc = (corsent / numsent) * 100
        print 'Token Acc : ', tokenacc
        print 'Sent Acc : ', tweetacc
 
        return tokenacc, tweetacc

    def train(self, trainsents, iterations=10, a0=1, alpha=0.2,
            rare_word_cutoff=5, rare_feat_cutoff=5):
            
            self.gen_feats(trainsents, rare_word_cutoff, rare_feat_cutoff)		
            self.sgd(iterations, a0, alpha)
            # print self.featurenum
            print 'Training done..'

def gridsearch(tagger, trainset, testset):

    iterations = 10
    tagger.train(trainset)
    for a0 in xrange(1, 6, 1):
        for alpha in np.linspace(0.2, 2.0, 10):
            
            print '\n---------------------------------------------------------------------'
            print 'Params: Iter= ', iterations , 'a0= ', a0, 'alpha= ', alpha
            tagger.sgd(iterations, a0, alpha)
            tagger.test(testset)

def main():
	print "MEMM based POS-TAGGER"
	tagger = MEMMTagger()
	trainset = read_data("oct27.train")	
	devset = read_data("oct27.dev")
        testset = read_data("oct27.test")
        
        tagger.train(trainset)
        tagger.test(devset)
        tagger.test(testset)

        # gridsearch(tagger, trainset, devset)


if __name__ == "__main__":
    main()


