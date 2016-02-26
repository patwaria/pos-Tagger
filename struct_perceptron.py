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


class StructuredPerceptronTagger(object):
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

	def gen_feats(self, trainsents, rare_word_cutoff, rare_feat_cutoff=5):
		features = defaultdict(int)
		self.X = []
		self.word_freqdist = self.gen_word_freqdist(trainsents)
		for row, sent in trainsents.items():
			history = None
			untagged = untag(sent)
			S = []
			# nltktags = pos_tag(untagged)
			# print nltktags
			for (i, (word, tag)) in enumerate(sent):
				x = dict()
				feature = self.extract_feat(untagged, i, history, rare_word_cutoff)
				# feature['nltk_tag'] = nltktags[i]
				x['features'] = feature
				x['tag'] = tag
				x['target_feat'] = self.phi(feature, tag)
										   
				S.append(x) 
				history = tag
				
				for f in x['target_feat']:
						features[f] += 1
			self.X.append(S)
		self.featuresets = OrderedDict(sorted(features.items(), key=lambda t: t[1], reverse = True))
		
		self.featurenum = dict()
		# for f, count in self.featuresets.items():
				# print f, count
		for (i, ((f, val), tag)) in enumerate(self.featuresets.iterkeys()):
			self.featurenum[((f, val), tag)] = i	
			self.tags[tag] += 1

		for S in self.X:
			for x in S:
				x['f'] = dict()
				for tag in self.tags.iterkeys():
					x['f'][tag] = self.getactivef( self.phi(x['features'], tag) )

				x['target_feat'] = x['f'][x['tag']]
				# print x['features'], x['target_feat']

		print '#Features', len(self.featuresets)
		return None

	def _argmax(self, S):
		y = []
		for x in S:
			maxval = 0.0; t = "START"
			for tag in self.tags.iterkeys():
				score = self.innerProd(x['f'][tag])
				if score > maxval:
					maxval = score
					t = tag
			y.append(t)
		return y

	def argmax(self, sentence, rare_word_cutoff=5):

		N = len(sentence)
		K = len(self.tags)
		
		y = []
		last = None
		for i in xrange(N):
			features = self.extract_feat(sentence, i, last, rare_word_cutoff)
			maxval = 0.0; t = "START"
			for tag in self.tags.iterkeys():
				score = self.innerProd( self.getactivef( self.phi(features, tag) ) )
				if score > maxval:
					maxval = score
					t = tag
			y.append(t)
			last = t;
		return y

	def train(self, trainset, iterations=10, a0=1, rare_word_cutoff=5, rare_feat_cutoff=5):
		self.gen_feats(trainset, rare_word_cutoff, rare_feat_cutoff)		
		self.M = len(self.featurenum)
		self.W = np.random.rand(self.M)
		W = self.W
                A = np.copy(W)
		for i in xrange(iterations):
			rate = a0 / (1 + sqrt(i))
			prevnorm = la.norm(W)	
			for S in self.X:
				gold = [x['tag'] for x in S]
				predict = self._argmax(S)
				if predict != gold:
					for j, x in enumerate(S):
						# promote gold
						for featind in x['f'][gold[j]]:
							W[featind] += rate
						# demote predicted
						for featind in x['f'][predict[j]]:
							W[featind] -= rate
			curnorm = la.norm(W)
                        A = add(A, W)
			print 'Train: iter ', i, ' prevnorm: ', prevnorm, ' curnorm: ', curnorm, ' del: ', abs(curnorm-prevnorm)
		self.W = np.copy(A / (iterations * len(self.W)))
                print 'Training done..'
                
        def test(self, testsents, clftype='argmax'):
            num = 0 
            numsent = corsent = numword = corword = 0.0
            for row, sent in testsents.items():
                # print '#', row    
                untagged = untag(sent)
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
                                    
                num += 1
                if num > 20: break
            tokenacc =  (corword / numword) * 100
            tweetacc = (corsent / numsent) * 100
            print 'Token Acc : ', tokenacc
            print 'Sent Acc : ', tweetacc
            return tokenacc, tweetacc

def gridsearch(tagger, trainset, testset):

    iterations = 10
    tagger.train(trainset)
    for a0 in xrange(1, 6, 1):
        for alpha in np.linspace(0.2, 2.0, 10):
            
            print '\n---------------------------------------------------------------------'
            print 'Params: Iter= ', iterations , 'a0= ', a0, 'alpha= ', alpha
            tagger.sgd(iterations, a0, alpha)
#            tagger.test(testset)

def main():
	print "Structured Perceptron based POS-TAGGER"
	tagger = StructuredPerceptronTagger()
	trainset = read_data("oct27.train")	
	devset = read_data("oct27.dev")
        testset = read_data("oct27.test")
        
        tagger.train(trainset, 10)
        tagger.test(devset)
        tagger.test(testset)

        # gridsearch(tagger, trainset, devset)


if __name__ == "__main__":
    main()


