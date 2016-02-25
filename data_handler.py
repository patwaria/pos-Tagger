#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2016 ayush <ayush@ayush-vm>
#
# Distributed under terms of the MIT license.

import json
import string
import re
import matplotlib.pyplot as plt
from sys import argv 

from collections import Counter, defaultdict

DATA_DIR = 'data/'

from sys import argv
from ast import literal_eval

def read_data(filename):

    with open(DATA_DIR + filename, 'r') as f:
        lines =  f.readlines()

    sentences = defaultdict(list)
    i = 0
    for j, line in enumerate(lines):
        if line == '\n':
            #print sentences[i], '\n'
            i += 1
        else:
            parts = tuple(line.rstrip('\n').split('\t'))
            
            uri = re.compile("[$,\'\"().!?]+")
            # if uri.search(parts[0]) != None:
                # print parts
            sentences[i].append(parts)

    # print len(sentences)
    return sentences

def main():
    # print "Hello World"
    # read_data(argv[1])
    with open("memm.py") as file:
        for line in file:
                line = line.rstrip()
                print line

if __name__ == "__main__":
    main()


