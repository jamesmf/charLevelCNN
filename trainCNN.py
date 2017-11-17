# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 19:56:26 2017

@author: jmf
"""

import numpy as np


def cos(vector, mat):
    n1 = np.linalg.norm(vector)
    n2 = np.linalg.norm(mat, axis=1)
    dot = mat.dot(vector.T)
    d = np.divide(dot, n1)
    d = np.divide(d, n2.reshape(-1,1))
    return d

def readWordVecs(fname="../recipeRNN/data/glove.6B.100d.txt", style="glove"):
    words = []
    with open(fname, "rb") as f:
        for line in f:
            try:
                line = line.decode("UTF-8").encode('ASCII')
                word = line[:line.find(' ')]
                words.append(word)
            except UnicodeError:
                pass
    n = len(words)
    print("read in "+str(n)+" words")
    out = np.zeros((n, 100))
    ind = 0
    with open(fname, "rb") as f:
        for line in f:
            try:
                line = line.decode("ASCII")
                vec = [np.float16(i) for i in line[line.find(' ')+1:].split()]
                out[ind,:] = vec
                ind += 1
                if divmod(ind, 100000)[1] == 0:
                    print(ind)
            except UnicodeError:
                pass
    return out, words

def getTopN(mat, n=10):
    out = np.zeros((mat.shape[0], n))
    for i in range(0, mat.shape[0]):
        if divmod(i, 10)[1] == 0:
            print(i)
        c = cos(mat[i:i+1], mat)[:,0]
        part = np.argpartition(-c, n)[:n]
        out[i, :] = np.array(part)
    return out


mat, words = readWordVecs()
print(words[:2], mat[:2])
c0 = cos(mat[:1], mat)
topN = getTopN(mat)