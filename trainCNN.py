# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 19:56:26 2017

@author: jmf
"""

import numpy as np
import os

def cos(vector, mat):
    n1 = np.linalg.norm(vector)
    n2 = np.linalg.norm(mat, axis=1)
    dot = mat.dot(vector.T)
    d = np.divide(dot, n1)
    d = np.divide(d, n2.reshape(-1,1))
    return d


def readWordVecs(fname="../recipeRNN/data/glove.6B.100d.txt", style="glove"):
    """
    Reads in a flat file of word vectors and returns a numpy array and a list
    of words. Also creates a list of the letters contained in those words.
    Currently ignored non-ASCII input. Assumes a GloVe style output, but can
    be modified to support .vec files.
    """
    words = []
    letters = set()
    # iterate over input file to see how many legal words we have
    with open(fname, "rb") as f:
        for line in f:
            try:
                line = line.decode("UTF-8").encode('ASCII')
                word = line[:line.find(' ')]
                words.append(word)
                letters = set(letters)|set(word)
            except UnicodeError:
                pass
    n = len(words)
    print("read in "+str(n)+" words")
    # define an array of the appropriate size
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
    return out, words, sorted(letters)


def getTopN(mat, k=10, chunksize=200):
    """
    Function to return a np array of size (vocab, k) corresponding to the k
    most cosine similar words to the word at index i. If the function has
    already computed it and stored it in the data/ folder, it loads the file
    instead.
    
    Increasing chunksize speeds up computation at the cost of quadratic
    increase in memory.
    """
    if os.path.isfile("data/topN.npy"):
        return np.load("data/topN.npy")
    out = np.zeros((mat.shape[0], k))
    numChunks = int(mat.shape[0]/chunksize) + (mat.shape[0] % chunksize != 0)*1
    for i in range(0, numChunks):
#        if divmod(i, 1000)[1] == 0:
        print(i, '/', numChunks)
        c = cos(mat[i*chunksize:(i+1)*chunksize], mat)
        part = np.argpartition(-c, k, axis=0)[:k].T
        out[i*chunksize:(i+1)*chunksize, :] = part
    np.save("data/topN", out)
    return out


def splitInds(words, train=0.8, val=0.1, test=0.1):
    """
    Random train/val/test split of the word indices
    """
    l = len(words)
    r = np.arange(l)
    np.random.shuffle(r)
    trainInd = int(l*train)
    valInd = int(int(l*train)+l*val)
    train = r[:trainInd]
    val = r[trainInd:valInd]
    test = r[valInd:]
    return train, val, test


def createExamples(mat, topN, indsSplit, indsAll, words, letters,
                   randToPosRatio, style="vec", maxCharLen=20):
    """
    Returns a random set of examples to train on. The outputs are (X1, X2, y).
    If style=="vec" then X1 is the vectors for the sampled words, X2 is the
    characters in the words associated with the ints in indsSplit, and y is
    the cosine similarity between the vectors for those two words. In the
    future, style=="shared" will return X1 and X2 both as the characters in
    the associated words. 
    
    indsSplit : array of the indices of the set you're in (train/val/test)

    indsAll : array of all the indices from which you can draw comparisons

    randToPosRatio: ratio at which to draw 'negative' random samples for each
                    positive sample (from the top-k most similar words)
    """
    # samples is length len(indsSplit)*(1+randToPosRatio)
    randToPosRatio = int(randToPosRatio)
    outSize = len(indsSplit)*(1+randToPosRatio)
    X1 = np.zeros((outSize, mat.shape[1]))
    X2 = np.zeros((outSize, maxCharLen), dtype=np.uint8)
    y = np.zeros(outSize)
    outIndex = 0
    for wordInd in indsSplit:
        posInd = int(np.random.choice(topN[wordInd]))
        vec1 = mat[posInd]
        vec2 = mat[wordInd]
        c = cos(vec1, vec2.reshape(-1,1).T)
        X1[outIndex] = vec1
        X2[outIndex] = vec2
        y[outIndex] = c
        outIndex += 1
        negInds = [int(i) for i in np.random.choice(indsAll,
                                                    size=randToPosRatio)]
        vecNeg = mat[negInds]
        cRand = cos(vec2, vecNeg.T)
        X1[outIndex:outIndex+randToPosRatio] = vecNeg
        
        
        


k = 10 # top k most similar words are considered "positives" in ratio below
randToPosRatio = 2 # for every word taken from top-k, this many random samples


mat, words, letters = readWordVecs()
topN = getTopN(mat, k=k)
train, val, test = splitInds(words)
