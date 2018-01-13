# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 19:56:26 2017

@author: jmf
"""
from keras.layers import Input, Dense, Conv1D, Embedding
from keras.layers import GlobalMaxPooling1D, Dot, Dropout
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
import numpy as np
import os
import pickle

def cos(vector, mat):
    n1 = np.linalg.norm(vector)
    n2 = np.linalg.norm(mat, axis=1)
    dot = mat.dot(vector.T)
    d = np.divide(dot, n1)
    try:
        d = np.divide(d, n2)
    except ValueError:
        d = np.divide(d, n2.reshape(-1,1))
    return d


def readWordVecs(fname="data/ft_100k.txt", style="glove"):
    """
    Reads in a flat file of word vectors and returns a numpy array and a list
    of words. Also creates a list of the letters contained in those words.
    Currently ignored non-ASCII input. Assumes a GloVe style output, but can
    be modified to support .vec files.
    """
    words = []
    letters = set()
    vecLen = 0
    # iterate over input file to see how many legal words we have
    with open(fname, "rb") as f:
        for line in f:
            try:
                line = line.decode("UTF-8").encode('ASCII')
                word = line[:line.find(' ')]
                words.append(word)
                if vecLen == 0:
                    vecLen = len(line.split(' ')) - 2
                letters = set(letters)|set(word)
            except UnicodeError:
                pass
    n = len(words)
    print("read in "+str(n)+" words")
    # define an array of the appropriate size
    out = np.zeros((n, vecLen))
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
    letters = sorted(letters)
    lettersDict = {k: letters.index(k)+1 for k in letters}
    return out, words, lettersDict


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


def getWordRep(word, letters, maxCharLen):
    out = np.zeros((1, maxCharLen))
    maxStart = max(0, maxCharLen - len(word))
    start = np.random.randint(0, maxStart+1)
    for i in range(0, len(word)):
        if word[i] in letters:
            out[0, start+i] = letters[word[i]]
        if (start+i+1) == maxCharLen:
            break
    return out


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
        wordRep = getWordRep(words[wordInd], letters, maxCharLen)
        X1[outIndex] = vec1
        X2[outIndex] = wordRep
        
#        y[outIndex] = c
        y[outIndex] = 1
        outIndex += 1
        negInds = [int(i) for i in np.random.choice(indsAll,
                                                    size=randToPosRatio)]
        vecNeg = mat[negInds]
#        print(words[wordInd], [words[int(i)] for i in negInds])
        cRand = cos(vec2, vecNeg)
#        print(cRand.shape, cRand)
        X1[outIndex:outIndex+randToPosRatio] = vecNeg
        for j in range(0, randToPosRatio):
            X2[outIndex+j] = getWordRep(words[wordInd], letters, maxCharLen)
#        y[outIndex: outIndex+randToPosRatio] = cRand
        outIndex += randToPosRatio

    return X1, X2, y


def exampleGen(*args, **kwargs):
    batchSize = 32
    while True:
        X1, X2, y = createExamples(*args, **kwargs)
        inds = np.arange(0, X1.shape[0])
        np.random.shuffle(inds)
        ind = 0
        while ind < X1.shape[0]:
            indSlice = inds[ind:ind+batchSize]
            yield ([X1[indSlice],
                   X2[indSlice]], y[indSlice])
            ind += batchSize


def checkScore(word, chars, letters, words, maxCharLen):
    numToCheck = 20
    x1 = []
    x2 = []
    for i in range(0, numToCheck):
        x1.append(mat[words.index(word)])
        x2.append(getWordRep(chars, letters, maxCharLen))
    x1 = np.array(x1).reshape(numToCheck,-1)
    x2 = np.array(x2).reshape(numToCheck,-1)
    p = model.predict([x1, x2])
    print(np.mean(p))

def defineModel(letters, maxCharLen, vecLen):
    sharedSize = 128
    charInp = Input(shape=(maxCharLen,))
    vecInp = Input(shape=(vecLen,))
    
    # character embedding
    char = Embedding(len(letters)+1, 10)(charInp)
    char = Conv1D(32, 3, padding="same", dilation_rate=1,
                  activation='relu')(char)
    char = Conv1D(64, 3, padding="same", dilation_rate=2,
                  activation='relu')(char)
    char = Conv1D(sharedSize, 3, padding="same", dilation_rate=4,
                  activation='relu')(char)
    char = Conv1D(sharedSize, 3, padding="same", dilation_rate=8,
                  activation='relu')(char)
    char = GlobalMaxPooling1D()(char)
    char = Dropout(0.5)(char)

    # vector representation
    d = Dense(sharedSize, activation='tanh')(vecInp)
    
    # merge
    dot = Dot(1, normalize=True)([char, d])
    dot = Dense(1, activation='sigmoid')(dot)
    
    model = Model([vecInp, charInp], dot)
    model.compile('adam', 'binary_crossentropy')
    return model

k = 10 #  top k most similar words are considered "positives" in ratio below
randToPosRatio = 4 #  for every word taken from top-k, this many random samples
maxCharLen = 20 #  maximum word length we'll allow

mat, words, letters = readWordVecs()
topN = getTopN(mat, k=k)
train, val, test = splitInds(words)

stepsPer = len(train)*(1+randToPosRatio)/32
allInds = np.arange(0, len(words))
X1val, X2val, yval = createExamples(mat, topN, val, allInds, words,
                                    letters, randToPosRatio)
X1test, X2test, ytest = createExamples(mat, topN, test, allInds,
                                    words, letters, randToPosRatio)

callbacks = [
    EarlyStopping(patience=8),
    ModelCheckpoint(filepath='models/charLevel.cnn', verbose=1,
                    save_best_only=True),
    TensorBoard() #  not all of the options work w/ TB+keras
]

model = defineModel(letters, maxCharLen, X1val.shape[1])
model.fit_generator(exampleGen(mat, topN, train, allInds, words,
                                   letters, randToPosRatio),
                    steps_per_epoch=stepsPer,
                    epochs=100,
                    callbacks=callbacks,
                    validation_data=([X1val, X2val],[yval]))
with open("models/letters.pkl", 'wb') as f:
    pickle.dump(letters, f)