# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 18:41:29 2018

@author: jmf
"""
from keras.layers import Input, Dense, Conv1D, Embedding, Flatten, Average, Concatenate
from keras.layers import GlobalMaxPooling1D, Dot, Dropout, MaxPooling1D
from keras.models import Model, load_model, clone_model
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from keras.utils import to_categorical
from keras.optimizers import Nadam
from sklearn.utils import shuffle
from sklearn import metrics
import numpy as np
import os
import pickle
import csv
import re
import tensorflow as tf
import tensorflow_hub as hub


def getData(fname, maxCharLen, tfhub_parts, letters=None):
    text_input, embedded_text, session = tfhub_parts
    newLetters = set()
    data = []
    y = []
    lens = []
    with open(fname, 'r') as f:
        rdr = csv.reader(f, delimiter=',', quotechar='"')
        for row in rdr:
            txt = ""
            for s in row[1:]:
                txt = txt + " " + re.sub("^\s*(.-)\s*$", "%1", s).replace("\\n", "\n")
#            txt = txt.lower()
            data.append(txt)
            y.append(int(row[0])-1)
            newLetters = set(newLetters)|set(txt)
            lens.append(len(txt))
    if letters is None:
        letters = sorted(newLetters)
        letters = {k: letters.index(k)+1 for k in letters}

    X = np.zeros((len(data), maxCharLen))
    X2 = np.zeros((len(data), 512))
    for n, d in enumerate(data):
        X[n] = getCharRep(d, letters, maxCharLen)
        X2[n] = session.run(embedded_text,
                            feed_dict={text_input: [d]})
    y = to_categorical(y, num_classes=4)
    print(np.mean(lens), np.max(lens), np.std(lens))
    return [X, X2], y, letters

def getCharRep(inp, letters, maxCharLen):
    out = np.zeros((1, maxCharLen))
    for i in range(0, len(inp)):
        if inp[i] in letters:
            out[0, i] = letters[inp[i]]
        if (i+1) == maxCharLen:
            break
        if (i+1) == len(inp):
            break
    return out


def defineModel(maxCharLen, lettersNew, mergeType="average",
                init_tfhub_dropout=0.4):
    sharedSize = 512
    charInp = Input(shape=(maxCharLen,))

    tfhubInp = Input(shape=(512,))
    tfhubside = Dropout(init_tfhub_dropout, name='dropout_tf')(tfhubInp)
    if mergeType == 'average':
        MergeLayer = Average
    else:
        MergeLayer = Concatenate

    # character embedding
    char = Embedding(len(lettersNew)+1, 10,
                     name="embedding_1_new")(charInp)
    char = Conv1D(32, 3, padding="same",
                  activation='relu', name="conv1d_1_new")(char)
    char = Conv1D(64, 3, padding="same",
                  activation='relu', name="conv1d_2_new")(char)
    char = Conv1D(sharedSize, 3, padding="same",
                  activation='relu', name="conv1d_3_new")(char)
    char = Dropout(0.2)(char)
    char = Conv1D(sharedSize, 3, padding="same",
                  activation='relu', name="conv1d_4_new")(char)
    char = MaxPooling1D(pool_size=3, strides=2)(char)
    char = Conv1D(sharedSize, 3, padding="valid",
                  activation='relu', name="conv1d_5_new")(char)
    char = MaxPooling1D(pool_size=3, strides=2)(char)
    char = Conv1D(sharedSize, 3, padding="valid",
                  activation='relu', name="conv1d_6_new")(char)
    char = Conv1D(sharedSize, 3, padding="valid",
                  activation='relu', name="conv1d_7_new")(char)
    char = Dropout(0.2)(char)
    char = Conv1D(sharedSize, 3, padding="valid",
                  activation='relu', name="conv1d_8_new")(char)
    char = GlobalMaxPooling1D()(char)
    char = Dense(sharedSize, activation='relu')(char)
    
    # combine both char-level and tfhub level input
    char = MergeLayer()([char, tfhubside])

    char = Dense(sharedSize, activation='relu')(char)
    char = Dropout(0.25)(char)
    char = Dense(sharedSize, activation='relu')(char)
    out = Dense(4, activation='softmax')(char)
    model = Model([charInp, tfhubInp], out)
    
    rms = Nadam(lr=0.0005)
    model.compile(optimizer=rms, loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model


def init_tfmod(name):
    g = tf.Graph()
    with g.as_default():
      # We will be feeding 1D tensors of text into the graph.
      text_input = tf.placeholder(dtype=tf.string, shape=[None])
      embed = hub.Module(name)
      embedded_text = embed(text_input)
      init_op = tf.group([tf.global_variables_initializer(), tf.tables_initializer()])
    g.finalize()
    # Create session and initialize.
    session = tf.Session(graph=g)
    session.run(init_op)
    return text_input, embedded_text, session


def compute_masked_X2(X2, perc):
    """
    
    """
    num_ex = X2.shape[0]
    out = np.zeros_like(X2)
    p = np.random.rand(num_ex)
    for ind in range(0, num_ex):
        if p[ind] > perc:
            out[ind] = X2[ind]
    return out

logDir = "logs/no_prep"
modelname = "tfhub_AG.cnn"
maxCharLen = 512
numEpochs = 50
hubModName = "https://tfhub.dev/google/universal-sentence-encoder/2"
cloneEvery = 3

callbacks = [
    ModelCheckpoint(filepath=modelname, verbose=1,
                    save_best_only=True, monitor="val_acc"),
    TensorBoard(log_dir=logDir) #  not all of the options work w/ TB+keras
]

 
tfhub_parts = init_tfmod(hubModName)
X, y, lettersNew = getData("data/demo_AG/train.csv", maxCharLen, tfhub_parts)
X, X2 = X
np.random.seed(123)
X, X2, y = shuffle(X, X2, y)
val_len = int(0.15*X.shape[0])
Xtrain = X[val_len:]
X2train = X2[val_len:]
print(np.mean(X2train))
ytrain = y[val_len:]
Xval = X[:val_len]
X2val = X2[:val_len]
yval = y[:val_len]
Xtest, ytest, _ = getData("data/demo_AG/test.csv", maxCharLen, tfhub_parts,
                          letters=lettersNew)
Xtest = Xtest[0], np.zeros_like(Xtest[1])

model = defineModel(maxCharLen, lettersNew, mergeType=None)
dropoutInc = cloneEvery*0.8/numEpochs  # how much dropout to add to the tfhub side
dropoutInc = 0.1
dropout_val = 0.

# the validation data should assume we dont' have tfhub input
X2val = np.zeros_like(X2val)
for ep in range(0, numEpochs):
    masked_X2train = compute_masked_X2(X2train, dropout_val)
    print(np.mean(masked_X2train))
    model.fit([Xtrain, masked_X2train], ytrain, epochs=1, callbacks=callbacks,
              validation_data=([Xval, X2val], yval), shuffle=True)
    if ep % cloneEvery == 0:
        dropout_val += dropoutInc

model = load_model(modelname)
p = model.predict(Xtest)
classes = p.argmax(axis=-1)
ytest2 = [i.tolist().index(1) for i in ytest]
score = metrics.accuracy_score(classes, ytest2)
print("final accuracy on test set: {}".format(score))