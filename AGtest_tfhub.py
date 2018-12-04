# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 18:41:29 2018

@author: jmf
"""
from keras.layers import Input, Dense, Conv1D, Embedding, Flatten, Average, Concatenate
from keras.layers import GlobalMaxPooling1D, Dot, Dropout, MaxPooling1D, Add
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
#    return X, y, letters

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
                init_tfhub_dropout=0.2, pretraining=False,
                existing_model=None):
    sharedSize = 128
    charInp = Input(shape=(maxCharLen,))

    if pretraining:
        tfhubInp = Input(shape=(512,))
        tfhubside = Dropout(init_tfhub_dropout, name='dropout_tf')(tfhubInp)
        if mergeType == 'average':
            MergeLayer = Average
        else:
            MergeLayer = Concatenate

    # character embedding
    char = Embedding(len(lettersNew)+1, 32,
                     name="embedding_1_new")(charInp)
    res1 = Conv1D(sharedSize, 3, padding="same",
                  activation='relu', name="conv1d_1_new")(char)
    char = Conv1D(sharedSize, 3, padding="same",
                  activation='relu', name="conv1d_2_new")(res1)
    char = Conv1D(sharedSize, 3, padding="same",
                  activation='relu', name="conv1d_3_new")(char)
    char = Dropout(0.2)(char)
    char = Conv1D(sharedSize, 3, padding="same",
                  activation='relu', name="conv1d_4_new")(char)
    char = Add(name='res_conn_1')([char, res1])
#    char = MaxPooling1D(pool_size=3, strides=2)(char)
    char = Conv1D(sharedSize, 3, padding="same",
                  activation='relu', name="conv1d_5_new")(char)
#    char = MaxPooling1D(pool_size=3, strides=2)(char)
    char = Conv1D(sharedSize, 3, padding="same",
                  activation='relu', name="conv1d_6_new")(char)
    char = Conv1D(sharedSize, 3, padding="same",
                  activation='relu', name="conv1d_7_new")(char)
    char = Add(name='res_conn_2')([char, res1])
    char = Dropout(0.2)(char)
    char = Conv1D(sharedSize, 3, padding="same",
                  activation='relu', name="conv1d_8_new")(char)
    char = GlobalMaxPooling1D()(char)
    char = Dense(sharedSize, activation='relu')(char)
    
    # combine both char-level and tfhub level input
    if pretraining:        
        char = MergeLayer()([char, tfhubside])

    char = Dense(sharedSize, activation='relu')(char)
    char = Dropout(0.25)(char)
    char = Dense(sharedSize, activation='relu')(char)
    
    rms = Nadam(lr=0.0005)
    if pretraining:
        # we're pretraining on whether the tfhub input matches the real input
        out = Dense(1, activation='sigmoid')(char)
        model = Model([charInp, tfhubInp], out)
        model.compile(optimizer=rms, loss='binary_crossentropy', metrics=['accuracy'])
    else:
        out = Dense(4, activation='softmax')(char)
        model = Model([charInp], out)
        model.compile(optimizer=rms, loss='categorical_crossentropy',
                      metrics=['accuracy'])
    if existing_model:
        for layer in model.layers:
            ln = layer.name
            try:
                w = existing_model.get_layer(ln).get_weights()
                layer.set_weights(w)
            except ValueError as e:
                print(e)
    return model


def pretrain(model, X, X2):
    """
    The pretraining task is given an input X and a vector representation X2,
    determine whether X and X2 are the same. X is the character-level
    representation of an example, X2 is the 512-dim output from the
    tensorflow-hub model.
    """
    y = np.concatenate((np.ones((X.shape[0],)), np.zeros((X.shape[0],))))
    shuffled_X2 = shuffle(X2.copy())
    X = np.vstack((X, X))
    X2 = np.vstack((X2, shuffled_X2))
    model.fit([X, X2], y, epochs=5, validation_split=0.1)
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

logDir = "logs/for_paper"
modelname = "tfhub_AG.cnn"
maxCharLen = 512
numEpochs = 40
hubModName = "https://tfhub.dev/google/universal-sentence-encoder/2"
cloneEvery = 3
pretraining = True

callbacks = [
    ModelCheckpoint(filepath=modelname, verbose=1,
                    save_best_only=True, monitor="val_acc"),
    TensorBoard(log_dir=logDir) #  not all of the options work w/ TB+keras
]

 
tfhub_parts = init_tfmod(hubModName)
#tfhub_parts = None
X, y, lettersNew = getData("data/demo_AG/train.csv", maxCharLen, tfhub_parts)
X, X2 = X
np.random.seed(123)
X, X2, y = shuffle(X, X2, y)
#X, y = shuffle(X, y)
val_len = int(0.15*X.shape[0])
Xtrain = X[val_len:]
X2train = X2[val_len:]
ytrain = y[val_len:]
Xval = X[:val_len]
X2val = X2[:val_len]
yval = y[:val_len]
Xtest, ytest, _ = getData("data/demo_AG/test.csv", maxCharLen, tfhub_parts,
                          letters=lettersNew)
tfhub_parts[2].close()
Xtest = Xtest[0], _
model = defineModel(maxCharLen, lettersNew, mergeType=None,
                    pretraining=pretraining)
if pretraining:
    model = pretrain(model, Xtrain, X2train)
    model = defineModel(maxCharLen, lettersNew, mergeType=None,
                        existing_model=model)

dropoutInc = cloneEvery*0.8/numEpochs  # how much dropout to add to the tfhub side
dropoutInc = 0.1
dropout_val = 0.

# the validation data should assume we dont' have tfhub input
#X2val = np.zeros_like(X2val)

#    masked_X2train = compute_masked_X2(X2train, dropout_val)
#    print(np.mean(masked_X2train))
model.fit([Xtrain], ytrain, epochs=numEpochs, callbacks=callbacks,
          validation_data=([Xval], yval), shuffle=True)
#    if ep % cloneEvery == 0:
#        dropout_val += dropoutInc

model = load_model(modelname)
p = model.predict(Xtest)
classes = p.argmax(axis=-1)
ytest2 = [i.tolist().index(1) for i in ytest]
score = metrics.accuracy_score(classes, ytest2)
print("final accuracy on test set: {}".format(score))