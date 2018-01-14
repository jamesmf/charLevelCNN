# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 18:41:29 2018

@author: jmf
"""
from keras.layers import Input, Dense, Conv1D, Embedding, Flatten
from keras.layers import GlobalMaxPooling1D, Dot, Dropout, MaxPooling1D
from keras.models import Model, load_model
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


def getData(fname, maxCharLen, letters=None):
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
            txt = txt.lower()
            data.append(txt)
            y.append(int(row[0])-1)
            newLetters = set(newLetters)|set(txt)
            lens.append(len(txt))
    if letters is None:
        letters = sorted(newLetters)
        letters = {k: letters.index(k)+1 for k in letters}

    X = np.zeros((len(data), maxCharLen))
    for n, d in enumerate(data):
        X[n] = getCharRep(d, letters, maxCharLen)
    y = to_categorical(y, num_classes=4)
    print(np.mean(lens), np.max(lens), np.std(lens))
    return X, y, letters

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


def defineModel(maxCharLen, lettersNew, lettersOld, pretrained=False):
    sharedSize = 128
    charInp = Input(shape=(maxCharLen,))

    layersToSetWeights = ["conv1d_1", "conv1d_2", "conv1d_3",
                          "conv1d_4", "embedding_1"]    
    
    # character embedding
    char = Embedding(len(lettersNew)+1, 10,
                     name="embedding_1_new")(charInp)
    char = Conv1D(32, 3, padding="same", dilation_rate=1,
                  activation='relu', name="conv1d_1_new")(char)
    char = Conv1D(64, 3, padding="same", dilation_rate=2,
                  activation='relu', name="conv1d_2_new")(char)
    char = Conv1D(sharedSize, 3, padding="same", dilation_rate=4,
                  activation='relu', name="conv1d_3_new")(char)
    char = Conv1D(sharedSize, 3, padding="same", dilation_rate=8,
                  activation='relu', name="conv1d_4_new")(char)
    char = MaxPooling1D(pool_size=6, strides=4)(char)
    char = Conv1D(sharedSize, 3, padding="valid", dilation_rate=2,
                  activation='relu', name="conv1d_5_new")(char)
    char = MaxPooling1D(pool_size=2, strides=2)(char)
    char = Conv1D(sharedSize, 3, padding="valid", dilation_rate=2,
                  activation='relu', name="conv1d_6_new")(char)
    char = GlobalMaxPooling1D()(char)
    char = Dense(128, activation='relu')(char)
    char = Dropout(0.5)(char)
    char = Dense(sharedSize, activation='relu')(char)
    char = Dropout(0.5)(char)
    char = Dense(sharedSize, activation='relu')(char)
    out = Dense(4, activation='softmax')(char)
    model = Model(charInp, out)
    
    rms = Nadam(lr=0.0005)
    model.compile(optimizer=rms, loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    if pretrained:
        oldModel = load_model("models/charLevel.cnn")
        for layerName in layersToSetWeights:
            if layerName.find("embedding") > -1:
#                gw = np.random.uniform(-0.1, 0.1, (len(lettersNew), 10))
                gw = model.get_layer(layerName+'_new').get_weights()
                gw[0][0:len(lettersOld)+1,:] = oldModel.get_layer(layerName).get_weights()[0]
                gw = [gw]
            else:
                gw = oldModel.get_layer(layerName).get_weights()
                model.get_layer(layerName+'_new').set_weights(gw)
    return model

pretraining = False
if pretraining:
    logDir = "logs/with_prep"
    modelname = "pretraining_AG.cnn"
else:
    logDir = "logs/no_prep"
    modelname = "no_pretraining_AG.cnn"

callbacks = [
    EarlyStopping(patience=10, monitor="val_acc"),
    ModelCheckpoint(filepath=modelname, verbose=1,
                    save_best_only=True, monitor="val_acc"),
    TensorBoard(log_dir=logDir) #  not all of the options work w/ TB+keras
]

 
maxCharLen = 650
Xtrain, ytrain, lettersNew = getData("data/demo_AG/train.csv", maxCharLen)
np.random.seed(123)
Xtrain, ytrain = shuffle(Xtrain, ytrain)
Xtest, ytest, _ = getData("data/demo_AG/test.csv", maxCharLen, letters=lettersNew)

with open("models/letters.pkl", 'rb') as f:
    lettersOld = pickle.load(f)

model = defineModel(maxCharLen, lettersNew, lettersOld, 
                    pretrained=pretraining)
model.fit(Xtrain, ytrain, epochs=50, callbacks=callbacks,
          validation_split=0.2)
p = model.predict(Xtest)
classes = p.argmax(axis=-1)
ytest2 = [i.tolist().index(1) for i in ytest]
score = metrics.accuracy_score(classes, ytest2)
print(score)