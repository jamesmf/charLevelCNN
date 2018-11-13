# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 19:56:26 2017

@author: jmf
"""
from keras.layers import Input, Dense, Conv1D, Embedding
from keras.layers import GlobalMaxPooling1D, Dot, Dropout, Flatten
from keras.regularizers import l1_l2
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
import numpy as np
import os
import pickle
from gensim.models import KeyedVectors


def read_word_vecs(fname="models/ag_w2v_gensim"):
    """
    Uses gensim to read in word vectors from a flat file. gensim has scripts
    to convert between formats, so we can put them all in the w2v format, which
    can then be loaded with KeyedVectors
    """
    model = KeyedVectors.load_word2vec_format(fname, binary=False)
    words = sorted(list(model.vocab.keys()))
    letters = set(['a'])
    for w in words:
        for l in w:
            letters.add(l)
    letters = sorted(letters)
    lettersDict = {k: letters.index(k)+1 for k in letters}
    return model, words, lettersDict


def split_inds(words, train=0.8, val=0.1, test=0.1):
    """
    Random train/val/test split of the words
    """
    l = len(words)
    r = np.arange(l)
    np.random.shuffle(r)
    trainInd = int(l*train)
    valInd = int(int(l*train)+l*val)
    train = r[:trainInd]
    val = r[trainInd:valInd]
    test = r[valInd:]
    train = [words[i] for i in train]
    val = [words[i] for i in val]
    test = [words[i] for i in test]
    return train, val, test


def get_char_level_rep(word, letters, max_char_len):
    """
    Given a word, the letter_to_embedding_index mapping, and the maximum
    sequence length, return a vector representing the word at character-level
    Args:
        word (str): word to represent
        letters (dict): mapping from character -> index in embedding layer
        max_char_len (int): maximum sequence length in our model
    """
    out = np.zeros((1, max_char_len))
    maxStart = max(0, max_char_len - len(word))
    start = np.random.randint(0, maxStart+1)
    for i in range(0, len(word)):
        if word[i] in letters:
            out[0, start+i] = letters[word[i]]
        if (start+i+1) == max_char_len:
            break
    return out


def create_examples(model, w1_words, w2_words, letters, max_char_len,
                    sim_examples_per_w=1, rand_examples_per_w=1):
    """
    Given a gensim KeyedVectors model, return ((w1, w2), cos(vw1, vw2)) pairs
    where w1 and w2 are character-level representations of words. Words are
    chosen from model.words[inds]. For each w1, we sample w2's in some ratio
    of similar:random expressed by sim_ and rand_examples_per_w. We do this
    using model.most_similar(w1) and sampling from the result for (similar) and
    sampling randomly for (random).
    
    When training we want w1_inds == w2_inds == training_inds. For dev we want
    w1_inds = dev_inds, w2_inds = [training+dev] and for test we want
    w1_inds = test_inds, w2_inds = [training+dev+test]. That way we are
    evaluating entirely unseen examples, but not restricting 
    Args:
        model (gensim.models.KeyedVectors): vector model
        w1_words (list): list of words in current set (train/dev/test)
        w2_words (list): all words the model has seen before
        letters (dict): mapping from character -> index in embedding layer
        max_char_len (int): max length of a word in characters
        sim_examples_per_w (int): number of similar examples per word in set_inds
        rand_examples_per_w (int): number of random examples per word in set_inds
    Returns:
        [[X1, X2], [y]]: X's are char-level reps, y is cosine_similarity
    """
    # get the words our model can see
    w2_set = set(w1_words+w2_words)

    # vector size
    vec_size = max_char_len
    
    # how many examples per word
    ex_per = sim_examples_per_w + rand_examples_per_w

    # initialize some big X1, X2, y matrices
    X1 = np.zeros((len(w1_words)*(ex_per), vec_size))
    X2 = np.zeros((len(w1_words)*(ex_per), vec_size))
    y = np.zeros((len(w1_words)*ex_per, 1))

    # iterate over them, inserting examples into X1, X2, y
    for word_ind, w in enumerate(w1_words):
        # for all the next samples, X1 should have the vector for w
        offset = ex_per * word_ind
        rep_x1 = get_char_level_rep(w, letters, max_char_len)
        X1[offset:offset+ex_per] = rep_x1

        # sample some similar words
        most_sim = model.most_similar(w, topn=100*ex_per)
        most_sim = [i for i in most_sim if i[0] in w2_set]

        sim_range = np.arange(0, len(most_sim))
        sim_w = np.random.choice(sim_range, size=sim_examples_per_w)
        sim_w = [most_sim[ind] for ind in sim_w]

        # put our sampled similar words into X2
        sim_reps_x2 = [get_char_level_rep(sw[0], letters, max_char_len) for sw in sim_w]
        X2[offset:offset+sim_examples_per_w] = sim_reps_x2
        y[offset:offset+sim_examples_per_w] = [i[1] for i in sim_w]

        # get some random words
        rand_words = np.random.choice(w2_words, size=rand_examples_per_w)
        rand_reps_x2 = [get_char_level_rep(rw, letters, max_char_len) for rw in rand_words]
        rand_sims = [model.similarity(w, r) for r in rand_words]
        y[offset+sim_examples_per_w:offset+ex_per] = [i for i in rand_sims]
        X2[offset+sim_examples_per_w:offset+ex_per] = rand_reps_x2
    X = [X1, X2]
    return X, y
        

def example_gen(*args, **kwargs):
    batchSize = 8
    while True:
        X, y = create_examples(*args, **kwargs)
        X1, X2 = X
        inds = np.arange(0, X1.shape[0])
        np.random.shuffle(inds)
        ind = 0
        while ind < X1.shape[0]:
            indSlice = inds[ind:ind+batchSize]
            yield ([X1[indSlice],
                   X2[indSlice]], y[indSlice])
            ind += batchSize


def get_model_layers(inps, embedding_size):
    layers = []
    layers.append(Embedding(len(letters)+1, embedding_size))
    layers.append(Conv1D(64, 3, dilation_rate=1, activation='relu',
                         activity_regularizer=l1_l2(0.05, 0.05)))
    layers.append(Conv1D(128, 3, dilation_rate=2, activation='relu',
                         activity_regularizer=l1_l2(0.05, 0.05)))
    layers.append(Dropout(0.1))
    layers.append(Conv1D(128, 3, dilation_rate=1, activation='relu',
                         activity_regularizer=l1_l2(0.05, 0.05)))
    layers.append(Conv1D(128, 3, dilation_rate=2, activation='relu',
                         activity_regularizer=l1_l2(0.05, 0.05)))
    layers.append(Dropout(0.1))

    for layer in layers:
        for n, side in enumerate(inps):
            inps[n] = layer(side)
    return inps


def define_model(letters, max_char_len):
    shared_size = 128
    char_inp_left = Input(shape=(max_char_len,))
    char_inp_right = Input(shape=(max_char_len,))
    inps = [char_inp_left, char_inp_right]
    left, right = get_model_layers(inps, 64)

    # share a dense layer to project from flatten to something to be dotted
    shared_dense = Dense(shared_size, activation='tanh')
    left = Flatten()(left)
    left = shared_dense(left)

    right = Flatten()(right)
    right = shared_dense(right)
    
    # merge
    dot = Dot(1, normalize=True)([left, right])
    
    model = Model([char_inp_left, char_inp_right], dot)
    model.compile('adam', 'mse')
    return model


max_char_len = 20 #  maximum word length we'll allow

word_model, words, letters = read_word_vecs()

train, val, test = split_inds(words)

steps_per = len(train)/16
w2_words = train+val
Xval, yval = create_examples(word_model, val, w2_words, letters, max_char_len)
w2_words += test
Xtest, ytest = create_examples(word_model, test, w2_words, letters, max_char_len)

callbacks = [
    EarlyStopping(patience=16),
    ModelCheckpoint(filepath='models/charLevel.cnn', verbose=1,
                    save_best_only=True),
    TensorBoard() #  not all of the options work w/ TB+keras
]

model = define_model(letters, max_char_len)
model.fit_generator(example_gen(word_model, train, train, letters, max_char_len),
                    steps_per_epoch=steps_per,
                    epochs=100,
                    callbacks=callbacks,
                    validation_data=(Xval, [yval]))
with open("models/letters.pkl", 'wb') as f:
    pickle.dump(letters, f)