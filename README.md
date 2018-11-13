# charLevelCNN
Pretraining method for character-level CNNs - converts word embeddings to character level embeddings

## Methodology
Word-level embeddings are extremely common, and large organizations have released `.vec` files trained using Word2Vec/fasttext/GloVe on enormous datasets (like Common Crawl). This makes it easy to pre-seed a word-level model with reasonable word embeddings without needing the time or resources to train on a dataset of that size yourself.

But if instead you want to train a character-level model, pretraining isn't as easy.

This repository attemps to solve that problem by using pretrained word embeddings as input to train a shallow character-level CNN that can then be used as the bottom layers of a deeper model.

The model takes pairs of (word_vector_i, character_representation_j) from the w2v/ft/glove .vec file and predicts cosine similarity between words `i` and `j`.

## Results
With a 4-layer CNN architecture, a maximum word length of 20 characters, and a negative sampling ratio of 4:1, the model converged on the validation set after 90 epochs (patience=10).

So without seeing more than just the words in the .vec file, we already have a decent character-level embedder:

```
# 'concedes' is in the test set, never seen by the model
checkScore("acknowledges", "concedes", letters, words, maxCharLen)
>0.898567

# we also have robustness to spelling errors (this misspelling was not in our data at all)
checkScore("acknowledges", "conceeds", letters, words, maxCharLen)
>0.873896

# and we can compare multi-word character inputs
checkScore("acknowledges", "concedes the point", letters, words, maxCharLen
>0.626232

# but unrelated words remain much lower
checkScore("acknowledges", "cat", letters, words, maxCharLen)
>0.0234105

checkScore("acknowledges", "dog", letters, words, maxCharLen)
>0.0299126

```

The model is far from perfect, but could provide a great jump-start to char-level models trained on non-web-scale datasets.

## Experiments

Tested the performance of a pretrained model on the AG news dataset (4-class headline+snippet categorization), available here: 
https://github.com/mhjabreel/CharCNN/tree/master/data/ag_news_csv

Caveats:
- only tried one simple architecture
- no optimization of the pre-trained bottom layers
- .vec file used in pretraining was all lower-cased

Results:
- without pretraining error rate: 13%
- with pretraining error rate: 17% (worse than without pretraining)
- without pretraining, we need a much lower learning rate for the model to converge

![to_train_or_not_to_pretrain_that_is_the_question](https://user-images.githubusercontent.com/7809188/34919946-c7a4951a-f938-11e7-9b66-c3df6d1ad312.png)

Hypotheses:
- if the vocabulary of the news dataset is limited, then much of the pretrained information is 'wasted.'
- without pretraining, any number of layers can model words, whereas with a pretrained model, you start with 4 layers devoted to just single-word embedding
- possibility of dead neurons

## Requirements

The repo requires a little more work to be conveniently repurposable, but it would work with a .vec file saved in `data/` and referenced in the `readWordVecs()` function.

Tested with:
```
keras=2.0.8
numpy=1.13.3
fastText=0.1.0
```

![PCA'd embedding of characters](https://user-images.githubusercontent.com/7809188/34907997-9229e8b4-f856-11e7-87e4-549cc8effb8a.png)
