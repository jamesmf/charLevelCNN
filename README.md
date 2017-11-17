# charLevelCNN
Pretraining method for character-level CNNs

## Methodology
This repository is designed to use a form of negative sampling to create reasonable 'feature extractor' bottom layers of a character-level CNN.

One input will be the character-level representation of a word `i` and the other will be the pretrained GloVe (or similar) vector for a word `j`.

The task will be to predict the cosine similarity between the vectors for words `i` and `j`. Doing well at that task is equivalent to mapping from character space to the pretrained vector space.

The character-input side of the graph can then be recycled as an 'embedder' in other NLP tasks much like pretrained w2v vectors would be in word-level CNN/LSTMs.
