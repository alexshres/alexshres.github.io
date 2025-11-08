## Attention Mechanism

The first usage of the term "attention" was done by Bahdanau et al. in the paper [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/pdf/1409.0473). At the time, neural machine translation was the new way of doing machine translation - basically using computers to translate text from one language to another.


The main reason for the creation of this new way of doing machine translation was that the previous state-of-the-art (SOTA) models used an encoder-decoder RNN architecture made famous by Sutskever et al [link here](). The issue was that the hidden representation of the combined input was one context vector fed into the decoder. This context vector lacked the ability to accurately represent or summarize the entirety of the input as the input sequence grew larger and larger. Attention was designed to circumvent this issue.


A very very quick background into embeddings first.  Given some sentence, each word needs to have a numerical value that "encodes" that specific word to a vector embedding using a technique such as [Word2vec](https://en.wikipedia.org/wiki/Word2vec). These embeddings are also referred to as tokens (there's more details into how a sentence gets split into tokens but for simplicty we will just assume each word is a token).


Before we get into how attention works, intuitively attention is allowing each token of an input to learn how it should think of itself based on the context of the entire input. For example, in the sentence "I ate an apple", we know that apple is a fruit and not a company primarily due to the verb "ate". The word "apple" attends to "ate" the most in order to move from some generic vector representation of "apple" to one that's more context specific based on the word "ate".