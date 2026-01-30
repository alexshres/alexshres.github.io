---
title: Implementing a Transformer in PyTorch
date: 2026-01-26 12:00:00 +0000
categories: [Artificial Intelligence]
tags: [transformers]
math: true
published: false
---

# Implementing a Transformer in PyTorch (WIP)

_Preliminaries_: Assumption is that you have some background/knowledge about how the transformer architecture works. Some good overviews are [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/), or [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/). The second option also includes its own implementation which differs from what I'll be going over. My implementation will follow the [ARENA curriculum](https://www.arena.education/curriculum).  This will most likely be one of two blog posts with this one going over the technical details of implementation and the next one going over training and sampling from the model (sampling might end up being its own post).


My goal in this post is to go through the questions and comments that I had as I try to implement my own small GPT-2 model by expanding upon the exercises I completed from the ARENA curriculum (which I highly recommend - worth every second).

### Architecture Overview

Before we get into the nitty-gritty architecture details, let's first go over some of the main components (or blocks) of the model:
* __Embedding Layer__: Takes a set of tokens (our input) and _embeds_ each token into some high-dimensional vector
* __Positional Embedding Layer__: Takes a set of tokens and creates a position vector embedding for each of the tokens
* __Layer Normalization__: Kind of weird form of normalization which normalizes an input over its features
* __Transformer Block (Residual)__: Where the magic happens, there are two components
    1. __Attention Block__: Each token of an input learns how to "update" itself based on the previous tokens
    2. __MLP Block__: A simple two-layer MLP that is very powerful
* __Unembedding Layer__: Takes our embeddings and turns them back into tokens - our prediction basically

The following image from Neel Nanda's excellent video on [implementing a GPT-2](https://www.youtube.com/watch?v=dsjUDacBw8o&list=PL7m7hLIqA0hoIUPhC26ASCVs_VrqcDpAz&index=3) is a great visualization of the entire process. We will explain everything (Neel does this as well in his video).

![](https://raw.githubusercontent.com/TransformerLensOrg/TransformerLens/refs/heads/clean-transformer-demo/transformer_overview.png)

### Embedding Layers

#### Token Embeddings

Some very basic background about embeddings. If we have a sequence (interchangeable with sentence for now) then each sequence can be broken down into small "pieces" called tokens. Usually these tokens first start off as the ASCII characters of the sequence. So if our sentence is "I ate an apple.", every character, including spaces and the period, is a token. The most common way of creating more complex tokens is using a process called __byte-pair encoding (BPE)__. This process iteratively goes through our sequence and merges these ASCII characters over and over again based on how frequently two adjacent pairs of characters occur in our dataset. A quick example would be we can assume that if we were training on a person's diary, the characters "I", "'", and "m" are probably together often thus maybe "I'm" is a token where as "diary" may be tokenized as "di", "a", and "ry" or "dia" and "ry", based on the frequencies in the dataset. The entire set of tokens, once created using BPE, is called our __vocabulary__ and the number of tokens in our vocabulary is the vocab size. Once we have our vocabulary, each vocab is assigned an ID starting from 0 to `n` where `n` is our vocab size minus one. The way we assign the ID's doesn't really matter as long as once we assign them, they stay consistent from there on out. 

Let's go through a quick example and say our dataset is `["hug", "pug", "pun", "bun", "rug"]` and our BPE process results in the tokens `["h", "u", "g", "p", "n", "r", "b", "un", "ug"]`. These tokens make up our vocabulary and vocab size here is 9. We can assign each token an ID so:

| Token | ID |
| :---: | :---: |
| h | 0 |
| u | 1 |
| g | 2 |
| . | . |
| un | 7 |
| ug | 8 |


There are other tokenization processes and for this post, we will just assume that the tokenization process has already been done for us.

So we now have tokenized our dataset, this basically means that the entire text in our dataset has been converted into numbers. Given some sentence, we can use the tokenization process to get a collection or a set of these numbers which represents our sentence. Using just these positive integers is mathematically "dumb" for neural networks in general. They can be misleading because models can assume that the ID 6 is greater than the ID 0, which makes sense for numbers, but in our context, this doesn't really mean anything. Also, because ID's 0 and 1 are numerically close to each other, the model might interpret that as some how meaning that the tokens "h" and "u" are closer than say "h" and "ug". Using just the integer ID's assumes a __linear__ relationship between the words which isn't true, these ID's are just a label and contain no information about what the word means or how it is being used in our sequence.

This is where embeddings come in. We turn each of these ID's into a vector in a high-dimensional space where each token is a coordinate in this space. The space where each of these tokens get projected onto is called the embedding space or `d_model`, for GPT-2, this was a 768 dimensional space, `d_model=768`. Trying to figure out what each of these dimensions represent is something that researchers are currently working on in a field called Mechanistic Interpretability. Theoretically different directions may represent different features of the text:
    * one direction might represent verb vs noun
    * another direction may represent sentiment

Being in these high-dimensional spaces allows us to do vector math. Classic example being:

$$King - Man + Woman \approx Queen$$

The way we are going to go about these embeddings are that we _learn_ where the model should place the tokens. Initially, these tokens are placed randomly in this large space, and as we train the model, these tokens are moved around in this space until tokens that are related are placed in useful locations relative to one another.

Let's get to the implementation.

```python
class Embedding(nn.Module):
    def __init__(self, d_vocab: int, d_model: int):
        super().__init__()

        # our vocab size
        self.d_vocab = d_vocab

        # our embedding space
        self.d_model = d_model

        # std dev for initialization
        self.init_range = 0.02

        # our learned matrix of where to embed each token
        self.W_E = nn.Parameter(t.empty((d_vocab, d_model)))
        nn.init.normal_(self.W_E, std=init_range)

    def forward(self,
                tokens: Int[Tensor, "batch seq"]
                ) -> Float [Tensor, "batch seq d_model"]:
            
        return self.W_E[tokens, :]
```

##### Syntax
```python
self.W_E = nn.Parameter(t.empty((d_vocab, d_model)))
nn.init.normal_(self.W_E, std=init_range)
```

We first initialize our weight matrix `self.W_E` to be $\in \mathbb{R}^{\text{d\_vocab} \times \text{d\_model}}$. Then `nn.init.normal_(...)` initializes our matrix to have values pulled from a normal distribution with $\mu=0$ and $\sigma=0.02$. Why $0.02$? As Neel says, "cargo culting" - GPT-2 did this so we just copy.

Our weight matrix has `d_vocab` rows and `d_model` columns. This is interpreted as each row represents an embedding and the $i_{th}$ token's embedding is the $i_{th}$ row of `self.W_E`.  `self.W_E[tokens, :]` is just a fancy way of extracting each row for a set of given tokens. Using the example of the personal diary again, each element of the entire diary's vocabulary has an embedding associated with it. If now, we are trying to predict the next token in the sequence "Today was a ____" then "Today was a " gets tokenized by whatever process we used. Then `self.W_E[tokens, :'` uses fancy indexing to extract a matrix of the shape `(seq, d_model)` where `seq` is the length of our sequence "Today was a " after it has been tokenized.

