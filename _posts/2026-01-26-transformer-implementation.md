---
title: Implementing a Transformer in PyTorch
date: 2026-01-26 12:00:00 +0000
categories: [Artificial Intelligence]
tags: [transformers]
math: true
---

# Implementing a Transformer in PyTorch (WIP)

_Preliminaries_: Assumption is that you have some background/knowledge about how the transformer architecture works. Some good overviews are [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/), or [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/). The second option also includes its own implementation which differs from what I'll be going over. My implementation will follow the [ARENA curriculum](https://www.arena.education/curriculum).


### Architecture Overview

Before we get into the nitty-gritty architecture details, let's first go over some of the main components (or blocks) of the model:
* __Embedding Layer__: Takes a set of tokens (our input) and _embeds_ each token into some high-dimensional vector
* __Positional Embedding Layer__: Takes a set of tokens and creates a position vector embedding for each of the tokens
* __Layer Normalization__: Kind of weird form of normalization which normalizes an input over its features
* __Transformer Block (Residual)__: Where the magic happens, there are two components
    1. __Attention Block__: Each token of an input learns how to "update" itself based on the previous tokens
    2. __MLP Block__: A simple two-layer MLP that is very powerful
* __Unembedding Layer__: Takes our embeddings and turns them back into tokens - our prediction basically

The following image from Neel Nanda's excellent video on [implementing a GPT-2](https://www.youtube.com/watch?v=dsjUDacBw8o&list=PL7m7hLIqA0hoIUPhC26ASCVs_VrqcDpAz&index=3) is a great visualization of the entire process. We will explain everything.

![](https://raw.githubusercontent.com/TransformerLensOrg/TransformerLens/refs/heads/clean-transformer-demo/transformer_overview.png)
