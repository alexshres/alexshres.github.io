---
title: Random Notes
date: 2025-09-20 12:00:00 +0000
categories: [Deep Learning, Math]
tags: [transformers, diffeq]
math: true
published: false
---

## random notes and things I want to remember

* Exponential growth basically means that the rate of growth is proportional to the current size - also from basics of differential equations but heard from Daniel Cremers MVG lecture on representing moving scenes
* What makes Transformers so great and so useful is that it is this generalized computation model where the main inductive bias it has is that whatever your inputs are, they need to be converted into a set of embeddings that can be pushed through the model (there's some bias in regards to positioning). 
    * The model learns the hidden biases and relationships within the data itself. 
    * This is analagous to how AlphaGo went from using Go experts to "hard code" game strategy into the model to eventually starting from scratch and learning how to play by playing against itself essentially 
    * Basically Transformers have moved us from an era where we used to include human biases into the design of our ML/DL models to one where we have a model that is very general and will learn the relationships within the data on its own
    * This is one of the main reasons why these models need so much data - they are learning these hidden relationships and domain specific knowledge from scratch
* NeRFs can be represented as a continuous dynamical system
