#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

### YOUR CODE HERE for part 1h
class Highway(nn.Module):
    """
    Highway Network that map output of a convolutional layer to word embeddings
    """
    def __init__(self, word_embed_size):
        """
        Init Highway Network.
        
        @param word_embed_size (int): Embedding size of word vectors (dimentionality)
        """
        super(Highway, self).__init__()
        self.word_embed_size = word_embed_size
        self.linear_proj = nn.Linear(word_embed_size, word_embed_size, bias=True)
        self.linear_gate = nn.Linear(word_embed_size, word_embed_size, bias=True)
        
    def forward(self, x_conv: torch.Tensor) -> torch.Tensor:
        """
        Take a mini-batch of output of the convolutional layer, compute the output by using the gate
        to combine the projection with the skip-connection
        
        @param x_conv (Tensor): Tensor of output of convolutional layer with shape (b, e), where
                            b = batch_size, e = embed_size.
        @return x_highway (Tensor): Tensor of output of highway network with shape (b, e)
        """
        x_proj = F.relu(self.linear_proj(x_conv))
        x_gate = torch.sigmoid(self.linear_gate(x_conv))
        x_highway = x_gate * x_proj + (1 - x_gate) * x_conv
        return x_highway
### END YOUR CODE 

