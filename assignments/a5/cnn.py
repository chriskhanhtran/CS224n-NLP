#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

### YOUR CODE HERE for part 1i
class CNN(nn.Module):
    """1D Convolutional Netword for character embeddings.
    """
    def __init__(self, char_embed_size=50, max_word_length=21, kernel_size=5, num_filters=256):
        """Init CNN.
        
        @params kernel_size (int): Size of kernels/windows
        @params num_filters (int): Number of filters used
        """
        super(CNN, self).__init__()
        self.char_embed_size = char_embed_size
        self.max_word_length = max_word_length
        self.kernel_size = kernel_size
        self.num_filters = num_filters
        
        self.conv1d = nn.Conv1d(
            in_channels=char_embed_size,
            out_channels=num_filters,
            kernel_size=kernel_size,
            bias=True
        )
        
        self.max_pool_1d = nn.MaxPool1d(
            kernel_size=(self.max_word_length - self.kernel_size - 1)
        )
    
    def forward(self, x):
        """Implement 1-dimensional convolutional network on character embeddings with ReLU activation
        and max pooling to compute word embeddings.
        
        @param x (Tensor): Tensor of padded character embeddings with shape (b, e_char, m), where
                            b is batch_size, e_char is char_embed_size, and m is the max_word_length.
        
        @return x_conv (Tensor): Tensor of output of the convolutional network with shape (b, num_filters),
                                 where b is batch_size, and num_filters is equal to word_embed_size.
        """
        # Apply a 1D convotional network over x with ReLU activation,
        # return a tensor with shape (b, f, m-k-1)
        x_conv = F.relu(self.conv1d(x))
        
        # Max pooling over the last dimension of the convolutional output,
        # return a tensor with shape (b, f)
        x_conv, _ = torch.max(x_conv, dim=2)
        #x_conv = self.max_pool_1d(x_conv).squeeze()
        
        return x_conv
        
### END YOUR CODE

