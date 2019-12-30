#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
model_embeddings.py: Embeddings for the NMT model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Anand Dhoot <anandd@stanford.edu>
Michael Hahn <mhahn2@stanford.edu>
"""

import torch.nn as nn

# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(j)

from cnn import CNN
from highway import Highway

# End "do not change" 

class ModelEmbeddings(nn.Module): 
    """
    Class that converts input words to their CNN-based embeddings.
    """
    def __init__(self, embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param embed_size (int): Embedding size (dimensionality) for the output 
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.
        """
        super(ModelEmbeddings, self).__init__()

        ## A4 code
        # pad_token_idx = vocab.src['<pad>']
        # self.embeddings = nn.Embedding(len(vocab.src), embed_size, padding_idx=pad_token_idx)
        ## End A4 code

        ### YOUR CODE HERE for part 1j
        self.embed_size = embed_size
        self.word_embed_size = embed_size
        self.char_embed_size = 50
        pad_token_idx = vocab.char2id['<pad>']
        
        self.char_embeddings = nn.Embedding(
            num_embeddings=len(vocab.char2id),
            embedding_dim=self.char_embed_size,
            padding_idx=pad_token_idx
        )
        
        self.cnn = CNN(
            char_embed_size=self.char_embed_size,
            max_word_length=21,
            kernel_size=5,
            num_filters=self.word_embed_size
        )  
        
        self.highway = Highway(self.word_embed_size)
        self.dropout = nn.Dropout(p=0.3)
        
        ### END YOUR CODE

    def forward(self, input):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, embed_size), containing the 
            CNN-based embeddings for each word of the sentences in the batch
        """
        ## A4 code
        # output = self.embeddings(input)
        # return output
        ## End A4 code

        ### YOUR CODE HERE for part 1j
        sentence_length, batch_size, max_word_length = input.shape
        
        # Embedding lookup, return a tensor with shape (sentence_length, batch_size, max_word_length, char_embed_size)
        x_emb = self.char_embeddings(input)
        
        # Reshape x_emb to shape (sentence_length*batch_size, char_embed_size, max_word_length)
        x_reshape = x_emb.view(-1, max_word_length, self.char_embed_size).transpose(1, 2)
        
        # Apply convolutional network on x_reshape,
        # return a tensor with shape (sentence_length*batch_size, word_embed_size)
        x_conv = self.cnn(x_reshape)
        
        # Apply highway network on x_conv,
        # return a tensor with the same shape
        x_highway = self.highway(x_conv)
        
        # Apply Dropout on x_highway
        x_word_embed = self.dropout(x_highway)
        
        # Reshape x_word_embed to shape (sentence_length, batch_size, word_embed_size)
        x_word_embed = x_word_embed.view(sentence_length, batch_size, self.word_embed_size)
        
        return x_word_embed
        ### END YOUR CODE

