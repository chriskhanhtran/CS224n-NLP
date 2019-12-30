#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
sanity_check_written.py: my version of sanity checks for assignment 5, question 1h, 1i, 1j
Usage:
    sanity_check.py highway
    sanity_check.py cnn
"""
import json
import math
import pickle
import sys
import time

import numpy as np

from docopt import docopt
from typing import List, Tuple, Dict, Set, Union
from tqdm import tqdm
from utils import pad_sents_char, read_corpus, batch_iter
from vocab import Vocab, VocabEntry

from char_decoder import CharDecoder
from nmt_model import NMT
from sanity_check import DummyVocab

import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F

from highway import Highway

#----------
# CONSTANTS
#----------
BATCH_SIZE = 5
EMBED_SIZE = 3
HIDDEN_SIZE = 3
DROPOUT_RATE = 0.0


def question_1h_sanity_check():
    """Sanity check for Highway layer
    """
    print ("-"*80)
    print("Running Sanity Check for Question 1h: Highway")
    print ("-"*80)
    
    # Generate inputs
    x_conv = torch.randn(BATCH_SIZE, EMBED_SIZE)
    
    # Generate weights with shape (out_features, in_features)
    W_proj = torch.randn(EMBED_SIZE, EMBED_SIZE)
    b_proj = torch.randn(EMBED_SIZE)
    
    W_gate = torch.randn(EMBED_SIZE, EMBED_SIZE)
    b_gate = torch.randn(EMBED_SIZE)
    
    # Generate gold outputs
    x_proj_gold = F.relu(x_conv @ W_proj.T + b_proj)
    x_gate_gold = torch.sigmoid(x_conv @ W_gate.T + b_gate)
    x_highway_gold = x_gate_gold * x_proj_gold + (1 - x_gate_gold) * x_conv
    output_size_gold = (BATCH_SIZE, EMBED_SIZE)
    
    # Initialize highway_model
    highway_model = Highway(EMBED_SIZE)
    
    # Load weights and biases for highway_model
    highway_model.linear_proj.weight.data = W_proj
    highway_model.linear_proj.bias.data = b_proj
    highway_model.linear_gate.weight.data = W_gate
    highway_model.linear_gate.bias.data = b_gate
    
    # Generate outputs from highway_model
    x_highway = highway_model(x_conv)
    
    # Check whether output from highway_model is correct
    # Shape of output
    assert(x_highway.size() == output_size_gold), \
        f"Shape of highway output is incorrect: it should be:\n{output_size_gold} but is:\n{x_highway.size()}"
    
    # Content of output
    assert((x_highway == x_highway_gold).all()), \
        f"Highway output is incorrect: it should be:\n{x_highway_gold} but is:\n{x_highway}"
    
    print("All Sanity Checks Passed for Question 1h: Highway!")
    print ("-"*80)

    
def main():
    """ Main func.
    """
    args = docopt(__doc__)

    # Check Python & PyTorch Versions
    assert (sys.version_info >= (3, 5)), "Please update your installation of Python to version >= 3.5"
    assert(torch.__version__ >= "1.0.0"), "Please update your installation of PyTorch. You have {} and you should have version 1.0.0".format(torch.__version__)

    # Seed the Random Number Generators
    seed = 1234
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed * 13 // 7)

    vocab = Vocab.load('./sanity_check_en_es_data/vocab_sanity_check.json') 

    # Create NMT Model
    model = NMT(
        embed_size=EMBED_SIZE,
        hidden_size=HIDDEN_SIZE,
        dropout_rate=DROPOUT_RATE,
        vocab=vocab)

    char_vocab = DummyVocab()

    # Initialize CharDecoder
    decoder = CharDecoder(
        hidden_size=HIDDEN_SIZE,
        char_embedding_size=EMBED_SIZE,
        target_vocab=char_vocab)

    if args['highway']:
        question_1h_sanity_check()
    else:
        raise RuntimeError('invalid run mode')

if __name__ == '__main__':
    main()