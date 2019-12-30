#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CharDecoder(nn.Module):
    def __init__(self, hidden_size, char_embedding_size=50, target_vocab=None):
        """ Init Character Decoder.

        @param hidden_size (int): Hidden size of the decoder LSTM
        @param char_embedding_size (int): dimensionality of character embeddings
        @param target_vocab (VocabEntry): vocabulary for the target language. See vocab.py for documentation.
        """
        ### YOUR CODE HERE for part 2a
        ### TODO - Initialize as an nn.Module.
        ###      - Initialize the following variables:
        ###        self.charDecoder: LSTM. Please use nn.LSTM() to construct this.
        ###        self.char_output_projection: Linear layer, called W_{dec} and b_{dec} in the PDF
        ###        self.decoderCharEmb: Embedding matrix of character embeddings
        ###        self.target_vocab: vocabulary for the target language
        ###
        ### Hint: - Use target_vocab.char2id to access the character vocabulary for the target language.
        ###       - Set the padding_idx argument of the embedding matrix.
        ###       - Create a new Embedding layer. Do not reuse embeddings created in Part 1 of this assignment.
        super(CharDecoder, self).__init__()
        self.char_embedding_size = char_embedding_size
        self.hidden_size = hidden_size
        self.char_vocab_size = len(target_vocab.char2id)
        self.charDecoder = nn.LSTM(char_embedding_size, hidden_size, bias=True)
        self.char_output_projection = nn.Linear(hidden_size, self.char_vocab_size)
        
        self.decoderCharEmb = nn.Embedding(
            num_embeddings=self.char_vocab_size,
            embedding_dim=char_embedding_size,
            padding_idx=target_vocab.char2id['<pad>']
        )
        
        self.target_vocab = target_vocab
        ### END YOUR CODE


    
    def forward(self, input, dec_hidden=None):
        """ Forward pass of character decoder.

        @param input: tensor of integers, shape (length, batch)
        @param dec_hidden: internal state of the LSTM before reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns scores: called s_t in the PDF, shape (length, batch, self.vocab_size)
        @returns dec_hidden: internal state of the LSTM after reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)
        """
        ### YOUR CODE HERE for part 2b
        ### TODO - Implement the forward pass of the character decoder.
        # Embedding look up for input, return a tensor with shape (length, batch, char_embedding_size)
        X = self.decoderCharEmb(input)
        
        # Apply charDecoder on the input, return:
        # output: hidden state for each time step (h_t), with shape (length, batch, hidden_size);
        # dec_hidden: the last states (h_n, c_n), each with shape (1, batch, hidden_size)
        output, dec_hidden = self.charDecoder(X, dec_hidden)
        
        # Apply linear projection on the output of LSTM,
        # return a tensor with shape (length, batch, self.char_vocab_size)
        scores = self.char_output_projection(output)
        
        return scores, dec_hidden
        ### END YOUR CODE 


    def train_forward(self, char_sequence, dec_hidden=None):
        """ Forward computation during training.

        @param char_sequence: tensor of integers, shape (length, batch). Note that "length" here and in forward() need not be the same.
        @param dec_hidden: initial internal state of the LSTM, obtained from the output of the word-level decoder. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns The cross-entropy loss, computed as the *sum* of cross-entropy losses of all the words in the batch.
        """
        ### YOUR CODE HERE for part 2c
        ### TODO - Implement training forward pass.
        ###
        ### Hint: - Make sure padding characters do not contribute to the cross-entropy loss.
        ###       - char_sequence corresponds to the sequence x_1 ... x_{n+1} from the handout (e.g., <START>,m,u,s,i,c,<END>).
        
        # Initialize input_sequence (chop <END>) and output_sequence (chop <START>),
        # return tensors with shape (length-1, batch)
        input_sequence = char_sequence[:-1, :]
        output_sequence = char_sequence[1:, :]
        
        # Apply a forward pass through char_sequence,
        # return a tensor with shape (length-1, batch, char_vocab_size)
        scores, _ = self.forward(input_sequence, dec_hidden)
        
        # To compute loss, scores needs to be reshaped into (length-1, char_vocab_size, batch)
        scores_input = scores.transpose(1, 2)

        # Compute loss for each input, return tensor of shape (length-1, batch)
        criterion = nn.CrossEntropyLoss(reduction='none')
        loss = criterion(scores_input, output_sequence)
        
        # Mask <pad> tokens in `loss`, sum losses of unpadded inputs
        target_masks = (input_sequence != self.target_vocab.char2id['<pad>'])
        loss = loss[target_masks].sum()
        
        return loss
        ### END YOUR CODE

    def decode_greedy(self, initialStates, device, max_length=21):
        """ Greedy decoding
        @param initialStates: initial internal state of the LSTM, a tuple of two tensors of size (1, batch, hidden_size)
        @param device: torch.device (indicates whether the model is on CPU or GPU)
        @param max_length: maximum length of words to decode

        @returns decodedWords: a list (of length batch) of strings, each of which has length <= max_length.
                              The decoded strings should NOT contain the start-of-word and end-of-word characters.
        """

        ### YOUR CODE HERE for part 2d
        ### TODO - Implement greedy decoding.
        ### Hints:
        ###      - Use target_vocab.char2id and target_vocab.id2char to convert between integers and characters
        ###      - Use torch.tensor(..., device=device) to turn a list of character indices into a tensor.
        ###      - We use curly brackets as start-of-word and end-of-word characters. That is, use the character '{' for <START> and '}' for <END>.
        ###        Their indices are self.target_vocab.start_of_word and self.target_vocab.end_of_word, respectively.
        
        batch_size = initialStates[0].shape[1]
        dec_hidden = initialStates
        
        # Create input tensor with shape (1, batch)
        current_char_input = torch.tensor([self.target_vocab.start_of_word]*batch_size, device=device).view(1, -1)
        
        decoded_ids = []
        for t in range(max_length):
            # Apply a forward pass of charDecoder on current input, return:
            # score: a tensor with shape (1, batch, char_vocab_size)
            # dec_hidden: a tuple of two tensors of size (1, batch, hidden_size)
            score, dec_hidden = self.forward(current_char_input, dec_hidden)
            
            # Apply softmax on score to compute probability distibution
            p = F.softmax(score, dim=2)
            
            # Predict the next character from probability distribution,
            # return a tensor with shape (1, batch)
            current_char_input = torch.argmax(p, dim=2)
            
            # Add the predicted ids to decoded_ids
            decoded_ids.append(current_char_input)   
            
        # Convert decoded_ids from a list of `max_length` tensors with shape (1, batch)
        # to a tensor with shape (max_length, batch)
        decoded_ids = torch.stack(decoded_ids, dim=0).squeeze(1)
        
        # Convert decoded_ids to decodedWords
        decodedWords = []
        for i in range(batch_size):
            input_ids = decoded_ids[:, i].cpu().numpy()    # np.array (max_length,)
            decoded_word = ''
            for id_ in input_ids:
                if id_ == self.target_vocab.end_of_word:
                    break
                decoded_word += self.target_vocab.id2char[id_]
            decodedWords.append(decoded_word)
            
        return decodedWords
        ### END YOUR CODE

