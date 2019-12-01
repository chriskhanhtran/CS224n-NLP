# CS224n Assignment 4: Neural Machine Translation

## 1. Neural Machine Translation with RNNs

### (a) See *utils.py*
### (b) See *model_embeddings.py*
### (c, d, e, f) See *nmt_model.py*
### (g)
By seting values of e_t where enc_masks = 1 at `-inf`, when we use softmax function to compute probability distribution from e_t values, the probability of padded tokens would be zeros (because `exp(-inf)` = 0). Therefore, the weighted average attention layer will ignore hidden layer of padded tokens, which doesn't hold any useful information.

### (h) Train NMT model with GPU
```
tmux new -s nmt
sh run.sh train
tmux detach    # close ssh connection to the server
tmux a -t nmt    # reopen ssh connection
```

### (i) Test NMT model
The BLEU score is 22.61575

### (j) Compare 3 attention mechanism:
- Dot product attention:
    - Pros: Simple and fast
    - Cons: Can not be implemented if the current hidden state and previous hidden states have different lengths (i.e. in NMT, hidden layers of the encoder are vectors of length `2h`, but hidden layers of the decoder are vectors of length `h`.
    - Used when current hidden layers and previous hidden layers are vectors of the same length.
    
- Multiplicative attention:
    - Pros: Can handle hidden layers of different lengths/shapes; Simple and fast by utilizing matrix multiplication
    - Cons: Need a projection step, creating more weights in our model.
    - Used when current hidden layers and previous hidden layers are vectors of the different length.
    
- Additive attention:
    - Pros: Learn separate transformations for current hidden layers and previous hidden layers; Perform better for larger dimesions
    - Cons: More complex than other mechanism, more parameters to learn, less efficient in computation.
    - Used when the dimensionality of the decoder is large.
    
## 2. Analyzing NMT Systems
### Identify error in the NMT translation
#### i.
- Reference Translation: So another one of my favorites, "The Starry Night".
- NMT Translation: Here's another favorite of my favorites, "The Starry Night".

Error: "favorite" is duplicated in the NMT translation

Reason: model limitation in preventing duplications

Solution: put penalty on duplicated translations during beam search

#### ii.
- Reference Translation: You know, what I do is write for children, and I'm probably America's most widely read children's author, in fact.
- NMT Translation: You know what I do is write for children, and in fact, I'm probably the author for children, more reading in the U.S.

Error: the superlative comparison is mistranslated, leading to meaning error

Reason: could be because specific linguistic construct of comparison in Spanish

Solution: train NMT with more comparison examples

#### iii.
- Reference Translation: A friend of mine did that - Richard Bolingbroke.
- NMT Translation: A friend of mine did that - Richard `<unk>`.
    
Error: Out-of-vocabulary word for named entity

Reason: limitations of NMT

Solution: Regconized named entity in the source sentence and copy it into the target sentence
    
#### iv.
- Reference Translation: You've just got to go around the block to see it as an epiphany.
- NMT Translation: You just have to go back to the apple to see it as a epiphany.

Error: "go around" is mistranslated to "go back," "block" is mistranslated to "apple"

Reason: "block" and "apple" could have similar representation in Spanish; however, it is a limitation of NMT because "apple" is unlikely to be after the verb "go" in the sentence

Solution: Add rules to differentiate "block" and "apple"
    
#### v.
- Reference Translation: She saved my life by letting me go to the bathroom in the teachers' lounge.
- NMT Translation: She saved my life by letting me go to the bathroom in the women's room.

Error: Mistranslate teachers with women

Reason: Many examples like this in training data

Solution: Add more training data and rules to correct this error

#### vi.
- Reference Translation: That's more than 250 thousand acres.
- NMT Translation: That's over 100,000 acres.

Error: "hectareas" is translated to "acres"

Reason: There is no "hectarea" in English

Solution: Identify measure unit in the source sentence and keep it as it is in the target sentence.



## 3. TAKE AWAY

- Initilize embeddings:
```python
    # https://pytorch.org/docs/stable/nn.html#torch.nn.Embedding
    # Variables: embedding.weights (Tensor): the learnable weights initialized from N(0,1)
    pad_token_idx = vocab['<pad>']
    embedding = nn.Embedding(len(vocab), embed_size, padding_idx=pad_token_idx)
    # Load from pretrained embeddings
    embedding.weights = nn.Parameter(torch.tensor(embeddings))
    # Get embeddings from `t` (src_len, b) - tensor of input tokens
    X = embedding(t)
```

-  Initilize LSTM:
```python
    # Encoder
    self.encoder = nn.LSTM(embed_size, hidden_size, bias=True, bidirectional=True)
    # Decoder: use LSTMCell because we now move step by step forward in the Decoder
    self.decoder = nn.LSTMCell(embed_size + hidden_size, hidden_size, bias=True)
```

- Encode: pack_padded_sequence() and pad_packed_sequence():

```python
    # pack_padded_sequence(): reduce computation, get correct last hidden layers on padded sentences
    X = pack_padded_sequence(X, lengths=source_lengths, batch_first=False)    # # (src_len, b, e)

    # pad_packed_sequence(): the output of LSTM(packed_padded_sequence) is a batch of packed sequences,
    #        we need to convert output back to the padded batch of hidden layers.
    enc_hiddens, (last_hidden, last_cell) = self.encoder(X)    # enc_hiddens: packed sequence (src_len, b, h*2); last_hidden, last_cell: (2, b, h)
    enc_hiddens, _ = pad_packed_sequence(enc_hiddens, batch_first=True) # enc_hiddens: (b, src_len, h*2)

    # More reading about pack_padded_sequence:
    #        https://stackoverflow.com/questions/51030782/why-do-we-pack-the-sequences-in-pytorch
```

- Other takeaways:
```python
    torch.split()
    torch.squeze()
    torch.unsqueeze()
    torch.bmm()
    tmux
```