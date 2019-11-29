# CS224n Assignment 3: Dependency Parsing

## 1. Machine learning and Neural Networks 
### (a) Adam Optimizer
- [Gradient Descent With Momentum](https://www.youtube.com/watch?v=k8fTYJPd3_I)
- [RMSProp](https://www.youtube.com/watch?v=_e-LFe_igno)
- [Adam Optimization Algorithm](https://www.youtube.com/watch?v=JXQT_vxqwIs)
- [An overview of gradient descent optimization algorithms](https://ruder.io/optimizing-gradient-descent/index.html#momentum)

#### i. Momentum :
By taking moving average of the gradients, the update on the dimension where gradients change direction will be small, and the update on the dimesion where gradients point in the same direction will be stronger. Therefore, momentum helps accelarate SGD in relevant direction and dampens oscillation.
    
#### ii. Adaptive learning rate:
By diving the update by $\sqrt{v}$, **Adam** will reduce the update for parameters with large gradients and increase the update for parameters with small gradients

### (b) Dropout
#### i.
$\gamma$ = 1/p
#### ii. 

In training, we apply dropout to prevent all neurons from being updated, helping prevent overfitting. However, in test time, we want to use all neurons to compute outputs. In addition, because there is randomness in dropout, we needs to disable dropout to get consistent predictions for evaluation.

## 2. Neural Transition-Based Dependency Parsing
### (a) Dependency Parsing

| Stack                                 | Buffer                                     | New dependency    | Transition |step|
|---------------------------------------|--------------------------------------------|-------------------|------------|----|
| [ROOT]                                |     [I, parsed, this, sentence, correctly] |                   | Initial    |0   |
| [ROOT, I]                             |     [parsed, this, sentence, correctly]    |                   | Shift      |1   |
| [ROOT, I, parsed]                     |     [this, sentence, correctly]            |                   | Shift      |2   |
| [ROOT, parsed]                        |     [this, sentence, correctly]            |parsed -> I        | Left-Arc   |3   |
| [ROOT, parsed, this]                  |     [ sentence, correctly]                 |                   | Shift      |4   |
| [ROOT, parsed, this, sentence]        |     [correctly]                            |                   | Shift      |5   |
| [ROOT, parsed, sentence]              |     [correctly]                            |sentence -> this   | Left-Arc   |6   |
| [ROOT, parsed]                        |     [correctly]                            |parsed -> sentence | Right-Arc  |7   |
| [ROOT, parsed, correctly]             |     []                                     |                   | Shift      |8   |
| [ROOT, parsed]                        |     []                                     |parsed -> correctly| Right-Arc  |9   |
| [ROOT]                                |     []                                     |Root -> parsed     | Right-Arc  |10  |

### (b) A sentence containing n words will be parsed in how many steps (in terms of n)? 

For a sentence with **n** words, we need:
- n SHIFT to move n words from *buffer* to *stack*
- (n - 1) ARC to make (n - 1) dependency connection between n words
- 1 ARC to connect the root words with *ROOT*

Thus, we need **2n** steps in total.

### (e) Report score:
88.77 UAS on DEV set and 89.06 UAS on Test set

### (f) Four types of parsing error:
- Prepositional Phrase Attachment Error
- Verb Phrase Attachment Error
- Modifier Attachment Error
- Coordination Attachment Error

#### i. I was heading to a wedding fearing my death .
- Error type: Verb Phrase Attachment Error
- Incorrect dependency: wedding -> fearing
- Correct dependency: heading -> fearing

#### ii. It makes me want to rush out and rescue people from dilemmas of their own making .
- Error type: Coordination Attachment Error
- Incorrect dependency: makes -> rescue
- Correct dependency: rush -> rescue

#### iii. It is on loan from a guy named Joe O'Neill in Midland , Texas .
- Error type: Prepositional Phrase Attachment Error
- Incorrect dependency: named -> Midland
- Correct dependency: guy -> Midland

#### iv. Brian has been one of the most crucial elements to the success of Mozilla software .
- Error type: Modifier Attachment Error
- Incorrect dependency: elements -> most
- Correct dependency: crucial -> most

[**Online CoreNLP**](https://corenlp.run/)

#### TAKE AWAY:
- Pipeline:
    - Parser Model: Predict what action to take with 1-layer MLP with input sized (n_features * embedding_size, ) and output sized (3,) for 3 actions (SHIFT / LEFT-ARC / RIGHT-ARC).
    - Parser Trainsition: Performs parse steps by applying predicted transitions to the partial parse
- PyTorch:
    ```python
    # Xavier Uniform Initialization
    self.embed_to_hidden = nn.Linear(self.embed_size * self.n_features, self.hidden_size)
    nn.init.xavier_uniform_(self.embed_to_hidden.weight)

    # Saving and loading weights
    torch.save(model.state_dict(), output_path)
    model.load_state_dict(torch.load(output_path))

    ## Turn off dropout in test time
    model.train() # before training
    model.eval() # before testing
    ```
- Progress bar:
    ```python
    with tqdm(total=(n_minibatches)) as prog:
        for i in range(10):
            prog.update(1)   
    ```