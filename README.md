# RNNs and Long Short-Term Memory (LSTM)

## Introduction
This project was the practical application of my theoretical study of recurrent
networks. My primary goal was to understand why LSTMs solve the vanishing 
gradient problem, which I did by mathematically deriving the BPTT gradients.

Once I had confirmed the theory, this project was my validation step. Instead 
of a redundant from-scratch build, I used a standard Keras/TensorFlow model to 
prove that I could apply that theoretical knowledge to train an effective 
token-level text generator.

## The structure
The structure of my word generator included the following layers:
1. Embedding Layer: Converts IDs into vectors. The vector size chosen was 100 
   as seen in the following code snippet:
   ```python
       keras.layers.Embedding(vocab_size, 100, input_length=max_len_sequence - 1),
   ```
2. LSTM Layer: Responsible for sequence prediction, The internal long and 
   short-term memory is of length 150 as seen in the following code snippet:
   ```python
       keras.layers.LSTM(150),
   ```
3. Dense Layer: This final layer acts as the classification head. It transforms 
   the LSTM's 150-dimensional output vector into a vocab_size-dimensional 
   vector. The softmax activation then converts these values into a probability
   distribution, where each element represents the model's predicted 
   probability that the corresponding token in the vocabulary is the next token
   in the sequence. The following is the configuration of the layer:
   ```python
       keras.layers.Dense(vocab_size, activation="softmax")
   ```

