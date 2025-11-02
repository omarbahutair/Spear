# In this project I'll implement text generative model based on Shakespear's
# novels
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow.keras as keras
import numpy as np
import pickle
import requests

# Part 1: Data preprocessing
# 0. Text Retrieval
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
text = requests.get(url).text[:300_000]

# 1. Tokenization Step: to transform words into numbers
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
word_index = tokenizer.word_index
vocab_size = len(word_index) + 1  # for the padding

# 2. Create Text Sequences
text_sequences = []
lines = text.lower().split("\n")

for line in lines:
    tokenized_line = tokenizer.texts_to_sequences([line])[0]

    for i in range(1, len(tokenized_line)):
        text_sequences.append(tokenized_line[:i + 1])

# 3. Padding Sequences: to ensure all sequences are of the same length
max_len_sequence = max([len(x) for x in text_sequences])

# Note: the reason why we used pre padding is because in LSTMs the most recent
# words are the most important
padded_sequences = np.array(pad_sequences(text_sequences, maxlen=max_len_sequence, padding="pre"))

X_train = padded_sequences[:, :-1]
y_train = padded_sequences[:, -1]

# One-hot encoding the output values
y_train = keras.utils.to_categorical(y_train, num_classes=vocab_size)

def train():
    # Part 2: Training

    # Setup the layers of the model
    model = keras.Sequential([
        keras.layers.Embedding(vocab_size, 100, input_length=max_len_sequence - 1),
        keras.layers.LSTM(150),
        keras.layers.Dense(vocab_size, activation="softmax")
    ])

    # Setup the configuration of the training process
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    print('Training the model starting:')
    model.fit(X_train, y_train, epochs=50)

    with open('spear', "wb") as f:
        pickle.dump(model, f)

