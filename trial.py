from main import tokenizer, max_len_sequence
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from typing import cast
import pickle
import numpy as np

with open('spear', 'rb') as f:
    model = pickle.load(f)

model = cast(keras.Sequential, model)

def generate_text(seed, n_words):
    text = seed

    for _ in range(n_words):
        tokenized_text = tokenizer.texts_to_sequences([text])[0]
        tokenized_text = pad_sequences([tokenized_text], maxlen=max_len_sequence - 1, padding="pre")

        prediction = model.predict(tokenized_text, verbose=0)
        highest_prediction_index = np.argmax(prediction)

        next_word = tokenizer.index_word[highest_prediction_index]

        text = text + " " + next_word

    return text


exit = False

while not exit:
    user_text = input('Please enter text: ')
    new_generated_text = generate_text(seed=user_text, n_words=5)

    print(new_generated_text)

    answer = input('Would you like to play again? (Y): ')

    exit = answer != 'Y'
