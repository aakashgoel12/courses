##############################
##### GENERAL TEXT PROCESSING 
##############################

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

sentences = [
    'I love my dog',
    'I love my cat',
    'You love my dog!',
    'Do you think my dog is amazing?'
]

tokenizer = Tokenizer(num_words = 100, oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)

## word_index will be dictionary with key as word and value as index & its size can be greater than num_words ....
word_index = tokenizer.word_index

## texts_to_sequences will take care of num_words
sequences = tokenizer.texts_to_sequences(sentences)

## truncating --> From where to loose in case of sentence > maxlen
padded = pad_sequences(sequences, padding = 'post', maxlen=5,truncating = 'post')
print("\nWord Index = " , word_index)
print("\nSequences = " , sequences)
print("\nPadded Sequences:")
print(padded)


# Try with words that the tokenizer wasn't fit to
test_data = [
    'i really love my dog',
    'my dog loves my manatee'
]

test_seq = tokenizer.texts_to_sequences(test_data)
print("\nTest Sequence = ", test_seq)

padded = pad_sequences(test_seq, maxlen=10)
print("\nPadded Test Sequence: ")
print(padded)

## bbc news link data -- http://mlg.ucd.ie/datasets/bbc.html
## another useful link -- https://rishabhmisra.github.io/projects/