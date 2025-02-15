
import string 
import re
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential 
from keras.layers import Dense, LSTM, Embedding, RepeatVector, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.callbacks import ModelCheckpoint 
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model 
from keras import optimizers 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Пример подготовки данных
data_1 = open('result1.csv', encoding = 'utf-8')

data_1 = data_1.read()
data_1 = [i.split(';') if len(i.split(';')) == 2 else ['Яблоко', '1'] for i in data_1.split('\n')]
data_1 = np.asarray(data_1)
texts = data_1[:, 0]


#texts = ['Пример текста для обработки нейронными сетями.', 'Еще один пример.']
#tokenizer = Tokenizer(num_words=1000)
#tokenizer.fit_on_texts(texts)
#sequences = tokenizer.texts_to_sequences(texts)
#seq = pad_sequences(sequences, maxlen=50)

# Prepare English tokenizer
tokenizer_t = Tokenizer()
tokenizer_t.fit_on_texts(data_1[:, 0])
vocab_size_t = len(tokenizer_t.word_index) + 1 
length1 = 15

# Prepare English tokenizer
tokenizer_l = Tokenizer()
tokenizer_l.fit_on_texts(data_1[:, 1])
vocab_size_l = len(tokenizer_l.word_index) + 1 
length2 = 15

def encode_sequences(tokenizer, length, lines):          
    # integer encode sequences
    seq = tokenizer.texts_to_sequences(lines)
    # pad sequences with 0 values
    seq = pad_sequences(seq, maxlen=length, padding='post')
    return seq


# Split data into train and test set 
train, test = train_test_split(data_1, test_size=0.2, random_state=12)

# prepare training data 
trainX = encode_sequences(tokenizer_t, length1, train[:, 0])
trainY = encode_sequences(tokenizer_l, length2, train[:, 1])

# prepare validation data 
testX = encode_sequences(tokenizer_t, length1, test[:, 0])
testY = encode_sequences(tokenizer_l, length2, test[:, 1])

def make_model(in_vocab, out_vocab, in_timesteps, out_timesteps, n):
    model = Sequential()
    model.add(Embedding(in_vocab, n, input_length=in_timesteps, mask_zero=True))
    model.add(LSTM(n))
    model.add(Dropout(0.3))
    model.add(RepeatVector(out_timesteps))
    model.add(LSTM(n, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(Dense(out_vocab, activation='softmax'))
    model.compile(optimizer=optimizers.RMSprop(learning_rate=0.001), loss='sparse_categorical_crossentropy')
    return model


# Model compilation (with 512 hidden units)
model = make_model(vocab_size_t, vocab_size_l, length1, length2, 512)

# Train model
num_epochs = 50
history = model.fit(trainX, trainY.reshape(trainY.shape[0], trainY.shape[1], 1), epochs=num_epochs, batch_size=512, validation_split=0.2, callbacks=None, verbose=1)
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.legend(['train','validation'])
# plt.show()
model.save('model2.h5')

# Load model
#model = load_model('model1.h5')


#loss, accuracy = model.evaluate(testX, testY)
#print(f'Loss: {loss}, Accuracy: {accuracy}')

#def get_word(n, tokenizer):
    #if n == 0:
        #return ""
    #for word, index in tokenizer.word_index.items():
        #if index == n:
            #return word
    #return ""

