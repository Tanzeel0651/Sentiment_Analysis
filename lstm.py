import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from keras.layers import Embedding, Input, Dropout
from keras.layers import LSTM, Dense, GlobalMaxPooling1D, Bidirectional
from keras.preprocessing.text import Tokenizer
from keras.models import Model
from sklearn.metrics import roc_auc_score
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.optimizers import Adam

# intializing
Max_sequence_length = 100
Max_vocab_size = 20000
Embedding_dim = 100
batch_size = 128
epochs = 250
validation_split = 0.15

# loading word vectors
print('Loading Word Vectors....')
word2vec = {}
with open('/home/tanzeel/Documents/glove_vector/glove.6B.%sd.txt' % Embedding_dim) as f:
    for line in f:
        value = line.split()
        word = value[0]
        vec = np.array(value[1:], dtype='float32')
        word2vec[word] = vec

print('Found %s word vectors'%len(word2vec))

# loading dataset
print('Loading the dataset')
dataframe = pd.read_csv('CleanTweet.csv', delimiter=',')
review = dataframe.iloc[:500,-1]
review = [str(text) for text in review]
review = pd.Series(review)
y = dataframe.iloc[:500,2]
#y1 = dataframe['airline_sentiment'].values[:500]

sentiment = []
for x in y:
    if x=='neutral':sentiment.append(0)
    if x=='positive':sentiment.append(1)
    if x=='negative':sentiment.append(-1)
#sentiment = pd.Series(sentiment)
#sentiment = to_categorical(sentiment)


# tokenizing
tokenizer = Tokenizer(num_words=Max_vocab_size)
tokenizer.fit_on_texts(review)
sequences = tokenizer.texts_to_sequences(review)

word2idx = tokenizer.word_index
print('Found %s unique tokens'%len(word2idx))

# pad sequences
data = pad_sequences(sequences, maxlen=Max_sequence_length)
print('shape of tensor ', data.shape)


print('Filling pre-trained embedding')
num_words = min(Max_vocab_size, len(word2idx)+1)
embedding_matrix = np.zeros((num_words, Embedding_dim))
for word, i in word2idx.items():
    if i < Max_vocab_size:
        try:
            embedding_vector = word2vec[word]
        except:
            embedding_vector = np.zeros((100,), dtype='float64')
            #del word2idx[word]
            #del word2vec[word]
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            
            
embedding_layer = Embedding(
        num_words,
        Embedding_dim,
        weights=[embedding_matrix],
        input_length=Max_sequence_length,
        trainable=False
 )

# Implementing LSTM
print('Building the model')
input_ = Input(shape=(Max_sequence_length,))
x = embedding_layer(input_)

x = Bidirectional(LSTM(30, return_sequences=True))(x)
x = GlobalMaxPooling1D()(x)
#x = Dropout(0.4)(x)
x = Dense(32, activation='sigmoid')(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(input_, output)

model.compile(loss='binary_crossentropy',
              optimizer=Adam(0.05),
              metrics=['accuracy'])

print('Training Model....')
r = model.fit(data,
              sentiment,
              batch_size=batch_size,
              epochs=epochs,
              validation_split=validation_split)

plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

plt.plot(r.history['acc'], label='acc')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

p = model.predict(data)
aucs = []
auc = roc_auc_score(sentiment[:], p[:])
aucs.append(auc)
print(np.mean(aucs))
