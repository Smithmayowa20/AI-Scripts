import keras
from keras.preprocessing.sequenceimportpad_sequences
from keras.modelsimportSequential
from keras.layersimportDense, Dropout, Embedding, LSTM, SpatialDropout1D
from keras.callbacksimportModelCheckpoint
import os
from sklearn.metricsimportroc_auc_score
import matplotlib.pyplotasplt
import pandasaspd
import numpyasnp
import re
from keras.preprocessing.textimportTokenizer
from sklearn.model_selectionimporttrain_test_split

tweets=pd.read_csv('./Dataset/Tweets.csv',sep=',')

data = tweets[['text','airline_sentiment']]

data = data[data.airline_sentiment != "neutral"]
data['text'] = data['text'].apply(lambda x: x.lower())
data['text'] = data['text'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))
print(data[ data['airline_sentiment'] == 'positive'].size)
print(data[ data['airline_sentiment'] == 'negative'].size)

max_fatures = 2000
tokenizer = Tokenizer(num_words=max_fatures, split=' ')
tokenizer.fit_on_texts(data['text'].values)
X = tokenizer.texts_to_sequences(data['text'].values)
X = pad_sequences(X)

embed_dim = 128
lstm_out = 196
model = Sequential()
model.add(Embedding(max_fatures, embed_dim,input_length = X.shape[1]))
model.add(Dropout(0.5))
model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(2,activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
print(model.summary())

Y = pd.get_dummies(data['airline_sentiment']).values
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.33, random_state = 42)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)


X_val = X_train[:500]
Y_val = Y_train[:500]

partial_X_train = X_train[500:]
partial_Y_train = Y_train[500:]

batch_size = 512
history = model.fit(partial_X_train, 
                    partial_Y_train, 
                    epochs = 10, 
                    batch_size=batch_size, 
                    validation_data=(X_val, Y_val))