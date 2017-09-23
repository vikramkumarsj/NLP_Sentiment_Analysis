# standard Python imports

from collections import Counter
from datetime import datetime 
import json
import numpy as np
import os

# keras deep learning library

import keras 
from keras.layers import Embedding, LSTM, Dense, Conv1D, MaxPooling1D, Dropout, Activation
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

#Check Data Path

os.getcwd()

# Load the reviews and parse JSON

t1 = datetime.now()
with open("yelp_academic_dataset_review.json.txt") as f:
    reviews = f.read().strip().split("\n")
    reviews = [json.loads(review) for review in reviews]
print(datetime.now() - t1)

# Each review in the Yelp dataset contains the text of the review and the associated 
# star rating,left by the reviewer. Our task is to teach a classifier to differentiate between 
# positive and negative reviews, looking only at the review text itself


# take a sample of the Yelp reviews which contains the same amount of
# positive (four or five-star reviews) and negative (one, two, or three-star reviews)


texts = [review['text'] for review in reviews]

# Convert our 5 classes into 2 (negative or positive) 
# Prepare Text and label list
binstars = [0 if review['stars'] <= 3 else 1 for review in reviews]
balanced_texts = []
balanced_labels = []
limit = 10000  # Change this to grow/shrink the dataset which will effect the accuracy
neg_pos_counts = [0, 0]

for i in range(len(texts)):
    polarity = binstars[i]
    if neg_pos_counts[polarity] < limit:
        balanced_texts.append(texts[i])
        balanced_labels.append(binstars[i])
        neg_pos_counts[polarity] += 1
        
# Verify that our new dataset is balanced by using a Python Counter
Counter(balanced_labels)

## Text Preprocessing

# Tokenizing the texts, we first need to split each text into words and represent each word by a number

# Here, we use the most common 20000 words and apply on our texts.
tokenizer = Tokenizer(num_words=20000)
tokenizer.fit_on_texts(balanced_texts)

# Convert texts to numerical sequences and “pad” our sequences. 
# Neural network can train more efficiently if all of the training examples are the same size.
# we pass maxlen=300 when we pad the sequences. This means that as well as padding the very short texts 
# with zeros, we’ll also truncate the # very long ones. 

sequences = tokenizer.texts_to_sequences(balanced_texts)
data = pad_sequences(sequences, maxlen=300)

## Building a Neural Network

# create an empty Sequential model. We’ll use this to add several “layers” to our network.
model = Sequential()

#Embedding layer. This layer lets the network expand each token to a larger vector (size 128), allowing the 
#network to represent a words in a meaningful way
model.add(Embedding(20000, 128, input_length=300))

# Dropout is a regularization technique for reducing overfitting in neural networks 
# we reset a random 20% of the weights from the LSTM layer with every iteration
model.add(Dropout(0.2))


#to speed up the training time add a “Convolutional” layer. 
#Adding a CNN layer before the LSTM, we allow the LSTM to see sequences of chunks instead of sequences of words
#MaxPooling layer, which combines all of the different chunked representations into a single chunk
model.add(Conv1D(64, 5, activation='relu'))
model.add(MaxPooling1D(pool_size=4))

#add an LSTM layer
# RNNs are designed to learn from sequences of data, where there is some kind of time dependency. 
# For example, they are used for time-series analysis, where each data point has some relation to those immediately before and after.
model.add(LSTM(128))

# Neural Network layer
# All neurons in the layer are connected to each other
model.add(Dense(1, activation='sigmoid'))

#compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Early stopping when Validation accuracy do not improve.
earlystopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')

#Training the neural network
model.fit(data, np.array(balanced_labels), validation_split=0.5, epochs=5, callbacks = [earlystopping])


#Save the Model for future use

import pickle
 
# save the tokenizer and model
with open("keras_tokenizer.pickle", "wb") as f:
   pickle.dump(tokenizer, f)
   
model.save("yelp_sentiment_model.hdf5")

#load our model and get a prediction
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import pickle

# load the tokenizer and the model
with open("keras_tokenizer.pickle", "rb") as f:
   tokenizer = pickle.load(f)

model = load_model("yelp_sentiment_model.hdf5")


#Case1:
newtexts = ["We want to Nagarjuna hotel and Biryani was fantastic and gobi dry was tasty and also hotel ambience was nice"]

# note that we shouldn't call "fit" on the tokenizer again
sequences = tokenizer.texts_to_sequences(newtexts)
data = pad_sequences(sequences, maxlen=300)
 
# get predictions for each of your new texts
predictions = model.predict(data)
#0.89
print(predictions)


#Case2
newtexts = ["We want to upahara Dharshini hotel and ordered masala dosa but it was not eatable"]

# note that we shouldn't call "fit" on the tokenizer again
sequences = tokenizer.texts_to_sequences(newtexts)
data = pad_sequences(sequences, maxlen=300)
 
# get predictions for each of your new texts
predictions = model.predict(data)
#0.17
print(predictions)

  
#Case3
newtexts = ["service technician noted down all the problems we encountered while driving. car technician was able to identify the problem and repaired our car to perfection. now our car is now running smoothly."]

# note that we shouldn't call "fit" on the tokenizer again
sequences = tokenizer.texts_to_sequences(newtexts)
data = pad_sequences(sequences, maxlen=300)
 
# get predictions for each of your new texts
predictions = model.predict(data)
#0.17
print(predictions)


#Case4
newtexts = ["service technician noted down all the problems we encountered while driving. we assume car technician was not able to identify the problem we encountered while driving. we faced same problems again while driving after service."]

# note that we shouldn't call "fit" on the tokenizer again
sequences = tokenizer.texts_to_sequences(newtexts)
data = pad_sequences(sequences, maxlen=300)
 
# get predictions for each of your new texts
predictions = model.predict(data)
#0.17
print(predictions)

#http://www.qualitycarservice.co.uk/testimonials



















