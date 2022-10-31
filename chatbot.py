import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle
import numpy as np
from keras.models import Sequential     #provides training and inference feature on this model 
from keras.layers import Dense, Activation, Dropout
'''Dense:- regular densely connected network of neurons in which 
each neuron receives input from all 
the neurons of previous layer

Activation:- The choice of activation function in the hidden 
layer will control how well the network model learns the training dataset.
It is used to determine the output of neural network
like yes or no. It maps the resulting values in between 0 to 1 or -1 to 1

Dropout:- randomly sets input units to 0 with a frequency
 of rate at each step during training time, 
 which helps prevent overfitting.'''

from keras.optimizers import SGD
'''Gradient Descent is a generic optimization algorithm capable 
of finding optimal solutions to a wide range of problems.'''
import random
words=[] #will contain all words in patterns tag
classes = [] #will contain all tags
documents = [] #will contain [[word],tag]
ignore_words = ['?','.',',']
data_file = open('datas.json').read()
intents = json.loads(data_file)
for intent in intents['datas']:
    for pattern in intent['patterns']:
        w = nltk.word_tokenize(pattern)
        words.extend(w) 
        # Extend list by appending elements from the iterable.
        #add documents in the corpus
        documents.append((w, intent['tag']))

        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# lemmatize, lower each word
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
#sort words and remove repeated words
words = sorted(list(set(words)))

classes = sorted(list(set(classes)))
print (len(documents), "documents")
print (len(classes), "classes", classes)
print (len(words), "unique lemmatized words", words)

pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))
# create our training data
training = []
# create an empty array for our output
output_empty = [0] * len(classes) # will be used to determine tag
# training set, bag of words for each sentence

for doc in documents:
    bag = []
    # list of tokenized words for the pattern
    pattern_words = doc[0]
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    # create our bag of words array with 1, if word match found in current pattern
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
    # output is a '0' for each tag and '1' for current tag (for each pattern)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])
# shuffle our features and turn into np.array
random.shuffle(training)
training = np.array(training)
# create train and test lists. X - patterns , Y-tags 
train_x = list(training[:,0])
train_y = list(training[:,1])
print("Training data created")
# Create model - 3 layers. First layer 128 neurons, second layer 64 neurons and 3rd output layer contains number of neurons
# equal to number of intents to predict output intent with softmax
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5', hist)
print("model created")