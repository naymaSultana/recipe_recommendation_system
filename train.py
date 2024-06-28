import streamlit as st
import nltk
from nltk.stem.snowball import SnowballStemmer
import numpy as np
import tflearn
import tensorflow as tf
import json
import re
import pandas as pd
#Download NLTK data
#nltk.download('punkt')
with open('/Users/alohomora/Downloads/combined/recipes.json') as f:
    data = json.load(f)


stemmer = SnowballStemmer("english")

words = []
labels = []
docs = []

for recipe in data["recipes"]:
    ner = recipe["NER"]
    wrds = nltk.word_tokenize(" ".join(ner))
    words.extend(wrds)

    docs.append((ner, recipe["steps"], recipe["recipe_name"]))

    if recipe["recipe_name"] not in labels:
        labels.append(recipe["recipe_name"])

words = [stemmer.stem(w.lower()) for w in words if w != "?"]
words = sorted(list(set(words)))
labels = sorted(labels)


training = []
output = []

out_empty = [0 for _ in range(len(labels))]

for doc in docs:
    bag = []

    wrds = [stemmer.stem(w.lower()) for w in doc[0]]  

    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)

    output_row = out_empty[:]
    output_row[labels.index(doc[2])] = 1

    training.append(bag)
    output.append(output_row)

training = np.array(training)
output = np.array(output)

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 2048, activation='relu', regularizer='L2', weight_decay=0.001)
net = tflearn.batch_normalization(net)
net = tflearn.dropout(net, 0.6)

net = tflearn.fully_connected(net, 512, activation='relu', regularizer='L2', weight_decay=0.001)
net = tflearn.batch_normalization(net)
net = tflearn.dropout(net, 0.6)

net = tflearn.fully_connected(net, len(labels), activation='softmax')
net = tflearn.regression(net, optimizer='adam', learning_rate=0.0001, loss='categorical_crossentropy')


model = tflearn.DNN(net)

try:
    model.load("model.tflearn")
except:
    model = tflearn.DNN(net)
    model.fit(training, output, n_epoch=50, batch_size=100, show_metric=True, shuffle=True)
    model.save("model.tflearn")



