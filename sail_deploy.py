# -*- coding: utf-8 -*-
"""sail_deploy.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1VYaUE-ZD9v2zIYvmtSVQsrXwoBqlteVA
"""

import streamlit as st
import tensorflow as tf
import pickle
import json
import random
import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy as np

import nltk
nltk.data.path.append("nltk_data")


# Load the trained model and preprocessed data
model = tf.keras.models.load_model('model.h5')
data = pickle.load(open('training_data', 'rb'))
words = data['words']
classes = data['classes']
intents = json.load(open('intents.json'))

# Preprocess the user input
def clean_up_sentence(sentence):
    stemmer = LancasterStemmer()
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

# Convert the user input to a bag of words
def bow(sentence, words, show_details=False):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

# Predict the class of the user input
def predict_class(sentence, model):
    p = bow(sentence, words)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

# Generate a response to the user input
def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

def app():
    st.title("STUDENT ASSISTANCE AND INFORMATION LIASION")
    user_input = st.text_input("You: ", "")
    res = ""  # Initialize res with a default value
    if user_input:
        ints = predict_class(user_input, model)
        res = get_response(ints, intents)
    st.text_area("Bot:", res)
    
if __name__ == '__main__':
    app()
