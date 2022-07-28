from flask import Flask, render_template, request
import random
import json
from keras.models import load_model
import numpy as np
import pickle
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('popular')
lemmatizer = WordNetLemmatizer()

model = load_model('model.h5')
intents = json.loads(open('data.json').read())
words = pickle.load(open('texts.pkl', 'rb'))
classes = pickle.load(open('labels.pkl', 'rb'))


def sentence_preprocessing(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(
        word.lower()) for word in sentence_words]
    return sentence_words


def bag_of_words(sentence, words):
    sentence_words = sentence_preprocessing(sentence)
    bag = [0]*len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return(np.array(bag))


def predict_class(sentence, model):
    p = bag_of_words(sentence, words)
    ans = model.predict(np.array([p]))[0]
    results = [[i, r] for i, r in enumerate(ans) if r > 0.25]
    results.sort(key=lambda x: x[1], reverse=True)
    return_array = []
    for r in results:
        return_array.append(
            {"intent": classes[r[0]], "probability": str(r[1])})
    return return_array


def get_Response(ints, intents_json):
    tag = ints[0]['intent']
    intents_list = intents_json['intents']
    for i in intents_list:
        if(i['tag'] == tag):
            ans = random.choice(i['responses'])
            break
    return ans


def chatbot_response(msg):
    ints = predict_class(msg, model)
    result = get_Response(ints, intents)
    return result


app = Flask(__name__)
app.static_folder = 'static'


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/get")
def get_bot_response():
    input_data = request.args.get('msg')
    return chatbot_response(input_data)


if __name__ == "__main__":
    app.run()
