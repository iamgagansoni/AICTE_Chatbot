import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle   #portable serialized objects
import numpy as np
import speech_recognition as sr
import pyttsx3

def read(line):
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[1].id)
    volume = engine.getProperty('volume')
    engine.setProperty('volume', 10.0)
    rate = engine.getProperty('rate')
    engine.setProperty('rate', rate - 25)
    engine.say(line)      #add an word to speak to queue
    engine.runAndWait()   #read till queue is empty

from keras.models import load_model
model = load_model('chatbot_model.h5')
import json
import random
intents = json.loads(open('datas.json').read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words) 
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                bag[i] = 1
    return(np.array(bag))

def predict_class(sentence, model):
    p = bow(sentence, words)
    res = model.predict(np.array([p]))[0]   #predicting probablity of tag 
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"datas": classes[r[0]], "probability": str(r[1])})
    return return_list


def getResponse(ints, intents_json):
    tag = ints[0]['datas']
    list_of_intents = intents_json['datas']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responces'])
            break
    return result

def chatbot_response(text):
    ints = predict_class(text, model)
    res = getResponse(ints, intents)
    return res

#Creating GUI
import tkinter
from tkinter import *

def listen():
    r=sr.Recognizer()
    with sr.Microphone() as source:
        msg=r.listen(source,timeout=5)
    try:
        msg=r.recognize_google(msg)      #to convert msg from microphone format to understandable format 
    except sr.RequestError as e:  
        print("error"+e)
    
    msg=str(msg)
    chatlog(msg)


def chatlog(msg):
    if msg != '':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "You: " + msg + '\n\n')
        ChatLog.config(foreground="black", font=("Comic Sans MS", 12 ))

        res = chatbot_response(msg)
        ChatLog.insert(END, "Jarvis: " + res + '\n\n')
        

        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)
        read(res)

def send():
    msg = EntryBox.get("1.0",'end-1c').strip()
    EntryBox.delete("0.0",END)
    chatlog(msg)

base = Tk()        
base.title("AICTE Assistant")
base.geometry("400x500")
base.resizable(width=FALSE, height=FALSE)

ChatLog = Text(base, bg="blue", height="8", width="50", font="Verdana")
voice=Button(base,font=("Verdana",12,'bold'), text="listen", width="12", height=5,
                     bg="red", activebackground="#3c9d9b",fg='#ffffff',
                    command= listen )

scrollbar = Scrollbar(base, command=ChatLog.yview)

EntryBox = Text(base, bg="blue",width="29", height="5", font="Arial")
SendButton = Button(base, font=("Verdana",12,'bold'), text="Send", width="12", height=5,
                 bg="#32de97", activebackground="#3c9d9b",fg='#ffffff',
                    command= send )

scrollbar.place(x=376,y=6, height=386)
ChatLog.place(x=6,y=6, height=386, width=370)
EntryBox.place(x=6, y=401, height=90, width=260)
SendButton.place(x=266, y=401, height=45)
voice.place(x=266,y=446,height=45)
base.mainloop()   #mainloop function of tinkter lib 