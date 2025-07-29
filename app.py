import streamlit as st
import pickle
import string
import nltk
import sklearn
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()
import nltk
nltk.download('punkt')

def transform_text(text):
  text = text.lower()              # Converting text to lower case
  text = nltk.word_tokenize(text)  # Tokenization

  y = []                           # Removing special characters
  for i in text:
    if i.isalnum():
      y.append(i)

  text = y[:]                      # Removing stopwords and punctuations
  y.clear()

  for i in text:
    if i not in nltk.corpus.stopwords.words('english') and i not in string.punctuation: # Using nltk.corpus.stopwords
      y.append(i)

  text = y[:]                      # Stemming words
  y.clear()

  for i in text:
    y.append(ps.stem(i))

  return " ".join(y)

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title("Email/SMS Spam Classifier")

input_sms = st.text_input("Enter the message")

if st.button("Predict"):
  # preprocess -> vectorize -> Predict -> Display

  transformed_sms = transform_text(input_sms)
  vector_input = tfidf.transform([transformed_sms])
  result = model.predict(vector_input)[0]

  if result == 1:
      st.header("Spam")
  else:
      st.header("Not Spam")
