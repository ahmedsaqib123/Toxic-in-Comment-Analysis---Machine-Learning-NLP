import numpy as np 
import pandas as pd 
import nltk 
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer 
import re  
import streamlit as st
from PIL import Image

#Heading
st.write("""
# Toxic Comment Classification
Information Retrival Project - Using Machine Learning
""")

#Importing Image
image = Image.open('C:/Users/Lenovo/Desktop/Toxic Comment Classification/toxic.png')
st.image(image,use_column_width=True)

#Read Data
df = pd.read_csv('C:/Users/Lenovo/Desktop/Toxic Comment Classification/train.csv')
st.subheader('Data Information')
st.dataframe(df.sample(20000))
st.write(df.describe())
st.subheader('Data Visualization')
st.bar_chart(df.sample(50))
st.area_chart(df.sample(50))

def text_processing(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

import string
df['processed_text'] = df['comment_text'].str.lower()
df['processed_text'] = df['processed_text'].apply(lambda x: text_processing(x))

dft = pd.read_csv('C:/Users/Lenovo/Desktop/Toxic Comment Classification/test.csv')
dft['processed_text'] = dft['comment_text'].str.lower()
dft['processed_text'] = dft['processed_text'].apply(lambda x: text_processing(x))

columns = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
targets = df[columns].values

train_df = df['processed_text']
test_df = dft['processed_text']

import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_union

word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    ngram_range=(1, 1),
    max_features=30000)

char_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='char',
    ngram_range=(1, 4),
    max_features=30000)

vectorizer = make_union(word_vectorizer, char_vectorizer, n_jobs=2)

all_text = pd.concat([df,dft])

vectorizer.fit(all_text)
train_features = vectorizer.transform(train_df)
test_features = vectorizer.transform(test_df)


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


scores = []
for class_name in columns:
    train_target = df[class_name]
    classifier = LogisticRegression(max_iter=3000,solver='sag')

    score = np.mean(cross_val_score(
        classifier, train_features, train_target, cv=3, scoring='roc_auc',
    ))
    scores.append(score)
    #print('Cross-Validation Score for Class {} is : {} '.format(class_name, score))

    classifier.fit(train_features, train_target)


scoreval=np.mean(scores)
st.subheader("Model-Accuracy")
st.write(str(scoreval*100)+'%')



