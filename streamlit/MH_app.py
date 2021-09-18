import pandas as pd 
import numpy as np
import streamlit as st
from afinn import Afinn
afinn = Afinn(language='en')
#tweet preprocessing 
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
import string
import re
from nltk.tokenize import word_tokenize,sent_tokenize,TweetTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

emo_lex=pd.read_excel('streamlit/NRC-Emotion-Lexicon-v0.92-In105Languages-Nov2017Translations.xlsx') # emotion lexicon 
emolex_df=emo_lex[['English (en)','Positive','Negative','Anger','Anticipation','Disgust','Fear','Joy','Sadness','Surprise','Trust']]
emotions=emolex_df.columns.drop('English (en)')
emolex_df.rename(columns={'English (en)':'word'},inplace=True)


def Lowercasing(words):
    string=re.sub("([^0-9A-Za-z \t])|(\w+:\/\/\S+)", "",str(words))
    word=string.lower()
    return word

#Tokenization and (@)handle extraction
def Tokenization(tweet):
    tokens=sent_tokenize(tweet)
    return tokens

#punctuations
def Punctuation_removal(tokens):
    words=[ word for word in tokens if word.isalnum()]
    return words

#stemming
def stemming(text):
    stemmer=PorterStemmer()
    for  word in text:
        stemmed_words=stemmer.stem(word)
        return stemmed_words

#stopword_removal
def remove_stopwords(words):
    stop_words=set(stopwords.words("english")) 
    result=[word for word in words if word not in stop_words ]
    return result


#lemmatization
def lemmatization(text):
    lemmatizer=WordNetLemmatizer()
    lemmatized_phrase=[]
    for word in text:
        lemmatized_word=lemmatizer.lemmatize(word)
        lemmatized_phrase.append(lemmatized_word)
    return lemmatized_phrase


def clean_tweet(tweet):
    tweet_tokens=Tokenization(tweet)
    lemmatized_tweet=lemmatization(tweet_tokens)#lemmatization
    clean_string=Lowercasing(lemmatized_tweet)#lowercasing and removing numbers
    return clean_string


def tweet_emotions(tweet):
    emo_df=pd.DataFrame(0,index=['word'],columns=emotions)
    words=word_tokenize(tweet) #the body of text for each individual tweet(row)
    for word in words:
        emo_score=emolex_df[emolex_df.word == word]   
        if emo_score.empty:
            continue
        else:
            for emotion in list(emotions):
                emo_df.at['word',emotion] += emo_score[emotion]
    
    emo_df['afinn_score'] = afinn.score(tweet)
    return emo_df

def analyze(text):
    message = clean_tweet(text)#clean_text
    emo_df=tweet_emotions(message)#calculate emotion scores
    tweet_score=emo_df.iloc[0,:].values
    weight_list=[1,5,24.28,5.71,15,9.28,0.71,39.28,3.57,2.14,1]#calculate weighted average
    weight_list= [float(item) for item in weight_list]
    final_val = np.dot(tweet_score,weight_list)
    return final_val


st.title('Depressive Texts Analyzer')
message = st.text_area('Enter text')
if st.button('Analyze'):
    with st.spinner('Analyzing the text'):
        prediction=analyze(message)
        if prediction < 50:
            st.write('Depressive sentiments not present in text')
        elif prediction < 100:
            st.write('Depressive sentiments present in text')
        else:
            st.write('Depressive sentiments highly present in text')
