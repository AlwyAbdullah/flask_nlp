# Flask code (app2.py)

from flask import Flask, request, jsonify
from flask_cors import CORS
import nltk
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import PyPDF2
import os
import joblib

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('words')

lemmatizer = WordNetLemmatizer()
stopwords = set(stopwords.words('english'))

app = Flask(__name__)
CORS(app)
def preprocess_text(text):
    text = text.lower()
    text = re.sub('[^A-Za-z]', ' ', text)
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stopwords]
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text

def load_nrc():
    file_path = 'static/csv_file/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt'
    nrc = pd.read_csv(file_path, names=['word', 'sentiment', 'value'], sep='\t', comment=';', usecols=['word', 'sentiment', 'value'], dtype={'word': str, 'sentiment': str, 'value':int})
    return nrc

def predict_personality(text):
    text = preprocess_text(text)
    cAGR = joblib.load('static/model/svc_model_agr.pkl')
    cCON = joblib.load('static/model/svc_model_con.pkl')
    cEXT = joblib.load('static/model/svc_model_ext.pkl')
    cNEU = joblib.load('static/model/svc_model_neu.pkl')
    cOPN = joblib.load('static/model/svc_model_opn.pkl')
    
    file_name = 'File_1'
    df = pd.DataFrame([[file_name, text]], columns=['file_name', 'Text'])
    df = df.reset_index(drop=True)
    df['tokenized_text'] = [word_tokenize(i) for i in df['Text']]
    df = df.explode('tokenized_text').reset_index(drop=True)
    df = df.groupby(['file_name', 'tokenized_text']).size().reset_index(name='count').reset_index(drop=True)
    
    nrc = load_nrc()
    nrc.drop(nrc[nrc['value'] == 0].index, inplace=True)
    essay_token_labeled = pd.merge(df, nrc, left_on='tokenized_text', right_on='word', how='inner')
    essay_sentiment_scores = essay_token_labeled.groupby(['file_name', 'sentiment']).size().reset_index(name='count')
    
    essay_sentiment = essay_sentiment_scores.pivot(index='file_name', columns='sentiment', values='count').fillna(0)
    essays_features = pd.merge(df, essay_sentiment, on='file_name', how='inner').drop(['tokenized_text'], axis=1)
    essays_features = essays_features.drop_duplicates()
    essays_features = essays_features.reset_index()
    
    df_new = pd.DataFrame([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], columns=['anger', 'anticipation', 'disgust', 'fear', 'joy', 'negative', 'positive', 'sadness', 'surprise', 'trust'])
    df_new.update(essays_features)

    anger = df_new.iloc[0]['anger']
    anticipation = df_new.iloc[0]['anticipation']
    disgust = df_new.iloc[0]['disgust']
    fear = df_new.iloc[0]['fear']
    joy = df_new.iloc[0]['joy']
    negative = df_new.iloc[0]['negative']
    positive = df_new.iloc[0]['positive']
    sadness = df_new.iloc[0]['sadness']
    surprise = df_new.iloc[0]['surprise']
    trust = df_new.iloc[0]['trust']

    predict_ext = cEXT.predict([[anger, anticipation, disgust, fear, joy, negative, positive, sadness, surprise, trust]])
    predict_agr = cAGR.predict([[anger, anticipation, disgust, fear, joy, negative, positive, sadness, surprise, trust]])
    predict_opn = cOPN.predict([[anger, anticipation, disgust, fear, joy, negative, positive, sadness, surprise, trust]])
    predict_con = cCON.predict([[anger, anticipation, disgust, fear, joy, negative, positive, sadness, surprise, trust]])
    predict_neu = cNEU.predict([[anger, anticipation, disgust, fear, joy, negative, positive, sadness, surprise, trust]])

    predict_ext = int(predict_ext[0])
    predict_agr = int(predict_agr[0])
    predict_opn = int(predict_opn[0])
    predict_con = int(predict_con[0])
    predict_neu = int(predict_neu[0])

    return [predict_ext, predict_agr, predict_opn, predict_con, predict_neu]


@app.route("/", methods=["POST"])
def process_data():
    text = request.form["text"]
    predictions = predict_personality(text)
    return jsonify(predictions)


if __name__ == "__main__":
    app.run()
