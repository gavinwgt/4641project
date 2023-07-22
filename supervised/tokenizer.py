import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

import pandas as pd

import re
import nltk
from nltk.corpus import stopwords

from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download('stopwords')

# def get_random_songs():

#     #Insert the file path
#     df = pd.read_csv(r"C:\Users\sneh_\OneDrive - Georgia Institute of Technology\2022-2023 2nd semester\summer\Machine Learning\spotifydata\supervised\rand_lyrics.csv")
#     lyrics = df[125:126]['Lyric']
#     return lyrics

def itter():
    df = pd.read_csv(r"C:\Users\sneh_\OneDrive - Georgia Institute of Technology\2022-2023 2nd semester\summer\Machine Learning\spotifydata\supervised\rand_lyrics.csv")
    return df['Lyric'].values

def token(ly):
    tokenizer = Tokenizer(oov_token="<OOV>")
    tokenizer.fit_on_texts(ly)
    sequences = tokenizer.texts_to_sequences(ly)
    lyr = tokenizer.sequences_to_texts(sequences)
    return lyr


def parse_text(text):
    # print(f'Input: {text}')

    text = re.sub("[^a-zA-Z]", ' ', text)
    # print(f'Remove punctuation and numbers: {text}')

    text = text.lower().split()
    # print(f'Lowercase and split: {text}')

    swords = set(stopwords.words("english"))
    text = [w for w in text if w not in swords]
    # print(f'Remove stop words: {text}')

    text = " ".join(text)
    # print(f'Final: {text}')

    return text
def senti(lyric):
    sia = SentimentIntensityAnalyzer()
    sentiment = sia.polarity_scores(lyric)
    return sentiment

def c_file():
    vals = itter()
    com = []
    for i in vals:
        pa = parse_text(i)
        com.append(senti(pa))
    file = open(r"C:\Users\sneh_\OneDrive - Georgia Institute of Technology\2022-2023 2nd semester\summer\Machine Learning\spotifydata\supervised\label.csv",'w')
    for i in com:
        file.write(str(i)+"\n")
    file.close()

c_file()