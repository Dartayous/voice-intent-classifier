import re
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

def load_and_preprocess_data(csv_path, num_words=1000, maxlen=20):
    df = pd.read_csv(csv_path)
    df['clean_query'] = df['query_text'].apply(clean_text)

    tokenizer = Tokenizer(num_words=num_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(df['clean_query'])
    sequences = tokenizer.texts_to_sequences(df['clean_query'])
    padded_sequences = pad_sequences(sequences, padding='post', maxlen=maxlen)

    label_encoder = LabelEncoder()
    df['intent_encoded'] = label_encoder.fit_transform(df['intent'])

    return padded_sequences, df['intent_encoded'].values, tokenizer, label_encoder, df
