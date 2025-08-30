import pandas as pd
import re
import numpy as np
import nvtx
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense


# Load the dataset
df = pd.read_csv("F:/ML_DATASETS/NLP/Voice Search AI Conversational Queries 2025/voice_search_query_captures.csv")


# Basic text cleaning
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
    return text

df['clean_query'] = df['query_text'].apply(clean_text)


# Tokenize
tokenizer = Tokenizer(num_words=1000, oov_token="<OOV>")
tokenizer.fit_on_texts(df['clean_query'])

sequences = tokenizer.texts_to_sequences(df['clean_query'])
padded_sequences = pad_sequences(sequences, padding='post', maxlen=20)


label_encoder = LabelEncoder()
df['intent_encoded'] = label_encoder.fit_transform(df['intent'])


X = padded_sequences
y = df['intent_encoded'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Parameters
vocab_size = 1000  # same as tokenizer num_words
embedding_dim = 16
max_length = 20    # same as pad_sequences maxlen
num_classes = len(df['intent'].unique())

# Model definition
model = Sequential([
    Embedding(vocab_size, embedding_dim),
    GlobalAveragePooling1D(),
    Dense(24, activation='relu'),
    Dense(num_classes, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.summary()


history = model.fit(X_train, y_train,
                    epochs=10,
                    validation_data=(X_test, y_test),
                    verbose=2)



new_queries = [
    "Play jazz music on Spotify",
    "What's the weather in Tokyo tomorrow?",
    "Turn off the living room lights",
    "Order me a pizza from Domino's",
    "How far is the moon from Earth?"
]

# Clean the text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

cleaned_queries = [clean_text(q) for q in new_queries]

# Tokenize and pad
new_sequences = tokenizer.texts_to_sequences(cleaned_queries)
new_padded = pad_sequences(new_sequences, padding='post', maxlen=20)


predictions = model.predict(new_padded)
predicted_labels = [label_encoder.inverse_transform([np.argmax(p)])[0] for p in predictions]

for query, label in zip(new_queries, predicted_labels):
    print(f"Query: '{query}' → Predicted Intent: '{label}'")



epochs = 10  # or however many you want to profile

for epoch in range(epochs):
    with nvtx.annotate(f"Epoch {epoch+1}", color="blue"):
        model.fit(X_train, y_train,
                  epochs=1,
                  validation_data=(X_test, y_test),
                  verbose=0)
