import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from utils import clean_text, load_and_preprocess_data

# Load model and tokenizer
model = load_model("intent_classifier.h5")
_, _, tokenizer, label_encoder, _ = load_and_preprocess_data(
    "F:/ML_DATASETS/NLP/Voice Search AI Conversational Queries 2025/voice_search_query_captures.csv"
)

# New queries
new_queries = [
    "Play jazz music on Spotify",
    "What's the weather in Tokyo tomorrow?",
    "Turn off the living room lights",
    "Order me a pizza from Domino's",
    "How far is the moon from Earth?"
]

# Preprocess
cleaned = [clean_text(q) for q in new_queries]
sequences = tokenizer.texts_to_sequences(cleaned)
padded = pad_sequences(sequences, padding='post', maxlen=20)

# Predict
predictions = model.predict(padded)
predicted_labels = [label_encoder.inverse_transform([np.argmax(p)])[0] for p in predictions]

for query, label in zip(new_queries, predicted_labels):
    print(f"Query: '{query}' → Predicted Intent: '{label}'")
