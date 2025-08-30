import numpy as np
import nvtx
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from utils import load_and_preprocess_data

# Load and preprocess
X, y, tokenizer, label_encoder, df = load_and_preprocess_data(
    "F:/ML_DATASETS/NLP/Voice Search AI Conversational Queries 2025/voice_search_query_captures.csv"
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model parameters
vocab_size = 1000
embedding_dim = 16
max_length = 20
num_classes = len(df['intent'].unique())

# Build model
model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=max_length),
    GlobalAveragePooling1D(),
    Dense(24, activation='relu'),
    Dense(num_classes, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

# NVTX-wrapped training loop
epochs = 10
for epoch in range(epochs):
    with nvtx.annotate(f"Epoch {epoch+1}", color="blue"):
        model.fit(X_train, y_train, epochs=1, validation_data=(X_test, y_test), verbose=0)

# Save model
model.save("intent_classifier.h5")
