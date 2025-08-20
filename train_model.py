import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from sklearn.preprocessing import MultiLabelBinarizer
import pickle

# Load data
df = pd.read_csv("final_hateXplain.csv")

# Ensure text and labels exist
df = df.dropna(subset=["text", "label"])

# Convert label strings to lists (e.g., ['race', 'gender'])
df["label"] = df["label"].apply(eval)

# Tokenize text
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(df["text"])
sequences = tokenizer.texts_to_sequences(df["text"])
X = pad_sequences(sequences, maxlen=100)

# Multi-label binarization
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(df["label"])

# Save tokenizer and label binarizer
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
with open("mlb.pkl", "wb") as f:
    pickle.dump(mlb, f)

# Build model
model = Sequential([
    Embedding(input_dim=10000, output_dim=128, input_length=100),
    LSTM(64, return_sequences=False),
    Dense(64, activation='relu'),
    Dense(len(mlb.classes_), activation='sigmoid')
])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train model
model.fit(X, y, epochs=5, batch_size=32, validation_split=0.2)

# Save model
model.save("lstm_multilabel_model.h5")
