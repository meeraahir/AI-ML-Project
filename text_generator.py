# =====================================================
# LSTM TEXT GENERATION USING SHAKESPEARE DATASET
# =====================================================

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import string

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

print("Program started...")

with open("shakespeare.txt", "r", encoding="utf-8") as file:
    text = file.read()

text = text.lower()

text = text.translate(str.maketrans("", "", string.punctuation))

tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])

total_words = len(tokenizer.word_index) + 1
print("Total unique words:", total_words)

input_sequences = []

for line in text.split("\n"):
    token_list = tokenizer.texts_to_sequences([line])[0]

    for i in range(1, len(token_list)):
        input_sequences.append(token_list[:i + 1])

max_sequence_len = max(len(seq) for seq in input_sequences)

input_sequences = pad_sequences(
    input_sequences,
    maxlen=max_sequence_len,
    padding="pre"
)

X = input_sequences[:, :-1]
y = input_sequences[:, -1]

print("Input shape:", X.shape)
print("Output shape:", y.shape)

model = Sequential([
    Embedding(total_words, 64, input_length=max_sequence_len - 1),
    LSTM(100),
    Dense(total_words, activation="softmax")
])

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

model.summary()

early_stop = EarlyStopping(
    monitor="loss",
    patience=3,
    restore_best_weights=True
)

model.fit(
    X,
    y,
    epochs=20,
    batch_size=128,
    callbacks=[early_stop],
    verbose=1
)

print("Training completed...")


def generate_text(seed_text, next_words=30):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences(
            [token_list],
            maxlen=max_sequence_len - 1,
            padding="pre"
        )

        predicted_index = np.argmax(
            model.predict(token_list, verbose=0)
        )

        output_word = tokenizer.index_word.get(predicted_index, "")
        seed_text += " " + output_word

    return seed_text


print("\n--- GENERATED TEXT SAMPLE 1 ---")
print(generate_text("to be or not to be", 40))

print("\n--- GENERATED TEXT SAMPLE 2 ---")
print(generate_text("love is a fire", 40))
