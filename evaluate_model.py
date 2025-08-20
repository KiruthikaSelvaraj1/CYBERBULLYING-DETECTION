import pickle
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, multilabel_confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("final_hateXplain.csv").dropna(subset=["text", "label"])
df["label"] = df["label"].apply(eval)

# Load tokenizer and model
tokenizer = pickle.load(open("tokenizer.pkl", "rb"))
mlb = pickle.load(open("mlb.pkl", "rb"))
model = load_model("lstm_multilabel_model.h5")

# Prepare input
X = tokenizer.texts_to_sequences(df["text"])
X = tf.keras.preprocessing.sequence.pad_sequences(X, maxlen=100)
y_true = mlb.transform(df["label"])

# Predict
y_pred = model.predict(X)
y_pred_binary = (y_pred > 0.5).astype(int)

# Classification report
report = classification_report(y_true, y_pred_binary, target_names=mlb.classes_)
print(report)

# Save confusion matrix image
cm = multilabel_confusion_matrix(y_true, y_pred_binary)
fig, axes = plt.subplots(1, len(mlb.classes_), figsize=(15, 5))
for i, (ax, label) in enumerate(zip(axes, mlb.classes_)):
    sns.heatmap(cm[i], annot=True, fmt='d', ax=ax, cbar=False)
    ax.set_title(label)
plt.tight_layout()
plt.savefig("static/plots/confusion_matrix.png")
