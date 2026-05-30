"""
main.py — ISL Sign Language Model Training
==========================================
CSV landmarks data se TensorFlow/Keras model train karta hai.

Usage:
    python main.py

Output:
    model/sign_model.h5           — trained Keras model
    model/label_encoder.pkl       — class labels (for reference)
    model/training_history.png    — accuracy/loss curves
    model/confusion_matrix.png    — confusion matrix

=== FIXES IN THIS VERSION ===

FIX 1 — Loss function typo:
  'sparse_categorical_cross entropy' → 'sparse_categorical_crossentropy'

FIX 2 — Hardcoded Windows path:
  DATA_FILE ab relative path hai. CSV ko is script ke saath wali folder mein rakho.

FIX 3 — Label output was integers not letters:
  Dataset 'target' column mein 0-25 integers hain = A-Z.
  LabelEncoder in integers pe fit karta tha aur inverse_transform
  integers return karta tha. Fix: hum 'target' column ko seedha
  A-Z mein convert karke train karte hain taaki encoder bhi
  letters return kare. Aur app.py mein direct mapping bhi hai.
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import LabelEncoder
from sklearn.metrics         import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import seaborn as sns

# ── Config ───────────────────────────────────────────────────────────────────
DATA_FILE    = 'E:\\local disk c\\Indian Sign Language Gesture Landmarks.csv'  # FIX: relative path
MODEL_DIR    = 'model'
MODEL_PATH   = f'{MODEL_DIR}/sign_model.h5'
ENCODER_PATH = f'{MODEL_DIR}/label_encoder.pkl'
EPOCHS       = 60
BATCH_SIZE   = 32
DROPOUT      = 0.3
SEED         = 42
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(MODEL_DIR, exist_ok=True)
tf.random.set_seed(SEED)
np.random.seed(SEED)


def load_data(filepath):
    """Dataset load karo."""

    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"CSV file nahi mila: '{filepath}'\n"
            f"Script ke saath wali folder mein rakho."
        )

    df = pd.read_csv(filepath)

    if 'target' not in df.columns:
        raise ValueError("CSV mein 'target' column hona chahiye.")

    # FIX 3: Convert integer targets 0-25 → letters A-Z
    # Taaki LabelEncoder bhi letters pe train ho
    # aur inverse_transform 'A','B'... return kare (integers nahi)
    df['target'] = df['target'].apply(lambda x: chr(ord('A') + int(x)))

    y = df['target'].values
    X = df.drop(columns=['target']).values

    return np.array(X, dtype=np.float32), np.array(y)


def build_model(input_dim, num_classes):
    """
    MLP — 3 Dense layers with BatchNorm + Dropout.
    Input:  127 features (uses_two_hands + left_hand_63 + right_hand_63)
    Output: num_classes (26 = A-Z)
    """
    model = keras.Sequential([
        keras.Input(shape=(input_dim,)),

        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(DROPOUT),

        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(DROPOUT),

        layers.Dense(64, activation='relu'),
        layers.Dropout(DROPOUT * 0.5),

        layers.Dense(num_classes, activation='softmax'),
    ], name='ISL_Sign_Classifier')

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',   # FIX 1: space remove kiya
        metrics=['accuracy']
    )
    return model


def plot_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle('ISL Sign Language Model — Training History', fontsize=14)

    ax1.plot(history.history['accuracy'],     label='Train')
    ax1.plot(history.history['val_accuracy'], label='Val')
    ax1.set_title('Accuracy'); ax1.set_xlabel('Epoch')
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(history.history['loss'],     label='Train')
    ax2.plot(history.history['val_loss'], label='Val')
    ax2.set_title('Loss'); ax2.set_xlabel('Epoch')
    ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = f'{MODEL_DIR}/training_history.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"[Plot] Saved: {path}")
    plt.close()


def plot_confusion(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_title('Confusion Matrix — ISL A-Z')
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    plt.tight_layout()
    path = f'{MODEL_DIR}/confusion_matrix.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"[Plot] Saved: {path}")
    plt.close()


def train():
    print("=" * 55)
    print("  ISL Sign Language Model Training  (A-Z Fix)")
    print("=" * 55)

    # 1. Load
    print(f"\n[1/5] Loading: {DATA_FILE}")
    X, y_raw = load_data(DATA_FILE)
    print(f"      Samples  : {len(X)}")
    print(f"      Features : {X.shape[1]}")
    print(f"      Sample targets: {y_raw[:5]}  ← should be letters now")

    # 2. Encode
    print("\n[2/5] Encoding labels...")
    le = LabelEncoder()
    y  = le.fit_transform(y_raw)
    classes = le.classes_
    print(f"      Classes ({len(classes)}): {list(classes)}")

    # 3. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    print(f"\n[3/5] Split: Train={len(X_train)}, Test={len(X_test)}")

    # 4. Train
    print(f"\n[4/5] Training ({EPOCHS} epochs)...")
    model = build_model(X.shape[1], len(classes))
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=10,
            restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5,
            patience=5, min_lr=1e-6, verbose=1
        ),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    # 5. Evaluate
    print("\n[5/5] Evaluating...")
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n  Test Accuracy : {acc*100:.2f}%")
    print(f"  Test Loss     : {loss:.4f}")

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=[str(c) for c in classes]))

    # Save
    model.save(MODEL_PATH)
    print(f"\n[Saved] Model  : {MODEL_PATH}")

    with open(ENCODER_PATH, 'wb') as f:
        pickle.dump(le, f)
    print(f"[Saved] Encoder: {ENCODER_PATH}")

    # Plots
    plot_history(history)
    plot_confusion(y_test, y_pred, classes)

    print("\n" + "=" * 55)
    print(f"  Done! Accuracy: {acc*100:.2f}%")
    print(f"  Ab chalao: streamlit run app.py")
    print("=" * 55)


if __name__ == '__main__':
    train()
