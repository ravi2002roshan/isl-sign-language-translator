

# 🤟 ISL Sign Language Translator

### Real-time Indian Sign Language Detection using TensorFlow + MediaPipe



**Developed by [Ravi Roshan](https://www.linkedin.com/in/ravi-roshan-710105347)**

[🔗 LinkedIn](https://www.linkedin.com/in/ravi-roshan-710105347) • [💻 GitHub](https://github.com/ravi2002roshan)


---

## 📌 About The Project

This project is a **real-time Indian Sign Language (ISL) Translator** that uses a webcam to detect hand gestures and translates them into **A–Z English alphabets** instantly. The system uses **MediaPipe** for hand landmark detection and a **TensorFlow/Keras MLP model** trained on 50,859 samples of ISL gesture data.

The web interface is built with **Streamlit** and includes a **Text-to-Speech** feature powered by **pyttsx3**, which speaks every confirmed sign out loud — making it accessible and interactive.

---

## ✨ Features

- 🎥 **Real-time webcam detection** — live hand sign recognition
- 🤟 **A–Z ISL alphabet support** — all 26 Indian Sign Language alphabets
- 🙌 **Both one-hand and two-hand sign support**
  - Single-hand signs: C, I, L, O, U, V
  - Two-hand signs: A, B, D, E, F, G, H, J, K, M, N, P, Q, R, S, T, W, X, Y, Z
- 🔊 **Text-to-Speech** — model speaks every detected sign using offline TTS
- 📝 **Sentence builder** — signs combine into a full sentence in real time
- 📊 **Confidence meter + capture progress bar** — visual feedback while signing
- 📜 **Sign history log** — timestamped record of all confirmed signs
- 🌙 **Dark themed UI** — clean, modern Streamlit interface

---

## 🗂️ Project Structure

```
isl-sign-language-translator/
│
├── app.py                  # Streamlit web application (main UI + camera loop)
├── main.py                 # Model training script
│
├── model/
│   ├── sign_model.h5       # Trained Keras model (generated after training)
│   └── label_encoder.pkl   # Label encoder (generated after training)
│
├── Indian Sign Language Gesture Landmarks.csv   # Dataset
│
├── requirements.txt        # All Python dependencies
├── .gitignore              # Git ignore rules
└── README.md               # This file
```

---

## 📊 Dataset

| Property | Value |
|---|---|
| Total Samples | 50,859 |
| Classes | 26 (A–Z) |
| Features per sample | 127 |
| Two-hand samples | 38,965 |
| Single-hand samples | 11,894 |
| Samples per class | ~1,758 – 2,000 |

**Feature breakdown (127 total):**
- `uses_two_hands` — 1 value (0 or 1)
- `left_hand` — 21 landmarks × 3 (x, y, z) = 63 values
- `right_hand` — 21 landmarks × 3 (x, y, z) = 63 values

Absent hand slots are filled with `-1.0` as sentinel values (matching training data exactly).

---

## 🧠 Model Architecture

```
Input (127 features)
    ↓
Dense(256, relu) → BatchNorm → Dropout(0.3)
    ↓
Dense(128, relu) → BatchNorm → Dropout(0.3)
    ↓
Dense(64,  relu) → Dropout(0.15)
    ↓
Dense(26, softmax)   ← Output: A–Z
```

**Training config:**
- Optimizer: Adam (lr=1e-3)
- Loss: Sparse Categorical Crossentropy
- Epochs: 60 (with EarlyStopping)
- Batch size: 32
- Callbacks: EarlyStopping + ReduceLROnPlateau

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/ravi2002roshan/isl-sign-language-translator.git
cd isl-sign-language-translator
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the model

```bash
python main.py
```

This will generate `model/sign_model.h5` and `model/label_encoder.pkl`.

### 4. Run the app

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

---

## 📦 Requirements

```
streamlit
opencv-python
mediapipe==0.10.21
tensorflow
numpy
scikit-learn
matplotlib
seaborn
pandas
pyttsx3
```

> ⚠️ Use **mediapipe==0.10.21** specifically. Newer versions break `mp.solutions` interface.

---

## 🖥️ How To Use

1. Run `streamlit run app.py` and open the browser
2. Click **▶ Start Camera**
3. Show your hand sign in front of the webcam
4. Hold the sign steady until the **progress bar** fills up
5. The confirmed letter appears on screen and is **spoken aloud**
6. Signs build into a **sentence** automatically
7. Use **Space / Delete / Clear** buttons to edit the sentence
8. Click **🔊 Speak Sentence** to hear the full sentence

---

## 🔧 Key Technical Fixes

During development, three critical bugs were identified and fixed by analyzing the raw dataset:

| Bug | Root Cause | Fix |
|---|---|---|
| Numerical output (0,1,2 instead of A,B,C) | LabelEncoder on integers returns integers | Direct mapping: `chr(ord('A') + class_idx)` |
| Wrong predictions | Code used `0.0` for absent hand; dataset uses `-1.0` | Fill absent hand with `-1.0` sentinel |
| Mirror/flip coordinate mismatch | MediaPipe was run on flipped frame, shifting x-coordinates | Run MediaPipe on raw frame, flip only for display |

--

| Real-time Detection | Sentence Builder |
|---|---|
| *(screenshot)* | *(screenshot)* |

---

## 🤝 Connect

**Ravi Roshan**
- 🔗 LinkedIn: [ravi-roshan-710105347](https://www.linkedin.com/in/ravi-roshan-710105347)
- 💻 GitHub: [ravi2002roshan](https://github.com/ravi2002roshan)

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

--

Made with ❤️ by **Ravi Roshan**

⭐ Star this repo if you found it helpful!
<img width="1919" height="1020" alt="isl" src="https://github.com/user-attachments/assets/dd620bc4-ce32-4fbc-a6e3-dc89a20b8aa5" />




