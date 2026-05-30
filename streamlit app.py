"""
app.py — ISL Hand Sign Language Translator
===========================================
Real-time webcam se Indian Sign Language detect karta hai.
Detected sign automatically SPEAK bhi karta hai (Text-to-Speech).

REQUIREMENTS:
    pip install streamlit opencv-python mediapipe==0.10.21 tensorflow numpy pyttsx3

Run:
    streamlit run app.py





"""

import streamlit as st
import cv2
import numpy as np
import pickle
import time
import os
import threading
from collections import deque
import tensorflow as tf

# ── Text-to-Speech Setup ─────────────────────────────────────────────────────
try:
    import pyttsx3

    def speak_text(text):
        """Background thread mein TTS — camera loop block nahi hota."""
        def _speak():
            try:
                engine = pyttsx3.init()
                engine.setProperty('rate', 140)
                engine.setProperty('volume', 1.0)
                engine.say(text)
                engine.runAndWait()
                engine.stop()
            except Exception:
                pass
        threading.Thread(target=_speak, daemon=True).start()

    TTS_OK = True

except ImportError:
    TTS_OK = False
    def speak_text(text):
        pass

# ── MediaPipe ────────────────────────────────────────────────────────────────
try:
    import mediapipe as mp
    mp_hands     = mp.solutions.hands
    mp_draw      = mp.solutions.drawing_utils
    MEDIAPIPE_OK = True
except AttributeError:
    MEDIAPIPE_OK = False

# ── ISL Class Map (0→A, 1→B … 25→Z) ─────────────────────────────────────────
# BUG FIX 1: Dataset targets are integers 0-25 = A-Z.
# LabelEncoder.inverse_transform() gives back integers, not letters.
# We bypass this by directly mapping index → letter.
ISL_CLASSES = {i: chr(ord('A') + i) for i in range(26)}

# ── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="ISL Sign Translator", page_icon="🤟", layout="wide")

# ── CUSTOM CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
.stApp { background-color: #0f172a; color: white; }
.sign-box {
    background: #1e293b; padding: 20px; border-radius: 15px;
    text-align: center; margin-bottom: 15px; border: 2px solid #334155;
}
.sign-letter { font-size: 80px; font-weight: bold; color: #38bdf8; }
.sign-label  { font-size: 18px; color: #94a3b8; margin-top: 5px; }
.sentence-box {
    background: #1e293b; padding: 15px; border-radius: 10px;
    font-size: 24px; min-height: 80px; color: white; word-wrap: break-word;
    border: 1px solid #334155;
}
.error-box {
    background: #7f1d1d; padding: 15px; border-radius: 10px;
    color: #fca5a5; font-size: 16px;
}
.status-detecting {
    background: #052e16; border: 1px solid #16a34a; padding: 10px 15px;
    border-radius: 8px; color: #4ade80; font-size: 15px; text-align: center;
}
.status-waiting {
    background: #1c1917; border: 1px solid #57534e; padding: 10px 15px;
    border-radius: 8px; color: #a8a29e; font-size: 15px; text-align: center;
}
.status-confirmed {
    background: #1e3a5f; border: 2px solid #38bdf8; padding: 10px 15px;
    border-radius: 8px; color: #7dd3fc; font-size: 15px;
    text-align: center; font-weight: bold;
}
.tts-badge {
    background: #4c1d95; border: 1px solid #7c3aed; padding: 6px 12px;
    border-radius: 20px; color: #c4b5fd; font-size: 13px;
    display: inline-block; margin-top: 8px;
}
.history-box {
    background: #1e293b; padding: 12px; border-radius: 10px;
    font-size: 14px; color: #94a3b8; max-height: 150px; overflow-y: auto;
}
</style>
""", unsafe_allow_html=True)

# ── MEDIAPIPE CHECK ───────────────────────────────────────────────────────────
if not MEDIAPIPE_OK:
    st.markdown("""
    <div class="error-box">
    ⚠️ <b>MediaPipe Error</b><br><br>
    Terminal mein chalao: <code>pip install mediapipe==0.10.21</code><br>
    Phir app restart karo.
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── MODEL LOAD ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = 'model/sign_model.h5'
    if not os.path.exists(model_path):
        st.error(f"❌ Model file nahi mila: `{model_path}`\nPehle `python main.py` chalao.")
        st.stop()
    model = tf.keras.models.load_model(model_path, compile=False)
    return model

# ── MEDIAPIPE HANDS ───────────────────────────────────────────────────────────
@st.cache_resource
def load_mediapipe():
    return mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6
    )

# ── FEATURE EXTRACTION — FIXED ────────────────────────────────────────────────
def extract_features(results):
    """
    127 features — exact same structure as training CSV:
      [0]      uses_two_hands  (1 value)
      [1..63]  left_hand       (21 landmarks × x,y,z = 63 values)
      [64..126] right_hand     (21 landmarks × x,y,z = 63 values)

    BUG FIX 2a — RAW coordinates:
      Training CSV stores RAW mediapipe coordinates (x,y,z as-is).
      Previous code subtracted wrist → WRONG features → bad predictions.
      Now we store lm.x, lm.y, lm.z directly (no subtraction).

    BUG FIX 2b — Left/Right swap after mirror flip:
      After cv2.flip(frame, 1), the image is horizontally mirrored.
      MediaPipe processes the FLIPPED image, so its 'Left' label
      actually corresponds to the person's RIGHT hand on screen.
      Training data was collected on a normal (non-flipped) camera
      where 'Left' = left hand slot in CSV.
      Fix: After flip, swap the MediaPipe labels:
        MediaPipe says 'Left'  → store in right_hand slot
        MediaPipe says 'Right' → store in left_hand slot
    """
    features = []

    num_hands = len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0
    uses_two_hands = 1.0 if num_hands == 2 else 0.0
    features.append(uses_two_hands)

    left_hand  = [0.0] * 63   # 21 landmarks × 3
    right_hand = [0.0] * 63

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(
            results.multi_hand_landmarks,
            results.multi_handedness
        ):
            # RAW coordinates — no wrist subtraction
            coords = []
            for lm in hand_landmarks.landmark:
                coords.extend([lm.x, lm.y, lm.z])   # ← BUG FIX 2a

            mp_label = handedness.classification[0].label  # 'Left' or 'Right'

            # BUG FIX 2b: Swap because image is mirror-flipped
            if mp_label == 'Left':
                right_hand = coords   # MediaPipe 'Left' → right slot after flip
            else:
                left_hand = coords    # MediaPipe 'Right' → left slot after flip

    features.extend(left_hand)
    features.extend(right_hand)

    return np.array(features, dtype=np.float32)   # shape: (127,)


# ── PREDICTION — FIXED ────────────────────────────────────────────────────────
def predict_sign(model, features):
    """
    BUG FIX 1: Instead of using LabelEncoder.inverse_transform() which
    returns integer 0-25, we directly map: index → chr(ord('A') + index).
    This gives 'A', 'B', ... 'Z' correctly.
    """
    inp        = features.reshape(1, -1)
    prediction = model.predict(inp, verbose=0)[0]
    class_idx  = int(np.argmax(prediction))
    confidence = float(prediction[class_idx])

    # Direct mapping: 0→A, 1→B ... 25→Z
    label = ISL_CLASSES.get(class_idx, str(class_idx))

    return label, confidence


# ── SESSION STATE ─────────────────────────────────────────────────────────────
defaults = {
    'sentence':      "",
    'running':       False,
    'last_sign':     "",
    'sign_history':  [],
    'speak_enabled': True,
    'tts_mode':      "Each letter/sign",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🤟 ISL Translator")
    st.markdown("### ⚙️ Settings")

    confidence_threshold = st.slider("Confidence Threshold", 0.50, 1.00, 0.75, 0.01)
    hold_frames = st.slider("Stable Frames Required", 5, 30, 15,
                            help="Sign kitne frames tak same rahe tab confirm hoga")

    st.markdown("---")
    st.markdown("### 🔊 Text-to-Speech")

    if TTS_OK:
        st.session_state.speak_enabled = st.toggle(
            "Speak detected signs", value=st.session_state.speak_enabled
        )
        tts_mode = st.radio("Speak mode", ["Each letter/sign", "Full sentence (on Stop)"])
        st.markdown('<div class="tts-badge">🔊 pyttsx3 (Offline)</div>', unsafe_allow_html=True)
    else:
        st.warning("⚠️ pyttsx3 not installed.\n`pip install pyttsx3`")
        tts_mode = "Each letter/sign"

    st.markdown("---")
    st.markdown("### 📋 Instructions")
    st.markdown("""
    1. **Start Camera** dabao
    2. Haath webcam ke saamne rakho
    3. Sign **stable** rakho jab tak progress bar bhare
    4. Sign confirm hone pe **screen + awaaz** dono milegi
    5. **Speak Sentence** se poora sentence bolo
    """)

    st.markdown("---")
    st.markdown("### 🐛 Bug Fixes in this version")
    st.markdown("""
    - ✅ A-Z letters output (numbers nahi)
    - ✅ Dono haath detect hote hain
    - ✅ Raw coordinates (training data se match)
    - ✅ Mirror flip hand swap fixed
    """)

# ── MAIN UI ───────────────────────────────────────────────────────────────────
st.title("🤟 Indian Sign Language Translator")
st.markdown("Real-time Sign Detection • **TensorFlow** + **MediaPipe** • 🔊 Text-to-Speech")

col1, col2 = st.columns([3, 2])

with col2:
    st.subheader("✋ Detected Sign")
    sign_placeholder = st.empty()

    st.subheader("📊 Confidence")
    conf_placeholder = st.empty()

    st.subheader("⏳ Capture Progress")
    capture_status_placeholder   = st.empty()
    capture_progress_placeholder = st.empty()

    tts_status_placeholder = st.empty()

    st.markdown("---")
    st.subheader("📝 Sentence")
    sentence_placeholder = st.empty()

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("␣ Space"):
            st.session_state.sentence += " "
    with c2:
        if st.button("⌫ Delete"):
            st.session_state.sentence = st.session_state.sentence[:-1]
    with c3:
        if st.button("🗑 Clear"):
            st.session_state.sentence   = ""
            st.session_state.sign_history = []

    if st.button("🔊 Speak Sentence", use_container_width=True):
        if st.session_state.sentence.strip():
            speak_text(st.session_state.sentence)

    st.markdown("---")
    st.subheader("📜 Sign History")
    history_placeholder = st.empty()

with col1:
    b1, b2 = st.columns(2)
    with b1:
        start_btn = st.button("▶ Start Camera", use_container_width=True)
    with b2:
        stop_btn  = st.button("⏹ Stop",         use_container_width=True)

    if start_btn:
        st.session_state.running = True
    if stop_btn:
        st.session_state.running   = False
        st.session_state.last_sign = ""
        if TTS_OK and st.session_state.speak_enabled:
            if tts_mode == "Full sentence (on Stop)" and st.session_state.sentence.strip():
                speak_text(st.session_state.sentence)

    frame_placeholder      = st.empty()
    cam_status_placeholder = st.empty()


# ── RIGHT PANEL RENDERER ──────────────────────────────────────────────────────
def render_panel(current_sign, confidence, buf_len, hold_total, confirmed=False):

    sign_placeholder.markdown(f"""
    <div class="sign-box">
        <div class="sign-letter">{current_sign}</div>
        <div class="sign-label">Current Detection</div>
    </div>
    """, unsafe_allow_html=True)

    conf_placeholder.progress(min(max(float(confidence), 0.0), 1.0))

    progress = min(buf_len / max(hold_total, 1), 1.0)

    if current_sign == "—":
        capture_status_placeholder.markdown(
            '<div class="status-waiting">👋 Haath dikhao camera ke saamne</div>',
            unsafe_allow_html=True)
        capture_progress_placeholder.progress(0.0)
    elif confirmed:
        capture_status_placeholder.markdown(
            f'<div class="status-confirmed">✅ Confirmed: <b>{current_sign}</b> 🔊</div>',
            unsafe_allow_html=True)
        capture_progress_placeholder.progress(1.0)
    else:
        pct = int(progress * 100)
        capture_status_placeholder.markdown(
            f'<div class="status-detecting">🔍 Detecting <b>{current_sign}</b> — {pct}% ({buf_len}/{hold_total} frames)</div>',
            unsafe_allow_html=True)
        capture_progress_placeholder.progress(progress)

    sentence_placeholder.markdown(f"""
    <div class="sentence-box">
        {st.session_state.sentence if st.session_state.sentence
         else "<span style='color:#475569'>Sentence yahan dikhegi...</span>"}
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.sign_history:
        html = "<div class='history-box'>"
        for e in reversed(st.session_state.sign_history[-10:]):
            html += f"<div>🕐 {e['time']} &nbsp;→&nbsp; <b style='color:#38bdf8'>{e['sign']}</b></div>"
        html += "</div>"
        history_placeholder.markdown(html, unsafe_allow_html=True)
    else:
        history_placeholder.markdown(
            "<div class='history-box' style='color:#475569'>Abhi koi sign capture nahi hua</div>",
            unsafe_allow_html=True)


# ── CAMERA LOOP ───────────────────────────────────────────────────────────────
if st.session_state.running:

    model = load_model()
    hands = load_mediapipe()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("❌ Webcam open nahi ho raha.")
        st.session_state.running = False
        st.stop()

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cam_status_placeholder.markdown("🟢 **Camera Active**")

    prediction_buffer = deque(maxlen=hold_frames)
    just_confirmed    = False

    try:
        while st.session_state.running:

            ret, frame = cap.read()
            if not ret:
                st.error("❌ Camera frame nahi aa raha.")
                break

            # Mirror flip — natural selfie view
            frame   = cv2.flip(frame, 1)
            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            current_sign  = "—"
            confidence    = 0.0
            just_confirmed = False

            if results.multi_hand_landmarks:

                # Draw landmarks for ALL detected hands
                for hand_lm in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)

                # Show how many hands detected (top-right corner)
                num_detected = len(results.multi_hand_landmarks)
                hand_text    = f"Hands: {num_detected}/2"
                cv2.putText(frame, hand_text,
                            (frame.shape[1] - 160, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (255, 200, 0), 2, cv2.LINE_AA)

                # Extract 127 features (fixed)
                features = extract_features(results)

                # Predict
                current_sign, confidence = predict_sign(model, features)

                # Stability buffer
                if confidence >= confidence_threshold:
                    prediction_buffer.append(current_sign)
                else:
                    prediction_buffer.clear()

                # Confirm sign
                if (
                    len(prediction_buffer) == hold_frames
                    and len(set(prediction_buffer)) == 1
                    and prediction_buffer[0] != st.session_state.last_sign
                ):
                    confirmed_sign = prediction_buffer[0]
                    st.session_state.sentence  += confirmed_sign
                    st.session_state.last_sign  = confirmed_sign
                    just_confirmed              = True
                    st.session_state.sign_history.append({
                        'sign': confirmed_sign,
                        'time': time.strftime('%H:%M:%S')
                    })
                    prediction_buffer.clear()

                    # Speak the sign
                    if TTS_OK and st.session_state.speak_enabled and tts_mode == "Each letter/sign":
                        speak_text(confirmed_sign)

                    tts_status_placeholder.markdown(
                        f'<div class="tts-badge">🔊 Speaking: {confirmed_sign}</div>',
                        unsafe_allow_html=True
                    )

                # Overlay: sign + confidence
                cv2.putText(frame,
                            f'{current_sign}  ({confidence*100:.1f}%)',
                            (20, 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                            (0, 255, 100), 2, cv2.LINE_AA)

                # Progress bar at bottom of frame
                bx, by = 20, frame.shape[0] - 30
                bw, bh = frame.shape[1] - 40, 18
                filled = int(bw * min(len(prediction_buffer) / hold_frames, 1.0))
                cv2.rectangle(frame, (bx, by), (bx+bw, by+bh), (40, 40, 40), -1)
                if filled > 0:
                    bar_color = (0, 255, 0) if just_confirmed else (0, 200, 255)
                    cv2.rectangle(frame, (bx, by), (bx+filled, by+bh), bar_color, -1)
                cv2.putText(frame,
                            f'Capture: {len(prediction_buffer)}/{hold_frames}',
                            (bx, by - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (180, 180, 180), 1, cv2.LINE_AA)

                # Green flash on confirm
                if just_confirmed:
                    cv2.rectangle(frame, (0,0),
                                  (frame.shape[1]-1, frame.shape[0]-1),
                                  (0, 255, 0), 8)
                    cv2.putText(frame,
                                f'CONFIRMED: {st.session_state.last_sign}',
                                (20, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                                (0, 255, 0), 3, cv2.LINE_AA)

            else:
                # No hand
                prediction_buffer.clear()
                st.session_state.last_sign = ""
                cv2.putText(frame, 'No hand detected',
                            (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            (80, 120, 255), 2, cv2.LINE_AA)

            render_panel(current_sign, confidence,
                         len(prediction_buffer), hold_frames, just_confirmed)

            frame_placeholder.image(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                channels='RGB',
                use_container_width=True
            )

            time.sleep(0.03)

    finally:
        cap.release()
        cam_status_placeholder.markdown("🔴 **Camera Stopped**")


# ── IDLE STATE ────────────────────────────────────────────────────────────────
if not st.session_state.running:

    sign_placeholder.markdown("""
    <div class="sign-box">
        <div class="sign-letter" style="color:#475569">—</div>
        <div class="sign-label">Camera band hai</div>
    </div>
    """, unsafe_allow_html=True)

    conf_placeholder.progress(0.0)
    capture_status_placeholder.markdown(
        '<div class="status-waiting">▶ Camera start karo</div>',
        unsafe_allow_html=True)
    capture_progress_placeholder.progress(0.0)

    sentence_placeholder.markdown(f"""
    <div class="sentence-box">
        {st.session_state.sentence if st.session_state.sentence
         else "<span style='color:#475569'>Sentence yahan dikhegi...</span>"}
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.sign_history:
        html = "<div class='history-box'>"
        for e in reversed(st.session_state.sign_history[-10:]):
            html += f"<div>🕐 {e['time']} &nbsp;→&nbsp; <b style='color:#38bdf8'>{e['sign']}</b></div>"
        html += "</div>"
        history_placeholder.markdown(html, unsafe_allow_html=True)
    else:
        history_placeholder.markdown(
            "<div class='history-box' style='color:#475569'>Abhi koi sign capture nahi hua</div>",
            unsafe_allow_html=True)


# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("---")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Model Input",  "127 Features")
c2.metric("Classes",      "A–Z (26)")
c3.metric("Framework",    "TensorFlow")
c4.metric("TTS",          "pyttsx3 ✅" if TTS_OK else "Not installed ❌")
