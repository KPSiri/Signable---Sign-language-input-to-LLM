import os
import cv2
import numpy as np
import streamlit as st
import anthropic
import joblib
import json
from collections import deque
from tensorflow.keras.models import load_model
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# --- INITIAL SETUP ---
st.set_page_config(page_title="Signable", layout="wide")

# Persistent Font Size Logic in Session State
if "font_size" not in st.session_state:
    st.session_state.font_size = 18

# Custom CSS for Font Size and Logo Centering
st.markdown(f"""
    <style>
    .stChatFloatingInputContainer {{ background-color: rgba(0,0,0,0); }}
    .stChatMessage {{ border-radius: 15px; margin-bottom: 10px; }}
    .chat-text {{ 
        font-size: {st.session_state.font_size}px !important; 
        line-height: 1.5;
    }}
    .centered-image {{
        display: flex;
        justify-content: center;
    }}
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def init_assets():
    # Keep original paths as per your script
    model = load_model("app/model/sign_language_recognition_new_40.keras")
    scaler = joblib.load("app/model/scaler_new_40.pkl")
    label_encoder = joblib.load("app/model/label_encoder_new_40.pkl")
    with open("app/model/feature_order.json") as f:
        feat_order = json.load(f)
    
    hand_opts = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path="hand_landmarker.task"),
        num_hands=2, min_hand_detection_confidence=0.5
    )
    pose_opts = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path="pose_landmarker_lite.task")
    )
    
    return {
        "model": model, "scaler": scaler, "label_encoder": label_encoder,
        "feat_order": feat_order,
        "hand_det": mp_vision.HandLandmarker.create_from_options(hand_opts),
        "pose_det": mp_vision.PoseLandmarker.create_from_options(pose_opts)
    }

assets = init_assets()

# --- CLAUDE LLM INTEGRATION ---
def call_claude(api_key, gloss_words):
    client = anthropic.Anthropic(api_key=api_key)
    gloss_str = " ".join(gloss_words).upper()
    current_model = "claude-sonnet-4-20250514" 

    format_resp = client.messages.create(
        model=current_model,
        max_tokens=100,
        messages=[{"role": "user", "content": f"Convert these ASL glosses into a natural English sentence typically an question user ask to a LLM : [{gloss_str}]. Reply with ONLY the sentence."}]
    )
    natural_q = format_resp.content[0].text.strip()

    answer_resp = client.messages.create(
        model=current_model,
        max_tokens=300,
        system="You are a helpful assistant for a deaf user. Be concise and warm.",
        messages=[{"role": "user", "content": natural_q}]
    )
    return natural_q, answer_resp.content[0].text.strip()

# --- FEATURE EXTRACTION ---
def get_frame_landmarks(frame, detectors):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    h_res = detectors["hand_det"].detect(mp_img)
    p_res = detectors["pose_det"].detect(mp_img)
    lm = {"hand_0": [], "hand_1": [], "pose": []}
    tips, joints = [4, 8, 12, 16, 20], [11, 12, 13, 14, 15, 16]
    for i, hand_lms in enumerate(h_res.hand_landmarks[:2]):
        for idx in tips:
            p = hand_lms[idx]
            lm[f"hand_{i}"].append({"x": p.x, "y": p.y, "z": p.z})
    if p_res.pose_landmarks:
        for idx in joints:
            p = p_res.pose_landmarks[0][idx]
            lm["pose"].append({"x": p.x, "y": p.y, "z": p.z})
    return lm

def compute_window_features(window, feat_order):
    feats = {k: 0.0 for k in feat_order}
    for ax in "xyz":
        vals = [p[ax] for fl in window for p in fl["pose"]]
        if vals: feats[f"joints_{ax}_mean"], feats[f"joints_{ax}_std"] = np.mean(vals), np.std(vals)
    for hi, prefix in [(0, "left"), (1, "right")]:
        for ax in "xyz":
            vals = [p[ax] for fl in window for p in fl[f"hand_{hi}"]]
            if vals:
                feats[f"{prefix}_tips_{ax}_mean"] = float(np.mean(vals))
                feats[f"{prefix}_tips_{ax}_std"]  = float(np.std(vals))
    return np.array([feats[k] for k in feat_order])

# --- UI & STATE ---
if "gloss_list" not in st.session_state: st.session_state.gloss_list = []
if "chat_history" not in st.session_state: st.session_state.chat_history = []

# (i) Font Size Sidebar Feature
with st.sidebar:
    st.header("App Settings")
    st.session_state.font_size = st.slider("Adjust Chat Font Size", 14, 40, 18)
    st.info("The text in the chat will update immediately.")

# 1) LOGO - BIG AND CENTERED

col_l1, col_l2, col_l3 = st.columns([2, 1, 2]) 
with col_l2:
    st.image("Logo.jpg", width=200) # Set 'width' to your preferred pixel size

col_cam, col_chat = st.columns([1.2, 1])

# --- BOX 1: CAMERA ---
with col_cam:
    with st.container(border=True):
        st.subheader("Live Recognition")
        run = st.checkbox('Camera Active', value=True)
        FRAME_WINDOW = st.image([])
        st.write("**Buffer:** " + " ".join([f"`{g}`" for g in st.session_state.gloss_list]))
        
        if st.button("Clear Buffer", use_container_width=True):
            st.session_state.gloss_list = []
            st.rerun()

# --- BOX 2: CHAT WINDOW ---
with col_chat:
    with st.container(border=True):
        st.subheader("Conversation")
        
        chat_container = st.container(height=400)
        with chat_container:
            for i, chat in enumerate(st.session_state.chat_history):
                with st.chat_message(chat["role"]):
                    # Apply dynamic font size to content
                    st.markdown(f'<div class="chat-text">{chat["content"]}</div>', unsafe_allow_html=True)
                    
                    # (ii) Read Out Aloud Option (Using Browser Synthesis)
                    if chat["role"] == "assistant":
                        tts_button = st.button(f"🔊 Read Aloud", key=f"tts_{i}")
                        if tts_button:
                            # Lightweight JS injection to trigger browser speech
                            st.components.v1.html(f"""
                                <script>
                                    var msg = new SpeechSynthesisUtterance("{chat['content']}");
                                    window.speechSynthesis.speak(msg);
                                </script>
                            """, height=0)

        api_key = "sk-xxxxxxxxxxxx"
        if st.button("Send to Claude ⚡", type="primary", use_container_width=True) and st.session_state.gloss_list:
            with st.spinner("Interpreting..."):
                try:
                    question, answer = call_claude(api_key, st.session_state.gloss_list)
                    st.session_state.chat_history.append({"role": "user", "content": question})
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                    st.session_state.gloss_list = []
                    st.rerun()
                except Exception as e:
                    st.error(f"API Error: {e}")

# --- CAMERA LOOP ---
cap = cv2.VideoCapture(0)
frame_buf, feat_buf = deque(maxlen=7), deque(maxlen=5) 

while run:
    ret, frame = cap.read()
    if not ret: break
    lms = get_frame_landmarks(frame, assets)
    frame_buf.append(lms)
    
    if len(frame_buf) == frame_buf.maxlen:
        feat_vector = compute_window_features(list(frame_buf), assets["feat_order"])
        feat_buf.append(feat_vector)
        frame_buf.clear()

    if len(feat_buf) == feat_buf.maxlen:
        X = np.array(list(feat_buf)).reshape(1, 5, -1)
        Xs = assets["scaler"].transform(X.reshape(5, -1)).reshape(1, 5, -1)
        probs = assets["model"].predict(Xs, verbose=0)[0]
        idx = np.argmax(probs)
        if probs[idx] > 0.65:
            label = assets["label_encoder"].inverse_transform([idx])[0]
            if not st.session_state.gloss_list or label != st.session_state.gloss_list[-1]:
                st.session_state.gloss_list.append(label)
                st.rerun()

    FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
cap.release()
