
import streamlit as st
import pickle

st.set_page_config(page_title="AI Emotion Detection", layout="centered")

# Modern UI Styling
st.markdown("""
<style>

.stApp {
    background: 
        linear-gradient(rgba(255,255,255,0.85), rgba(255,255,255,0.85)),
        url("https://images.unsplash.com/photo-1506744038136-46273834b3fb");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

/* Main card */
.main-card {
    background-color: white;
    padding: 30px;
    border-radius: 20px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.2);
}

/* Title */
h1 {
    text-align: center;
    color: #2c3e50;
}

/* Text area */
.stTextArea textarea {
    border-radius: 12px;
    border: 2px solid #d0d7e2;
    padding: 10px;
}

/* Buttons */
.stButton>button {
    border-radius: 10px;
    height: 3em;
    font-weight: 600;
    transition: 0.3s;
}

.stButton>button:hover {
    transform: scale(1.05);
}

</style>
""", unsafe_allow_html=True)

# Load model
model = pickle.load(open("model/emotion_model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))

label_map = {
    0: ("Sadness", "😢"),
    1: ("Joy", "😊"),
    2: ("Love", "❤️"),
    3: ("Anger", "😠"),
    4: ("Fear", "😨"),
    5: ("Surprise", "😲")
}

emotion_colors = {
    "Sadness": "#5DADE2",
    "Joy": "#F4D03F",
    "Love": "#EC7063",
    "Anger": "#E74C3C",
    "Fear": "#8E44AD",
    "Surprise": "#48C9B0"
}

if "text_input" not in st.session_state:
    st.session_state.text_input = ""

st.markdown('<div class="main-card">', unsafe_allow_html=True)

st.title("🧠 AI Emotion Recognition System")
st.markdown("Detect the emotion behind your text using Machine Learning.")

st.markdown("### 🎭 Try Sample Emotions")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("😢 Sadness"):
        st.session_state.text_input = "I feel completely broken and empty today."
        st.rerun()
    if st.button("😠 Anger"):
        st.session_state.text_input = "I am furious about how unfair this situation is!"
        st.rerun()

with col2:
    if st.button("😊 Joy"):
        st.session_state.text_input = "This is the happiest day of my life!"
        st.rerun()
    if st.button("😨 Fear"):
        st.session_state.text_input = "I am really scared about what might happen next."
        st.rerun()

with col3:
    if st.button("❤️ Love"):
        st.session_state.text_input = "I love you more than words can describe."
        st.rerun()
    if st.button("😲 Surprise"):
        st.session_state.text_input = "Wow! I did not see that coming at all!"
        st.rerun()

st.markdown("---")

user_input = st.text_area(
    "Enter your sentence:",
    value=st.session_state.text_input,
    height=150
)

if st.button("🔍 Predict Emotion"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        vector = vectorizer.transform([user_input])
        prediction = model.predict(vector)[0]
        probabilities = model.predict_proba(vector)[0]

        emotion, emoji = label_map[prediction]
        confidence = round(max(probabilities) * 100, 2)

        color = emotion_colors[emotion]



        st.markdown(f"""
    <div style="
    background-color:white;
    border-left:6px solid {color};
    padding:20px;
    border-radius:12px;
    box-shadow:0 2px 10px rgba(0,0,0,0.05);
    margin-top:20px;
    ">
    <div style="font-size:22px; font-weight:600;">
        {emoji} {emotion}
    </div>
    <div style="margin-top:8px; color:#555;">
        Confidence: {confidence}%
    </div>
</div>
""", unsafe_allow_html=True)

if st.button("♻ Clear"):
    st.session_state.text_input = ""
    st.rerun()

st.markdown("</div>", unsafe_allow_html=True)

st.caption("Built by Disha • Emotion Recognition Project")