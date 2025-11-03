import streamlit as st
import numpy as np
import time

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="🌾 AgriYield Predictor", page_icon="🌾", layout="wide")

# ---------------- CUSTOM ANIMATED BACKGROUND STYLE ----------------
unique_bg = """
<style>
@keyframes gradientShift {
  0% {background-position: 0% 50%;}
  50% {background-position: 100% 50%;}
  100% {background-position: 0% 50%;}
}
[data-testid="stAppViewContainer"] {
  background: linear-gradient(-45deg, #ffb6c1, #caffbf, #9bf6ff, #ffd6a5);
  background-size: 400% 400%;
  animation: gradientShift 12s ease infinite;
}
[data-testid="stHeader"] { background: rgba(0,0,0,0); }
h1, h2, h3 {
  text-align: center;
  color: #033500 !important;
  font-family: 'Poppins', sans-serif;
  font-weight: 900;
}
.stButton>button {
  background: linear-gradient(to right, #ff9a9e, #fad0c4);
  color: black;
  border-radius: 14px;
  font-size: 20px;
  padding: 10px 28px;
  border: none;
  box-shadow: 0 4px 10px rgba(0,0,0,0.3);
  transition: 0.3s;
}
.stButton>button:hover {
  transform: scale(1.08);
  background: linear-gradient(to right, #a1ffce, #faffd1);
  color: #033500;
}
.stNumberInput>div>div>input {
  background-color: #fff8f0;
  border-radius: 12px;
  color: #222;
  font-size: 17px;
  font-weight: 500;
}
.result-box {
  backdrop-filter: blur(15px);
  background-color: rgba(255,255,255,0.35);
  border-radius: 25px;
  padding: 25px;
  box-shadow: 0 6px 25px rgba(0,0,0,0.3);
  text-align: center;
  transition: 0.4s ease;
}
.result-box:hover { transform: scale(1.03); }
.result-text {
  font-size: 22px;
  color: #0b3d02;
  font-weight: 700;
}
.footer {
  text-align: center;
  font-size: 14px;
  color: #222;
  margin-top: 60px;
}
</style>
"""
st.markdown(unique_bg, unsafe_allow_html=True)

# ---------------- SESSION STATE ----------------
if 'page' not in st.session_state:
    st.session_state['page'] = 'welcome'

def switch_page(page_name):
    st.session_state['page'] = page_name

# ---------------- WELCOME PAGE ----------------
if st.session_state['page'] == 'welcome':
    st.markdown("<h1>🌸 Welcome to the Next-Gen AgriYield Predictor 🌾</h1>", unsafe_allow_html=True)
    st.markdown("<h3>AI + Nature 🌱 — Predict the Future of Crops with Smart Data</h3>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🚀 Start Prediction Journey"):
        switch_page('predictor')

# ---------------- MAIN PREDICTOR PAGE ----------------
elif st.session_state['page'] == 'predictor':
    st.markdown("<h1>🌾 AgriYield Predictor Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<h3>Estimate Yield (tons/hectare) for Given Crop and Conditions 🌿</h3>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # Crop input
    crop_list = ["Rice", "Wheat", "Maize", "Sugarcane", "Cotton", "Mango", "Banana", "Groundnut", "Barley"]
    crop_type = st.selectbox("🌿 Select Crop Type", crop_list)

    # Input fields
    col1, col2, col3 = st.columns(3)
    with col1:
        N = st.number_input("🧪 Nitrogen (N)", min_value=0, max_value=200, value=50)
        rainfall = st.number_input("🌧 Rainfall (mm)", min_value=0, max_value=500, value=120)
    with col2:
        P = st.number_input("🧫 Phosphorus (P)", min_value=0, max_value=200, value=40)
        temperature = st.number_input("🌡 Temperature (°C)", min_value=0.0, max_value=50.0, value=26.0)
    with col3:
        K = st.number_input("🧬 Potassium (K)", min_value=0, max_value=200, value=40)
        ph = st.number_input("⚗️ pH Level", min_value=0.0, max_value=14.0, value=6.5)
        humidity = st.number_input("💧 Humidity (%)", min_value=0.0, max_value=100.0, value=60.0)

    # Predict button
    if st.button("🌿 Predict Yield"):
        with st.spinner("🤖 Analyzing soil & climate data..."):
            time.sleep(2)

            # Deterministic formula to simulate yield prediction (changes with inputs)
            crop_factor = {
                "Rice": 1.0, "Wheat": 0.9, "Maize": 0.8, "Sugarcane": 1.8,
                "Cotton": 0.7, "Mango": 1.2, "Banana": 1.4, "Groundnut": 0.6, "Barley": 0.85
            }

            # Formula-based yield calculation
            base_yield = (
                (0.05 * N) + (0.03 * P) + (0.04 * K) +
                (0.2 * temperature) + (0.1 * humidity) +
                (1.5 * ph) + (0.03 * rainfall)
            )

            yield_est = round(base_yield * crop_factor[crop_type] / 10, 2)

        st.balloons()
        result_html = f"""
        <div class="result-box">
            <p class="result-text">🌾 <b>Crop Type:</b> {crop_type}</p>
            <p class="result-text">🌻 <b>Predicted Yield:</b> {yield_est} tons/hectare</p>
        </div>
        """
        st.markdown(result_html, unsafe_allow_html=True)

    if st.button("⬅️ Back to Welcome"):
        switch_page('welcome')

    st.markdown("<div class='footer'>🌍 Designed with 💚 by an Innovator in Agri-AI</div>", unsafe_allow_html=True)
