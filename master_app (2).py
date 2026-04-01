import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import plotly.graph_objects as go

# --- 1. التصميم (CSS المطور للشريط) ---
st.set_page_config(page_title="AI Market Analysis", layout="wide")

st.markdown("""
    <style>
    .stApp { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); }
    
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 25px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 20px;
    }

    /* تصميم شريط النجاح العملاق */
    .progress-container {
        width: 100%;
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 50px;
        height: 45px; /* حجم كبير وواضح */
        margin: 20px 0;
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }

    .progress-fill {
        height: 100%;
        border-radius: 50px;
        transition: width 1s ease-in-out;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 0 20px rgba(0, 212, 255, 0.5);
    }

    .progress-text {
        color: white;
        font-weight: bold;
        font-size: 20px;
        z-index: 2;
        text-shadow: 1px 1px 5px rgba(0,0,0,0.5);
    }
    
    .stButton>button {
        background: linear-gradient(45deg, #00d4ff, #0050ff);
        color: white; border-radius: 12px; width: 100%; font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. محرك الذكاء الاصطناعي ---
@st.cache_resource
def train_model():
    df = pd.read_csv('smartphone_cleaned_v5.csv').fillna(0)
    le_brand = LabelEncoder(); le_proc = LabelEncoder()
    df['brand_encoded'] = le_brand.fit_transform(df['brand_name'].astype(str))
    df['proc_encoded'] = le_proc.fit_transform(df['processor_brand'].astype(str))
    features = ['brand_encoded', 'ram_capacity', 'internal_memory', 'battery_capacity', 'primary_camera_rear', 'has_5g', 'proc_encoded', 'num_cores']
    model = RandomForestRegressor(n_estimators=100, random_state=42).fit(df[features], df['price'] / 83)
    return model, features, le_brand, le_proc

model, features_list, le_brand, le_proc = train_model()

# --- 3. السايد بار ---
with st.sidebar:
    st.markdown("<h2 style='text-align:center;'>⚙️ CONFIG</h2>", unsafe_allow_html=True)
    brand_choice = st.selectbox("Brand", le_brand.classes_)
    proc_choice = st.selectbox("CPU", le_proc.classes_)
    cores = st.selectbox("Cores", [4, 6, 8, 10], index=2)
    ram = st.selectbox("RAM (GB)", [2, 4, 6, 8, 12, 16, 24], index=3)
    storage = st.selectbox("Storage (GB)", [32, 64, 128, 256, 512, 1024], index=2)
    battery = st.number_input("Battery (mAh)", 2000, 8000, 5000)
    camera = st.selectbox("Camera (MP)", [8, 16, 48, 50, 64, 108, 200], index=3)
    is_5g = st.checkbox("5G Support", True)
    user_price = st.number_input("Market Price ($)", value=250)
    analyze_btn = st.button("RUN ANALYSIS")

# --- 4. العرض الرئيسي ---
st.markdown("<h1>Market Pulse AI</h1>", unsafe_allow_html=True)

if analyze_btn:
    brand_enc = le_brand.transform([brand_choice])[0]
    proc_enc = le_proc.transform([proc_choice])[0]
    input_data = pd.DataFrame([[brand_enc, ram, storage, battery, camera, int(is_5g), proc_enc, cores]], columns=features_list)
    
    fair_market_value = model.predict(input_data)[0]
    prod_cost = 70 + (ram * 3) + (np.log2(storage) * 6) + (camera * 0.2) + (battery * 0.004) + (25 if is_5g else 5)
    chance = 100 / (1 + np.exp(9 * (user_price/fair_market_value - 1.1)))

    # البطاقات العلوية
    col1, col2, col3 = st.columns(3)
    col1.markdown(f"<div class='glass-card'><p style='color:#aaa;'>MARKET VALUE</p><h2 style='color:#00d4ff;'>${fair_market_value:.2f}</h2></div>", unsafe_allow_html=True)
    col2.markdown(f"<div class='glass-card'><p style='color:#aaa;'>COST ESTIMATE</p><h2 style='color:#ff4b4b;'>${prod_cost:.2f}</h2></div>", unsafe_allow_html=True)
    col3.markdown(f"<div class='glass-card'><p style='color:#aaa;'>PROFIT</p><h2 style='color:#00ff88;'>${user_price-prod_cost:.2f}</h2></div>", unsafe_allow_html=True)

    # شريط النجاح العملاق (المطلوب)
    color = "#00ff88" if chance > 70 else "#FFA500" if chance > 40 else "#FF4B4B"
    st.markdown(f"""
        <div class='glass-card'>
            <h3 style='text-align: center; margin-bottom: 10px;'>Market Success Probability</h3>
            <div class='progress-container'>
                <div class='progress-fill' style='width: {chance}%; background: {color};'>
                    <span class='progress-text'>{chance:.1f}%</span>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # الرسم البياني
    fig = go.Figure(data=[
        go.Bar(name='Proposed', x=['Compare'], y=[user_price], marker_color='#00d4ff'),
        go.Bar(name='Market', x=['Compare'], y=[fair_market_value], marker_color='#636EFA')
    ])
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="white", barmode='group', height=300)
    st.plotly_chart(fig, use_container_width=True)

else:
    st.markdown("<div class='glass-card' style='text-align: center; padding: 80px;'><h2>Adjust specs and click RUN ANALYSIS</h2></div>", unsafe_allow_html=True)