import subprocess
import string
import re
import streamlit as st
import joblib
import json
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, confusion_matrix, roc_curve, auc
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from collections import Counter
import pickle

# 設定路徑
MODELS_DIR = Path("models")
ARTIFACTS_DIR = Path("models/artifacts")

# 確保目錄存在
MODELS_DIR.mkdir(exist_ok=True)
ARTIFACTS_DIR.mkdir(exist_ok=True)

# 資料集 URL
DATASET_URL = "https://raw.githubusercontent.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/refs/heads/master/Chapter03/datasets/sms_spam_no_header.csv"

st.set_page_config(
    page_title="📧 Spam Detector",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "# Spam Email/SMS Classifier\nBuilt with Streamlit & Machine Learning"
    }
)

# 現代化 CSS 樣式
st.markdown("""
<style>
    /* 主要配色 */
    :root {
        --primary-color: #FF4B4B;
        --secondary-color: #0068C9;
        --success-color: #21C354;
        --warning-color: #FFA600;
        --danger-color: #FF4B4B;
        --dark-bg: #0E1117;
        --card-bg: #262730;
    }
    
    /* 隱藏 Streamlit 預設元素 */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* 標題樣式 */
    .main-title {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        padding: 1rem 0;
    }
    
    .subtitle {
        font-size: 1.2rem;
        color: #a0a0a0;
        text-align: center;
        margin-bottom: 3rem;
    }
    
    /* 卡片樣式 */
    .custom-card {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        backdrop-filter: blur(4px);
    }
    
    /* 按鈕樣式增強 */
    .stButton > button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        font-weight: 600;
        font-size: 1.1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: none;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3);
    }
    
    /* Metric 卡片 */
    [data-testid="stMetricValue"] {
        font-size: 2.5rem;
        font-weight: 700;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 1rem;
        font-weight: 500;
    }
    
    /* 輸入框樣式 */
    .stTextArea textarea {
        border-radius: 10px;
        border: 2px solid rgba(102, 126, 234, 0.3);
        font-size: 1rem;
        padding: 1rem;
    }
    
    .stTextArea textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 0.2rem rgba(102, 126, 234, 0.25);
    }
    
    /* 側邊欄樣式 */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(102, 126, 234, 0.05) 0%, rgba(118, 75, 162, 0.05) 100%);
    }
    
    /* 標籤頁樣式 */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px 10px 0 0;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
    }
    
    /* 進度指示器 */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* 數據表格樣式 */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
    }
    
    /* 警告框樣式 */
    .stAlert {
        border-radius: 10px;
        border-left: 5px solid;
    }
    
    /* 分隔線 */
    hr {
        margin: 2rem 0;
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, rgba(102, 126, 234, 0.5), transparent);
    }
    
    /* 圖表容器 */
    .chart-container {
        background: rgba(255, 255, 255, 0.02);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    /* 徽章樣式 */
    .badge {
        display: inline-block;
        padding: 0.35em 0.65em;
        font-size: 0.85em;
        font-weight: 700;
        line-height: 1;
        text-align: center;
        white-space: nowrap;
        vertical-align: baseline;
        border-radius: 0.5rem;
        margin: 0.2rem;
    }
    
    .badge-success {
        background-color: #21C354;
        color: white;
    }
    
    .badge-danger {
        background-color: #FF4B4B;
        color: white;
    }
    
    /* 動畫 */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .fade-in {
        animation: fadeIn 0.5s ease-out;
    }
</style>
""", unsafe_allow_html=True)

# ==================== 資料載入 ====================
@st.cache_data
def load_data():
    """載入資料集"""
    try:
        df = pd.read_csv(DATASET_URL, names=['label', 'text'], encoding='utf-8')
        return df
    except Exception as e:
        st.error(f"❌ Error loading dataset: {e}")
        return None

# ==================== 模型載入 ====================
@st.cache_resource
def load_model_artifacts():
    """載入模型相關檔案"""
    model_path = MODELS_DIR / "spam_classifier.pkl"
    vectorizer_path = MODELS_DIR / "vectorizer.pkl"
    label_encoder_path = MODELS_DIR / "label_encoder.pkl"
    
    # 檢查檔案是否存在
    missing_files = []
    if not model_path.exists():
        missing_files.append("spam_classifier.pkl")
    if not vectorizer_path.exists():
        missing_files.append("vectorizer.pkl")
    if not label_encoder_path.exists():
        missing_files.append("label_encoder.pkl")
    
    if missing_files:
        st.warning(f"⚠️ Missing files: {', '.join(missing_files)}")
        st.info("💡 Please train the model first by running: `python train_baseline.py`")
        return None, None, None
    
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)
        with open(label_encoder_path, 'rb') as f:
            label_encoder = pickle.load(f)
        return model, vectorizer, label_encoder
        
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.info("💡 Try retraining the model: `python train_baseline.py`")
        return None, None, None

# ==================== 載入 Metrics ====================
def load_metrics():
    """載入訓練指標"""
    metrics_path = ARTIFACTS_DIR / "metrics.json"
    if metrics_path.exists():
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None

# ==================== 文字預處理 ====================
def preprocess_text(text, lowercase=True, remove_punct=True, remove_stopwords=False):
    """預處理文字"""
    if lowercase:
        text = text.lower()
    if remove_punct:
        text = re.sub(r"[{}]".format(re.escape(string.punctuation)), " ", text)
    if remove_stopwords:
        tokens = [t for t in text.split() if t not in ENGLISH_STOP_WORDS]
        text = " ".join(tokens)
    return text

# ==================== 視覺化函數 ====================
def get_top_tokens(df, label, n_tokens=20):
    """取得 Top N tokens"""
    texts = df[df['label'] == label]['text'].values
    all_text = ' '.join(texts)
    tokens = re.findall(r'\b\w+\b', all_text.lower())
    token_counts = Counter(tokens)
    return token_counts.most_common(n_tokens)

def plot_top_tokens(df, n_tokens=20):
    """繪製 Top Tokens 對比圖"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.patch.set_facecolor('#0E1117')
    
    # HAM tokens
    ham_tokens = get_top_tokens(df, 'ham', n_tokens)
    ham_words, ham_counts = zip(*ham_tokens)
    colors_ham = plt.cm.viridis(np.linspace(0.3, 0.9, len(ham_words)))
    ax1.barh(range(len(ham_words)), ham_counts, color=colors_ham, edgecolor='white', linewidth=0.5)
    ax1.set_yticks(range(len(ham_words)))
    ax1.set_yticklabels(ham_words, fontsize=11)
    ax1.invert_yaxis()
    ax1.set_xlabel('Frequency', fontsize=13, fontweight='bold', color='white')
    ax1.set_ylabel('Token', fontsize=13, fontweight='bold', color='white')
    ax1.set_title('✅ Legitimate Messages (HAM)', fontsize=15, fontweight='bold', color='#21C354', pad=15)
    ax1.grid(axis='x', alpha=0.2, color='white')
    ax1.set_facecolor('#1a1a2e')
    ax1.tick_params(colors='white')
    ax1.spines['bottom'].set_color('white')
    ax1.spines['left'].set_color('white')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # SPAM tokens
    spam_tokens = get_top_tokens(df, 'spam', n_tokens)
    spam_words, spam_counts = zip(*spam_tokens)
    colors_spam = plt.cm.plasma(np.linspace(0.3, 0.9, len(spam_words)))
    ax2.barh(range(len(spam_words)), spam_counts, color=colors_spam, edgecolor='white', linewidth=0.5)
    ax2.set_yticks(range(len(spam_words)))
    ax2.set_yticklabels(spam_words, fontsize=11)
    ax2.invert_yaxis()
    ax2.set_xlabel('Frequency', fontsize=13, fontweight='bold', color='white')
    ax2.set_ylabel('Token', fontsize=13, fontweight='bold', color='white')
    ax2.set_title('⚠️ Spam Messages', fontsize=15, fontweight='bold', color='#FF4B4B', pad=15)
    ax2.grid(axis='x', alpha=0.2, color='white')
    ax2.set_facecolor('#1a1a2e')
    ax2.tick_params(colors='white')
    ax2.spines['bottom'].set_color('white')
    ax2.spines['left'].set_color('white')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    plt.suptitle('📊 Top Frequent Words by Category', fontsize=18, fontweight='bold', y=0.98, color='white')
    plt.tight_layout()
    return fig

def plot_class_distribution(df):
    """繪製類別分布圖"""
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor('#0E1117')
    ax.set_facecolor('#0E1117')
    
    class_counts = df['label'].value_counts()
    colors = ['#21C354', '#FF4B4B']
    explode = (0.05, 0.05)
    
    wedges, texts, autotexts = ax.pie(
        class_counts.values, 
        labels=['✅ Legitimate (HAM)', '⚠️ Spam'],
        autopct='%1.1f%%',
        colors=colors,
        startangle=90,
        textprops={'fontsize': 14, 'weight': 'bold', 'color': 'white'},
        explode=explode,
        shadow=True
    )
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(16)
        autotext.set_weight('bold')
    
    ax.set_title('📊 Dataset Distribution', fontsize=16, fontweight='bold', pad=20, color='white')
    return fig

def plot_message_length_distribution(df):
    """繪製訊息長度分布"""
    df['message_length'] = df['text'].str.len()
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor('#0E1117')
    ax.set_facecolor('#1a1a2e')
    
    ham_lengths = df[df['label'] == 'ham']['message_length']
    spam_lengths = df[df['label'] == 'spam']['message_length']
    
    ax.hist(ham_lengths, bins=50, alpha=0.7, label='✅ HAM', color='#21C354', edgecolor='white', linewidth=1.2)
    ax.hist(spam_lengths, bins=50, alpha=0.7, label='⚠️ SPAM', color='#FF4B4B', edgecolor='white', linewidth=1.2)
    
    ax.set_xlabel('Message Length (characters)', fontsize=13, fontweight='bold', color='white')
    ax.set_ylabel('Frequency', fontsize=13, fontweight='bold', color='white')
    ax.set_title('📏 Message Length Distribution', fontsize=16, fontweight='bold', color='white', pad=15)
    ax.legend(fontsize=13, loc='upper right', framealpha=0.9)
    ax.grid(axis='y', alpha=0.3, color='white', linestyle='--')
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    return fig

# ==================== 主程式 ====================
# 標題區域
st.markdown('<h1 class="main-title">📧 Smart Spam Detector</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Powered by Machine Learning | Protect Your Inbox from Spam</p>', unsafe_allow_html=True)

# 載入資料
df = load_data()

if df is not None:
    # ==================== 側邊欄配置 ====================
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/spam.png", width=100)
        st.markdown("### ⚙️ Configuration Panel")
        st.markdown("---")
        
        # 模式選擇
        st.markdown("#### 🎯 Detection Mode")
        mode = st.radio(
            "Choose mode", 
            ["🔍 Single Message", "📦 Batch Processing"], 
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # 模型參數
        st.markdown("#### 🤖 Model Settings")
        model_choice = st.selectbox(
            "Algorithm", 
            ["Logistic Regression", "SVM"],
            help="Choose the machine learning algorithm"
        )
        confidence_threshold = st.slider(
            "Confidence Threshold", 
            0.0, 1.0, 0.5, 0.01,
            help="Minimum confidence required for prediction"
        )
        
        st.markdown("---")
        
        # 文字預處理選項
        st.markdown("#### 🔧 Preprocessing")
        with st.expander("Text Processing Options", expanded=False):
            opt_lowercase = st.checkbox("Convert to lowercase", value=True)
            opt_remove_punct = st.checkbox("Remove punctuation", value=True)
            opt_remove_stopwords = st.checkbox("Remove stopwords", value=False)
        
        st.markdown("---")
        
        # 顯示選項
        st.markdown("#### 👁️ Display Settings")
        with st.expander("Visualization Options", expanded=False):
            opt_show_preproc = st.checkbox("Show preprocessing", value=True)
            opt_show_probs = st.checkbox("Show probabilities", value=True)
            opt_show_cm = st.checkbox("Show confusion matrix", value=False)
            opt_show_roc = st.checkbox("Show ROC curve", value=False)
        
        st.markdown("---")
        
        # Data Overview 控制
        st.markdown("#### 📊 Data Settings")
        with st.expander("Dataset Options", expanded=False):
            show_raw_data = st.checkbox("Show raw data", value=False)
            n_samples = st.slider("Sample size", 5, 50, 10)
            n_tokens = st.slider("Top tokens", 10, 50, 20, 5)
        
        st.markdown("---")
        
        # 圖表控制
        st.markdown("#### 📈 Charts")
        show_class_dist = st.checkbox("Class distribution", value=True)
        show_length_dist = st.checkbox("Length distribution", value=True)
        show_token_chart = st.checkbox("Token frequency", value=True)
        
        st.markdown("---")
        st.markdown("#### ℹ️ About")
        st.info("Built with ❤️ using Streamlit\n\nVersion 2.0")
    
    # ==================== 主要統計概覽 ====================
    st.markdown("### 📊 Quick Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📨 Total Messages",
            value=f"{len(df):,}",
            delta="Dataset loaded"
        )
    
    with col2:
        ham_count = len(df[df['label'] == 'ham'])
        st.metric(
            label="✅ Legitimate",
            value=f"{ham_count:,}",
            delta=f"{ham_count/len(df)*100:.1f}%"
        )
    
    with col3:
        spam_count = len(df[df['label'] == 'spam'])
        st.metric(
            label="⚠️ Spam",
            value=f"{spam_count:,}",
            delta=f"{spam_count/len(df)*100:.1f}%"
        )
    
    with col4:
        avg_length = df['text'].str.len().mean()
        st.metric(
            label="📏 Avg Length",
            value=f"{avg_length:.0f}",
            delta="characters"
        )
    
    st.markdown("---")
    
    # ==================== 標籤頁區域 ====================
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔮 Spam Detection", 
        "📊 Data Analysis", 
        "📈 Visualizations", 
        "🤖 Model Info"
    ])
    
    # ==================== Tab 1: 預測功能 ====================
    with tab1:
        model, vectorizer, label_encoder = load_model_artifacts()
        
        if model is None:
            st.error("❌ Model not found. Please train the model first.")
            st.code("python train_baseline.py", language="bash")
        else:
            if "🔍 Single Message" in mode:
                st.markdown("### 🔍 Single Message Classification")
                st.markdown("Enter your message below to check if it's spam or legitimate.")
                
                # 輸入區域
                col1, col2 = st.columns([3, 1])
                with col1:
                    input_text = st.text_area(
                        "Message Content",
                        placeholder="Type or paste your message here...",
                        height=200,
                        label_visibility="collapsed"
                    )
                
                with col2:
                    st.markdown("#### Quick Examples")
                    if st.button("📱 Example 1", use_container_width=True):
                        input_text = "Congratulations! You've won a $1000 gift card. Click here to claim now!"
                    if st.button("📧 Example 2", use_container_width=True):
                        input_text = "Hi, are we still meeting for lunch tomorrow at 1pm?"
                    if st.button("💰 Example 3", use_container_width=True):
                        input_text = "URGENT! Your account will be closed. Call now to verify your information."
                
                if st.button("🚀 Analyze Message", type="primary", use_container_width=True):
                    if input_text.strip():
                        with st.spinner("🔄 Analyzing message..."):
                            try:
                                # 預處理
                                x_proc = preprocess_text(
                                    input_text, 
                                    opt_lowercase, 
                                    opt_remove_punct, 
                                    opt_remove_stopwords
                                )
                                
                                # 向量化
                                text_vector = vectorizer.transform([x_proc])
                                
                                # 預測
                                prediction = model.predict(text_vector)[0]
                                proba = model.predict_proba(text_vector)[0]
                                
                                # 解碼標籤
                                label = label_encoder.inverse_transform([prediction])[0]
                                confidence = max(proba) * 100
                                spam_prob = proba[1]
                                ham_prob = proba[0]
                                
                                st.markdown("---")
                                st.markdown("### 📋 Analysis Results")
                                
                                # 結果顯示
                                col1, col2, col3 = st.columns([2, 2, 1])
                                
                                with col1:
                                    if label == 'spam':
                                        st.markdown(f"""
                                        <div class="custom-card" style="background: linear-gradient(135deg, rgba(255, 75, 75, 0.2) 0%, rgba(255, 0, 0, 0.1) 100%); border-left: 5px solid #FF4B4B;">
                                            <h2 style="color: #FF4B4B; margin: 0;">⚠️ SPAM DETECTED</h2>
                                            <p style="font-size: 1.2rem; margin: 0.5rem 0;">This message appears to be spam</p>
                                            <p style="font-size: 2rem; font-weight: bold; margin: 0; color: #FF4B4B;">{confidence:.1f}% Confident</p>
                                        </div>
                                        """, unsafe_allow_html=True)
                                    else:
                                        st.markdown(f"""
                                        <div class="custom-card" style="background: linear-gradient(135deg, rgba(33, 195, 84, 0.2) 0%, rgba(0, 255, 0, 0.1) 100%); border-left: 5px solid #21C354;">
                                            <h2 style="color: #21C354; margin: 0;">✅ LEGITIMATE</h2>
                                            <p style="font-size: 1.2rem; margin: 0.5rem 0;">This message appears to be safe</p>
                                            <p style="font-size: 2rem; font-weight: bold; margin: 0; color: #21C354;">{confidence:.1f}% Confident</p>
                                        </div>
                                        """, unsafe_allow_html=True)
                                
                                with col2:
                                    # 信心度檢查
                                    if spam_prob >= confidence_threshold:
                                        st.warning(f"⚠️ Above threshold ({confidence_threshold:.0%})")
                                        st.markdown("**Recommendation:** Block or review")
                                    else:
                                        st.success(f"✅ Below threshold ({confidence_threshold:.0%})")
                                        st.markdown("**Recommendation:** Likely safe")
                                
                                with col3:
                                    # 風險等級
                                    if spam_prob > 0.8:
                                        risk = "🔴 High"
                                        risk_color = "#FF4B4B"
                                    elif spam_prob > 0.5:
                                        risk = "🟡 Medium"
                                        risk_color = "#FFA600"
                                    else:
                                        risk = "🟢 Low"
                                        risk_color = "#21C354"
                                    
                                    st.markdown(f"""
                                    <div style="text-align: center; padding: 1rem; background: rgba(255,255,255,0.05); border-radius: 10px;">
                                        <p style="margin: 0; font-size: 0.9rem; color: #a0a0a0;">Risk Level</p>
                                        <p style="margin: 0; font-size: 1.5rem; font-weight: bold; color: {risk_color};">{risk}</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                # 機率顯示
                                if opt_show_probs:
                                    st.markdown("---")
                                    st.markdown("#### 📊 Probability Breakdown")
                                    
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.markdown(f"""
                                        <div style="text-align: center; padding: 1.5rem; background: linear-gradient(135deg, rgba(33, 195, 84, 0.1) 0%, rgba(0, 255, 0, 0.05) 100%); border-radius: 10px; border: 2px solid rgba(33, 195, 84, 0.3);">
                                            <p style="margin: 0; font-size: 1.1rem; color: #21C354; font-weight: 600;">✅ Legitimate</p>
                                            <p style="margin: 0.5rem 0 0 0; font-size: 2.5rem; font-weight: bold; color: #21C354;">{ham_prob*100:.1f}%</p>
                                        </div>
                                        """, unsafe_allow_html=True)
                                    
                                    with col2:
                                        st.markdown(f"""
                                        <div style="text-align: center; padding: 1.5rem; background: linear-gradient(135deg, rgba(255, 75, 75, 0.1) 0%, rgba(255, 0, 0, 0.05) 100%); border-radius: 10px; border: 2px solid rgba(255, 75, 75, 0.3);">
                                            <p style="margin: 0; font-size: 1.1rem; color: #FF4B4B; font-weight: 600;">⚠️ Spam</p>
                                            <p style="margin: 0.5rem 0 0 0; font-size: 2.5rem; font-weight: bold; color: #FF4B4B;">{spam_prob*100:.1f}%</p>
                                        </div>
                                        """, unsafe_allow_html=True)
                                    
                                    # 進度條視覺化
                                    st.markdown("##### Confidence Visualization")
                                    st.progress(spam_prob, text=f"Spam Confidence: {spam_prob*100:.1f}%")
                                
                                # 預處理步驟
                                if opt_show_preproc:
                                    st.markdown("---")
                                    st.markdown("#### 🔍 Processing Details")
                                    
                                    with st.expander("View Preprocessing Steps", expanded=False):
                                        col1, col2 = st.columns(2)
                                        with col1:
                                            st.markdown("**Original Text:**")
                                            st.code(input_text, language="text")
                                        
                                        with col2:
                                            st.markdown("**Processed Text:**")
                                            st.code(x_proc, language="text")
                                        
                                        st.markdown("**Tokens Extracted:**")
                                        tokens = x_proc.split()
                                        st.write(f"Total tokens: {len(tokens)}")
                                        st.write(" • ".join(tokens[:20]) + ("..." if len(tokens) > 20 else ""))
                                        
                                        # Top TF-IDF features
                                        try:
                                            feature_names = vectorizer.get_feature_names_out()
                                            x_vec = text_vector.toarray()[0]
                                            top_idx = np.argsort(x_vec)[-10:][::-1]
                                            top_features = [(feature_names[i], float(x_vec[i])) for i in top_idx if x_vec[i] > 0]
                                            
                                            if top_features:
                                                st.markdown("**Top TF-IDF Features:**")
                                                features_df = pd.DataFrame(top_features, columns=['Feature', 'TF-IDF Score'])
                                                features_df['TF-IDF Score'] = features_df['TF-IDF Score'].round(4)
                                                st.dataframe(features_df, use_container_width=True, hide_index=True)
                                        except Exception:
                                            pass
                                
                            except Exception as e:
                                st.error(f"❌ Analysis failed: {e}")
                    else:
                        st.warning("⚠️ Please enter a message to analyze")
            
            else:  # Batch upload
                st.markdown("### 📦 Batch Processing")
                st.markdown("Upload a CSV file containing multiple messages for batch analysis.")
                
                uploaded_file = st.file_uploader(
                    "Choose a CSV file",
                    type=['csv', 'txt'],
                    help="File should contain messages in the first or second column"
                )
                
                if uploaded_file is not None:
                    try:
                        uploaded_df = pd.read_csv(uploaded_file, header=None)
                        if uploaded_df.shape[1] >= 2:
                            texts = uploaded_df.iloc[:, 1].astype(str).tolist()
                        else:
                            texts = uploaded_df.iloc[:, 0].astype(str).tolist()
                        
                        st.success(f"✅ Loaded {len(texts)} messages")
                        
                        with st.expander("📄 Preview Uploaded Data"):
                            st.dataframe(uploaded_df.head(10), use_container_width=True)
                        
                        if st.button("🚀 Analyze All Messages", type="primary", use_container_width=True):
                            with st.spinner("🔄 Processing batch..."):
                                # 預處理
                                texts_proc = [
                                    preprocess_text(t, opt_lowercase, opt_remove_punct, opt_remove_stopwords) 
                                    for t in texts
                                ]
                                
                                # 建立進度條
                                progress_bar = st.progress(0)
                                status_text = st.empty()
                                
                                # 向量化
                                progress_bar.progress(33)
                                status_text.text("Vectorizing messages...")
                                text_vectors = vectorizer.transform(texts_proc)
                                
                                # 預測
                                progress_bar.progress(66)
                                status_text.text("Making predictions...")
                                predictions = model.predict(text_vectors)
                                probas = model.predict_proba(text_vectors)
                                
                                # 解碼
                                progress_bar.progress(100)
                                status_text.text("Finalizing results...")
                                labels = label_encoder.inverse_transform(predictions)
                                confidences = np.max(probas, axis=1) * 100
                                spam_probs = probas[:, 1]
                                
                                # 建立結果 DataFrame
                                result_df = pd.DataFrame({
                                    'Message': texts,
                                    'Prediction': labels,
                                    'Spam Probability': spam_probs,
                                    'Confidence': confidences
                                })
                                
                                progress_bar.empty()
                                status_text.empty()
                                
                                st.success("✅ Batch analysis completed!")
                                
                                # 統計摘要
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("Total Analyzed", len(result_df))
                                with col2:
                                    spam_detected = len(result_df[result_df['Prediction'] == 'spam'])
                                    st.metric("Spam Detected", spam_detected)
                                with col3:
                                    ham_detected = len(result_df[result_df['Prediction'] == 'ham'])
                                    st.metric("Legitimate", ham_detected)
                                with col4:
                                    spam_rate = spam_detected / len(result_df) * 100
                                    st.metric("Spam Rate", f"{spam_rate:.1f}%")
                                
                                st.markdown("---")
                                st.markdown("#### 📊 Results Table")
                                st.dataframe(result_df, use_container_width=True)
                                
                                # 下載按鈕
                                csv = result_df.to_csv(index=False)
                                col1, col2, col3 = st.columns([1, 1, 2])
                                with col1:
                                    st.download_button(
                                        label="📥 Download Full Results",
                                        data=csv,
                                        file_name="spam_detection_results.csv",
                                        mime="text/csv",
                                        use_container_width=True
                                    )
                                with col2:
                                    spam_only = result_df[result_df['Prediction'] == 'spam'].to_csv(index=False)
                                    st.download_button(
                                        label="📥 Download Spam Only",
                                        data=spam_only,
                                        file_name="spam_messages.csv",
                                        mime="text/csv",
                                        use_container_width=True
                                    )
                    
                    except Exception as e:
                        st.error(f"❌ Error processing file: {e}")
    
    # ==================== Tab 2: 數據分析 ====================
    with tab2:
        st.markdown("### 📊 Dataset Analysis")
        
        if show_raw_data:
            st.markdown("#### 📄 Raw Data Sample")
            st.dataframe(df.head(n_samples), use_container_width=True)
            st.markdown("---")
        
        st.markdown("#### 📈 Detailed Statistics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="custom-card">
                <h4 style="color: #21C354;">✅ Legitimate Messages (HAM)</h4>
            </div>
            """, unsafe_allow_html=True)
            
            ham_df = df[df['label'] == 'ham']
            ham_lengths = ham_df['text'].str.len()
            
            stats_data = {
                'Metric': ['Count', 'Percentage', 'Avg Length', 'Min Length', 'Max Length', 'Std Dev'],
                'Value': [
                    f"{len(ham_df):,}",
                    f"{len(ham_df)/len(df)*100:.1f}%",
                    f"{ham_lengths.mean():.0f} chars",
                    f"{ham_lengths.min()} chars",
                    f"{ham_lengths.max()} chars",
                    f"{ham_lengths.std():.0f} chars"
                ]
            }
            st.table(pd.DataFrame(stats_data))
        
        with col2:
            st.markdown("""
            <div class="custom-card">
                <h4 style="color: #FF4B4B;">⚠️ Spam Messages</h4>
            </div>
            """, unsafe_allow_html=True)
            
            spam_df = df[df['label'] == 'spam']
            spam_lengths = spam_df['text'].str.len()
            
            stats_data = {
                'Metric': ['Count', 'Percentage', 'Avg Length', 'Min Length', 'Max Length', 'Std Dev'],
                'Value': [
                    f"{len(spam_df):,}",
                    f"{len(spam_df)/len(df)*100:.1f}%",
                    f"{spam_lengths.mean():.0f} chars",
                    f"{spam_lengths.min()} chars",
                    f"{spam_lengths.max()} chars",
                    f"{spam_lengths.std():.0f} chars"
                ]
            }
            st.table(pd.DataFrame(stats_data))
        
        st.markdown("---")
        
        # 隨機樣本展示
        st.markdown("#### 🎲 Random Samples")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Random HAM Message:**")
            random_ham = ham_df.sample(1)['text'].values[0]
            st.info(random_ham)
        
        with col2:
            st.markdown("**Random SPAM Message:**")
            random_spam = spam_df.sample(1)['text'].values[0]
            st.warning(random_spam)
    
    # ==================== Tab 3: 視覺化 ====================
    with tab3:
        st.markdown("### 📈 Data Visualizations")
        
        if show_token_chart:
            st.markdown("#### 📊 Most Frequent Words")
            with st.spinner("🎨 Generating word frequency chart..."):
                fig = plot_top_tokens(df, n_tokens)
                st.pyplot(fig, use_container_width=True)
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if show_class_dist:
                st.markdown("#### 🥧 Class Distribution")
                with st.spinner("🎨 Generating distribution chart..."):
                    fig = plot_class_distribution(df)
                    st.pyplot(fig, use_container_width=True)
        
        with col2:
            if show_length_dist:
                st.markdown("#### 📏 Message Length")
                with st.spinner("🎨 Generating length distribution..."):
                    fig = plot_message_length_distribution(df)
                    st.pyplot(fig, use_container_width=True)
        
        # 模型評估圖表
        st.markdown("---")
        st.markdown("### 🎯 Model Evaluation Metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if opt_show_cm:
                cm_path = ARTIFACTS_DIR / "confusion_matrix.png"
                if cm_path.exists():
                    st.markdown("#### 📊 Confusion Matrix")
                    st.image(str(cm_path), use_container_width=True)
                else:
                    st.info("ℹ️ Train the model to generate confusion matrix")
        
        with col2:
            if opt_show_roc:
                roc_path = ARTIFACTS_DIR / "roc_curve.png"
                if roc_path.exists():
                    st.markdown("#### 📈 ROC Curve")
                    st.image(str(roc_path), use_container_width=True)
                else:
                    st.info("ℹ️ Train the model to generate ROC curve")
    
    # ==================== Tab 4: 模型資訊 ====================
    with tab4:
        st.markdown("### 🤖 Model Information")
        
        model, vectorizer, label_encoder = load_model_artifacts()
        
        if model is not None:
            st.success("✅ Model loaded successfully")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🔧 Model Configuration")
                model_info = {
                    "Model Type": type(model).__name__,
                    "Vectorizer": type(vectorizer).__name__,
                    "Max Features": vectorizer.max_features if hasattr(vectorizer, 'max_features') else "N/A",
                    "Vocabulary Size": len(vectorizer.vocabulary_) if hasattr(vectorizer, 'vocabulary_') else "N/A"
                }
                
                for key, value in model_info.items():
                    st.markdown(f"**{key}:** `{value}`")
            
            with col2:
                metrics = load_metrics()
                if metrics is not None:
                    st.markdown("#### 📊 Performance Metrics")
                    
                    metrics_df = pd.DataFrame({
                        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC'],
                        'Score': [
                            f"{metrics.get('accuracy', 0):.4f}",
                            f"{metrics.get('precision', 0):.4f}",
                            f"{metrics.get('recall', 0):.4f}",
                            f"{metrics.get('f1', 0):.4f}",
                            f"{metrics.get('roc_auc', 0):.4f}"
                        ]
                    })
                    st.dataframe(metrics_df, use_container_width=True, hide_index=True)
            
            st.markdown("---")
            st.markdown("#### 📋 Model Details")
            
            with st.expander("View Full Model Configuration"):
                st.json(model_info)
                
                if hasattr(model, 'get_params'):
                    st.markdown("**Model Parameters:**")
                    st.json(model.get_params())
        
        else:
            st.error("❌ No model found")
            st.markdown("""
            ### 🚀 Train Your Model
            
            To use the spam detector, you need to train a model first:
            
            ```bash
            python train_baseline.py
            ```
            
            This will:
            - Download and process the dataset
            - Train the classification model
            - Save model artifacts
            - Generate evaluation metrics
            """)

else:
    st.error("❌ Failed to load dataset. Please check your internet connection.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; color: #a0a0a0;">
    <p>Made with ❤️ using Streamlit | © 2025 Smart Spam Detector</p>
    <p style="font-size: 0.9rem;">Protecting your inbox with the power of AI</p>
</div>
""", unsafe_allow_html=True)
