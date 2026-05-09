"""
HSRIS - Hybrid Semantic Retrieval and Intelligence System
Streamlit App - Interactive Demo
Authors: Ali Naqi (23F-3052) & Muhammad Aamir (23F-3073) | SE-6B
"""
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import re
import pickle
import os
from collections import Counter

# ======================================================================
# PAGE CONFIG
# ======================================================================
st.set_page_config(page_title="HSRIS | Hybrid Search", page_icon="H", layout="wide")

# ======================================================================
# INJECT CUSTOM CSS (Advanced Styling)
# ======================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* ── Global Reset ── */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .stApp {
        background: linear-gradient(135deg, #0a0a1a 0%, #0d1b2a 30%, #1b263b 60%, #0d1b2a 100%);
    }

    /* ── Hide default streamlit branding ── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* ── Animated Header Banner ── */
    .hero-banner {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        background-size: 200% 200%;
        animation: gradientFlow 6s ease infinite;
        border-radius: 24px;
        padding: 48px 40px 40px;
        margin-bottom: 32px;
        position: relative;
        overflow: hidden;
        box-shadow: 0 20px 60px rgba(102, 126, 234, 0.3), 0 0 0 1px rgba(255,255,255,0.08);
    }
    .hero-banner::before {
        content: '';
        position: absolute; top: 0; left: 0; right: 0; bottom: 0;
        background: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='0.05'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
        opacity: 0.4;
    }
    .hero-banner h1 {
        font-size: 2.4rem;
        font-weight: 900;
        color: #fff;
        letter-spacing: -0.02em;
        line-height: 1.1;
        margin: 0 0 8px 0;
        position: relative;
        text-shadow: 0 2px 20px rgba(0,0,0,0.2);
    }
    .hero-banner .subtitle {
        color: rgba(255,255,255,0.85);
        font-size: 1.05rem;
        font-weight: 400;
        margin: 0;
        position: relative;
    }
    .hero-banner .badge-row {
        display: flex;
        gap: 10px;
        margin-top: 20px;
        position: relative;
        flex-wrap: wrap;
    }
    .hero-badge {
        background: rgba(255,255,255,0.15);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255,255,255,0.2);
        border-radius: 100px;
        padding: 6px 16px;
        font-size: 0.8rem;
        font-weight: 600;
        color: #fff;
        letter-spacing: 0.02em;
    }

    @keyframes gradientFlow {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* ── Glass Cards ── */
    .glass-card {
        background: rgba(255, 255, 255, 0.04);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 28px 32px;
        margin-bottom: 20px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .glass-card:hover {
        border-color: rgba(102, 126, 234, 0.3);
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.15);
        transform: translateY(-2px);
    }
    .glass-card h3 {
        font-size: 1.1rem;
        font-weight: 700;
        color: #e0e6ff;
        margin: 0 0 4px 0;
        letter-spacing: -0.01em;
    }
    .glass-card .card-subtitle {
        font-size: 0.82rem;
        color: rgba(255,255,255,0.45);
        margin-bottom: 16px;
    }

    /* ── Result Cards ── */
    .result-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02));
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 24px 28px;
        margin-bottom: 16px;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    .result-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0;
        width: 4px; height: 100%;
        border-radius: 4px 0 0 4px;
    }
    .result-card.rank-1::before { background: linear-gradient(to bottom, #ffd700, #ffaa00); }
    .result-card.rank-2::before { background: linear-gradient(to bottom, #c0c0c0, #a0a0a0); }
    .result-card.rank-3::before { background: linear-gradient(to bottom, #cd7f32, #a0622e); }
    .result-card:hover {
        border-color: rgba(102, 126, 234, 0.25);
        box-shadow: 0 4px 24px rgba(0,0,0,0.2);
        transform: translateY(-1px);
    }
    .result-rank {
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 8px;
    }
    .rank-1 .result-rank { color: #ffd700; }
    .rank-2 .result-rank { color: #c0c0c0; }
    .rank-3 .result-rank { color: #cd7f32; }
    .result-score {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.6rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 12px;
    }
    .result-type {
        display: inline-block;
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 4px 14px;
        border-radius: 100px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-bottom: 12px;
    }
    .meta-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 8px;
        margin-bottom: 14px;
    }
    .meta-item {
        background: rgba(255,255,255,0.04);
        border-radius: 10px;
        padding: 10px 14px;
        text-align: center;
    }
    .meta-label {
        font-size: 0.68rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: rgba(255,255,255,0.4);
        margin-bottom: 2px;
    }
    .meta-value {
        font-size: 0.88rem;
        font-weight: 600;
        color: #e0e6ff;
    }
    .result-desc {
        font-size: 0.88rem;
        color: rgba(255,255,255,0.65);
        line-height: 1.6;
        border-top: 1px solid rgba(255,255,255,0.06);
        padding-top: 14px;
    }

    /* ── Prediction Banner ── */
    .prediction-banner {
        background: linear-gradient(135deg, rgba(102,126,234,0.15), rgba(118,75,162,0.15));
        border: 1px solid rgba(102,126,234,0.25);
        border-radius: 16px;
        padding: 24px 28px;
        margin-bottom: 24px;
        text-align: center;
        animation: fadeSlideIn 0.5s ease-out;
    }
    .prediction-label {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: rgba(255,255,255,0.5);
        margin-bottom: 6px;
    }
    .prediction-value {
        font-size: 1.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea, #764ba2, #f093fb);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    @keyframes fadeSlideIn {
        from { opacity: 0; transform: translateY(12px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* ── Alpha Bar ── */
    .alpha-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: rgba(255,255,255,0.04);
        border-radius: 12px;
        padding: 14px 20px;
        margin-top: 16px;
    }
    .alpha-label {
        font-size: 0.82rem;
        color: rgba(255,255,255,0.5);
    }
    .alpha-value {
        font-family: 'JetBrains Mono', monospace;
        font-weight: 700;
        font-size: 1.1rem;
        color: #e0e6ff;
    }

    /* ── Footer ── */
    .app-footer {
        margin-top: 60px;
        padding: 30px 0;
        border-top: 1px solid rgba(255,255,255,0.06);
        text-align: center;
    }
    .app-footer p {
        color: rgba(255,255,255,0.3);
        font-size: 0.82rem;
        margin: 4px 0;
    }
    .app-footer .footer-brand {
        font-weight: 700;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* ── Stats Pill Row ── */
    .stats-row {
        display: flex;
        gap: 12px;
        margin-bottom: 28px;
        flex-wrap: wrap;
    }
    .stat-pill {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 100px;
        padding: 8px 20px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .stat-pill .stat-num {
        font-family: 'JetBrains Mono', monospace;
        font-weight: 700;
        font-size: 0.95rem;
        color: #e0e6ff;
    }
    .stat-pill .stat-label {
        font-size: 0.78rem;
        color: rgba(255,255,255,0.4);
    }

    /* ── Streamlit Widget Tweaks ── */
    div[data-baseweb="textarea"] {
        background-color: #1b263b !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 14px !important;
    }
    div[data-baseweb="textarea"] > div {
        background-color: transparent !important;
    }
    .stTextArea textarea {
        background-color: transparent !important;
        color: #ffffff !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.95rem !important;
        padding: 16px !important;
        -webkit-text-fill-color: #ffffff !important;
    }
    .stTextArea textarea::placeholder {
        color: rgba(255,255,255,0.5) !important;
        -webkit-text-fill-color: rgba(255,255,255,0.5) !important;
    }
    .stTextArea textarea:focus {
        border-color: transparent !important;
        box-shadow: none !important;
    }
    div[data-baseweb="textarea"]:focus-within {
        border-color: rgba(102,126,234,0.5) !important;
        box-shadow: 0 0 0 2px rgba(102,126,234,0.25) !important;
    }
    .stSlider > div > div {
        background: rgba(255,255,255,0.06) !important;
    }
    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 14px !important;
        padding: 14px 28px !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        letter-spacing: 0.02em !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.3) !important;
    }
    div.stButton > button[kind="primary"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 30px rgba(102, 126, 234, 0.45) !important;
    }

    /* ── Section Headers ── */
    .section-header {
        font-size: 1.1rem;
        font-weight: 700;
        color: #e0e6ff;
        margin-bottom: 16px;
        padding-bottom: 10px;
        border-bottom: 1px solid rgba(255,255,255,0.06);
    }

    /* ── Formula Display ── */
    .formula-box {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 16px 20px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.9rem;
        color: #c5ceff;
        text-align: center;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)


# ======================================================================
# LOAD DATA
# ======================================================================
@st.cache_resource
def load_data():
    pkl_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'hsris_app_data.pkl')
    if not os.path.exists(pkl_path):
        st.error(f"Missing: {pkl_path}")
        st.stop()
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    emb = nn.Embedding(len(data['glove_tokens']), data['EMB_DIM'], padding_idx=0)
    emb.weight.data.copy_(torch.tensor(data['emb_matrix']))
    emb.weight.requires_grad = False
    data['embedding_layer'] = emb
    return data

data = load_data()
df = data['df']
vocab = data['vocab']
vocab2idx = data['vocab2idx']
idf = data['idf']
tfidf_dense = data['tfidf_dense']
glove_matrix = data['glove_matrix']
embedding_layer = data['embedding_layer']
glove_tok2idx = data['glove_tok2idx']
TOP_K = data['TOP_K']
EMB_DIM = data['EMB_DIM']
device = torch.device('cpu')


# ======================================================================
# NLP FUNCTIONS
# ======================================================================
def tokenize(text):
    return re.findall(r'[a-z]+', str(text).lower())

def get_ngrams(tokens, n):
    return ['_'.join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

def tokenize_with_ngrams(text):
    tokens = tokenize(text)
    all_t = list(tokens)
    all_t.extend(get_ngrams(tokens, 2))
    all_t.extend(get_ngrams(tokens, 3))
    return all_t

def get_glove_indices(tokens):
    return [glove_tok2idx.get(t, 0) for t in tokens]

def get_sentence_vector(tokens, tfidf_row_dict):
    total_w = 0.0
    ws = torch.zeros(EMB_DIM)
    for tok, idx in zip(tokens, get_glove_indices(tokens)):
        if idx == 0:
            continue
        w = tfidf_row_dict.get(tok, 1e-5)
        ws += w * embedding_layer(torch.tensor([idx]))[0]
        total_w += w
    return ws / total_w if total_w > 0 else ws

def hybrid_search(query, alpha=0.4, top_k=5):
    q_toks = tokenize_with_ngrams(query)
    qt = Counter(q_toks)
    total = sum(qt.values()) or 1
    qv = torch.zeros(TOP_K)
    for tok, cnt in qt.items():
        if tok in vocab2idx:
            qv[vocab2idx[tok]] = (cnt / total) * idf[vocab2idx[tok]]
    qv = F.normalize(qv.unsqueeze(0), dim=1)

    q_uni = tokenize(query)
    rd = {vocab[j]: qv[0, j].item() for j in qv[0].nonzero(as_tuple=True)[0] if j.item() < len(vocab)}
    gv = get_sentence_vector(q_uni, rd)
    gv = F.normalize(gv.unsqueeze(0), dim=1)

    ts = torch.mm(qv, tfidf_dense.T).squeeze(0)
    gs = torch.mm(gv, glove_matrix.T).squeeze(0)
    fs = alpha * ts + (1 - alpha) * gs
    top_i = fs.topk(top_k).indices.tolist()

    res = df.iloc[top_i][['Ticket Description', 'Ticket Subject',
                           'Ticket Type', 'Ticket Priority', 'Ticket Channel']].copy()
    res['Score'] = [fs[i].item() for i in top_i]
    return res, df.iloc[top_i]['Ticket Type'].mode()[0]


# ======================================================================
# HERO BANNER
# ======================================================================
st.markdown("""
<div class="hero-banner">
    <h1>HSRIS</h1>
    <p class="subtitle">Hybrid Semantic Retrieval and Intelligence System</p>
    <div class="badge-row">
        <span class="hero-badge">PyTorch</span>
        <span class="hero-badge">GloVe 300D</span>
        <span class="hero-badge">TF-IDF</span>
        <span class="hero-badge">Dual GPU</span>
        <span class="hero-badge">Ali Naqi (23F-3052)</span>
        <span class="hero-badge">Muhammad Aamir (23F-3073)</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Stats Pills ──
n_tickets = len(df)
n_vocab = TOP_K
n_emb = EMB_DIM
st.markdown(f"""
<div class="stats-row">
    <div class="stat-pill"><span class="stat-num">{n_tickets:,}</span><span class="stat-label">tickets indexed</span></div>
    <div class="stat-pill"><span class="stat-num">{n_vocab:,}</span><span class="stat-label">vocabulary tokens</span></div>
    <div class="stat-pill"><span class="stat-num">{n_emb}D</span><span class="stat-label">GloVe embeddings</span></div>
    <div class="stat-pill"><span class="stat-num">SE-6B</span><span class="stat-label">batch</span></div>
</div>
""", unsafe_allow_html=True)

# ── Formula ──
st.markdown("""
<div class="formula-box">
    FinalScore = &alpha; &times; CosineSim(TF-IDF) + (1 - &alpha;) &times; CosineSim(GloVe)
</div>
""", unsafe_allow_html=True)


# ======================================================================
# SEARCH INTERFACE
# ======================================================================
col1, col2 = st.columns([2.5, 1])

with col1:
    st.markdown('<div class="glass-card"><h3>Enter Ticket Description</h3><div class="card-subtitle">Type or paste a customer support ticket to find similar past resolutions</div></div>', unsafe_allow_html=True)
    query = st.text_area("Query", height=130, placeholder="e.g. I cannot login to my account, password reset not working...", label_visibility="collapsed")

with col2:
    st.markdown('<div class="glass-card"><h3>Search Parameters</h3><div class="card-subtitle">Adjust the hybrid weighting</div></div>', unsafe_allow_html=True)
    alpha = st.slider("Alpha", 0.0, 1.0, 0.4, 0.05, label_visibility="collapsed")
    tfidf_pct = str(int(alpha * 100)) + "%"
    glove_pct = str(int((1 - alpha) * 100)) + "%"
    st.markdown(f"""
    <div class="alpha-bar">
        <div>
            <div class="alpha-label">TF-IDF (Keyword)</div>
            <div class="alpha-value">{tfidf_pct}</div>
        </div>
        <div>
            <div class="alpha-label">GloVe (Semantic)</div>
            <div class="alpha-value">{glove_pct}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

search_clicked = st.button("Search", type="primary", use_container_width=True)


# ======================================================================
# RESULTS
# ======================================================================
if search_clicked and query.strip():
    results, predicted_type = hybrid_search(query, alpha=alpha, top_k=3)

    # Prediction Banner
    st.markdown(f"""
    <div class="prediction-banner">
        <div class="prediction-label">Predicted Ticket Type</div>
        <div class="prediction-value">{predicted_type}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Top 3 Similar Past Resolutions</div>', unsafe_allow_html=True)

    rank_labels = ["1st Match", "2nd Match", "3rd Match"]
    for i, (_, row) in enumerate(results.iterrows()):
        rank_class = f"rank-{i+1}"
        score_val = f"{row['Score']:.4f}"
        subj_short = str(row['Ticket Subject'])[:30]
        st.markdown(f"""
        <div class="result-card {rank_class}">
            <div class="result-rank">{rank_labels[i]}</div>
            <div class="result-score">{score_val}</div>
            <span class="result-type">{row['Ticket Type']}</span>
            <div class="meta-grid">
                <div class="meta-item">
                    <div class="meta-label">Priority</div>
                    <div class="meta-value">{row['Ticket Priority']}</div>
                </div>
                <div class="meta-item">
                    <div class="meta-label">Channel</div>
                    <div class="meta-value">{row['Ticket Channel']}</div>
                </div>
                <div class="meta-item">
                    <div class="meta-label">Subject</div>
                    <div class="meta-value">{subj_short}</div>
                </div>
            </div>
            <div class="result-desc">{row['Ticket Description']}</div>
        </div>
        """, unsafe_allow_html=True)


# ======================================================================
# FOOTER
# ======================================================================
st.markdown("""
<div class="app-footer">
    <p><span class="footer-brand">HSRIS</span> &mdash; Hybrid Semantic Retrieval and Intelligence System</p>
    <p>Assignment 3 | Data Science for Software Engineering | SE-6B</p>
    <p>Ali Naqi (23F-3052) &bull; Muhammad Aamir (23F-3073)</p>
    <p>Built with PyTorch, GloVe, and Streamlit</p>
</div>
""", unsafe_allow_html=True)
