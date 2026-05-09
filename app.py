import streamlit as st
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datetime import datetime
import streamlit.components.v1 as components

MODEL_NAME = "Neelam0404/distilbert-misinfo-model"

st.set_page_config(
    page_title="Misinformation Detector",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

if "theme" not in st.session_state:
    st.session_state.theme = "dark"
if "history" not in st.session_state:
    st.session_state.history = []
if "text" not in st.session_state:
    st.session_state.text = ""
if "last_result" not in st.session_state:
    st.session_state.last_result = None


def inject_css(theme: str):
    dark = theme == "dark"

    if dark:
        bg = "#0b0f1a"
        panel = "#0f172a"
        panel2 = "#0b1220"
        text = "#e5e7eb"
        muted = "#a3aab8"
        border = "rgba(255,255,255,0.12)"
        shadow = "rgba(0,0,0,0.45)"
        chip_bg = "rgba(255,255,255,0.06)"
        accent = "#ff4fd8"
        accent2 = "#ff7ae6"
        accent_soft = "rgba(255,79,216,0.18)"
    else:
        bg = "#fff7fd"
        panel = "#ffffff"
        panel2 = "#ffffff"
        text = "#121826"
        muted = "#5b6472"
        border = "rgba(16,24,40,0.10)"
        shadow = "rgba(16,24,40,0.08)"
        chip_bg = "rgba(16,24,40,0.03)"
        accent = "#ff2fbf"
        accent2 = "#ff64d6"
        accent_soft = "rgba(255,47,191,0.14)"

    good = "rgba(16,185,129,0.16)"
    bad = "rgba(239,68,68,0.16)"
    warn = "rgba(245,158,11,0.18)"

    st.markdown(
        f"""
        <style>
        .stApp {{
            background: radial-gradient(1200px 700px at 12% 5%, {accent_soft} 0%, transparent 55%),
                        radial-gradient(900px 650px at 88% 18%, rgba(99,102,241,0.12) 0%, transparent 55%),
                        radial-gradient(900px 650px at 50% 100%, rgba(16,185,129,0.10) 0%, transparent 60%),
                        {bg} !important;
            color: {text} !important;
        }}
        div[data-testid="stAppViewContainer"] {{
            background: transparent !important;
        }}
        header, footer, section[data-testid="stSidebar"] {{
            background: transparent !important;
        }}
        h1, h2, h3, h4, h5, h6, p, label, div {{
            color: {text} !important;
        }}
        .muted {{
            color: {muted} !important;
        }}
        textarea, input {{
            background-color: {panel} !important;
            color: {text} !important;
            border: 1px solid {border} !important;
            border-radius: 14px !important;
        }}
        .stButton>button {{
            border-radius: 14px !important;
            border: 1px solid {border} !important;
            padding: 0.6rem 0.9rem !important;
            font-weight: 700 !important;
        }}
        .primary-btn .stButton>button {{
            background: linear-gradient(135deg, {accent}, {accent2}) !important;
            color: white !important;
            border: none !important;
        }}
        .card {{
            border-radius: 18px;
            padding: 18px 18px;
            border: 1px solid {border};
            background: rgba(255,255,255,0.02);
            box-shadow: 0 10px 30px {shadow};
        }}
        .panel {{
            border-radius: 18px;
            padding: 18px 18px;
            border: 1px solid {border};
            background: {panel2};
            box-shadow: 0 10px 30px {shadow};
        }}
        .hero {{
            border-radius: 22px;
            padding: 22px 22px;
            border: 1px solid {border};
            background: linear-gradient(135deg, rgba(255,255,255,0.04), rgba(255,255,255,0.01));
            box-shadow: 0 12px 34px {shadow};
            margin-bottom: 14px;
        }}
        .hero-title {{
            font-size: 1.65rem;
            font-weight: 850;
            line-height: 1.2;
        }}
        .hero-sub {{
            margin-top: 6px;
            font-size: 1.0rem;
            color: {muted} !important;
        }}
        .badge {{
            display: inline-block;
            padding: 7px 12px;
            border-radius: 999px;
            font-size: 0.86rem;
            font-weight: 800;
            border: 1px solid {border};
            background: {chip_bg};
            margin-left: 10px;
        }}
        .chip {{
            display: inline-block;
            padding: 6px 10px;
            border-radius: 999px;
            border: 1px solid {border};
            background: {chip_bg};
            font-size: 0.85rem;
            margin-right: 8px;
            margin-top: 8px;
        }}
        .result-good {{ background: {good}; }}
        .result-bad {{ background: {bad}; }}
        .result-warn {{ background: {warn}; }}
        .token {{
            display:inline-block;
            padding: 5px 8px;
            margin: 4px 6px 0 0;
            border-radius: 12px;
            border: 1px solid {border};
            font-size: 0.92rem;
            background: rgba(255,79,216,0.08);
        }}
        .block-container {{
            padding-top: 1.0rem !important;
            padding-bottom: 2.2rem !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def copy_to_clipboard(text_to_copy: str):
    safe = text_to_copy.replace("\\", "\\\\").replace("`", "\\`")
    components.html(
        f"""
        <button id="copyBtn" style="
            width:100%;
            padding:0.65rem 0.85rem;
            border-radius:14px;
            border:1px solid rgba(255,255,255,0.18);
            cursor:pointer;
            font-weight:800;
            background: linear-gradient(135deg, #ff4fd8, #ff7ae6);
            color:white;
        ">📋 Copy to clipboard</button>

        <script>
        const btn = document.getElementById("copyBtn");
        btn.onclick = async () => {{
            try {{
                await navigator.clipboard.writeText(`{safe}`);
                btn.innerText = "✅ Copied!";
                setTimeout(() => btn.innerText = "📋 Copy to clipboard", 1400);
            }} catch (e) {{
                btn.innerText = "❌ Copy failed";
                setTimeout(() => btn.innerText = "📋 Copy to clipboard", 1400);
            }}
        }};
        </script>
        """,
        height=64,
    )


inject_css(st.session_state.theme)


@st.cache_resource
def load_model():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    mdl = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    mdl.eval()
    return tok, mdl


tokenizer, model = load_model()

LABELS = {0: "Real ✅", 1: "Fake ⚠️"}


def predict(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
    pred = int(np.argmax(probs))
    conf = float(probs[pred])
    return pred, conf, probs


def token_importance(text: str, target_label: int, max_len: int = 128, top_k: int = 18):
    model.eval()
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len, add_special_tokens=True)
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    emb_layer = model.get_input_embeddings()
    embeds = emb_layer(input_ids)
    embeds.requires_grad_(True)
    embeds.retain_grad()

    outputs = model(inputs_embeds=embeds, attention_mask=attention_mask)
    score = outputs.logits[0, target_label]

    grads = torch.autograd.grad(outputs=score, inputs=embeds, retain_graph=False, create_graph=False)[0]
    grads = grads[0]
    embs = embeds.detach()[0]

    token_scores = torch.sum(torch.abs(grads * embs), dim=1).cpu().numpy()
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu().tolist())

    cleaned = []
    for t, s, m in zip(tokens, token_scores, attention_mask[0].cpu().tolist()):
        if m == 0:
            continue
        if t in ["[CLS]", "[SEP]"]:
            continue
        cleaned.append((t, float(s)))

    if not cleaned:
        return []

    scores = np.array([s for _, s in cleaned], dtype=float)
    scores = scores / (scores.max() + 1e-9)
    cleaned_norm = [(t, float(s)) for (t, _), s in zip(cleaned, scores)]
    cleaned_norm.sort(key=lambda x: x[1], reverse=True)
    return cleaned_norm[:top_k]


def render_tokens(tokens_with_scores):
    if not tokens_with_scores:
        st.info("No explanation tokens found. Try a longer text.")
        return

    html = ""
    for tok, score in tokens_with_scores:
        display_tok = tok.replace("##", "")
        alpha = 0.10 + 0.55 * score
        html += f'<span class="token" style="background: rgba(255,79,216,{alpha});">{display_tok}</span>'

    st.markdown(html, unsafe_allow_html=True)


top_l, top_r = st.columns([3.8, 1.2], vertical_alignment="center")

with top_l:
    st.markdown(
        """
        <div class="hero">
          <div class="hero-title">🧠 Misinformation Detector <span class="badge">DistilBERT • Fine-tuned</span></div>
          <div class="hero-sub">Paste a claim or short article — get prediction, confidence, and lightweight explanations in a clean dashboard.</div>
          <div style="margin-top:10px;">
            <span class="chip">⚡ Fast inference</span>
            <span class="chip">🎀 Pink UI</span>
            <span class="chip">📊 Confidence meter</span>
            <span class="chip">🧩 Token insights</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with top_r:
    st.markdown('<div class="panel">', unsafe_allow_html=True)

    theme_choice = st.radio(
        "Theme",
        ["dark", "light"],
        index=0 if st.session_state.theme == "dark" else 1,
        horizontal=True,
        label_visibility="collapsed",
    )

    if theme_choice != st.session_state.theme:
        st.session_state.theme = theme_choice
        st.rerun()

    if st.button("🧹 Reset app state", use_container_width=True):
        st.session_state.clear()
        st.rerun()

    st.markdown(
        f'<div class="muted" style="margin-top:8px;font-size:0.9rem;">Model: <b>{MODEL_NAME}</b><br/>Max length: <b>256</b></div>',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


total = len(st.session_state.history)
fake_n = sum(1 for r in st.session_state.history if r.get("label") == 1)
real_n = sum(1 for r in st.session_state.history if r.get("label") == 0)
avg_conf = float(np.mean([r.get("confidence", 0.0) for r in st.session_state.history])) if total else 0.0

k1, k2, k3, k4 = st.columns(4)
k1.metric("Analyses", total)
k2.metric("Fake detected", fake_n)
k3.metric("Real detected", real_n)
k4.metric("Avg confidence", f"{avg_conf*100:.1f}%" if total else "—")

st.markdown("")

left, right = st.columns([1.25, 1], gap="large")

with left:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("📝 Analyze input")

    mode = st.tabs(["Text", "URL (coming soon)"])

    with mode[0]:
        with st.expander("✨ Try sample inputs"):
            a, b, c = st.columns(3)

            if a.button("Sample 1", use_container_width=True):
                st.session_state.text = "Breaking: Government secretly approved a law to ban all cash transactions tomorrow."
                st.rerun()

            if b.button("Sample 2", use_container_width=True):
                st.session_state.text = "Reuters reports the central bank kept interest rates unchanged after the policy meeting."
                st.rerun()

            if c.button("Sample 3", use_container_width=True):
                st.session_state.text = "Scientists confirm a new planet will collide with Earth next week, experts say."
                st.rerun()

        st.text_area(
            "Paste claim / headline / paragraph",
            key="text",
            height=190,
            placeholder="Example: A new study proves drinking only coffee cures all diseases overnight.",
        )

        b1, b2, b3 = st.columns([1.2, 1, 1])

        with b1:
            st.markdown('<div class="primary-btn">', unsafe_allow_html=True)
            predict_clicked = st.button("🔍 Analyze", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with b2:
            copy_clicked = st.button("📋 Copy result", use_container_width=True)

        with b3:
            download_clicked = st.button("⬇️ Export CSV", use_container_width=True)

        st.markdown(
            '<div class="muted" style="font-size:0.9rem;margin-top:10px;">Tip: Longer text usually gives more stable predictions and better token insights.</div>',
            unsafe_allow_html=True,
        )

    with mode[1]:
        st.info("URL analysis is a future upgrade. For now, paste the content as text.")

    st.markdown("</div>", unsafe_allow_html=True)


with right:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("📊 Result")

    last = st.session_state.last_result

    if last is None:
        st.markdown(
            """
            <div class="card">
              <div style="font-size:1.05rem;font-weight:800;">No analysis yet</div>
              <div class="muted" style="margin-top:6px;">Run an analysis to see prediction, confidence meter, probabilities, and token insights.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        is_fake = last["label"] == 1
        conf = float(last["confidence"])
        badge = "High confidence" if conf >= 0.85 else "Medium confidence" if conf >= 0.65 else "Low confidence"
        card_class = "result-bad" if is_fake else "result-good"

        st.markdown(
            f"""
            <div class="card {card_class}">
              <div style="font-size:1.15rem;font-weight:900;">
                Prediction: {last["label_name"]}
                <span class="badge">{badge}</span>
              </div>
              <div style="margin-top:6px;">Confidence: <b>{conf*100:.2f}%</b></div>
              <div class="muted" style="margin-top:6px;font-size:0.9rem;">
                Confidence reflects model certainty based on learned patterns, not live fact-checking.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("**Confidence meter**")
        st.progress(min(max(conf, 0.0), 1.0))

        st.markdown("**Probabilities**")
        prob_df = pd.DataFrame(
            {
                "Class": ["Real ✅", "Fake ⚠️"],
                "Probability": [float(last["prob_real"]), float(last["prob_fake"])],
                "Percent": [f"{last['prob_real']*100:.2f}%", f"{last['prob_fake']*100:.2f}%"],
            }
        )
        st.table(prob_df)

        st.markdown("**Quick interpretation**")
        if is_fake:
            st.warning("⚠️ The model thinks this resembles misinformation-style writing.")
        else:
            st.success("✅ The model thinks this resembles real/news-style writing patterns.")

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("")

if "predict_clicked" in locals() and predict_clicked:
    if not st.session_state.text.strip():
        st.warning("Please paste some text first.")
    else:
        pred, conf, probs = predict(st.session_state.text)

        st.session_state.last_result = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "text": st.session_state.text,
            "label": int(pred),
            "label_name": LABELS[pred],
            "confidence": float(conf),
            "prob_real": float(probs[0]),
            "prob_fake": float(probs[1]),
        }

        st.session_state.history.append(st.session_state.last_result)
        st.rerun()


last = st.session_state.last_result

if "copy_clicked" in locals() and copy_clicked:
    if not last:
        st.warning("Run an analysis first, then copy the result.")
    else:
        copy_text = (
            f"Prediction: {last['label_name']}\n"
            f"Confidence: {last['confidence']*100:.2f}%\n"
            f"Probabilities -> Real: {last['prob_real']:.4f}, Fake: {last['prob_fake']:.4f}\n"
            f"Timestamp: {last['timestamp']}\n"
            f"Text: {last['text']}"
        )

        st.info("Click to copy 👇")
        copy_to_clipboard(copy_text)
        st.text_area("Preview", value=copy_text, height=140)


if "download_clicked" in locals() and download_clicked:
    if not st.session_state.history:
        st.warning("No predictions yet. Run an analysis first.")
    else:
        df_hist = pd.DataFrame(st.session_state.history)
        csv_bytes = df_hist.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="✅ Download predictions CSV",
            data=csv_bytes,
            file_name="misinfo_predictions.csv",
            mime="text/csv",
            use_container_width=True,
        )


st.markdown("")

exp_col, hist_col = st.columns([1.25, 1], gap="large")

with exp_col:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("🧩 Token insights")

    if not last:
        st.info("Analyze a text to see token-level importance highlights.")
    else:
        with st.spinner("Computing token importance..."):
            toks = token_importance(last["text"], target_label=last["label"], max_len=128, top_k=18)

        render_tokens(toks)

        st.markdown(
            '<div class="muted" style="margin-top:10px;font-size:0.9rem;">Token insights are approximate gradient-based explanations.</div>',
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)


with hist_col:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("📜 History")

    if not st.session_state.history:
        st.info("No history yet. Your analyses will appear here.")
    else:
        df = pd.DataFrame(st.session_state.history).copy()
        df["confidence_%"] = (df["confidence"] * 100).round(2)
        df["snippet"] = df["text"].astype(str).str.replace("\n", " ").str.slice(0, 60) + "…"
        view_df = df[["timestamp", "label_name", "confidence_%", "snippet"]].iloc[::-1].reset_index(drop=True)

        st.dataframe(view_df, use_container_width=True, height=360)

    st.markdown("</div>", unsafe_allow_html=True)


st.markdown("---")
st.markdown(
    """
<div class="muted" style="font-size:0.95rem;">
<b>About</b><br/>
This app uses a <b>DistilBERT transformer</b> fine-tuned on the <b>WELFake dataset</b> to classify text into <b>Real vs Fake</b>.
It outputs prediction, confidence, probabilities, and token insights.
<br/><br/>
⚠️ <b>Disclaimer:</b> This model predicts based on linguistic/contextual patterns. It does <b>not</b> fact-check against live web sources.
</div>
""",
    unsafe_allow_html=True,
)