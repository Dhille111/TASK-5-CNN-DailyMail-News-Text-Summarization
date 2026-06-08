import streamlit as st
from transformers import BartTokenizer, BartForConditionalGeneration
import fitz
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import re

MODEL_PATH = "models/final_model"

st.set_page_config(
    page_title="AI News Summarizer",
    page_icon="📰",
    layout="wide"
)

@st.cache_resource
def load_model():
    tokenizer = BartTokenizer.from_pretrained(MODEL_PATH)
    model = BartForConditionalGeneration.from_pretrained(MODEL_PATH)
    return tokenizer, model

tokenizer, model = load_model()

if "history" not in st.session_state:
    st.session_state.history = []

# =========================
# HERO SECTION
# =========================

st.markdown("""
<div style='text-align:center;padding:20px;'>
    <h1>📰 AI News Text Summarizer</h1>
    <h4>Fine-Tuned BART Transformer Model</h4>
    <p>Generate concise summaries from long news articles instantly.</p>
</div>
""", unsafe_allow_html=True)

# =========================
# SIDEBAR
# =========================

with st.sidebar:

    st.title("🤖 Project Info")

    st.success("Transformer Based Summarization")

    st.markdown("---")

    st.write("📚 Dataset: CNN/DailyMail")
    st.write("🤖 Model: BART Base")

    st.write("📈 ROUGE-1: 38.5%")
    st.write("📈 ROUGE-2: 17.7%")
    st.write("📈 ROUGE-L: 27.0%")

    st.markdown("---")

    st.info("Built using Hugging Face Transformers + Streamlit")

    st.markdown("---")

    st.subheader("📜 Summary History")

    for item in st.session_state.history[-5:]:
        st.write("• " + item[:60] + "...")

# =========================
# MAIN AREA
# =========================

summary_length = st.slider(
    "🎯 Select Summary Length",
    min_value=30,
    max_value=200,
    value=100
)

st.markdown("### 📂 Upload TXT or PDF")

uploaded_file = st.file_uploader(
    "Choose a TXT or PDF file",
    type=["txt", "pdf"]
)

article = ""

if uploaded_file:

    if uploaded_file.name.endswith(".txt"):
        article = uploaded_file.read().decode("utf-8")

    elif uploaded_file.name.endswith(".pdf"):

        pdf = fitz.open(
            stream=uploaded_file.read(),
            filetype="pdf"
        )

        for page in pdf:
            article += page.get_text()

article = st.text_area(
    "📝 Enter News Article",
    value=article,
    height=300,
    placeholder="Paste your article here..."
)

word_count = len(article.split())

st.info(f"📄 Input Words: {word_count}")

# =========================
# GENERATE SUMMARY
# =========================

if st.button("🚀 Generate Summary"):

    if article.strip() == "":
        st.warning("Please enter an article.")

    else:

        with st.spinner("Generating Summary..."):

            inputs = tokenizer(
                article,
                return_tensors="pt",
                truncation=True,
                max_length=256
            )

            summary_ids = model.generate(
                inputs["input_ids"],
                max_length=summary_length,
                min_length=summary_length // 2,
                num_beams=5,
                length_penalty=2.0,
                early_stopping=True
            )

            summary = tokenizer.decode(
                summary_ids[0],
                skip_special_tokens=True
            )

        st.session_state.history.append(summary)

        # SUMMARY

        st.markdown("## ✨ Generated Summary")

        st.text_area(
            "Generated Summary",
            summary,
            height=200
        )

        st.code(summary)

        # METRICS

        summary_words = len(summary.split())

        compression = round(
            (summary_words / max(word_count, 1)) * 100,
            2
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Input Words",
            word_count
        )

        col2.metric(
            "Summary Words",
            summary_words
        )

        col3.metric(
            "Compression %",
            f"{compression}%"
        )

        # KEYWORDS

        words = re.findall(
            r"\b[a-zA-Z]{4,}\b",
            article.lower()
        )

        top_words = Counter(words).most_common(10)

        st.markdown("## 🔑 Top Keywords")

        for word, count in top_words:
            st.write(f"{word} ({count})")

        # WORD CLOUD

        st.markdown("## ☁️ Word Cloud")

        if len(article.strip()) > 0:

            wc = WordCloud(
                width=800,
                height=400,
                background_color="white"
            ).generate(article)

            fig, ax = plt.subplots(figsize=(10, 5))

            ax.imshow(wc)

            ax.axis("off")

            st.pyplot(fig)

        # DOWNLOAD

        st.download_button(
            "📥 Download Summary",
            summary,
            file_name="summary.txt",
            mime="text/plain"
        )