# 📰 AI News Text Summarizer

An end-to-end Text Summarization system built using a fine-tuned **BART (Bidirectional and Auto-Regressive Transformers)** model. The project features a modern, interactive web application built with **Streamlit** to generate concise summaries from long news articles or documents (TXT/PDF format) and display key text analytics.

---

## 🚀 Key Features

* **Fine-Tuned Transformer**: Employs a `facebook/bart-base` model fine-tuned on the CNN/DailyMail news dataset.
* **Flexible Input Methods**: 
  * Paste articles directly into the text area.
  * Upload `.txt` or `.pdf` files directly.
* **Customizable Summary Length**: Dynamically adjust output summary length using a slider.
* **Rich Text Analytics**: 
  * Displays input word count, summary word count, and compression ratio.
  * Extracts top keywords from the source text.
  * Generates and renders a visual **Word Cloud**.
* **Download Option**: Easily download the generated summary as a `.txt` file.
* **Summary History**: Tracks and displays recent summaries in the sidebar.

---

## 🛠️ Project Structure

```
├── .streamlit/
│   └── config.toml             # Streamlit theme & UI configuration
├── assets/
│   ├── input_screenshot.png    # Input interface screenshot
│   └── output_screenshot.png   # Output interface screenshot
├── src/
│   ├── train.py                # Seq2Seq training script
│   ├── predict.py              # Model inference utility
│   ├── preprocess.py           # Text preprocessing module
│   ├── model_evaluation.py     # Evaluation utilities
│   ├── eda.py                  # Exploratory Data Analysis
│   ├── rouge_evaluation.py     # ROUGE metric calculations
│   └── ...                     # Additional data analysis scripts
├── app.py                      # Main Streamlit web application
├── .gitignore                  # Git ignore file
└── README.md                   # Project documentation
```

---

## 📦 Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Dhille111/TASK-5-CNN-DailyMail-News-Text-Summarization.git
cd "TASK 5CNN-DailyMail News Text Summarization"
```

### 2. Set Up Virtual Environment
Create and activate a virtual environment:
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
Make sure you install the required packages:
```bash
pip install streamlit transformers datasets torch pandas matplotlib wordcloud pymupdf
```

### 4. Download / Train Model
The web application expects the fine-tuned model weights to be present in `models/final_model`. To train the model locally using the subset dataset:
```bash
python src/train.py
```

---

## 🖥️ Running the Web Application

Start the Streamlit application using:
```bash
streamlit run app.py
```
Once the server starts, access the app in your browser at `http://localhost:8501`.

---

## 📊 Output & Screenshots

Here is the user interface of the web application in action:

### 1. Web App Input Interface
![Web App Input Interface](assets/input_screenshot.png)

### 2. Generated Summary & Metrics
![Generated Summary & Metrics](assets/output_screenshot.png)
