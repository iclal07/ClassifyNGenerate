# **ClassifyNGenerate: AI-Powered News Classification and Text Generation**

## 📖 **Overview**
NewsExpander is an AI-powered project designed to:
1. Classify news articles into predefined categories.
2. Generate expanded versions of the input news text using a large language model.

With the power of **LLaMA 3.2 (3B)** for text generation and a **Logistic Regression** classifier, this project demonstrates how AI can transform news processing tasks.

---

## 🚀 **Features**
- **News Classification**: Classifies news articles into one of the following categories:
  - `World`
  - `Sports`
  - `Business`
  - `Sci/Tech`
- **Text Expansion**: Generates expanded and coherent versions of news articles.
- **Dynamic Model Training**: Automatically trains and saves classification models if no pre-trained models are available.
- **Evaluation Metrics**: Includes BLEU and ROUGE scores to evaluate the quality of generated text.

---

## 🛠️ **Technologies Used**
1. **Large Language Model (LLM)**:
   - **LLaMA 3.2 (3B)** for text generation.
2. **Machine Learning for Classification**:
   - **Logistic Regression** with **TF-IDF Vectorization**.
3. **Python Libraries**:
   - `Transformers`: For LLaMA model handling.
   - `Scikit-learn`: For classification and vectorization.
   - `NLTK`: For text preprocessing.
   - `ROUGE` & `BLEU`: For evaluating text quality.

---

## 📂 **Project Structure**
```plaintext
Project/
├── main2.py                      # Main execution file
├── modules/                      # Core modules
│   ├── data_processing.py        # Text preprocessing utilities
│   ├── model_training.py         # Model training and saving
│   ├── predict.py                # Prediction and classification
│   ├── text_generation.py        # Text expansion using LLaMA
│   ├── model_loader.py           # LLaMA model loader
├── data/                         # Data directory
│   ├── train.parquet             # Training data
│   ├── test.parquet              # Test data
├── models/                       # Saved models
│   ├── logistic_model.pkl        # Logistic Regression model
│   ├── vectorizer.pkl            # TF-IDF vectorizer
│   ├── Llama-3.2-3B/             # LLaMA 3.2 model files
