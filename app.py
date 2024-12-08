import gradio as gr
import torch
import gc
from modules.data_processing import TextProcessor
from modules.predict import Predictor
from modules.text_generation import TextGenerator
from modules.model_loader import LlamaModelLoader

# GPU kullanımı ve bellek temizleme
device = "mps" if torch.backends.mps.is_available() else "cpu"

def clear_memory():
    torch.mps.empty_cache()
    gc.collect()

# Modül ve model yükleme
classification_model_path = "models/logistic_model.pkl"
vectorizer_path = "models/vectorizer.pkl"
llama_model_path = "models/Llama-3.2-3B"

text_processor = TextProcessor()
predictor = Predictor(classification_model_path, vectorizer_path)
llama_loader = LlamaModelLoader(llama_model_path, device)
llama_model = llama_loader.model
llama_tokenizer = llama_loader.tokenizer
text_generator = TextGenerator(llama_model, llama_tokenizer, device)

# Analiz Fonksiyonu
def analyze_text(input_text):
    try:
        # Sınıf tahmini
        predicted_class = predictor.predict(input_text, text_processor.clean_text)

        # Genişletilmiş metin
        expanded_text = text_generator.generate_text(input_text)

        # Performans skorları
        rouge_scores, bleu_score = text_generator.evaluate_text(input_text, expanded_text)
        rouge_scores_formatted = f"ROUGE-1: {rouge_scores['rouge1'].fmeasure:.4f}\n" \
                                 f"ROUGE-L: {rouge_scores['rougeL'].fmeasure:.4f}"

        # Belleği temizle
        clear_memory()

        return (
            f"{predicted_class}",
            f"{expanded_text}",
            rouge_scores_formatted,
            f"{bleu_score:.4f}"
        )
    except Exception as e:
        clear_memory()
        return "Hata", str(e), "-", "-"

# CSS Özelleştirme
demo_css = """
    body { background-color: #E6E6FA; margin: 0; padding: 0; }
    .gradio-container { width: 100%; max-width: 800px; margin: 0 auto; padding: 20px; background-color: white; border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); }
    .gr-button { width: 100%; }
    .gr-textbox { width: 100%; }
    #header { text-align: center; }
"""

# Gradio Arayüzü
with gr.Blocks(css=demo_css) as demo:
    # Başlık ve Açıklama
    gr.Markdown(
        """
        # ✨ 🚀 ClassifyNGenerate 🚀✨

        Haber metinlerini analiz edin, genişletin ve performans skorlarını öğrenin! 
        """,
        elem_id="header",
    )
    
    # Giriş Kutusu
    input_text = gr.Textbox(
        placeholder="Buraya haber metnini giriniz...",
        label="Haber Metni",
        lines=2
    )
    
    # Submit ve Clear Butonları (Yan Yana)
    with gr.Row():
        submit_button = gr.Button("Analiz Et")
    
    # Çıktı Kutuları
    predicted_class = gr.Textbox(label="Tahmin Edilen Sınıf", lines=2, interactive=False)
    gr.Markdown("<br>")  # Boşluk ekler
    expanded_text = gr.Textbox(label="Genişletilmiş Metin", lines=6, interactive=False)
    gr.Markdown("<br>")  # Boşluk ekler
    rouge_score = gr.Textbox(label="ROUGE Skoru", lines=2, interactive=False)
    bleu_score = gr.Textbox(label="BLEU Skoru", lines=1, interactive=False)
    

    # Submit butonunun işlevselliği
    submit_button.click(
        analyze_text, 
        inputs=[input_text], 
        outputs=[predicted_class, expanded_text, rouge_score, bleu_score]
    )

# Uygulamayı çalıştır
demo.launch()
