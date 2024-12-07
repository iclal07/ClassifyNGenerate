from fastapi import FastAPI
from pydantic import BaseModel
from modules.data_processing import TextProcessor
from modules.predict import Predictor
from modules.text_generation import TextGenerator
from modules.model_loader import LlamaModelLoader
import gradio as gr
import torch

# API Başlat
app = FastAPI()

# Modüller Yükleniyor
classification_model_path = "models/logistic_model.pkl"
vectorizer_path = "models/vectorizer.pkl"
llama_model_path = "models/Llama-3.2-3B"
device = "mps"  if torch.backends.mps.is_available() else "cpu"

print("Modeller yükleniyor...")
text_processor = TextProcessor()
predictor = Predictor(classification_model_path, vectorizer_path)

# LLaMA modelini yükleme
llama_loader = LlamaModelLoader(llama_model_path, device)
llama_model = llama_loader.model
llama_tokenizer = llama_loader.tokenizer
text_generator = TextGenerator(llama_model, llama_tokenizer, device)

print("Modeller başarıyla yüklendi!")

# Gradio Fonksiyonu
def gradio_interface(input_text):
    response = {
        "predicted_class": predictor.predict(input_text, text_processor.clean_text),
        "generated_text": text_generator.generate_text(input_text),
    }
    rouge_scores, bleu_score = text_generator.evaluate_text(
        input_text, response["generated_text"]
    )
    response["rouge_scores"] = {
        "rouge1": rouge_scores["rouge1"].fmeasure,
        "rougeL": rouge_scores["rougeL"].fmeasure,
    }
    response["bleu_score"] = bleu_score
    return (
        response["predicted_class"],
        response["generated_text"],
        f"ROUGE-1: {response['rouge_scores']['rouge1']:.4f}\n"
        f"ROUGE-L: {response['rouge_scores']['rougeL']:.4f}\n"
        f"BLEU: {response['bleu_score']:.4f}",
    )


# Gradio Arayüzü
with gr.Blocks(css=".gradio-container {background-color: #D6A2E8; justify-content: center;}") as gr_interface:
    gr.Markdown(
        """
        <h1 style="text-align: center;">Haber Sınıflandırma ve Metin Genişletme</h1>
        <p style="text-align: center;">Bir haber metni girin ve sonuçları görün!</p>
        """
    )

    with gr.Row():
        with gr.Column():
            input_box = gr.Textbox(
                lines=5,
                placeholder="Haber metnini buraya girin...",
                label="Haber Metni",
            )
            submit_button = gr.Button("Sonuçları Göster")

    with gr.Row(visible=False) as output_row:
        with gr.Column():
            gr.Markdown("<br>")  # Boşluk ekler
            output_class = gr.Textbox(label="Tahmin Edilen Sınıf")
            gr.Markdown("<br>")  # Boşluk ekler
            output_text = gr.Textbox(label="Üretilen Metin")
            gr.Markdown("<br>")  # Boşluk ekler
            output_scores = gr.Textbox(label="Skorlar (ROUGE & BLEU)")

    def handle_click(input_text):
        predicted_class, generated_text, scores = gradio_interface(input_text)
        output_row.visible = True
        return predicted_class, generated_text, scores

    submit_button.click(
        handle_click,
        inputs=input_box,
        outputs=[output_class, output_text, output_scores],
    )

# Gradio başlatma
if __name__ == "__main__":
    gr_interface.launch(server_name="0.0.0.0", server_port=7860)