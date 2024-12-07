from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from textblob import TextBlob


class TextGenerator:
    def __init__(self, model, tokenizer, device="cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        # Pad token ayarı
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token


    @staticmethod
    def post_process_text(text):
        
        blob = TextBlob(text)
        return str(blob.correct())

    def generate_text(self, prompt, max_length=500, temperature=0.85, top_p=0.95):
        inputs = self.tokenizer(
            prompt, return_tensors="pt", padding=True, truncation=True, max_length=100
        ).to(self.device)
        outputs = self.model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return self.post_process_text(generated_text)

    def evaluate_text(self, prompt, generated_text):
        rouge_scorer_instance = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
        rouge_scores = rouge_scorer_instance.score(prompt, generated_text)
        bleu_score = sentence_bleu([prompt.split()], generated_text.split())
        return rouge_scores, bleu_score
