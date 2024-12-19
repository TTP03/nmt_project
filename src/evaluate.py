import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

def load_model_and_tokenizer(model_path):
    """Load the pretrained model and tokenizer."""
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer

def translate(sentence, model, tokenizer, device):
    """Translate a sentence using the model."""
    model.eval()
    inputs = tokenizer(
        sentence,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=128,
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=128,
            num_beams=4,
            early_stopping=True
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    # Đường dẫn mô hình đã huấn luyện
    model_path = "saved_model"  # Thay bằng đường dẫn mô hình đã lưu của bạn
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load mô hình và tokenizer
    model, tokenizer = load_model_and_tokenizer(model_path)
    model.to(device)

    # Câu cần dịch
    sentence = "I am going to school."  # Thay câu này bằng câu tiếng Anh bất kỳ
    translation = translate(sentence, model, tokenizer, device)

    # Hiển thị kết quả
    print(f"English: {sentence}")
    print(f"Vietnamese: {translation}")
