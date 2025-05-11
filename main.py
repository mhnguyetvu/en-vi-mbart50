from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from fastapi import FastAPI
from pydantic import BaseModel
import logging
from datetime import datetime

# Cấu hình logging
logging.basicConfig(
    filename="./logs/translation_api.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Load model và tokenizer đã fine-tune
model_dir = "./models/mbart_en_vi"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)

# Thiết lập ngôn ngữ cho mBART
tokenizer.src_lang = "en_XX"
tokenizer.tgt_lang = "vi_VN"

app = FastAPI()

class TranslationRequest(BaseModel):
    text: str

@app.post("/translate")
def translate(req: TranslationRequest):
    try:
        logging.info(f"Received text: {req.text}")

        inputs = tokenizer(req.text, return_tensors="pt", padding=True, truncation=True)

        generated = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.lang_code_to_id["vi_VN"],
            max_length=75
        )

        translation = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]

        logging.info(f"Translation: {translation}")
        return {"translation": translation}

    except Exception as e:
        logging.error(f"Translation failed: {e}")
        return {"error": "Translation failed. Check server logs for details."}
