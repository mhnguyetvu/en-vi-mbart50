import sys
sys.stdout = open("training.log", "w")
import logging
from datetime import datetime

# Tạo logger
logfile = f"logs/train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    filename=logfile,
    filemode="w",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

# Ghi log test thử
logging.info("Training started.")

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, Seq2SeqTrainer
import os
from utils import preprocessing_function,  build_compute_metrics
from functools import partial

# Tắt wandb nếu không dùng
os.environ["WANDB_DISABLED"] = "true"

# Load dataset train và test
train_ds = load_dataset("thainq107/iwslt2015-en-vi", split="train")
valid_ds = load_dataset("thainq107/iwslt2015-en-vi", split="test")

# Load tokenizer và model
model_name = "facebook/mbart-large-50-many-to-many-mmt"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Thiết lập ngôn ngữ cho mBART
tokenizer.src_lang = "en_XX"
tokenizer.tgt_lang = "vi_VN"

# Preprocess dữ liệu
train_ds = train_ds.map(lambda x: preprocessing_function(x, tokenizer), batched=True)
valid_ds = valid_ds.map(lambda x: preprocessing_function(x, tokenizer), batched=True)

# Thiết lập huấn luyện
training_args = Seq2SeqTrainingArguments(
    output_dir="./models/mbart_en_vi",
    logging_dir="./logs",
    eval_strategy="steps",
    save_strategy="steps",
    logging_steps=1000,
    eval_steps=1000,
    save_steps=1000,
    save_total_limit=1,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    predict_with_generate=True,
    load_best_model_at_end=True,
)

# Collator để xử lý padding cho batch
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Huấn luyện
compute_metrics = build_compute_metrics(tokenizer)
train_preprocessed = train_ds.map(lambda x: preprocessing_function(x, tokenizer), batched=True)
valid_preprocessed = valid_ds.map(lambda x: preprocessing_function(x, tokenizer), batched=True)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_preprocessed,
    eval_dataset=valid_preprocessed,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

# Lưu model và tokenizer
trainer.save_model("./models/mbart_en_vi")
tokenizer.save_pretrained("./models/mbart_en_vi")




















