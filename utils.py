import numpy as np
import evaluate
import logging
MAX_LEN = 75

def preprocessing_function(examples, tokenizer):
    input_ids = tokenizer(
        examples['en'], padding="max_length", truncation=True, max_length=MAX_LEN
    ).input_ids

    label_ids = tokenizer(
        examples['vi'], padding="max_length", truncation=True, max_length=MAX_LEN
    ).input_ids

    labels = [
        [-100 if token == tokenizer.pad_token_id else token for token in label]
        for label in label_ids
    ]
    return {'input_ids': input_ids, 'labels': labels}


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels


def build_compute_metrics(tokenizer):
    metric = evaluate.load("sacrebleu")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        result = {"bleu": result["score"]}

        # Optional: log loss or custom metric
        logging.info(f"Eval BLEU: {result['bleu']:.2f}")
    
        return result
     

    return compute_metrics
