# English-Vietnamese Translation with mBART50

This project fine-tunes `facebook/mbart-large-50-many-to-many-mmt` for English-to-Vietnamese translation using the IWSLT2015 dataset.

## 📦 Setup

```bash
pip install -r requirements.txt
```

## Translation API Usage
```bash
curl -X POST http://127.0.0.1:8000/translate \
    -H "Content-Type: application/json" \
    -d '{"text": "Hello, how are you?"}'
```
