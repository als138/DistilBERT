# DistilBERT

# DistilBERT Fine-Tuning on GLUE Tasks (MRPC, QNLI, SST-2)

A complete repository for fine-tuning the `distilbert-base-uncased` model on various GLUE benchmark tasks using the Hugging Face Transformers library.

---

## 🔍 Overview

This project focuses on fine-tuning the DistilBERT model for:

- **MRPC** (Microsoft Research Paraphrase Corpus)
- **QNLI** (Question Natural Language Inference)
- **SST-2** (Stanford Sentiment Treebank 2) – optional

We include:

- Dataset loading and preprocessing
- Tokenization of input sentence pairs
- Fine-tuning configuration and training
- Evaluation and metrics (Accuracy, F1)
- Inference pipeline for custom inputs
- Saving and reusing the final model

---

## 📦 Installation

```bash
pip install transformers datasets evaluate accelerate
```

Make sure you're using Python 3.8 or higher.

---

## 🧠 Supported Tasks

### 1. MRPC – Paraphrase Detection
- Input: Sentence1, Sentence2
- Output: Whether the two sentences are paraphrases
- Metrics: Accuracy, F1

### 2. QNLI – Natural Language Inference
- Input: Question, Sentence
- Output: Entailment or not
- Metrics: Accuracy

### 3. SST-2 – Sentiment Classification (optional)
- Input: Single sentence
- Output: Positive or Negative sentiment
- Metrics: Accuracy

---

## 🚀 Training Example

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import numpy as np
import evaluate

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# Load and tokenize dataset
dataset = load_dataset("glue", "mrpc")

def tokenize_fn(examples):
    return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True)

tokenized_dataset = dataset.map(tokenize_fn, batched=True)

# Metrics
accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy.compute(predictions=predictions, references=labels)["accuracy"],
        "f1": f1.compute(predictions=predictions, references=labels)["f1"]
    }

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    load_best_model_at_end=True,
    metric_for_best_model="f1"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()
```

---

## 📊 Evaluation & Inference

```python
# Evaluate the model
eval_results = trainer.evaluate()
print(eval_results)

# Inference with pipeline
from transformers import pipeline
classifier = pipeline("text-classification", model=trainer.model, tokenizer=tokenizer)

# Test on custom sentence pairs
pairs = [
    {"text": "He traveled to France.", "text_pair": "He went to Paris."},
    {"text": "The cat is sleeping.", "text_pair": "A dog is barking loudly."}
]
results = classifier(pairs)

for pair, res in zip(pairs, results):
    label = "Paraphrase" if res['label'] == 'LABEL_1' else "Not Paraphrase"
    print(f"{pair['text']} / {pair['text_pair']} => {label} ({res['score']:.4f})")
```

---

## 💾 Saving & Loading the Model

```python
trainer.save_model("distilbert-finetuned")
tokenizer.save_pretrained("distilbert-finetuned")
```

To load again:

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained("distilbert-finetuned")
tokenizer = AutoTokenizer.from_pretrained("distilbert-finetuned")
```

---

## 🧪 Results

| Task  | Accuracy | F1 (if applicable) |
|-------|----------|--------------------|
| MRPC  | ~0.84    | ~0.91              |
| QNLI  | ~0.90    | —                  |
| SST-2 | ~0.92    | —                  |

---

## 📁 Directory Structure

```
DistilBERT/
├── train_mrpc.py
├── train_qnli.py
├── train_sst2.py
├── requirements.txt
└── README.md
```

---

## 🤝 Contribution

Pull requests are welcome! To contribute:

1. Fork the repo
2. Create a new branch: `git checkout -b feature/task`
3. Make your changes
4. Commit: `git commit -m "Add new feature"`
5. Push: `git push origin feature/task`
6. Submit a pull request

---

## 📚 References

- [Hugging Face Transformers](https://huggingface.co/transformers)
- [GLUE Benchmark](https://gluebenchmark.com/)
- [DistilBERT Paper](https://arxiv.org/abs/1910.01108)

---

## 📜 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgements

Special thanks to:

- Hugging Face for the Transformers ecosystem
- GLUE benchmark creators
- The open-source community for inspiration
