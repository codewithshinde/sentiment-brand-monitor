"""
Skeleton for transformer fine-tuning (DistilBERT) using Hugging Face `datasets` + `transformers`.
Fill in and run on a GPU runtime for best results.
"""
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
import evaluate

MODEL_NAME = "distilbert-base-uncased"

def main():
    ds = load_dataset("tweet_eval", "sentiment")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(batch):
        return tok(batch["text"], truncation=True)
    ds_tok = ds.map(tokenize, batched=True)

    collator = DataCollatorWithPadding(tokenizer=tok)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)

    metric = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {"f1_macro": metric.compute(predictions=preds, references=labels, average="macro")}

    args = TrainingArguments(
        output_dir="models/distilbert-finetune",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=2,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        weight_decay=0.01,
        logging_steps=50,
    )

    tr = Trainer(
        model=model,
        args=args,
        train_dataset=ds_tok["train"],
        eval_dataset=ds_tok["validation"],
        tokenizer=tok,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )
    tr.train()
    tr.evaluate()
    tr.save_model("models/distilbert-finetune")

if __name__ == "__main__":
    main()
