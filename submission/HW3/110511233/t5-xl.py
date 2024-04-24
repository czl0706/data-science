# %%
import os 
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# %%
import torch
from dotenv import load_dotenv

load_dotenv()

assert torch.cuda.is_available() == True

# %%
from transformers import T5ForConditionalGeneration, AutoTokenizer
import torch

model_name = "google/flan-t5-xl"

model = T5ForConditionalGeneration.from_pretrained(model_name, 
                                                   device_map="auto")     
                                                            
tokenizer = AutoTokenizer.from_pretrained(model_name,
                                          device_map="auto")

# %%
print(model.dtype)

# %%
prefix = "summarize: "

def preprocess_function(examples):
    inputs = [prefix + doc for doc in examples["body"]]
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True)

    labels = tokenizer(text_target=examples["headline"], max_length=128, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

import os 

if not os.path.exists("train_tokenized"):
    from datasets import load_dataset
    dataset = load_dataset("json", data_files="train.json", split="train")
    
    os.makedirs("train_tokenized")
    tokenized_datasets = dataset.map(preprocess_function, batched=True) \
                                .remove_columns(dataset.column_names)
                                
    tokenized_datasets.save_to_disk("train_tokenized")
else:
    from datasets import load_from_disk
    tokenized_datasets = load_from_disk("train_tokenized")

# %%
tokenized_datasets = tokenized_datasets.train_test_split(test_size=0.01)

tokenized_datasets

# %%
from transformers import DataCollatorForSeq2Seq

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# %%
import numpy as np
import evaluate

metric_rouge = evaluate.load("rouge", rouge_types=["rouge1", "rouge2", "rougeL"])
# metric_bertscore = evaluate.load("bertscore")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    result = metric_rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}

# %%
# generation_config = model.generation_config 

# generation_config.max_length = 40
# generation_config.min_length = 10
# generation_config.do_sample = True
# generation_config.top_k = 30
# generation_config.top_p = 0.95

# %%
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

output_dir=f"./headline-generation-{model_name.split('/')[-1]}"

training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="steps",
    eval_steps=1000,
    # eval_steps=3,
    report_to="wandb", # enables logging to W&B 😎
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    # weight_decay=0.01,
    # learning_rate=2e-4,
    num_train_epochs=3,
    gradient_accumulation_steps=32, # simulate larger batch sizes
    save_total_limit=3,
    predict_with_generate=True,
    metric_for_best_model="rougeL",
    # fp16=True,
    # fp16_full_eval=True
    # generation_config=generation_config,
    # load_best_model_at_end=True,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test'],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

# %%
model.save_pretrained(output_dir)

# %%
from datasets import load_dataset
dataset = load_dataset("json", data_files="train.json", split="train[-50:]")

# %%
dataset

# %%
metric_rouge = evaluate.load("rouge", rouge_types=["rouge1", "rouge2", "rougeL"])
metric_bertscore = evaluate.load("bertscore")

# %%
model = T5ForConditionalGeneration.from_pretrained(output_dir)
model.eval()

# test with training data
from transformers import pipeline

summarizer = pipeline("summarization", model=model, tokenizer=model_name)

# %%
results = []
for x in dataset:
    body = x["body"]
    headline = x["headline"]

    summary = summarizer("summarize: " + body, 
                         max_length=40, 
                         min_length=10, 
                         do_sample=True,
                         top_k=30,
                         top_p=0.95,
                         
                         
                        #  num_beams=4,
                        #  length_penalty=0.6,
                         )[0]['summary_text']
    
    # compute and store
    results.append({
        "summary_len": len(summary),
        "headline_len": len(headline),
        "rouge": metric_rouge.compute(predictions=[summary], references=[headline], use_stemmer=True),
        "bertscore": metric_bertscore.compute(predictions=[summary], references=[headline], lang="en")
    })

    print("Body:", body)
    print("Headline:", headline)
    print("Generated:", summary)
    print()

# %%
# compute average
rouge1 = sum([x["rouge"]["rouge1"] for x in results]) / len(results)
rouge2 = sum([x["rouge"]["rouge2"] for x in results]) / len(results)
rougeL = sum([x["rouge"]["rougeL"] for x in results]) / len(results)
rougeLsum = sum([x["rouge"]["rougeLsum"] for x in results]) / len(results)
f1 = sum([x["bertscore"]["f1"][0] for x in results]) / len(results)

print("Rouge-1:", rouge1)
print("Rouge-2:", rouge2)
print("Rouge-L:", rougeL)
print("Rouge-Lsum:", rougeLsum)
print("BertScore F1:", f1)


