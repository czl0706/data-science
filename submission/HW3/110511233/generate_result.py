from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import pipeline
from transformers import logging
from tqdm import tqdm
import json

model_dir = f"./headline-generation-flan-t5-xl"
test_filename = "test.json"
submit_filename = "110511233.json"

logging.set_verbosity_info(logging.ERROR)

model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)

# get tokenizer by model name
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")

summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)

config = model.generation_config 
config.max_length = 40
config.min_length = 10
config.do_sample = True
config.top_k = 30
config.top_p = 0.95

# read jsonline file into a list
with open(test_filename, "r") as f:
    data = f.readlines()

# parse jsonlines into a list of dictionaries
data = [json.loads(line) for line in data]
data = [d['body'] for d in data]

results = []
for x in tqdm(data):

    summary = summarizer("summarize: " + x, 
                         max_length=40, 
                         min_length=10, 
                         do_sample=True,
                         top_k=30,
                         top_p=0.95,
                        #  num_beams=4,
                        #  length_penalty=0.6,
                         )[0]['summary_text']
    
    # print(summary)
    
    results.append({
        "headline": summary,
    })
    
# write the results to a json file, in jsonlines format
with open(submit_filename, "w") as f:
    for result in results:
        f.write(json.dumps(result))
        f.write("\n")