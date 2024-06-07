import json
from datasets import load_dataset, DatasetDict
import os
import math


# Number of chunks to split the data into
num_chunks = 10

# Load the SQuAD train set from the input file
with open("/home/mila/m/maziar.sargordi/scratch/chunk_bert_storage/squad.1.1.split.train.json", "r", encoding="utf-8") as f:
    squad_train_data = json.load(f)["data"]

# Split the data into chunks
chunk_size = math.ceil(len(squad_train_data) / num_chunks)
data_chunks = [squad_train_data[i:i+chunk_size] for i in range(0, len(squad_train_data), chunk_size)]

# Process each chunk separately and write the examples to separate JSON files
for i, chunk in enumerate(data_chunks):
    datasets_train_data = []
    for article in chunk:
        for paragraph in article["paragraphs"]:
            for qa in paragraph["qas"]:
                answers = qa.get("answers", [])
                answer_texts = [answer["text"] for answer in answers]
                answer_starts = [answer["answer_start"] for answer in answers]
                example = {
                    "id": qa["id"],
                    "title": article["title"],
                    "context": paragraph["context"],
                    "question": qa["question"],
                    "answers": {"text": answer_texts, "answer_start": answer_starts},
                }
                datasets_train_data.append(example)

    # Save the examples to a JSON file
    with open(f"train_data_chunk_{i}.json", "w", encoding="utf-8") as f:
        json.dump(datasets_train_data, f, ensure_ascii=False)

# Load the train dataset from the separate JSON files
train_data_files = [f for f in os.listdir() if f.startswith("train_data_chunk_") and f.endswith(".json")]
train_dataset = load_dataset("json", data_files=train_data_files)['train']



# Load the SQuAD validation set from the input file
with open("/home/mila/m/maziar.sargordi/scratch/chunk_bert_storage/squad.1.1.split.valid.json", "r", encoding="utf-8") as f:
    squad_validation_data = json.load(f)["data"]

datasets_validation_data = []
for article in squad_validation_data:
    for paragraph in article["paragraphs"]:
        for qa in paragraph["qas"]:
            answers = qa.get("answers", [])
            answer_texts = [answer["text"] for answer in answers]
            answer_starts = [answer["answer_start"] for answer in answers]
            example = {
                "id": qa["id"],
                "title": article["title"],
                "context": paragraph["context"],
                "question": qa["question"],
                "answers": {"text": answer_texts, "answer_start": answer_starts},
            }
            datasets_validation_data.append(example)

# Save the examples to a JSON file
with open("validation_data.json", "w", encoding="utf-8") as f:
    json.dump(datasets_validation_data, f, ensure_ascii=False)

# Load the validation dataset from the JSON file
validation_dataset = load_dataset("json", data_files="validation_data.json")['train']

raw_datasets = DatasetDict({"train": train_dataset, "validation": validation_dataset})
raw_datasets.save_to_disk("/home/mila/m/maziar.sargordi/scratch/chunk_bert_storage/squad_dataset")
print(raw_datasets)