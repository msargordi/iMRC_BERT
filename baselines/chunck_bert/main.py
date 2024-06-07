from datasets import load_dataset, Dataset, DatasetDict, load_from_disk
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from transformers import TrainingArguments
from tqdm.auto import tqdm
import torch
import evaluate
import numpy as np
import collections
from torch.optim import AdamW
from transformers import get_scheduler
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import default_data_collator
import json
import codecs
import os
import argparse
from IPython import embed


parser = argparse.ArgumentParser()
parser.add_argument("--epochs", default=4, type=int, help="Number of training epochs.")
parser.add_argument("--batch_size", default=64, type=int, help="Mini-batch size.")
parser.add_argument("--n_best", default=100, type=int, help="Number of highest n_best logits.")
parser.add_argument("--model_checkpoint", default='bert-base-uncased', type=str, help="Path to model.")
parser.add_argument("--output_dir", default='/home/mila/m/maziar.sargordi/scratch/chunk_bert_storage/logs', type=str,
                    help="Path to save the model.")
parser.add_argument('--train', type=str,
                    default='/home/mila/m/maziar.sargordi/scratch/chunk_bert_storage/squad_processed_train',
                    help='Path to load train set')
parser.add_argument('--eval', type=str,
                    default='/home/mila/m/maziar.sargordi/scratch/chunk_bert_storage/squad_processed_valid',
                    help='Path to load eval set')
args = parser.parse_args()


num_train_epochs = args.epochs
batch_size = args.batch_size
n_best = args.n_best

raw_datasets = load_from_disk("/home/mila/m/maziar.sargordi/scratch/chunk_bert_storage/squad_dataset")
train_dataset = load_from_disk(args.train)
validation_dataset = load_from_disk(args.eval)

tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
metric = evaluate.load("squad")



def compute_metrics(start_logits, end_logits, features, examples, n_best=n_best, max_answer_length=30):
    example_to_features = collections.defaultdict(list)
    for idx, feature in enumerate(features):
        example_to_features[feature["example_id"]].append(idx)

    predicted_answers = []
    for example in tqdm(examples):
        example_id = example["id"]
        context = example["context"]
        answers = []

        # Loop through all features associated with that example
        for feature_index in example_to_features[example_id]:
            start_logit = start_logits[feature_index]
            end_logit = end_logits[feature_index]
            offsets = features[feature_index]["offset_mapping"]

            start_indexes = np.argsort(start_logit)[-1 : -n_best - 1 : -1].tolist()
            end_indexes = np.argsort(end_logit)[-1 : -n_best - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Skip answers that are not fully in the context
                    if offsets[start_index] is None or offsets[end_index] is None:
                        continue
                    # Skip answers with a length that is either < 0 or > max_answer_length
                    if (
                        end_index < start_index
                        or end_index - start_index + 1 > max_answer_length
                    ):
                        continue

                    answer = {
                        "text": context[offsets[start_index][0] : offsets[end_index][1]],
                        "logit_score": start_logit[start_index] + end_logit[end_index],
                    }
                    answers.append(answer)

        # Select the answer with the best score
        if len(answers) > 0:
            best_answer = max(answers, key=lambda x: x["logit_score"])
            predicted_answers.append(
                {"id": example_id, "prediction_text": best_answer["text"]}
            )
        else:
            predicted_answers.append({"id": example_id, "prediction_text": ""})

    theoretical_answers = [{"id": ex["id"], "answers": ex["answers"]} for ex in examples]
    return metric.compute(predictions=predicted_answers, references=theoretical_answers)


train_dataset.set_format("torch")
validation_set = validation_dataset.remove_columns(["example_id", "offset_mapping"])
validation_set.set_format("torch")

train_dataloader = DataLoader(
    train_dataset,
    shuffle=True,
    collate_fn=default_data_collator,
    batch_size=batch_size,
)
eval_dataloader = DataLoader(
    validation_set, collate_fn=default_data_collator, batch_size=batch_size
)


model = AutoModelForQuestionAnswering.from_pretrained(args.model_checkpoint)
optimizer = AdamW(model.parameters(), lr=2e-5)

# accelerator = Accelerator(fp16=True)
accelerator = Accelerator()#fp16=True)
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader
)


num_update_steps_per_epoch = len(train_dataloader)
num_training_steps = num_train_epochs * num_update_steps_per_epoch

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)


output_dir = args.output_dir
progress_bar = tqdm(range(num_training_steps))

for epoch in range(num_train_epochs):
    # Training
    model.train()
    for step, batch in enumerate(train_dataloader):
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

    # Evaluation
    model.eval()
    start_logits = []
    end_logits = []
    accelerator.print("Evaluation!")
    for batch in tqdm(eval_dataloader):
        with torch.no_grad():
            outputs = model(**batch)

        start_logits.append(accelerator.gather(outputs.start_logits).cpu().numpy())
        end_logits.append(accelerator.gather(outputs.end_logits).cpu().numpy())

    start_logits = np.concatenate(start_logits)
    end_logits = np.concatenate(end_logits)
    start_logits = start_logits[: len(validation_dataset)]
    end_logits = end_logits[: len(validation_dataset)]

    metrics = compute_metrics(
        start_logits, end_logits, validation_dataset, raw_datasets["validation"]
    )
    with open(output_dir + '/accuracy.txt', 'a') as f:
        f.write(json.dumps(metrics) + '\n')

    print(f"epoch {epoch}:", metrics)

    # Save and upload
    # accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(output_dir + '/model', save_function=accelerator.save)
    if accelerator.is_main_process:
        tokenizer.save_pretrained(output_dir)
