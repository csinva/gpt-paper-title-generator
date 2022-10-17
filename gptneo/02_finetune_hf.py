# Follow tutorial from here: https://huggingface.co/docs/transformers/v4.17.0/en/tasks/language_modeling
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import json
import datasets
import os.path
import pickle as pkl
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from transformers import TrainerCallback
import data

# hyperparams
# checkpoint = "EleutherAI/gpt-j-6B"
checkpoint = "EleutherAI/gpt-neo-2.7B"
# checkpoint = 'facebook/opt-2.7b'
# checkpoint = "gpt2-xl"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokenizer.pad_token = tokenizer.eos_token

if os.path.exists('dset_tokenized.pkl'):
    dset_tokenized = pkl.load(open('dset_tokenized.pkl', 'rb'))
else:
    # set up dset
    df = pd.read_json(open('titles_prepared.jsonl', 'r'),
                      lines=True).astype(str)
    # df = df.sample(n=500)
    # prompt ends with two newlines, completion ends with one
    df['text'] = df['prompt'] + '\n\n' + df['completion']
    df = df.drop(columns=['prompt', 'completion'])
    dset = datasets.Dataset.from_pandas(df)

    # set up tokenized data
    print('tokenizing data...')

    def tokenize_function(ex):
        return tokenizer(ex["text"], padding='max_length', truncation=True)
    dset_tokenized = dset.map(tokenize_function, batched=True)
    dset_tokenized = dset_tokenized.add_column(
        'labels', dset_tokenized['input_ids'])
    pkl.dump(dset_tokenized, open('dset_tokenized.pkl', 'wb'))
print('training on', len(dset_tokenized), 'examples')
print('ex', repr(dset_tokenized[0]['text']))
print('\t(both prompt and text-gen should end with something specific, like newline)')
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# load model
print('loading model...')
model = AutoModelForCausalLM.from_pretrained(checkpoint)
# model = model.half()


class TestGenerationCallback(TrainerCallback):
    "A callback that prints a message at the beginning of training"

    def on_save(self, args, state, control, **kwargs):
        print('example samples',
              data.generate_samples(
                  model,
                  tokenizer,
                  prompt='2020\n\n',
                  num_return_sequences=10,
              ))


# set up trainer
training_args = TrainingArguments(
    output_dir='title-' + checkpoint.replace('/', '__'),
    learning_rate=5e-06,  # 1e-05 was used for 2.7B model
    # warmup_steps=100,
    weight_decay=0.01,  # 0.01 was used for 20B
    per_device_train_batch_size=1,
    gradient_accumulation_steps=20,
    num_train_epochs=4,
    save_strategy='steps',
    save_steps=500,
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dset_tokenized,
    data_collator=data_collator,
    callbacks=[TestGenerationCallback],
)

# launch training
trainer.train()
