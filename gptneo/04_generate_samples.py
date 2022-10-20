import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import json
import torch
import datasets
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from transformers import AutoModelForCausalLM, pipeline, AutoTokenizer
import data
import os
from os.path import join
from tqdm import tqdm

num_return_sequences = 5000
checkpoint = "csinva/gpt-neo-2.7B-titles"
model = AutoModelForCausalLM.from_pretrained(checkpoint)
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")

years = ['2022', '2023',
         '2020', '2021', '2010', '2024', '2030', '2050']
for year in tqdm(years):
    print('generating for', year)
    pipe = pipeline('text-generation', model=model, tokenizer=tokenizer)
    text = pipe(f'{year}\n\n', return_full_text=False, num_return_sequences=num_return_sequences,
                max_new_tokens=30, eos_token_id=198)  # 628 is \n\n

    # process texts
    texts = []
    for t in text:
        s = t['generated_text']
        if '\n' in s:
            s = s[:s.index('\n')]
        texts.append(s.strip(' \'\"'))

    out_dir = f'../samples/gptneo/{year}'
    os.makedirs(out_dir, exist_ok=True)
    with open(join(out_dir, 'titles.txt'), 'w') as f:
        f.write('\n\n'.join(texts))
