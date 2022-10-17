import pandas as pd
import openai
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import re
import string
from collections import defaultdict
import itertools
from typing import List
from tqdm import tqdm
import os.path
from adjustText import adjust_text
import data
import pickle as pkl

# authors sorted in descending order by num papers
authors_dict_titles = pkl.load(open('../data/authors_dict_titles.pkl', 'rb'))
prompt = 'Here is a list of related machine-learning papers:\n\n> '
authors_save = {}
authors = list(authors_dict_titles.keys())

# settings to save
authors = [a for a in authors if len(authors_dict_titles[a]) > 2]
gens_per_author = 5
papers_in_context = 5

# run
for i, author in enumerate(tqdm(authors)):
    if not author in authors_save:
        query = prompt + '\n> '.join(authors_dict_titles[author][-papers_in_context:]) + '\n>'
        completion = openai.Completion.create(
            engine="text-davinci-002", prompt=query,
            n=gens_per_author, stop='>'
        )
        authors_save[author] = [completion.choices[i].text for i in range(len(completion.choices))]
    if i % 200 == 0:
        pkl.dump(authors_save, open(f'gen_titles/authors_save_{i}.pkl', 'wb'))
pkl.dump(authors_save, open(f'gen_titles/authors_save_full.pkl', 'wb'))