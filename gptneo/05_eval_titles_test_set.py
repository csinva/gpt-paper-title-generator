import pandas as pd
import numpy as np
import re
import string
import itertools
import os.path
import pickle as pkl
import json
from eval import title_bleu
from tqdm import tqdm
import matplotlib.pyplot as plt
years = ['2022', '2023',
         '2020', '2021', '2010', '2024', '2030', '2050']
for year in years:

    # load titles
    titles_df = pd.read_csv(
        f'../samples/gptneo/{year}/titles.txt', sep='\n\n', header=None)
    titles_gen = titles_df[0].str.strip('\"\' ')
    titles_len = titles_df[0].str.split().apply(len)

    titles_new_df = pd.read_pickle('../data/df_test_recent.pkl')
    titles_new = titles_new_df.title

    # plt.hist(titles_len)
    # plt.xlabel('Title num words')
    # plt.ylabel('Count')
    # plt.show()

    # Look for exact matches
    # For this to be kosher, need to make sure the model never saw
    # any titles within the dates of the test set.
    titles_overlap = []
    tg = titles_gen.str.lower().str.strip().values
    tn = titles_new.str.lower().str.strip().values
    print('# generated titles', tg.size, '# GT titles', tn.size)

    # exact matches
    for title_gen in tg:
        if title_gen in tn:
            print(title_gen)
    print('# exact matches', len(titles_overlap))

    # closest matches ##################################
    bleus = np.zeros((tg.size, tn.size))
    for r, title_gen in enumerate(tqdm(tg)):
        for c, title_new in enumerate(tn):
            bleus[r, c] = title_bleu(title_gen, title_new)

    # find top matches
    matches = np.argmax(bleus, axis=1)
    bleu_matches = bleus[np.arange(bleus.shape[0]), matches]
    args = np.argsort(bleu_matches)[::-1]

    closest_matches = {
        'tg': tg[args],
        'tn': tn[matches[args]],
        'bleu': bleu_matches[args],
        'bleus_arr': bleus,
    }
    pkl.dump(closest_matches, open(f'closest_matches_{year}.pkl', 'wb'))

    N = 50
    for i in range(N):
        print(i)
        print('\t', closest_matches['tg'][i])
        print('\t', closest_matches['tn'][i])
        print('\t', closest_matches['bleu'][i])
