
import pandas as pd
import os.path
import re

def get_metadata():
    if os.path.exists('../data/arxiv_metatadata_2022_clean.pkl'):
        df = pd.read_pickle('../data/arxiv_metatadata_2022_clean.pkl')
    else:
        df = pd.read_pickle('../data/arxiv_metadata_2022.pkl')
        # df['update_date'] = pd.to_datetime(df['update_date'])
        df['date'] = pd.to_datetime(df['versions'].apply(lambda x: x[0]['created'])) # converting to datetime takes a while
        df = df.drop(columns=['versions', 'update_date'])
        df = df.drop_duplicates(subset=['title', 'authors'], keep='first') # replicated submissions
        # df['num_versions'] = df['versions'].apply(len)
        df['year'] = pd.DatetimeIndex(df['date']).year
        def clean_title(s):
            s = s.replace('\n', '')
            s = s.replace('\t', '')
            s = re.sub(' +', ' ', s)
            return s
        df['title'] = df['title'].apply(clean_title)
        df['title_len'] = df['title'].str.split(' ').apply(len)
        df = df[~df['title'].str.lower().str.startswith('comment')]
        df = df.sort_values(by='date') # most recent papers at bottom
        df = df.to_pickle('../data/arxiv_metatadata_2022_clean.pkl')
    return df


def generate_samples(model, tokenizer, prompt='2020\n\n', num_return_sequences=1):

    # decode some examples
    eos_token_id = tokenizer('\n')['input_ids'][0]
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    outputs = model.generate(
        **inputs,
        do_sample=True,
        eos_token_id=eos_token_id,
        max_new_tokens=25,
        num_return_sequences=num_return_sequences,
        # temperature=0.1,
    )
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)