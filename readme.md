We explore generating new paper titles given past titles on arXiv.
We first explore generating titles conditioned on a specific author (using GPT-3 without finetuning).
We then generate titles conditioned only on their publication year (using GPT-Neo with finetuning)

## Author-specific paper titles (prompting gpt3 text-davinci-002)
To generate author-specific titles, we take the five most recent titles from each author with atleast 3 arXiv AI papers (cs.ML, cs.LG, stat.ML).
We then feed then format the papers using the following template and query for the next title using GPT-3 with the OpenAI API:

```
Here is a list of related machine-learning papers:

> [title 1]
> [title 2]
...
> [title 5]
> ____
```

See the results in [the demo](http://localhost:4000/docs/#demo:~:text=Type%20in%20the%20name%20of%20an%20author%20to%20see%20the%20predicted%20titles%20of%20their%20future%20papers) above or the full results in this [json file](https://github.com/csinva/gpt-paper-title-generator/blob/master/samples/gpt3/authors_save_full.json). 

<img src="http://localhost:4000/docs/figs/author_counts.svg">

As an example, when prompting with these 5 recent titles:

```
> Hierarchical Shrinkage: improving the accuracy and interpretability of tree-based methods
> Group Probability-Weighted Tree Sums for Interpretable Modeling of Heterogeneous Data
> Beyond the Imitation Game: Quantifying and extrapolating the capabilities of language models
> Emb-GAM: an Interpretable and Efficient Predictor using Pre-trained Language Models
> Explaining Patterns in Data with Language Models via Interpretable Autoprompting
> ____
```

We get these 5 (independent) random generations for the blank:

```
1. Towards Interpretable Natural Language Processing: A Survey
2. A Unified Framework for Interpretable Machine Learning
3. Compositional Attention Networks for Machine Reasoning
4. Achieving Open Vocabulary Neural Machine Translation
5. A Deep Understanding of Neural Networks through Deep Visualization
```

The results are often interesting but fall into failure modes where they generate irrelevant titles for an author, often leaning towards popular topics such as deep learning, multi-task learning, and reinforcement learning.
Note: the model used was GPT `text-davinci-002` on Oct 14 2022. It likely was not up to date with the most current advances and could be improved with finetuning on more recent titles. We explore that below with a smaller model:

## Finetuned paper title generation (gptneo 2.7B)

To improve the model, we now turn to finetuning a model specifically for paper-title generation. We start from the [gpt-neo-2.7B checkpoint](https://huggingface.co/EleutherAI/gpt-neo-2.7B) (see the [the training script](https://github.com/csinva/gpt-paper-title-generator/blob/91d8aa78d83f16778a120ec4a3dc41be28f5e8f2/gptneo/02_finetune_hf.py) for hyperparameters). We finetune on all [paper titles on arXiv](https://www.kaggle.com/datasets/Cornell-University/arxiv) in the categories cs.AI, cs.LG, stat.ML. However, we exclude some titles for testing: we exclude all papers after Apr 1, 2022 and an additional random 5\% of titles. Note that papers are very skewed towards recent years:

<img src="http://localhost:4000/docs/figs/paper_metadata.svg">

We also exclude titles with a length of less than 6 words or greater than 20 words. This results in 98,388 papers for finetuning. 





- inference
    - should prepend with a year and two newlines before querying for a title, e.g. `2022\n\n`

**Data**
- 
    - date cutoff: only finetuned on papers with dat on or before Apr 1, 2022
    - random 5% of papers also excluded
    - this results in 98,388 papers for finetuning
- during finetuning each paper title was given starting with the prompt `<year>\n\n <title>\n` (e.g. `2022\n\n Emb-GAM: an Interpretable and Efficient Predictor using Pre-trained Language Models\n`)

**Code**
- to rerun, first run the code in the `scrape` folder
- then pick one of the model folders (e.g. `gptneo`) and run the notebooks in that folder
    - this will ave results into the `samples` folder

**Inference example**
```python
from transformers import AutoModelForCausalLM, pipeline, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("csinva/gpt-neo-2.7B-titles")
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")
pipe = pipeline('text-generation', model=model, tokenizer=tokenizer)
pipe('2022\n\n')
```

These samples are considerably improved over the samples we made with GPT2 [back in 2019)(https://csinva.io/gpt-paper-title-generator/web/gpt2).

## Todos



## Reference

- troubleshooting: if you get an error, it might help to run `export PYTHONIOENCODING=ASCII` before finetuning
- uses [gpt-2 simple](https://github.com/minimaxir/gpt-2-simple)
- uses [arxiv-scraper](https://github.com/Mahdisadjadi/arxivscraper)
- uses [tweetscraper](https://gist.github.com/yanofsky/5436496)
- website based on this [example](https://codepen.io/michaeltombor/pen/yoMrMj)
- adorable robot from [here](http://pngimg.com/uploads/robot/robot_PNG94.png)