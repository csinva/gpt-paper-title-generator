## Author-specific paper titles (prompting gpt3 text-davinci-002)

To generate author-specific titles, we use the following template:

```
Here is a list of related machine-learning papers:\n\n> 
> [title 1]
> [title 2]
...
> [title 5]
> ____
```
The model must then fill in the blank. We take the five titles to be the most recent five titles from an author, with the most recnet coming last.


[samples/gpt3/authors_save_full.json](samples/gpt3/authors_save_full.json) contains a list of 5 generated papers for **all authors with atleast 3 arXiv AI papers** (cs.ML, cs.LG, stat.ML). For example, here are the entries for a few authors:

```
yoshua bengio
	 Semi-Supervised Reinforcement Learning with Deep Representation Learning
	 Centralized training of deep neural networks with low communication overhead
	 Context-aware politeness: A learning-based approach
	 Neural Variational Inference and Learning in Boltzmann Machines
	 Watch Your Step: Guarding Model-Based Reinforcement Learning from Step-wise
chelsea finn
	 Neural-Symbolic Virtual Machines
	 Scalable Inverse Reinforcement Learning for Street Networks
	 Neural Program Repair
	 Neural Module Networks
	 Neural Networks for Pattern Recognition
joshua tenenbaum
	 Neural Symbolic VLIW Architectures for Programmable Inference
	 Understanding Overfitting in Deep Learning
	 Deep Learning for Physical Processes: Integrating Physics and Data-Driven Methods
	 Deep Generative Models for Physical Scene Understanding
	 Joint Optimization of Multiple Objectives withpolicy gradient Methods
demis hassabis
	 How transferable are features in deep neural networks?
	 User modelling in dialog systems: A survey
	 Learning to play Go from scratch by self-play reinforcement learning
	 Multiagent collaboration understaing with inverse reinforcement learning
	 Achieving deep and structured representation learning in online problems
jianfeng gao
	 Pre-trained Transformer Models for Language Understanding
	 A Neural Retriever for Context-Dependent Semantic Completion
	 Retrofitting Distributed Representations for Zero-Shot Cross-lingual Entity Link
	 Achieving Zero Shot Learning via Pre-trained Language Models
	 Modeling Adjacencies for Efficient Contextual Representation Learning
```

Note: model used was GPT `text-davinci-002` on Oct 14 2022. It likely was not up to date with the most current advances and could be improved with finetuning on more recent titles. We explore that below with a smaller model:

## Finetuned paper title generation (Finetuned, gptneo 2.7B)

**Model**
- finetunes starting from the [gpt-neo-2.7B checkpoint](https://huggingface.co/EleutherAI/gpt-neo-2.7B)
    - for training details see [the training script](https://github.com/csinva/gpt-paper-title-generator/blob/0157f26be9b0763b4ea6480e5b149fdb8dff4626/gptneo/02_finetune_hf.py)
- inference
    - should prepend with a year and two newlines before querying for a title, e.g. `2022\n\n`

**Data**
- all [papers on arXiv](https://www.kaggle.com/datasets/Cornell-University/arxiv) in the categories cs.AI, cs.LG, stat.ML
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

These samples are considerably improved over the samples we made with GPT2 <a href="https://csinva.io/gpt-paper-title-generator/web/gpt2">back in 2019</a>.

## Reference

- troubleshooting: if you get an error, it might help to run `export PYTHONIOENCODING=ASCII` before finetuning
- uses [gpt-2 simple](https://github.com/minimaxir/gpt-2-simple)
- uses [arxiv-scraper](https://github.com/Mahdisadjadi/arxivscraper)
- uses [tweetscraper](https://gist.github.com/yanofsky/5436496)
- website based on this [example](https://codepen.io/michaeltombor/pen/yoMrMj)
- adorable robot from [here](http://pngimg.com/uploads/robot/robot_PNG94.png)
- other things to train on
    - limericks
    - haikus
    - song lyrics