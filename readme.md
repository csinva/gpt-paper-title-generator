<h1 align="center">
	Auto-generating paper titles
</h1>

## Paper title by author (gpt3 text-davinci-002)

In one experiment, we ask gpt3 to generate title for different authors following the following template:

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

## Finetuned paper title generation (gptneo 2.7B)

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

## Finetuned paper title generation (gpt2, from 2019 AKA "Back in the day")

Well, all the cool kids seem to be training their own text bots so here's one which finetunes gpt-2 to generate titles of scientific papers (or anything else). All code and instructions are in [gpt2/scrape_finetune_sample.ipynb](gpt2/scrape_finetune_sample.ipynb). Works with python 3.6 and tf 1.15.0.

**Data details**
- all title-generating models are trained by finetuning the 117M model
- **ml**: Trained for 1000 steps on article titles from stat.ML betweem 2017-08-01 and 2019-07-01 with 'learning' in the abstract.
- **neuro/genomics**: Trained for 100 steps on 2,870 article titles from arXiv q-bio GN (genomics) and q-bio NC (neurons and cognition) between 2016-08-01 and 2019-07-01.
- **quantum**: Trained for 150 steps on a couple thousand article titles from arXiv quant-ph between 2019-05-01 to 2019-07-03.
- [witty quotes](https://raw.githubusercontent.com/akhiltak/inspirational-quotes/master/Quotes.csv) (note: some of the produced samples may be offensive)
- **tweets**: Trained on the super funny tweets by [@SICKOFWOLVES](https://twitter.com/SICKOFWOLVES) (uses 355M model)
- **kdrama plots**: Trained on [kdrama synopsis first paragraphs](https://en.wikipedia.org/wiki/List_of_South_Korean_dramas)

**Samples**

Here are some samples for ml titles (more in the [samples](samples) folder, e.g. [quantum titles](samples/samples_quantum/all.txt), [kdrama synopses](samples/samples_kdrama_synopses/all.txt), [witty quotes](samples/samples_witty_quotes/all.txt))

- A Machine Learning Framework For Computer Vision
- A Semi-Supervised Csi-On-Icb Test For Differential Privacy
- Neural Control Variates For Efficient Inference And Meaningful Decision Making
- Unlearn What You Have Learned: Recurrent Neural Networks For Pre-Processing Speech Using Sparsified Emotional Speech
- Learning Representations For Long Short-Term Memory Linked Index Circuits
- Stochastic Gradient Mcmc For Compressing Neural Network Training For Training Fine-Grained Relationships
- Bayesian Inference And Wavelet Decomposition Of Iterative Random Code
- Robust And Parallel K-Svm Through Sparsely Correlated Matrix Decomposition
- Towards A Practical $K$-Dimensional Deep Neural Network Model Parameterization Theory Emphasizing Sequence Interaction
- Deep Signal Recovery With Dual Momentum: A Complexity And Stability Analysis Based Analysis
- Fault Diagnosis Using Deep Signal Recovery
- Sparse Least Squares Regression: Robust Regularization And Recommendations For Improved Statistical Validation
- Stacking With A Neural Network For Sequential Intent Classification
- Deep Residual Auto-Encoders For Multi-Label Image Classification
- Sparse Least Squares Regression For Robust And Adaptive Classification
- Anatomical Coronary Stearage Reconstruction Using Deep 3D Convolutional Neural Networks
- Deep Neural Processes
- Gated-Cgroup Communications Solution For Ciliary Dysfunction Detection Using Hierarchical Directed Acyclic Graph Convolutional Network
- Understanding Batch Normalization
- Improving Gan Performance With Stochastic Gradients And More Via Optimized Alternative Estimation
- Graphoisheter Encephalization Encephalization
- Improving Gans Using Covariate Shift And Multivariate Spatial Embeddings
- Dynamic Event Graph Convolutional Networks For Adverse Event Forecasting
- Fast Asynchronous Parallel Training Of Deep Networks
- On The Non-Parametric Power Of Logistic Regression For Smooth Events
- Unifying Pac And Learning Mdps Using Influence Functions
- Machine Learning To Plan And Downlink Using Intrinsic Motivation
- Classifier Readiness Testing For Imbalanced Data
- Fast And Scalable Bayesian Deep Learning With Limited Observations
- A Comparison Of Deep Neural Networks And Adaptive Graph Neural Networks For Anomaly Detection
- Distributed Deep Learning With Gossip Networks Using Bidirectional Lstm Sensors
- Revisiting Reuse Of Super Categories
- Anatomical Visual Exploration
- Multimodal Social Learning With Active Interest Discovery
- Stochastic Variance-Reduced Cubic Regularization For Approximate Inference
- Predicting County Level Corn Yields Based On Time Series Data
- A Deep Residual Network Approach For Predicting County Level Eegs Using Sparse And Incomplete Data

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