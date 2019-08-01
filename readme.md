# auto-generating paper titles

Well, all the cool kids seem to be training their own text bots so here's one which finetunes gpt-2 to generate titles of ml papers. Main code in [scrape_finetune_pred.ipynb](scrape_finetune_pred.ipynb).

Trained by finetuning the 117M model for 1000 steps on article titles from stat.ML betweem 2017-08-01 and 2019-07-01.

# samples
- see more in [samples](samples) folder


```
a machine learning framework for computer vision
a semi-supervised csi-on-icb test for differential privacy
neural control variates for efficient inference and meaningful decision making
unlearn what you have learned: recurrent neural networks for pre-processing speech using sparsified emotional speech
learning representations for long short-term memory linked index circuits
stochastic gradient mcmc for compressing neural network training for training fine-grained relationships
bayesian inference and wavelet decomposition of iterative random code
robust and parallel k-svm through sparsely correlated matrix decomposition
towards a practical $k$-dimensional deep neural network model parameterization theory emphasizing sequence interaction
deep signal recovery with dual momentum: a complexity and stability analysis based analysis
fault diagnosis using deep signal recovery
sparse least squares regression: robust regularization and recommendations for improved statistical validation
stacking with a neural network for sequential intent classification
deep residual auto-encoders for multi-label image classification
sparse least squares regression for robust and adaptive classification
anatomical coronary stearage reconstruction using deep 3d convolutional neural networks
deep neural processes
gated-cgroup communications solution for ciliary dysfunction detection using hierarchical directed acyclic graph convolutional network
understanding batch normalization
improving gan performance with stochastic gradients and more via optimized alternative estimation
graphoisheter encephalization encephalization
improving gans using covariate shift and multivariate spatial embeddings
dynamic event graph convolutional networks for adverse event forecasting
fast asynchronous parallel training of deep networks
on the non-parametric power of logistic regression for smooth events
unifying pac and learning mdps using influence functions
machine learning to plan and downlink using intrinsic motivation
classifier readiness testing for imbalanced data
fast and scalable bayesian deep learning with limited observations
a comparison of deep neural networks and adaptive graph neural networks for anomaly detection
distributed deep learning with gossip networks using bidirectional lstm sensors
revisiting reuse of super categories
anatomical visual exploration
multimodal social learning with active interest discovery
stochastic variance-reduced cubic regularization for approximate inference
predicting county level corn yields based on time series data
a deep residual network approach for predicting county level eegs using sparse and incomplete data
```

# reference

- uses [gpt-2 simple](https://github.com/minimaxir/gpt-2-simple)
- uses [arxiv-scraper](https://github.com/Mahdisadjadi/arxivscraper)

- unfortunately, flask seems to throw an error when loading tensorflow > 1.5
- app based on [this tutorial](https://towardsdatascience.com/develop-a-nlp-model-in-python-deploy-it-with-flask-step-by-step-744f3bdd7776)
- could deploy to heroku following [this tutorial](https://medium.com/the-andela-way/deploying-a-python-flask-app-to-heroku-41250bda27d0)