# auto-generating paper titles

A textbot which finetunes gpt-2 to generate titles of ml papers. Main code in [scrape_finetune_pred.ipynb](scrape_finetune_pred.ipynb).

Trained by finetuning the [117M GPT-2 model](https://openai.com/blog/better-language-models/) for 1000 steps on article titles from stat.ML between 2017-08-01 and 2019-07-01.

# samples
see more in [samples](samples) folder

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

# reference

- uses [gpt-2 simple](https://github.com/minimaxir/gpt-2-simple)
- uses [arxiv-scraper](https://github.com/Mahdisadjadi/arxivscraper)
- app based on [this tutorial](https://towardsdatascience.com/develop-a-nlp-model-in-python-deploy-it-with-flask-step-by-step-744f3bdd7776)
- website based on this [example](https://codepen.io/michaeltombor/pen/yoMrMj)