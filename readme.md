# Auto-generating paper titles

## Paper title by author (gpt3 text-davinci-002)

In one experiment, we ask gpt3 to generate title for different authors following the following template:


[samples/gpt3/authors_save_full.json](samples/gpt3/authors_save_full.json) contains a list of 5 generated papers for **all authors with atleast 3 arXiv AI papers** (cs.ML, cs.LG, stat.ML). For example, the first few entries are:

```
"yoshua bengio": [" Semi-Supervised Reinforcement Learning with Deep Representation Learning\n", " Centralized training of deep neural networks with low communication overhead", " Context-aware politeness: A learning-based approach\n", " Neural Variational Inference and Learning in Boltzmann Machines\n", " Watch Your Step: Guarding Model-Based Reinforcement Learning from Step-wise"], 

{"yang liu": [" Models of Explanation for Deep Neural Networks", " Neural Networks for Surrogate Modeling: A Comparative Review\n", " A Study on Generative Adversarial Nets with Wasserstein Distance\n", " On the Universality of Deep Learning\n", " Representation Learning for Time-Series Classification\n"], "sergey levine": [" Combining Model-Based and Model-Free Updates in Deep Reinforcement Learning\n", " A Neural Algorithm of Artistic Style", " Quadrupedal Locomotion by Reinforcement Learning", " RLPy: A Multi-platform Toolkit for Reinforcement Learning", " Efficient Exploration in Deep Reinforcement Learning via Bootstrapped DQN\n"], "bo li": [" Segmenting out Task-Independent Representations in Multi-Task Learning", " Measuring Representational Similarity in Deep Neural Networks\n", " ...", " On the Convergence of Gradient Descent for Deep Linear Neural Networks\n", " On Certified Adversarial Defenses for Deep Neural Networks\n"], "pieter abbeel": [" hierarchicalrl: A Framework for Multi-Level Reinforcement Learning\n", " Investigate Neural Representation for Sequential Decision Making Tasks", " Context-Aware Neural Machine Translation\n", " A Simple Approach for Time-Varying Graphical Models\n", " Tracking the World State with Neural Maps"], "dacheng tao": [" Accurate Uncertainties for Deep Learning using Stochastic Gradient Flow\n", " Understanding Generalization in Deep Learning: A Survey\n", " Projecting Implicit Generative Models", " Relaxed Label Propagation for Semi-Supervised Learning\n", " Towards Understanding the Effectiveness ofdeep Convolutional Neural Networks for Stock Prediction"], "jun wang": [" NCF: Neural Collaborative Filtering", " Guided Cost Learning: Deep Inverse Optimal Control via Policy Optimization\n", " Neural-Network Based Container Allocation in Unikernels\n", " Bridging the Gap Between Off-Policy and On-Policy Reinforcement Learning", " Coverage-Based Neural Machine Translation\n"], "masashi sugiyama": [" Semi-Supervised Learning under Domain Shift\n", " Domain Adaptation from Unlabeled Data with a Generative Adversarial Framework", " Controllable Synonym Replacement for Neural Machine Translation\n", " Learning Translation Equivariant Representations of Images\n", " Distributionally Robust Multi-Armed Bandits with Coverage Constraints"], "jun zhu": [" Neural Collaborative Filtering with Rating Inflation Detection\n", " Learning to Route Traffic in Mixed Real-Time settings\n\n1. On the", " Stochastic Backpropagation for Energy-Based Models\n", " Neural Ordinary Differential Equations for Few-Shot Verbal Representation Learning", " Neural Ordinary Differential Equations with Applications to Learning Hamiltonian Monte Carlo\n"], "yu zhang": [" S3VM: Sparse Support Vector Machine with Stable Soft-threshold", " MIX: A Benchmark and Competition on Explainable AI\n", " I know what you did last summer: Sentiment and phrase-level event prediction", " Achieving Non-Myopic Submodular Maximization over Massive Data with", " Modeling the Effect of Quantum Machine Learning on the Human Brain\n"], "wei wang": [" boosting Tree-based methods for detecting out-of-distribution samples in streaming", " CompressiveGANS: Towards Real-Time 3D Compressive Imaging with Gener", " Robust PCA via Outlier Pursuit\n", " Towards Optimal Sample Complexity in Adversarial Learning\n", " Learning to Map Images to 3D Scenes via Unsupervised Geometry-A"], "shie mannor": [" Reinforcement Learning with Deep Energy-Based Policies\n", " Experience Replay in Distributed Reinforcement Learning\n", " A Neural Network Approach to Automated Machine Teaching\n", " Finally, some studies have proposed using reinforcement learning to tackle the credit assignment problem in", " Reinforcement Learning with Continuous Actions by Policy Gradition\n"], "uwe aickelin": [" Learning from Multiple Domains: A Unified Approach\n", " A Unified Information-Theoretic Framework for Domain Adaptation and Semi-Super", " Information Theoretic Learning for Domain Adaptation: Representation Bounds and Al", " A Survey of Multi-Source and Transfer Learning\n", " From Domain Adaptation to Instance Adaptation: A Survey on Transfer Learning\n"], "tong zhang": [" Hierarchical Reinforcement Learning with the Maximum Entropy Principle\n", " Exploration by Distributional Reinforcement Learning\n", " Contrastive Divergence Learning of Restricted Boltzmann Machines\n", " Reinforcement Learning with Model Misspecification\n", " A Temporal Point Process Approach to Generative Modeling of Time Series\n"], "xin wang": [" Neural Generative Modeling of 3D Molecular Structures\n\nThese papers all", " Adversarial Temporal Augmentation for Zero-Shot Video Action Recognition\n", " Hybridattack: A Novel Composite Method for Generating Physical Adversarial Examples with", " Automatic Musical Genre RecognitionUsing Convolutional Neural Networks", " An Energy-based Model for Deep Learning with Sparse Connectivity\n"], "max welling": [" Discovering meaningful visual patterns in the wild with t-SNE\n", " Hierarchical Graphical Models for Efficient 3D Body Pose Tracking\n", " Rational Approximations for Deep Learning\n", " Tractable Propagation for Bayesian Deep Learning\n", " Reweighted Wake-Sleep for Energy-Based Models\n"], "kyunghyun cho": [" Latent Variable Language Models\n", " Generalizing from a Single Demonstration through Meta-learning\n", " Semi-Supervised Learning with Compact Prototypes\n", " Scalable Bayesian Optimization Using Deep Neural Networks\n", " Neural Program Interpreter"], "xi chen": [" Neural Truth Serum: A Survey of Methods for Generating Unbiased Estimates\n", " LMSR-VAE: A Latent Variable Model with Sparse Regular", " C3AE: Contextualized Causal Representation Learning for Action Effect Prediction", " Adversarial One-hot Selection for GANs\n", " Classifying Uncertain Inputs with Deep Probabilistic Neural Networks\n"], "stefano ermon": [" Conditioned Similarity Networks for Image Retrieval\n", " A Deep Learning Framework for Forest Biomass Monitoring Using Satellite Imagery\n", " Neural Processes as Temporal Kernel Machines", " Variational Approaches for Joint Wavelength and Spatial Resolution Recovery\n", " A General-Purpose Similarity Function on Symbolic Sequences\n"], "andreas krause": [" Risk-Aware Generative Adversarial Imitation Learning\n", " Neural Network Controllers for Safe Reinforcement Learning\n", " Safe Bayesian Reinforcement Learning\n", " A Unified Framework for Safe Reinforcement Learning and Safe imitation Learning", " Risk-Aware Multi-Armed Bandits\n"], "marcus hutter": [" A Deep Reinforcement Learning Framework for the Financial Portfolio Management Problem\n", " Beating Montezuma's Revenge with Deep Reinforcement Learning\n", " End-To-End Training of Deep Visuomotor Policies\n", " Policy Gradient Methods for Reinforcement Learning with Function Approximation\n", " A Formal Definition of the PAC Model for Reinforcement Learning"], "wei chen": [" Safe Interval Prediction for Stochastic Convex Optimization\n", "Fairness Constraints in Non-Convex Optimization: Regularization", " A Survey of Real-Time Machine Learning", " Multiplicative Bandits with Smooth Components\n", " Achieving Safe Exploration via Randomized Value Functions\n"], "hao wang": [" Efficient Similarity Search on Large Graphs with Graph Neural Networks\n", " Neural Logic Machines", " Neural Tree Edit Distance\n", " Meta-learning for Quadratic Programming with Neural Networks", " Neural Ordinary Differential Equations"], "lawrence carin": [" Neural Relational Inference for Interacting Systems\n", " Latent Variable Modeling of Neural Data with Gaussian Processes", " Multivariate LSTM-FCNs for Time Series Classification\n", " A Meta-Learning Approach for Low-Resource Cross-Lingual Named Entity", " Neural Designs for Learning Algorithms\n"], "yang li": [" Learning to Manage Energy Consumption of Mobile Applications\n", " Neural Discrete Representation Learning\n", " Learning to Interpret Body Language\n", " On the Personalization of Human-Centered Machine Learning Interfaces", " Neural Text-to-Speech Synthesis with Meta-Learning\n"], "qiang liu": [" A Unified optimization Framework for Generative Modeling\n", " On the Transferability of Deep Representations across Medical Imaging Modalities", " Representation Learning for Electronic Health Records\n", " Adversarial User Modeling for Personalized Recommendation\n", " A Unified Flow-based Model for Joint Distribution Matching"], "mohit bansal": [" Neural Abstractive Document Summarization with Query-Completion\n", " A Text-to-Image Synthesis Approach to Paraphrasing with Visual", " GPT-f: A psycholinguistically-inspired language model\n\n", " Improving Abstractive Summarization with Semantic Edit Distance\n", " Learning to Ask Questions: A Neural Question Generator\n"], "jianfeng gao": [" Pre-trained Transformer Models for Language Understanding\n", " A Neural Retriever for Context-Dependent Semantic Completion\n", " Retrofitting Distributed Representations for Zero-Shot Cross-lingual Entity Link", " Achieving Zero Shot Learning via Pre-trained Language Models", " Modeling Adjacencies for Efficient Contextual Representation Learning\n"], "chelsea finn": [" Neural-Symbolic Virtual Machines\n", " Scalable Inverse Reinforcement Learning for Street Networks\n", " Neural Program Repair\n", " Neural Module Networks\n", " Neural Networks for Pattern Recognition"], "mihaela van der schaar": [" Interpretable Prediction of Molecular Subtypes from Breast Cancer Images\n", " Context-Based Explanations for Deep Learning Models\n", " A Unified Framework For Counterfactual Explanations\n", " A Survey of Neural Architectures for Interpretable Machine Learning\n", " \"Why Should I Trust You?\": Explaining the Predictions of Any Class"], "wei zhang": [" Learning Image Embeddings via Marginal Fisher Analysis", " HLat: A Hierarchical Latent Space Model for Representation Learning", " Group representation learning for multi-class image classification", " Improving Generalization Performance by Eliminating Spurious Correlations through Canonical Cor", " Efficient Context Modeling for Scene Recognition with Limited Annotation\n"], "ruslan salakhutdinov": [" Reinforcement Learning with Latent Variable Actions\n", " Neural Representation Learning: A survey", " A Generative Query Network for Task-Oriented RL\n", " End-to-end User Experience Optimization\n", " Learning to Learn in Model-based Reinforcement Learning with Latent Variable Models\n"], "doina precup": [" Safe Imitation Learning from Clinical Examples with Deep Reinforcement Learning\n", " Safe and Efficient Off-Policy Reinforcement Learning through Maximum Causal Entropy", " Optimal transport for learning with noisy target distributions\n", " Action Elimination in Reinforcement Learning\n", " Safe Imitation Learning from Human Preferences\n"], "ping li": [" A Comparison of Regularization Strategies for Deep Learning\n", " Beyond the Perceptron: Multiplicative Update Rules and Polyak Update", " Range-Based Boosting for Regression Tasks\n", " Boosting with the L1-Loss: Regression and Classification", " Pattern discovery in sequential data via the Viterbi algorithm and Bayesian inference\n"], "zhangyang wang": [" Data-Driven Discretization of Continuous-Valued Sensors for E", " Achieving Efficient In-Memory Tensor Processing with Hybrid Compositional T", " Efficient Neural Beam Search for Structured Prediction\n", " Adaptive Neural Networks with Memory for Context-Aware Prediction of Human Traject", " Learning to Generate Reviews from Rating Histograms\n\n1. Partial and As"], "yarin gal": [" A General Framework for Learning Transferable Representations from Unlabeled Data\n", " A Two-Stage Approach for Weakly Supervised Object Detection\n", " Achieving High Generalization Performance via Learning Data Augmentation Policies\n", " Curriculum Deep Learning", " Learning Representations for Multi-Task Reinforcement Learning\n"], "aaron courville": [" Towards Value-based Model-based Reinforcement Learning\n", " Constantly Learning to Predict by the Long Term\n", " Continual Learning with Deep Generative Replay\n", " A Survey of Model-based Reinforcement Learning\n", " Neural Network Interpolation for Rapid Change Detection\n"], "ji liu": [" Reinforcement Learning with Deep Neural Networks: A Survey", " Large Scale Deep Reinforcement Learning with Hybrid Reward Function\n", " A General Framework for Multi-party Computation with Secure Maximum Error\n", " Framework for Scalable and Efficient Accelerated Distributed Deep Learning\n", " A Truncated Backpropagation Algorithm for Deep Networks\n"],
```

## Finetuned paper title generation (gptneo 2.7B)


2022 update -- I guess a lot happened since 2019. Let's see how modern models do at this task...


**Code overview**
- to rerun, first run the code in the `scrape` folder
- then pick one of the model folders (e.g. `gptneo`) and run the notebooks in that folder
    - this will ave results into the `samples` folder

## Finetuned paper title generation (gpt2, from 2019 AKA "Back in the day")

Well, all the cool kids seem to be training their own text bots so here's one which finetunes gpt-2 to generate titles of scientific papers (or anything else). All code and instructions are in [gpt2/scrape_finetune_sample.ipynb](gpt2/scrape_finetune_sample.ipynb). Works with python 3.6 and tf 1.15.0.

### Settings for data gathering

- all title-generating models are trained by finetuning the 117M model
- **ml**: Trained for 1000 steps on article titles from stat.ML betweem 2017-08-01 and 2019-07-01 with 'learning' in the abstract.
- **neuro/genomics**: Trained for 100 steps on 2,870 article titles from arXiv q-bio GN (genomics) and q-bio NC (neurons and cognition) between 2016-08-01 and 2019-07-01.
- **quantum**: Trained for 150 steps on a couple thousand article titles from arXiv quant-ph between 2019-05-01 to 2019-07-03.
- [witty quotes](https://raw.githubusercontent.com/akhiltak/inspirational-quotes/master/Quotes.csv) (note: some of the produced samples may be offensive)
- **tweets**: Trained on the super funny tweets by [@SICKOFWOLVES](https://twitter.com/SICKOFWOLVES) (uses 355M model)
- **kdrama plots**: Trained on [kdrama synopsis first paragraphs](https://en.wikipedia.org/wiki/List_of_South_Korean_dramas)

## Samples

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

# Other things to train on

- limericks
- haikus
- song lyrics


## Reference

- troubleshooting: if you get an error, it might help to run `export PYTHONIOENCODING=ASCII` before finetuning
- uses [gpt-2 simple](https://github.com/minimaxir/gpt-2-simple)
- uses [arxiv-scraper](https://github.com/Mahdisadjadi/arxivscraper)
- uses [tweetscraper](https://gist.github.com/yanofsky/5436496)
- website based on this [example](https://codepen.io/michaeltombor/pen/yoMrMj)
- adorable robot from [here](https://csinva.github.io/gpt2-paper-title-generator/index.html)