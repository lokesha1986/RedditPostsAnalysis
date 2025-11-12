# Multi-class mental health classification on social media using ensemble transformer models and emoji sentiment analysis

## Description
This repository contains exploratory analysis, model comparisons, and ensemble methods applied to Reddit posts data. 
The goal is to investigate how different transformer-based and ensemble architectures perform on NLP tasks such as classification or sentiment analysis using Reddit data.
A collection of Jupyter Notebooks exploring Reddit posts using advanced NLP and deep learning methods: ensembles, BERT, RoBERTa, and voting models.

The repository includes:
- avgensemble.ipynb → Implements average-based ensemble of models.
- bertvsensemble.ipynb → Compares standalone BERT model vs ensemble methods.
- robertavsensemble.ipynb → Compares RoBERTa model vs ensemble approaches.
- rpbertkaggle1.ipynb → Kaggle-style pipeline using Reddit posts with BERT.
- rprobertakaggle2.ipynb → Kaggle-style pipeline using Reddit posts with RoBERTa.
- voting-ensemble2.ipynb → Voting ensemble combining multiple model predictions.

## Dataset Information
- Dataset used: https://www.kaggle.com/datasets/lokeshaweerasinghe/reddit
- Data preprocessing steps are included in the notebooks.
- Ensure that you have access to the dataset before running the notebooks.

## Code Information
- Code is implemented in Python using Jupyter Notebooks.
- Includes model training, validation, and evaluation pipelines.
- Provides comparisons between baseline RoBERTa and ensemble approaches.

## Usage Instructions
1. Clone this repository:
    git clone https://github.com/lokesha1986/RedditPostsAnalysis.git
    cd RedditPostsAnalysis
   
2. Launch Jupyter:

       jupyter notebook

3. Open any notebook (e.g., bertvsensemble.ipynb) and run all cells.
4. Run the cells step by step to preprocess data, train the models, and evaluate results.

## Requirements
Requirements:
- Python 3.7 or above
- Jupyter Notebook / JupyterLab
- Libraries: transformers, torch, scikit-learn, pandas, numpy, matplotlib, seaborn (optional)
- GPU support (CUDA) is recommended for training.

1. Materials & Methods

**Computing Infrastructure**

- **Operating System:** Ubuntu 22.04 LTS (Linux-based environment)
- **CPU:** Intel(R) Core i7 / AMD Ryzen 7 or higher
- **GPU:** NVIDIA Tesla T4 (used in Google Colab) or local CUDA-compatible GPU
- **RAM:** Minimum 16 GB recommended (Colab Pro or equivalent)
- **Python Version:** 3.8+
- **Notebook Environment:** Jupyter Notebook / Google Colab

**Software Libraries**

The experiments were performed using the following Python packages:
numpy
scipy
matplotlib
pandas
scikit-learn
torch
transformers
jupyter
notebook
seaborn
tqdm

** Methods **

1. **Data Preprocessing** – Cleaning and structuring the dataset for text based analysis.
2. **Model Development** – Fine-tuning RoBERTa for classification tasks.
3. **Ensemble Methods** – Combining multiple models to improve robustness and accuracy.
4. **Evaluation** – Metrics such as accuracy, precision, recall, and F1-score are computed.
5. **Comparison** – Performance of RoBERTa vs. ensemble models is analyzed.

## License
This repository is licensed under the MIT License.

## Contribution Guidelines
Contributions are welcome!

You can contribute by:
- Adding new transformer models (e.g., XLNet, DistilBERT, GPT).
- Improving preprocessing or evaluation pipelines.
- Enhancing visualization or adding interpretability tools.
- Adding a unified Python script for batch runs.

Steps:
1. Fork the repository
2. Create a new branch (`git checkout -b feature-branch`)
3. Make your changes and commit (`git commit -m "Added new feature"`)
4. Push and submit a Pull Request


