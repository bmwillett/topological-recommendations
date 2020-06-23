
# topological recommendations

## INTRODUCTION

The goal of this project is to create a recommendation system which makes use of recent advances in topological data analysis.  Specifically, we will use the Mapper-Classifier algorithm (MCA) of [arXiv:1910.08103](https://arxiv.org/pdf/1910.08103.pdf) to suggest products to users in a way that is robust against outlier user purchases.

The author of the package is Brian Willett (bmwillett1 at gmail).


## DIRECTORY STRUCTURE

```console
├── README.md 
├── requirements.txt
├── data
│   ├── instacart_small - small version of instacart dataset for running tests
├── lib
│   ├── data_class.py - class for product datasets
│   ├── mapper_class.py - Mapper-based classifier algorithm
│   ├── process_data.py - helper methods for loading data
│   └── tools.py - general helper methods
├── models
│   ├── base_model.py - base class for all models
│   ├── baseline_models.py - standard algorithms for baseline comparison
│   ├── latent_models.py - models to encode user/products in latent space, used for topological embeddinng
│   └── main_models.py - main models incorporating mapper-based classifier
└── tests
    ├── model_test.py - runs simple tests of models
    └── test_notebook.ipynb
```

## SETUP

To install top-choice package:

- In the command line, run:
```console
pip install top-choice
```

- To run tests in this repo:

```console
git clone https://github.com/bmwillett/topological-recommendations
pip install -r requirements.txt
python tests/runtests.py
```

## DESCRIPTION



