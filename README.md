
# top-choice

## INTRODUCTION

The author of the repo is Brian Willett (bmwillett1 at gmail).

## DESCRIPTION

A product recommendation system is an algorithm which determines for each user a set
of products or services which they would like to purchase or interact with.  Typically these
rely on a large dataset of users and products, for example, a history of
past interactions between users and products.  A common approach to this problem is
to use a machine learning algorithm trained on this dataset.  In particular, 
modern approaches often use neural networks and deep learning techniques to 
achieve impressive accuracy.

A potential pitfall of these techniques occurs when a user has some behavior that
lies outside their normal preferences.  For example, a friend may watch a
video on your streaming account, or you may misclick on an advertisement you were not really 
interested in.  The recommendation system may then be led to suggest products based on 
this behavior that are not desired by the user.  These outliers can be interpreted as 
"noise" in the user-product dataset, and it is desirable for the recommendation algorithm to 
be somewhat robust against this noise.

Many techniques have been developed for improving the performance of machine-learning
algorithms on noisy data.  In particular, topological data analysis (TDA) uses mathematical analysis
of patterns in data to extract features of data robust to small variations such as noise.  In 
particular, we will be interested in the Mapper-Classifier algorithm (MCA) of [arXiv:1910.08103](https://arxiv.org/pdf/1910.08103.pdf), 
which uses the concept of a Mapper graph from TDA to achieve improved robustness in image classification.  We will apply this algorithm to determine which products a user is likely to be interested in.
 
In this repo we implement some models with and without the MCA to evaluate
the performance of the recommendation systems in the presence of noise.  Concretely, we will
focus on the Instacart dataset used in a previous Kaggle challenge.  Our task will be
to predict what products a user reorders given their previous orders.

Let us now briefly describe the models, including an outline of the MCA.

### Models

BW: include pictures of models

### MCA

BW: borrow picture
 
 
 


## DIRECTORY STRUCTURE

```
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




