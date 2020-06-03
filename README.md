
# topological recommendations

## INTRODUCTION

The goal of this project is to create a recommendation system which makes use of recent advances in topological data analysis.  Specifically, we will use the Mapper-Classifier algorithm (MCA) of [arXiv:1910.08103](https://arxiv.org/pdf/1910.08103.pdf) to suggest products to users in a way that is robust against outlier user purchases.


The author of the package is Brian Willett (bmwillett1 at gmail).


## (temporary) PROJECT OUTLINE BRAINSTORMING: 

- Goal: recommendation system that:
    - predicts users next purchases based on previous orders, and/or
    - suggests related products to a given product
    - other tasks?
- Datasets:
    - [The Instacart Online Grocery Shopping Dataset 2017](https://www.instacart.com/datasets/grocery-shopping-2017)
    - experiment with others later... 
- Simpler approaches (baselines):
    - gradient boost trained on order data
    - clustering (k-means or dbscan) to group similar items
- Novel approach:
    - model 1: embeds products in latent space encoding product similarity
        - use NLP on product/aisle/department text data
        - train VAE or word2vec on sequences of products appearing in orders
    - model 2a (user recommendations): 
        - given set of products bought by user, embed in latent space
        - train MCA to classify products into ones user likes/dislikes
            - concerns: too little data? class imbalance?
    - model 2b (product recommendations):
        - recommend similar products based on latent space embedding
            - concerns: how to incorporate MCA into this task?
- Metrics:
    - split given user/order data into train/test/val
    - use test set user orders to evaluate model
    - compare baseline and novel approaches
- Deliverables:
    - Python package with MCA algorithms built in
        - what to present in demo
        - how to incorporate ML infrastructure (AWS, Docker, etc.)
    - API to serve predictions
       
   
## (temporary) TO DO
- decide on concrete goals (ie, user vs product recommendations)
- get baseline models working (mostly done)
- get MCA algorithm up and running (in progress)
- experiment more with latent space embeddings (fair progress)
- combine latent space/MCA to get predictions 

## DIRECTORY STRUCTURE

- To be added

## SETUP

- To be added
        




