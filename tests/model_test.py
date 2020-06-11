"""
*********************************

IDEA FOR BASIC CLASSIFIER-BASED MODELS:

MODEL 1:
   input:
    concatenate the following:
    - user feature/latent vector (if any)
    - product feature/latent vector (if any)
    - user/product interaction (decide)
    hidden layers:
     - dense neural network
     output:
      logit of desirability of product

    usage:
    - given user and list of products, feed through (in batches) to get desirability
    - take top N choices

MODEL 2:
    - first create product latent space
    - given a user, look at all (projections of) products as train set
    - use to predict their preference on unseen products

*********************************

INCLUDING MAPPER-CLASSIFIER

- basic idea would be that it would be plugged in at certain stage in place of a NN classifier
- other ideas that more intrinsically use it?

*********************************

MORE DETAILS FOR INSTACART:

 - latent/feature space for products
    - just department
    - just tf-idf or other simple NLP from product name
    - just word2vec model
    - combination
 - user features -> None
 - user/product:
    - autoencoder?
    - PCA
    - matrix fact?

*********************************

TESTS:

 - what to compare:
    - model with ordinary classifier
    - model with topological classifier
    - gradient boost/etc.
 - accuracy:
    - test/val/train split
    - MAE, NDCG?
 - robustness
    - accuracy after changing random items


*********************************

DELIVERABLE:

 - streamlit app
  - search products, add to order
  - present 3 rating engines
 - what else?


*********************************

TO DO NEXT:

 - set up simple model with NN classifier
 - see how hard to plug in MC
 - set up code to run tests
 - set up simple streamlit demo
 - set up training on AWS (after getting approved)
 - find other baselines/metrics
 - refine model
 - refine deliverable

"""

import sys

sys.path.append('../lib/')

from latent_space import *
from process_data import *
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

verbose=0
IC_DATA_DIR = '../data/instacart_2017_05_01/'
SMALL_IC_DATA_DIR = '../data/instacart_2017_05_01_small/'

# MODEL 2

# load data
order_data, product_data = instacart_process(data_dir=SMALL_IC_DATA_DIR)
product_ids = order_data.product_id.unique()
user_ids = order_data.user_id.unique()

# split data
order_data_train, order_data_test = first_last_split(order_data)

# get product latent model (modify to take processed data above)
model = latent_model(order_data_train, product_data, retrain=False)
ordered_products = list(model.wv.vocab.keys())

# run trials (for now only one)
# n_trials=1
# for i in range(1,n_trials+1):
i = 1

# select a user and get list of products
selected_user_id = np.random.choice(user_ids)
user_order_data_train = order_data_train[order_data_train.user_id == selected_user_id]
user_order_data_test = order_data_test[order_data_test.user_id == selected_user_id]

user_products_dict = user_order_data_train.groupby('product_id')['product_id'].count().to_dict()
user_products = list(user_products_dict.keys())

print("trial %d: user %d bought %d products" % (i, selected_user_id, len(user_products)))

max_ordered = max(user_products_dict.values())
X = np.array([model.wv[str(prod)] for prod in user_products])
y = np.array([user_products_dict[int(prod)] / max_ordered for prod in user_products])

# add one spurious item for adversarial test
user_products_dict_adv = user_products_dict.copy()
user_products_dict_adv[int(np.random.choice(ordered_products))] = 1 / max_ordered
X_adv = np.array([model.wv[str(prod)] for prod in user_products_dict_adv.keys()])
y_adv = np.array([user_products_dict_adv[int(prod)] / max_ordered for prod in user_products_dict_adv.keys()])

# generate test data from user's last order
user_products_dict_test = user_order_data_test.groupby('product_id')['product_id'].count().to_dict()
user_products_test = list(user_products_dict_test.keys())

max_ordered = max(user_products_dict_test.values())
X_test = np.array([model.wv[str(prod)] if str(prod) in ordered_products else np.zeros(100) for prod in user_products_test])
y_test = np.array([user_products_dict_test[int(prod)] / max_ordered for prod in user_products_test])

X_all = np.array([model.wv[prod] for prod in ordered_products])

# train neural net on normal data
batch_size = 16

clf = Sequential()
clf.add(Dense(100, activation='relu'))
clf.add(Dense(50, activation='relu'))
clf.add(Dense(1, activation='sigmoid'))

clf.compile(loss=keras.losses.categorical_crossentropy,
            optimizer='adam',
            metrics=['accuracy'])

clf.fit(X, y,
        batch_size=batch_size,
        epochs=5,
        validation_split=0.2)

# find rankings of products in test set
preds = clf.predict(X_all)
ranking = np.argsort(preds, axis=None)[::-1]
print("out of %d products, items in last order ranked:" % len(ranking))
print([np.where(ranking==prod) for prod in user_products_test])

## mapper classifier tests
mapper = MapperClassifier(n_components=n_components, NRNN=NRNN)


# to do right now:
# X get latent model working ([product_ids] in, [latent_vectors] out)
# X write function to make X and y for tensorflow model
# X set up simple tensorflow neural network as follows:
# X input = product latent vector
# X some hidden layers
# X output = single logit
# - try to get to overfit?
# X test:withhold 10% of products of user
# - look for metrics
# - get working with MC
# - try a non-NN method as baseline
# - set up robustness tests
# - put in streamlit app