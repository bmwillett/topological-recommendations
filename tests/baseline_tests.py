import sys

sys.path.append('./lib')
sys.path.append('./models')

from process_data import instacart_process
from baseline_models import *

"""
MVP code organization:

- load instacart dataset
- get latent space encoding
- use to recommend similar products

- process data into standard form:
- user_id
    - for each user, list of order_id's
        - for each order, list of product_id's
- product_id
    - for each product_id, relevant text data

"""
# baseline tests


verbose=0
data_dir = '../data/instacart_2017_05_01_small/'

df_train, labels, df_val, labels_val = get_features(0,0,0, IDIR=data_dir, verbose=verbose)
print(df_train.columns)
# logistic regression
df_val, labels_val, preds = logistic_model_predict(df_train, labels, df_val, labels_val, verbose=verbose)


df_val['labels'] = labels_val
df_val['preds'] = preds
pd.set_option('display.max_columns', None)

true_pos = df_val[(df_val.labels == 1) & (df_val.preds > 0.22)].shape[0]
false_pos = df_val[(df_val.labels == 0) & (df_val.preds > 0.22)].shape[0]
true_neg = df_val[(df_val.labels == 0) & (df_val.preds <= 0.22)].shape[0]
false_neg = df_val[(df_val.labels == 1) & (df_val.preds <= 0.22)].shape[0]

LOG_prec = true_pos / (true_pos + false_pos)
LOG_rec = true_pos / (true_pos + false_neg)

# random forest
df_val, labels_val, preds = random_forest_model_predict(df_train, labels, df_val, labels_val, verbose=verbose)

df_val['labels'] = labels_val
df_val['preds'] = preds
pd.set_option('display.max_columns', None)

true_pos = df_val[(df_val.labels == 1) & (df_val.preds > 0.22)].shape[0]
false_pos = df_val[(df_val.labels == 0) & (df_val.preds > 0.22)].shape[0]
true_neg = df_val[(df_val.labels == 0) & (df_val.preds <= 0.22)].shape[0]
false_neg = df_val[(df_val.labels == 1) & (df_val.preds <= 0.22)].shape[0]

RF_prec = true_pos / (true_pos + false_pos)
RF_rec = true_pos / (true_pos + false_neg)

# gradient boost
df_val, labels_val, preds = random_forest_model_predict(df_train, labels, df_val, labels_val, verbose=verbose)

df_val['labels'] = labels_val
df_val['preds'] = preds
pd.set_option('display.max_columns', None)

true_pos = df_val[(df_val.labels == 1) & (df_val.preds > 0.22)].shape[0]
false_pos = df_val[(df_val.labels == 0) & (df_val.preds > 0.22)].shape[0]
true_neg = df_val[(df_val.labels == 0) & (df_val.preds <= 0.22)].shape[0]
false_neg = df_val[(df_val.labels == 1) & (df_val.preds <= 0.22)].shape[0]

GB_prec = true_pos / (true_pos + false_pos)
GB_rec = true_pos / (true_pos + false_neg)

print("\n\n\nsummary:\n")

print("logistic regression:")
print("precision = ", LOG_prec)
print("recall = ", LOG_rec)

print("random forest:")
print("precision = ", RF_prec)
print("recall = ", RF_rec)

print("gradient boost:")
print("precision = ", GB_prec)
print("recall = ", GB_rec)


# # load and process instacart dataset
# order_data, product_data = instacart_process()
#
# # TEMP: decrease size for debugging...
# order_data = order_data[:10000]
#
# train_order_data, train_last_order, val_order_data, val_last_order = train_val_split(order_data)
#
# # TEMP: later make model class
# preds = lgboost_model_predict(train_order_data, train_last_order, val_order_data, product_data):
#
# # compare preds and actual order in example

# latent space tests
# latent_model(retrain=False, data_dir= './data/instacart_2017_05_01/')
#
