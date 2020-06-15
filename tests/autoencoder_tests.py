import sys
sys.path.append('../lib')
from process_data import *
from latent_space import *


verbose=1
IC_DATA_DIR = '../data/instacart_2017_05_01/'
SMALL_IC_DATA_DIR = '../data/instacart_2017_05_01_small/'


# load data
if verbose>0:
    print("loading data...")
order_data, product_data = instacart_process(data_dir=SMALL_IC_DATA_DIR)
product_ids = order_data.product_id.unique()
user_ids = order_data.user_id.unique()

# split into train and test sets
if verbose>0:
    print("splitting into test/train sets...")
order_data_train, order_data_test = first_last_split(order_data)

order_data_train.groupby(["user_id", "product_id"])["product_id"].count().reset_index(name="count")


# Create user product matrix
if verbose>0:
    print("creating user-product matrix...")
updf = order_data_train.groupby(["user_id", "product_id"])["product_id"].count().reset_index(name="count")
user_product_dict = {}
for row in updf.itertuples():
    uid, pid, cnt = row.user_id, row.product_id, row.count
    if row.user_id not in user_product_dict:
        user_product_dict[uid] = {}
    user_product_dict[uid][pid] = cnt

user_total = {}
for uid in user_ids:
    user_total[uid] = sum([user_product_dict[uid][pid] for pid in user_product_dict[uid].keys()])

user_idx = {uid:i for i,uid in enumerate(user_ids)}
prod_idx = {pid:i for i,pid in enumerate(product_ids)}

X=np.zeros((len(user_ids),len(product_ids)))
for uid in user_ids:
    for pid in user_product_dict[uid]:
        X[user_idx[uid],prod_idx[pid],] = user_product_dict[uid][pid]/user_total[uid]

if verbose>0:
    print("created X of size", X.shape)

# create user autoencoder
if verbose>0:
    print("creating autoencoder...")
user_encoder = UserAE(len(product_ids),encoding_dim = 100)
user_encoder.fit(X, epochs=10)

encoded_users = user_encoder.encoder.predict(X)
decoded_users = user_encoder.decoder.predict(encoded_users)




