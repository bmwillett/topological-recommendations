"""

 For various datasets, takes input in provided format and outputs in following
 standard format (below are both pandas DataFrames):

 order_data:
    - rows: user/order/product combinations
    - columns:
        - user_id - id uniquely identifying users
        - order_number - order number for given user
        - add_to_cart_order - order product was added in given order
        - product_id - id identifying products (see below)

 product_data:
    - index : product_id numbers
    - columns:
        - feature1, feature2, ... (text/categorical features for products)
    
"""

import numpy as np
import pandas as pd
import os

IC_DATA_DIR = '../data/instacart_2017_05_01/'
SMALL_IC_DATA_DIR = '../data/instacart_2017_05_01_small/'

AWS = True
if AWS:
    import boto3
    from io import StringIO

    client = boto3.client('s3')

    bucket_name = 'bmwillett1'

    IC_DATA_DIR_AWS = 'instacart/instacart_2017_05_01_small/'


def getCSVs3(filename, ic_data_dir=IC_DATA_DIR_AWS):
    object_key = ic_data_dir + filename
    csv_obj = client.get_object(Bucket=bucket_name, Key=object_key)
    body = csv_obj['Body']
    csv_string = body.read().decode('utf-8')
    return StringIO(csv_string)

def instacart_process_AWS():
    """
    load and preprocess instacart dataset

    :param data_dir: directory containing dataset
    :return: order_data, product_data : Pandas DataFrames
    """

    # load create order_data
    orders = pd.read_csv(getCSVs3('orders.csv'), dtype={
        'order_id': np.int32,
        'user_id': np.int32,
        'order_number': np.int16},
                         usecols=['order_id', 'user_id', 'order_number'])

    priors = pd.read_csv(getCSVs3('order_products__prior.csv'), dtype={
        'order_id': np.int32,
        'product_id': np.uint16,
        'add_to_cart_order': np.int16},
                         usecols=['order_id', 'product_id', 'add_to_cart_order'])

    train = pd.read_csv(getCSVs3('order_products__train.csv'), dtype={
        'order_id': np.int32,
        'product_id': np.uint16,
        'add_to_cart_order': np.int16},
                        usecols=['order_id', 'product_id', 'add_to_cart_order'])

    orders.set_index('order_id', inplace=True)
    order_data = priors.append(train)
    order_data['user_id'] = order_data.order_id.map(orders.user_id)
    order_data['order_number'] = order_data.order_id.map(orders.order_number)
    order_data.drop(['order_id'], axis=1, inplace=True)
    order_data = order_data[['user_id', 'order_number', 'add_to_cart_order', 'product_id']]

    # load and create product_data
    products = pd.read_csv(getCSVs3('products.csv'))
    aisles = pd.read_csv(getCSVs3('aisles.csv')).set_index('aisle_id')
    departments = pd.read_csv(getCSVs3('departments.csv')).set_index('department_id')

    product_data = pd.DataFrame()
    product_data['product_id'] = products['product_id']
    product_data['feature1'] = products['product_name']
    product_data['feature2'] = products.aisle_id
    product_data['feature3'] = products.aisle_id.map(aisles.aisle)
    product_data['feature4'] = products.department_id
    product_data['feature5'] = products.department_id.map(departments.department)
    product_data.set_index('product_id', inplace=True)

    return order_data, product_data


def instacart_process(data_dir=IC_DATA_DIR):
    """
    load and preprocess instacart dataset

    :param data_dir: directory containing dataset
    :return: order_data, product_data : Pandas DataFrames
    """

    # load create order_data
    orders = pd.read_csv(data_dir + 'orders.csv', dtype={
        'order_id': np.int32,
        'user_id': np.int32,
        'order_number': np.int16},
        usecols=['order_id', 'user_id', 'order_number'])

    priors = pd.read_csv(data_dir + 'order_products__prior.csv', dtype={
        'order_id': np.int32,
        'product_id': np.uint16,
        'add_to_cart_order': np.int16},
        usecols=['order_id', 'product_id','add_to_cart_order'])

    train = pd.read_csv(data_dir + 'order_products__train.csv', dtype={
        'order_id': np.int32,
        'product_id': np.uint16,
        'add_to_cart_order': np.int16},
        usecols=['order_id', 'product_id','add_to_cart_order'])

    orders.set_index('order_id',inplace=True)
    order_data = priors.append(train)
    order_data['user_id'] = order_data.order_id.map(orders.user_id)
    order_data['order_number'] = order_data.order_id.map(orders.order_number)
    order_data.drop(['order_id'], axis=1, inplace=True)
    order_data = order_data[['user_id','order_number','add_to_cart_order','product_id']]

    # load and create product_data
    products = pd.read_csv(data_dir + 'products.csv')
    aisles = pd.read_csv(data_dir + 'aisles.csv').set_index('aisle_id')
    departments = pd.read_csv(data_dir + 'departments.csv').set_index('department_id')

    product_data = pd.DataFrame()
    product_data['product_id'] = products['product_id']
    product_data['feature1'] = products['product_name']
    product_data['feature2'] = products.aisle_id
    product_data['feature3'] = products.aisle_id.map(aisles.aisle)
    product_data['feature4'] = products.department_id
    product_data['feature5'] = products.department_id.map(departments.department)
    product_data.set_index('product_id', inplace=True)

    return order_data, product_data

def make_instacart_small(user_frac=0.01, order_frac=1, product_frac=1, data_dir=IC_DATA_DIR,
                         output_dir=SMALL_IC_DATA_DIR):
    """
    creates smaller version of instacart dataset for faster testing

    :param user_frac: fraction of users to keep
    :param order_frac: fraction of orders to keep
    :param product_frac: fraction of products to keep
    :param data_dir: input data directory
    :param output_dir: output directory for small dataset
    :return: None
    """
    priors = pd.read_csv(data_dir + 'order_products__prior.csv')
    train = pd.read_csv(data_dir + 'order_products__train.csv')
    orders = pd.read_csv(data_dir + 'orders.csv')
    products = pd.read_csv(data_dir + 'products.csv')
    aisles = pd.read_csv(data_dir + 'aisles.csv')
    departments = pd.read_csv(data_dir + 'departments.csv')

    product_ids = products.product_id.unique()
    user_ids = orders.user_id.unique()
    product_ids_small = np.random.choice(product_ids, int(len(product_ids) * product_frac), replace=False)
    user_ids_small = np.random.choice(user_ids, int(len(user_ids) * user_frac), replace=False)

    orders = orders[((orders.user_id.isin(user_ids_small)) | (user_frac == 1))]
    order_ids = orders.order_id.unique()
    order_ids_small = np.random.choice(order_ids, int(len(order_ids) * order_frac), replace=False)

    priors_small = priors[
        (priors.order_id.isin(order_ids_small)) & ((product_frac == 1) | (priors.product_id.isin(product_ids_small)))]
    train_small = train[
        (train.order_id.isin(order_ids_small)) & ((product_frac == 1) | (train.product_id.isin(product_ids_small)))]
    orders_small = orders[(orders.order_id.isin(order_ids_small))]
    products_small = products[(product_frac == 1) | (products.product_id.isin(product_ids_small))]

    priors_small.to_csv(output_dir + 'order_products__prior.csv', index=False)
    train_small.to_csv(output_dir + 'order_products__train.csv', index=False)
    orders_small.to_csv(output_dir + 'orders.csv', index=False)
    products_small.to_csv(output_dir + 'products.csv', index=False)
    aisles.to_csv(output_dir + 'aisles.csv', index=False)
    departments.to_csv(output_dir + 'departments.csv', index=False)

def train_val_split(order_data, train_frac=0.8):
    """
    splits users in order_data into train and validation sets:
        train_order_data, val_order_data

    drops users with 1 order
    for each user with >1 order, removes last order, to be predicted, produces:
        train_last_order, val_last_order

    NOTE: somewhat slow, maybe should save as csv files

    :param order_data: order_data produced by data loading function
    :param train_frac: fraction of data in train set
    :return: Pandas DataFrames: order_data_train, order_data_val, last_order_train, last_order_val
    """

    order_numbers = {}
    for row in order_data.itertuples():
        if row.user_id in order_numbers:
            order_numbers[row.user_id] = max(order_numbers[row.user_id], row.order_number)
        else:
            order_numbers[row.user_id] = row.order_number

    user_ids = order_data.user_id.unique()
    train_user_ids = np.random.choice(user_ids, int(train_frac * user_ids.shape[0]), replace=False)

    train_order_data_rows, train_last_order_rows, val_order_data_rows, val_last_order_rows = [], [], [], []
    for row in order_data.itertuples():
        if row.user_id in train_user_ids:
            if row.order_number < order_numbers[row.user_id]:
                train_order_data_rows.append(row)
            else:
                train_last_order_rows.append(row)
        else:
            if row.order_number < order_numbers[row.user_id]:
                val_order_data_rows.append(row)
            else:
                val_last_order_rows.append(row)

    train_order_data = pd.DataFrame(train_order_data_rows)
    train_last_order = pd.DataFrame(train_last_order_rows)
    val_order_data = pd.DataFrame(val_order_data_rows)
    val_last_order = pd.DataFrame(val_last_order_rows)

    return train_order_data, train_last_order, val_order_data, val_last_order


if __name__=='__main__':
    # run tests using small data set
    if not os.path.isfile(SMALL_IC_DATA_DIR+'orders.csv'):
        print("creating small dataset...")
        make_instacart_small()

    print("processing instacart data...")
    order_data, product_data = instacart_process(data_dir=SMALL_IC_DATA_DIR)
    pd.set_option('display.max_columns', None)
    print("order_data shape = ", order_data.shape)
    print(order_data.head())
    print("product_data shape = ", product_data.shape)
    print(product_data.head())

    print("doing train/val split...")
    train_order_data, train_last_order, val_order_data, val_last_order = train_val_split(order_data)
    print("train_order_data shape = ", train_order_data.shape)
    # print(train_order_data.head())
    print("train_last_order shape = ", train_last_order.shape)
    # print(train_last_order.head())
    print("val_order_data shape = ", val_order_data.shape)
    # print(val_order_data.head())
    print("val_last_order shape = ", val_last_order.shape)
    # print(val_last_order.head())
