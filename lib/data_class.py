"""
Definition of standard DataSet class, used by all models
"""
import numpy as np
import pandas as pd
import logging

log = logging.getLogger("TR_logger")

class DataSet:
    """
    Dataset class

    methods:

        train_test_split : splits dataset into two new datasets for fitting and training

        make_adversarial : returns dataset with num_switches of each user's products replaced by a random product

        _prior_last_split : splits order history for each user into last order and all others, for predicting final order
                            saves in variable prior_order_df.  also creates labels and size

        _make_user_prod_matrix : makes user product matrix, used for autoencoder
                                saves in user_prod_matrix

    variables:

        user_ids: numpy array of user ids
        prod_ids: numpy array of product ids

        prior_order_df: dataframe consisting of all but last order for each user
        prior_user_prod: dataframe with user-product pairs from prior orders
        labels: binary vector keeping track of which products were purchased on last order, target for prediction
        size: number of data points.  depends on specific targets, but here we take it to be size of user_prod_pairs

        user_prod_matrix : matrix of user and product interactions
    """

    def __init__(self, order_df=None, product_df=None, user_df=None):
        log.debug("creating new dataset...")
        self.order_df, self.product_df, self.user_df = None, None, None
        if order_df is not None:
            self._add_order_data(order_df)
        if product_df is not None:
            self._add_product_data(product_df)
        if user_df is not None:
            self._add_user_data(user_df)

        # set when called by _prior_last_split()
        self._prior_order_df, self._prior_user_prod, self._labels, self._size = None, None, None, None

        # set when called by _make_user_prod_matrix()
        self._user_prod_matrix = None

    def _add_order_data(self, order_df):
        log.debug("adding order dataframe...")
        self.order_df = order_df
        self.user_ids = order_df.user_id.unique()
        self.prod_ids = order_df.product_id.unique()
        self.user_idx = {uid: i for i, uid in enumerate(self.user_ids)}
        self.prod_idx = {pid: i for i, pid in enumerate(self.prod_ids)}

    def _add_product_data(self, product_df):
        log.debug("adding product dataframe...")
        self.product_df = product_df

    def _add_user_data(self, user_df):
        log.debug("adding user dataframe...")
        self.user_df = user_df

    def train_test_split(self, test_frac=0.25, inherit_products=True, shuffle=True):
        """
         returns two Dataset objects, train and test, split into fractions (1-test_size) and test_size of the users
        if inherit_products, preserves list of all products from parent

        :param test_size: fraction of users to place in test dataset
        :param inherit_products: whether to inherit list of products from original dataset
        :return: test and train DataSet objects
        """
        log.debug(f"train/test split with test_frac={test_frac}...")
        user_ids = np.copy(self.user_ids)
        if shuffle:
            np.random.shuffle(user_ids)

        # number of train samples
        num_train = int((1-test_frac)*len(user_ids))

        # split user_ids among train and test sets
        train_user_ids, test_user_ids = user_ids[:num_train], user_ids[num_train:]

        # create train and test dataframes
        train_order_df = self.order_df[self.order_df.user_id.isin(train_user_ids)].copy()
        test_order_df = self.order_df[self.order_df.user_id.isin(test_user_ids)].copy()

        # create train and test DataSets
        train_dataset = DataSet(order_df=train_order_df, product_df=self.product_df)
        test_dataset = DataSet(order_df=test_order_df, product_df=self.product_df)

        # if inherit_products, replace prod_ids and prod_idx by those of parent
        if inherit_products:
            train_dataset.prod_ids = self.prod_ids.copy()
            train_dataset.prod_idx = self.prod_idx.copy()

            test_dataset.prod_ids = self.prod_ids.copy()
            test_dataset.prod_idx = self.prod_idx.copy()

        return train_dataset, test_dataset


    def make_adversarial(self, num_switches=1, inherit_ids=True):
        """
        returns dataset with num_switches of each user's prior products (all but last order) replaced by a random product
        used for testing robustness of models
        
        :param num_switches: number of random switches (with replacement) to perform for each user
        :param inherit_ids: if true, inherits user_ids and prod_ids from original
        :return: new Dataset object
        """
        log.debug(f"making adversarial dataset with {num_switches} switches...")

        idxs = {user_id : [] for user_id in self.user_ids}

        for i, row in enumerate(self.order_df.itertuples()):
            idxs[row.user_id].append(i)

        to_swap = []
        for uid in self.user_ids:
            to_swap += list(np.random.choice(idxs[uid], size=num_switches))

        all_pids = self.order_df.product_id.unique() # do not use self.prod_ids as this can be larger if inherited
        swapped_pids = np.random.choice(all_pids, size=len(to_swap))

        new_pids = np.copy(self.order_df.product_id.values)

        to_swap = sorted(to_swap)

        for i, pid in zip(to_swap, swapped_pids):
            new_pids[i] = pid

        new_order_df = self.order_df.copy()
        new_order_df.product_id = new_pids

        new_dataset = DataSet(order_df=new_order_df, product_df=self.product_df)

        if inherit_ids:
            new_dataset.user_ids = self.user_ids.copy()
            new_dataset.user_idx = self.user_idx.copy()
            new_dataset.prod_ids = self.prod_ids.copy()
            new_dataset.prod_idx = self.prod_idx.copy()

        return new_dataset

    @property
    def prior_order_df(self):
        if self._prior_order_df is None:
            self._prior_last_split()
        return self._prior_order_df

    @property
    def prior_user_prod(self):
        if self._prior_user_prod is None:
            self._prior_last_split()
        return self._prior_user_prod

    @property
    def labels(self):
        if self._labels is None:
            self._prior_last_split()
        return self._labels

    @property
    def size(self):
        if self._size is None:
            self._prior_last_split()
        return self._size

    def _prior_last_split(self):
        """
        Splits each users orders into all but last and last orders
        creates new dataframe prior_order_df, as well as labels and size

        :return: None
        """
        log.debug("performing split into prior and last orders...")

        # TODO: improve using groupby to get max order_number for each user
        log.debug("finding number of orders for each user...")
        order_numbers = {}
        for row in self.order_df.itertuples():
            if row.user_id in order_numbers:
                order_numbers[row.user_id] = max(order_numbers[row.user_id], row.order_number)
            else:
                order_numbers[row.user_id] = row.order_number

        log.debug("creating rows of new dataframes...")
        prior_order_df_rows, last_order_df_rows = [], []
        for row in self.order_df.itertuples():
            if row.order_number < order_numbers[row.user_id]:
                prior_order_df_rows.append(row)
            else:
                last_order_df_rows.append(row)

        self._prior_order_df = pd.DataFrame(prior_order_df_rows).drop(columns=['Index'])
        self._prior_user_prod = self._prior_order_df[['user_id', 'product_id']].drop_duplicates().reset_index(drop=True)
        last_order_df = pd.DataFrame(last_order_df_rows).set_index(['user_id', 'product_id'])

        log.debug("creating labels...")
        self._labels = []
        for row in self._prior_user_prod.itertuples():
            self._labels.append(int((row.user_id, row.product_id) in last_order_df.index))
        self._labels = np.array(self._labels)

        self._size = np.shape(self._labels)[0]

    @property
    def user_prod_matrix(self):
        if self._user_prod_matrix is None:
            self._make_user_prod_matrix()
        return self._user_prod_matrix

    def _make_user_prod_matrix(self):
        """
        makes matrix of user and product purchase history, saves in self._user_prod_matrix
        used for user autoencoder

        :return: None
        """
        if self.order_df is None:
            log.error("Order dataframe not defined!")
            return

        log.debug("creating user_prod_matrix...")

        updf = self.order_df.groupby(["user_id", "product_id"])["product_id"].count().reset_index(name="count")

        user_product_dict = {}
        for row in updf.itertuples():
            uid, pid, cnt = row.user_id, row.product_id, row.count
            if row.user_id not in user_product_dict:
                user_product_dict[uid] = {}
            user_product_dict[uid][pid] = cnt

        user_total = {}
        for uid in self.user_ids:
            user_total[uid] = sum([user_product_dict[uid][pid] for pid in user_product_dict[uid].keys()])

        self._user_prod_matrix = np.zeros((len(self.user_ids), len(self.prod_ids)))
        for uid in self.user_ids:
            for pid in user_product_dict[uid]:
                self.user_prod_matrix[self.user_idx[uid], self.prod_idx[pid]] = user_product_dict[uid][pid] / user_total[uid]

        log.debug(f"created user_prod_matrix of shape {self._user_prod_matrix.shape}")


# TESTING

def run_tests(IC_DATA_DIR):
    from lib.process_data import instacart_process

    # set random seed for consistent tests
    np.random.seed(42)

    # load data from instacart csv files (values below use testing directory)
    order_data, product_data = instacart_process(data_dir=IC_DATA_DIR)

    # create dataset
    ic_dataset = DataSet(order_df=order_data, product_df=product_data)

    # check dataframes created correctly
    assert ic_dataset.order_df.shape == (31032, 4)
    assert ic_dataset.product_df.shape == (6126, 5)

    # check user and product ids created correctly
    assert ic_dataset.user_ids.shape == (206,)
    assert len(ic_dataset.user_idx) == 206
    assert ic_dataset.prod_ids.shape == (6126,)
    assert len(ic_dataset.prod_idx) == 6126

    # perform train-test split
    train_dataset, test_dataset = ic_dataset.train_test_split()
    assert train_dataset.order_df.shape == (24939, 4)
    assert train_dataset.product_df.shape == (6126, 5)
    assert test_dataset.order_df.shape == (6093, 4)
    assert test_dataset.product_df.shape == (6126, 5)

    # check that prod_ids inherited correctly
    assert (train_dataset.prod_ids == ic_dataset.prod_ids).all()
    assert (test_dataset.prod_ids == ic_dataset.prod_ids).all()

    # create adversarial dataset
    adv_dataset = ic_dataset.make_adversarial()
    assert adv_dataset.order_df.shape == (31032, 4)
    assert adv_dataset.product_df.shape == (6126, 5)
    assert (adv_dataset.user_ids == ic_dataset.user_ids).all()

    # number of places old and new dfs differ, should be number of users (unless swap product for self; very unlikely)
    assert np.sum(ic_dataset.order_df.product_id.values != adv_dataset.order_df.product_id.values) == 206

    # test prior-last order split
    assert ic_dataset.prior_order_df.shape == (28906, 4)
    assert ic_dataset.prior_user_prod.shape == (12214, 2)
    assert ic_dataset.labels.shape == (12214,)
    assert ic_dataset.size == 12214

    # test user-product matrix
    assert ic_dataset.user_prod_matrix.shape == (206, 6126)

    log.info("data_class tests passed!")

if __name__ == '__main__':
    IC_DATA_DIR = '../data/instacart_2017_05_01_testing/'
    logging.basicConfig()
    log.setLevel(logging.DEBUG)
    run_tests(IC_DATA_DIR)
