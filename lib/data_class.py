import pandas as pd
import numpy as np


class DataSet:
    def __init__(self, order_data=None, product_data=None, user_data=None):
        self.order_data = self.product_data = self.user_data = None
        if order_data is not None:
            self.add_order_data(order_data)
        if product_data is not None:
            self.add_product_data(product_data)
        if user_data is not None:
            self.add_user_data(user_data)

    def add_order_data(self, order_data):
        self.order_data = order_data
        self.user_ids = order_data.user_id.unique()
        self.product_ids = order_data.product_id.unique()
        self.user_idx = {uid: i for i, uid in enumerate(self.user_ids)}
        self.prod_idx = {pid: i for i, pid in enumerate(self.product_ids)}

    def add_product_data(self, product_data):
        self.product_data = product_data

    def add_user_data(self, user_data):
        self.user_data = user_data

    def get_user_product_matrix(self):
        try:
            return self.UserProductMatrix
        except:
            self._make_user_product_matrix()
            return self.UserProductMatrix

    def _make_user_product_matrix(self, verbose=0):
        if self.order_data is None:
            # TODO: implement as exception
            print("Error: order data not defined")
            return

        if verbose > 0:
            print("creating user-product matrix...")

        updf = self.order_data.groupby(["user_id", "product_id"])["product_id"].count().reset_index(name="count")

        user_product_dict = {}
        for row in updf.itertuples():
            uid, pid, cnt = row.user_id, row.product_id, row.count
            if row.user_id not in user_product_dict:
                user_product_dict[uid] = {}
            user_product_dict[uid][pid] = cnt

        user_total = {}
        for uid in self.user_ids:
            user_total[uid] = sum([user_product_dict[uid][pid] for pid in user_product_dict[uid].keys()])

        self.UserProductMatrix = np.zeros((len(self.user_ids), len(self.product_ids)))
        for uid in self.user_ids:
            for pid in user_product_dict[uid]:
                self.UserProductMatrix[self.user_idx[uid], self.prod_idx[pid],] = user_product_dict[uid][pid] / user_total[uid]

        if verbose > 0:
            print("created UserProductMatrix of size", self.UserProductMatrix.shape)

    def get_prior_products(self):
        prior_dataset, last_dataset = self.split()

        prior_orders = prior_dataset.order_data[['user_id', 'product_id']].drop_duplicates().reset_index(drop=True)
        last_dataset.order_data.set_index(['user_id', 'product_id'], inplace=True)

        labels = []
        for row in prior_orders.itertuples():
            labels.append(int((row.user_id, row.product_id) in last_dataset.order_data.index))
        labels = np.array(labels)

        return prior_orders, labels

    def split(self, mode='last_order', reset_ids=False):
        """
        for mode == 'last_order':
            drops users with 1 order
            for each user with >1 order, removes last order, to be predicted, produces new DataSet's:
                prior_order_data, last_order_data

        :return: Pandas DataFrames:  first_order_data, last_order_data
        """


        if mode == 'last_order':
            order_numbers = {}
            for row in self.order_data.itertuples():
                if row.user_id in order_numbers:
                    order_numbers[row.user_id] = max(order_numbers[row.user_id], row.order_number)
                else:
                    order_numbers[row.user_id] = row.order_number

            prior_order_data_rows, last_order_data_rows = [], []
            for row in self.order_data.itertuples():
                if row.order_number < order_numbers[row.user_id]:
                    prior_order_data_rows.append(row)
                else:
                    last_order_data_rows.append(row)

            prior_order_data = pd.DataFrame(prior_order_data_rows)
            last_order_data = pd.DataFrame(last_order_data_rows)

            prior_dataset = DataSet(order_data=prior_order_data, product_data=self.product_data)
            last_dataset = DataSet(order_data=last_order_data, product_data=self.product_data)

            if not reset_ids:
                prior_dataset.user_ids = self.user_ids
                prior_dataset.user_idx = self.user_idx
                prior_dataset.product_ids = self.product_ids
                prior_dataset.prod_idx = self.prod_idx

                last_dataset.user_ids = self.user_ids
                last_dataset.user_idx = self.user_idx
                last_dataset.product_ids = self.product_ids
                last_dataset.prod_idx = self.prod_idx

        return prior_dataset, last_dataset

    def test_train_val_split(self, train_frac=0.7 , val_frac=0.2):
        assert train_frac+val_frac<=1
        # test_frac=1-train_frac-test_frac (not needed)

        uids_shuffle = np.copy(self.user_ids)
        np.random.shuffle(uids_shuffle)
        num_uids = len(self.user_ids)
        num_train, num_val = int(train_frac*num_uids), int(val_frac*num_uids)
        train_uids, val_uids, test_uids = uids_shuffle[:num_train], uids_shuffle[num_train:num_train+num_val], uids_shuffle[num_train+num_val:]

        train_order_data = self.order_data[self.order_data.user_id.isin(train_uids)].copy()
        val_order_data = self.order_data[self.order_data.user_id.isin(val_uids)].copy()
        test_order_data = self.order_data[self.order_data.user_id.isin(test_uids)].copy()

        train_dataset = DataSet(order_data=train_order_data, product_data=self.product_data)
        val_dataset = DataSet(order_data=val_order_data, product_data=self.product_data)
        test_dataset = DataSet(order_data=test_order_data, product_data=self.product_data)

        train_dataset.product_ids = self.product_ids
        train_dataset.product_idx = self.prod_idx
        val_dataset.product_ids = self.product_ids
        val_dataset.product_idx = self.prod_idx
        test_dataset.product_ids = self.product_ids
        test_dataset.product_idx = self.prod_idx

        return train_dataset, val_dataset, test_dataset

    def make_adversarial(self, num_switches=1, reset_ids=False):

        idxs = {uid: [] for uid in self.user_ids}

        for i, row in enumerate(self.order_data.itertuples()):
            uid = row.user_id
            idxs[uid].append(i)

        toswap = []
        for uid in self.user_ids:
            toswap += list(np.random.choice(idxs[uid], size=num_switches))

        all_pids = self.order_data.product_id.unique() # do not use self.product_ids as this can be larger
        new_pids = np.random.choice(all_pids, size=len(toswap))

        pids = np.copy(self.order_data.product_id.values)

        toswap = sorted(toswap)

        for i, pid in zip(toswap, new_pids):
            pids[i] = pid

        new_order_data = self.order_data.copy()
        new_order_data.product_id = pids

        new_dataset = DataSet(order_data=new_order_data, product_data=self.product_data)

        if not reset_ids:
            new_dataset.user_ids = self.user_ids
            new_dataset.user_idx = self.user_idx
            new_dataset.product_ids = self.product_ids
            new_dataset.prod_idx = self.prod_idx

        return new_dataset