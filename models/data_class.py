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