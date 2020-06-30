"""
Define Base recommendation model, all other recommendation models inherit from this
"""
import numpy as np
import logging

log = logging.getLogger("TR_logger")

class RecModel:
    """
    Base model which all main and baseline models inherit from

    Methods:

        find_threshold : pick threshold in [0,1] for prediction (ie, output of neural net > threshold => predict buy product).
                        grid search over given data set to find threshold maximizing f1 score

        evaluate : computes precision, recall, f1 score, and NCDG (normalized discountef cumulative gain) for model
                        on given data set
    """
    def __init__(self):
        self.preds = None
        self.threshold = 0.5  # default threshold, should be set by find_threshold()

    def find_threshold(self, dataset, pts=20, lower=0, upper=1):
        """
        use a give (validation) dataset to search for the threshold which gives the highest metric
        currently uses f1 score as metric to optimize

        :param dataset: validation dataset used in optimization
        :param pts: number of points in grid search of thresholds
        :param lower: lower bound of grid seacrh
        :param upper: upper bound of grid seacrh

        :return: best threshold (also set to self.threshold and used as default threshold for model)
        """
        log.debug(f"getting threshold: trying {pts} values in range ({lower},{upper})...")
        to_try = np.linspace(lower, upper, num=pts)
        best_t, best_f1 = 0, 0
        for t in to_try:
            self.evaluate(dataset, threshold=t)
            if self.metrics[3] > best_f1:  # maximize f1 score
                best_t, best_f1 = t, self.metrics[3]
        self.threshold = best_t
        return self.threshold

    def evaluate(self, dataset, threshold=None):
        """
        given dataset (with known labels assigned to dataset.labels), perform the following tests:

        [accuracy, precision, recall, f1 score, NDCG]

        place this list in variable model.metrics

        :param dataset: dataset used for testing
        :param threshold: threshold to use for testing, or if None, use self.threshold
        :return: None
        """
        log.debug("getting metrics...")
        if threshold is None:
            threshold = self.threshold

        preds = self.predict(dataset)

        prior_user_prod = dataset.prior_user_prod

        user_true = {}
        user_pred = {}
        user_pred_values = {}
        for i, row in enumerate(prior_user_prod.itertuples()):
            uid = row.user_id
            pid = row.product_id
            if uid not in user_true:
                user_true[uid], user_pred[uid], user_pred_values[uid] = [], [], {}
            user_pred_values[uid][pid] = preds[i]
            if dataset.labels[i] == 1:
                user_true[uid].append(pid)
            if preds[i] > threshold:
                user_pred[uid].append(pid)

        accs, precs, recs, f1s, ndcgs = [], [], [], [], []
        for uid in user_true:
            trues = set(user_true[uid])
            preds = set(user_pred[uid])

            tp = len(trues.intersection(preds))
            fp = len(preds) - tp
            fn = len(trues) - tp
            tn = len(user_pred_values) - tp - fp - fn

            acc = (tp+tn)/(tp+tn+fp+fn) if (tp+tn+fp+fn)>0 else 1
            prec = tp/(tp+fp) if tp+fp>0 else 1
            rec = tp/(tp+fn) if tp+fn>0 else 1
            f1 = (2*prec*rec)/(prec+rec) if prec+rec>0 else 0

            accs.append(acc)
            precs.append(prec)
            recs.append(rec)
            f1s.append(f1)

            # compute dcg
            ordered_preds = sorted((-user_pred_values[uid][pid], pid) for pid in user_pred_values[uid])
            pred_rank = {x[1]: i for i, x in enumerate(ordered_preds)}  # gives predicted rank of ith product
            ndcgs.append(sum([1/np.log(pred_rank[pid]+2) for pid in trues])/sum([1/np.log(i+2) for i in range(len(trues))])) if len(trues)>0 else 1

        self.metrics = [np.mean(accs), np.mean(precs), np.mean(recs), np.mean(f1s), np.mean(ndcgs)]
        return self.metrics
