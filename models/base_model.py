import numpy as np


class BaseModel:
    def __init__(self):
        self.test_dataset = None
        self.X_pred = None
        self.threshold = 0.5 # default threshold; should be updated by validation
        self.prec, self.rec, self.f1 = 0, 0, 0

    def fit(self, dataset):
        pass

    def predict(self, dataset):
        pass

    def find_threshold(self, dataset, pts=20, min=0, max=1):
        totry = np.linspace(min, max, num=pts)
        best_t, best_f1 = 0, 0
        for t in totry:
            self.accuracy_test(dataset, threshold=t)
            if self.f1>best_f1:
                best_t, best_f1 = t, self.f1
        self.threshold = best_t

    def accuracy_test(self, test_dataset, threshold=None):
        if threshold is None:
            threshold = self.threshold

        preds, test_labels, prior_orders = self.predict(test_dataset, getdf=True)

        user_true = {}
        user_pred = {}
        user_pred_values = {}
        for i,row in enumerate(prior_orders.itertuples()):
            uid = row.user_id
            pid = row.product_id
            if uid not in user_true:
                user_true[uid], user_pred[uid], user_pred_values[uid]  = [], [], {}
            user_pred_values[uid][pid] = preds[i]
            if test_labels[i] == 1:
                user_true[uid].append(pid)
            if preds[i] > threshold:
                user_pred[uid].append(pid)

        precs, recs, f1s, ndcgs = [], [], [], []
        for uid in user_true:
            trues = set(user_true[uid])
            preds = set(user_pred[uid])

            tp = len(trues.intersection(preds))
            fp = len(preds) - tp
            fn = len(trues) - tp

            prec = tp/(tp+fp) if tp+fp>0 else 1
            rec = tp/(tp+fn) if tp+fn>0 else 1
            f1 = (2*prec*rec)/(prec+rec) if prec+rec>0 else 0

            precs.append(prec)
            recs.append(rec)
            f1s.append(f1)

            # compute dcg
            ordered_preds = sorted((-user_pred_values[uid][pid], pid) for pid in user_pred_values[uid])
            pred_rank = {x[1]: i for i, x in enumerate(ordered_preds)}  # gives predicted rank of ith product
            ndcgs.append(sum([1/np.log(pred_rank[pid]+2) for pid in trues])/sum([1/np.log(i+2) for i in range(len(trues))])) if len(trues)>0 else 1

        self.prec, self.rec, self.f1, self.ndcg = np.mean(precs), np.mean(recs), np.mean(f1s), np.mean(ndcgs)