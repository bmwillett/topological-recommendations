"""
Define Base recommendation model, all other recommendation models inherit from this
"""
import numpy as np
import logging
import matplotlib.pyplot as plt
from bisect import bisect_left

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
        self.preds, metrics, stats, class_sizes = None, None, None, None
        self.threshold = 0.5  # default threshold, should be set by find_threshold()


    def find_threshold(self, dataset, on_metric='f1'):
        """
        use a give (validation) dataset to search for the threshold which gives the highest metric
        currently uses f1 score as metric to optimize

        :param dataset: validation dataset used in optimization
        :param on_metric: metric maximized in picking threshold (default = 'f1' (f1 score))

        :return: best threshold (also set to self.threshold and used as default threshold for model)
        """
        self.get_stats(dataset)

        best_pred, best = None, 0
        for pred, fp, tp in self.stats:
            tn, fn = self.class_sizes[0] - fp, self.class_sizes[1] - tp
            prec = tp/(tp+fp)
            rec = tp/(tp+fn)
            f1 = 2*prec*rec/(prec+rec)
            acc = (tp+tn)/sum(self.class_sizes)
            test = {'prec': prec, 'rec': rec, 'f1': f1, 'acc': acc}[on_metric]
            if test > best:
                best_pred, best = pred, test
                self.metrics = [prec, rec, f1, acc]

        self.threshold = best_pred
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
        if threshold is not None:
            self.threshold = threshold
        if self.threshold is None:
            raise ValueError("threshold not set")

        self.get_stats(dataset)

        idx = bisect_left(self.stats[:, 0], self.threshold)
        fp, tp = self.stats[idx, 1:]
        tn, fn = self.class_sizes[0] - fp, self.class_sizes[1] - tp
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        f1 = 2 * prec * rec / (prec + rec)
        acc = (tp + tn) / sum(self.class_sizes)

        res_df = dataset.prior_user_prod.drop(columns=['product_id'])
        res_df['preds'] = self.preds
        res_df['labels'] = dataset.labels
        res_df.sort_values(by=['user_id', 'preds'], ascending=False, inplace=True)

        def _ndcg(labels):
            return sum([1 / np.log(i + 2) if label == 1 else 0 for i, label in enumerate(labels.values)]) / sum(
                [1 / np.log(i + 2) for i in range(sum(labels.values))]) if sum(labels.values) > 0 else 0


        ndcg = res_df.groupby('user_id')['labels'].apply(_ndcg).values.mean()

        self.metrics = {'prec': prec,
                        'rec': rec,
                        'f1': f1,
                        'acc': acc,
                        'ndcg': ndcg}
        return self.metrics


    def get_stats(self, dataset, plot_hist=False, plot_roc=False):
        preds = self.predict(dataset)

        classes = {0: [], 1: []}
        labels = {}
        for pred, label in zip(preds, dataset.labels):
            classes[label].append(pred)
            labels[pred] = label

        classes[0], classes[1] = sorted(classes[0]), sorted(classes[1])
        self.class_sizes = [len(classes[0]), len(classes[1])]

        if plot_hist:
            plt.hist(classes[0])
            plt.hist(classes[1])
            plt.show()

        self.stats, fp, tp = [], self.class_sizes[0], self.class_sizes[1]
        for pred in sorted(labels.keys()):
            if labels[pred] == 0:
                fp -= 1
            else:
                tp -= 1
            self.stats.append([pred, fp, tp])
        self.stats = np.array(self.stats)

        if plot_roc:
            plt.plot(self.stats[:, 1]/self.class_sizes[0], self.stats[:, 2]/self.class_sizes[1])
            plt.show()
