import numpy as np


class NearestNeighbor(object):
    def __init__(self):
        pass

    def train(self, X, y):
        self.X = X
        self.y = y

    def predict(self, Sample):
        sample_num = Sample.shape[0]
        predict_results = np.zeros(sample_num, dtype=self.y.dtype)

        for i in xrange(sample_num):
            print "current index: %d" % i
            total_diffs = np.sum(np.abs(self.X - Sample[i, :]), axis=1)
            type_label = np.argmin(total_diffs)
            predict_results[i] = self.y[type_label]

        return predict_results