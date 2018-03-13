# -*- coding: utf-8 -*-
import numpy as np

from classifiers.KNearestNeighbor import *
from data_utils import *

SAMPLE_CNT = 10

CLASSIFIER_CNT = 10

Xtr, Ytr, Xte, Yte = load_CIFAR10('datasets/')
Xtr_rows = Xtr.reshape(Xtr.shape[0], 32*32*3)
Xte_rows = Xte.reshape(Xte.shape[0], 32*32*3)

knn = NearestNeighbor()

knn.train(Xtr_rows, Ytr)
predict_results = knn.predict(Xte_rows[0:SAMPLE_CNT, :])

print Yte[0:SAMPLE_CNT]
print predict_results
print "accuracy %f" % np.mean(Yte[0:SAMPLE_CNT]==predict_results)