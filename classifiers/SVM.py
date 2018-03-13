# -*- coding: utf-8 -*-

import numpy as np


def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).
  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.
  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength
  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    diff_count = 0
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        diff_count += 1
        dW[:, j] += X[i] # gradient update for incorrect rows
        loss += margin
    # gradient update for correct row
    dW[:, y[i]] += -diff_count * X[i]

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train
  dW += reg*W # regularize the weights
  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)

  return loss, dW

def svm_loss_vectorized(W, X, Y, reg):
    """
    :param X: 200 X 3073
    :param Y: 200
    :param W: 3073 X 10
    :return: reg: 正则化损失系数（无法通过拍脑袋设定，需要多试几个值，然后找个最优的）
    """
    delta = 1.0
    num_train = X.shape[0]

    patch_X = X  # 200 X 3073
    patch_Y = Y  # 200

    patch_result = patch_X.dot(W)  # 200 X 3073 3073 X 10 -> 200 X 10

    sample_label_value = patch_result[[xrange(patch_result.shape[0])], patch_Y]  # 1 X 200 切片操作，将得分array中标记位置的得分取出来作为新的array
    loss_array = np.maximum(0, patch_result - sample_label_value.T + delta)  # 200 X 10 计算误差
    loss_array[[xrange(patch_result.shape[0])], patch_Y] = 0  # 200 X 10 将label值所在的位置误差置零

    loss = np.sum(loss_array)

    loss /= num_train  # get mean

    # regularization: 这里给损失函数中正则损失项添加了一个0.5参数，是为了后面在计算损失函数中正则化损失项的梯度时和梯度参数2进行抵消
    loss += 0.5 * reg * np.sum(W * W)

    # 将loss_array大于0的项（有误差的项）置为1，没误差的项为0
    loss_array[loss_array > 0] = 1  # 200 X 10

    # 没误差的项中有一项是标记项，计算标记项的权重分量对误差也有共享，也需要更新对应的权重分量
    # loss_array中这个参数就是当前样本结果错误分类的数量
    loss_array[[xrange(patch_result.shape[0])], patch_Y] = -np.sum(loss_array, 1)

    # patch_X:200X3073  loss_array:200 X 10   -> 10*3072
    dW = np.dot(np.transpose(patch_X), loss_array)  # 3073 X 10
    dW /= num_train  # average out weights
    dW += reg * W  # regularize the weights

    return loss, dW