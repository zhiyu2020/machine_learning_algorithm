# -*- coding: UTF-8 -*-

import  numpy as np
from math import  sqrt
from collections import  Counter
from   .metrics  import  accuracy_score

class KNNClassifier:
    def __init__(self,k):
        """初始化KNN分类器"""
        assert  k>=1, \
            "k must be valid"
        self.k = k
        """        x_train y_train训练数据集
        """
        self._X_train = None
        self._y_train = None

    def fit(self,X_train,y_train):
        """根据训练数据集X_train 和 y_train训练KNN分类器
        遵守siciket-learn 标准 可以送到其他方法中"""
        assert  X_train.shape[0] == y_train.shape[0], \
          "the size of X_train must be equal to the size of y_train"
        assert  self.k <= X_train.shape[0] , \
          "the size of X_train must be at least k"

        self._X_train = X_train
        self._y_train = y_train
        return self

    def predict(self,X_predict):
        """        给定待预测数据集X_predict,返回表示X_predict的结果向量
"""

        assert self._X_train is not None and self._y_train is not None, \
           "must fit before predict!"
        assert  X_predict.shape[1] == self._X_train.shape[1] , \
           "the feature number of X_predict must be equal to X_train"

        y_predict = [self._predict(x) for x in X_predict]
        return  np.array(y_predict)


    def _predict(self,x):
        """给定单个待预测数据x,返回x的预测结果值"""
        assert  x.shape[0] == self._X_train.shape[1] , \
          "the feature number of x must be equal to X_train"

        distances = [sqrt(np.sum((x_train - x)**2))
                     for x_train in self._X_train]

        nearest = np.argsort(distances)

        topK_y = [self._y_train[i] for i in nearest[:self.k]]
        votes = Counter(topK_y)

        return votes.most_common(1)[0][0]

    def score(self,x_test,y_test):
        """封装在方法中"""
        y_predict = self.predict(x_test)
        return accuracy_score(y_predict,y_test)



    def __repr__(self):
        return "KNN(k =%d)" %self.k

# if __name__ == '__main__':
#         knn_clf = KNNClassifier(k = 6)
#
#         raw_data_x = [[3.333, 2.313],
#                       [3.111, 1.781],
#                       [1.343, 3.368],
#                       [3.582, 4.679],
#                       [2.280, 2.866],
#                       [7.423, 4.696],
#                       [5.745, 3.533],
#                       [9.172, 2.511],
#                       [7.792, 3.424],
#                       [7.939, 0.791]]
#         raw_data_y = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
#
#         X_train = np.array(raw_data_x)
#         y_train = np.array(raw_data_y)
#
#         x = np.array([8.093, 3.365])
#         x_predict = x.reshape(1,-1)
#         knn_clf.fit(X_train, y_train)
#
#
#         y_predict = knn_clf.predict(x_predict)
#         print(y_predict[0])