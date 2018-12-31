# -*- coding: UTF-8 -*-

import  numpy as np
from math import sqrt
#准确度 分类结果的信息
def accuracy_score(y_true,y_predict):
    assert y_true.shape[0] == y_predict.shape[0] ,\
    "the size of y_true must be equal to the size of y_predict"

    return sum(y_true == y_predict)/len(y_true)


def mean_squared_error(y_true,y_predict):
    """计算y_true和y_predict之间的MSE"""
    assert len(y_true) == len(y_predict) , \
        "the size of y_true must be equal to the size of y_predict."
    return np.sum((y_true - y_predict)**2) / len(y_true)

def root_mean_squared_error(y_true,y_predict):
    """计算y_true和y_predict之间的RMSE"""
    return  sqrt(mean_squared_error(y_true,y_predict))

def mean_absolute_error(y_true,y_predict):
    """计算y_true和y_predict之间的RMSE"""
    assert len(y_true) == len(y_predict) ,\
        "the size of y_true must be equal to the size fo y_predict"

    return  np.sum(np.absolute(y_predict - y_true)) / len(y_true)

def r2_square(y_true,y_predict):
    """计算y_true和y_predict之间的R Square"""
    return  1 - mean_squared_error(y_true,y_predict)/np.var(y_true)









