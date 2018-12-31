import  numpy as np
from .metrics import  r2_square
class SimpleLinearRegression1:

    def __init__(self):
        """初始化Simple Linear Regression 模型"""
        self.a_ = None
        self.b_ = None

    def fit(self,x_train,y_train):
        """只有一个特征 只是一个线性向量 根据输入数据训练出模型 """
        assert x_train.ndim == 1 , \
            "simple linear regressor can only solve single feature training data"
        assert  len(x_train) == len(y_train) , \
            "the size of x_train must be equal to the size of y_train"


        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)

        num = 0.0
        d = 0.0
        for x,y in zip(x_train,y_train):
            num += (x - x_mean) * (y - y_mean)
            d += (x - x_mean)**2

        self.a_ = num/d
        self.b_ = y_mean - self.a_ * x_mean

        return self

    def score(self,x_test,y_test):
        """根据测试数据集x_test y_test 确定当前模型的准确度"""
        y_predict = self.predict(x_test)
        return r2_square(y_test,y_predict)
    def predict(self,x_predict):
        """给定x_predict 返回表示x_predict的结果表示的向量"""
        assert  x_predict.ndim == 1 ,\
            "Simple Linear Regressor can only solve single feature training data"
        assert  self.a_ is not None and self.b_ is not None ,\
            "must fit before predict!"

        return np.array([self._predict(x) for x in x_predict])



    def _predict(self,x_single):
        """给定单个待预测数据x_single,返回x_single的预测结果值"""
        return self.a_ * x_single + self.b_


    def __repr__(self):
        return "SimpleLinearRegression1()"


class SimpleLinearRegression2:


    def __init__(self):
        """初始化Simple Linear Regression 模型"""
        self.a_ = None
        self.b_ = None

    def fit(self,x_train,y_train):
        """只有一个特征 只是一个线性向量 根据输入数据训练出模型 """
        assert x_train.ndim == 1 , \
            "simple linear regressor can only solve single feature training data"
        assert  len(x_train) == len(y_train) , \
            "the size of x_train must be equal to the size of y_train"


        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)

        num = (x_train - x_mean).dot(y_train - y_mean)
        d = (x_train - x_mean).dot(x_train - x_mean)


        self.a_ = num/d
        self.b_ = y_mean - self.a_ * x_mean

        return self


    def predict(self,x_predict):
        """给定x_predict 返回表示x_predict的结果表示的向量"""
        assert  x_predict.ndim == 1 ,\
            "Simple Linear Regressor can only solve single feature training data"
        assert  self.a_ is not None and self.b_ is not None ,\
            "must fit before predict!"

        return np.array([self._predict(x) for x in x_predict])

    def score(self,x_test,y_test):
        """根据测试数据集x_test y_test 确定当前模型的准确度"""
        y_predict = self.predict(x_test)
        return r2_square(y_test,y_predict)

    def _predict(self,x_single):
        """给定单个待预测数据x_single,返回x_single的预测结果值"""
        return self.a_ * x_single + self.b_


    def __repr__(self):
        return "SimpleLinearRegression1()"