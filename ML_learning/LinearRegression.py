import  numpy as np
from .metrics import  r2_square


class LinearRegression:

    def __init__(self):
        """初始化Linear Regression模型"""
        self.coef_ = None#系数
        self.interception_ = None#截距
        self._theta = None

    def fit_normal(self,X_train,y_train):#X_train矩阵 对应一个样本只有一个输出
        """根据训练数据集X_train,y_train训练Linear Regression模型"""
        assert X_train.shape[0] == y_train.shape[0] ,\
            "the size of X_train must be equal to the size of y_train"

        X_b = np.hstack([np.ones((len(X_train),1)),X_train])#x_train + one column
        #hstack horizontal 水平多加一列
        self._theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)
        #np.linalg.inv 求逆 对应
        self.interception_ = self._theta[0]
        self.coef_ = self._theta[1:]

        return  self



    def fit_gd(self,X_train,y_train,eta=0.01,n_iters=1e4):
        """根据训练数据集X_train y_train 使用梯度下降法训练linear regression模型"""
        assert X_train.shape[0] == y_train.shape[0] , \
            "the size of X_train must be equal to the size of y_train"
        def J(theta,x_b,y):
            try:
                return np.sum((y - x_b.dot(theta))**2) / len(y)
            except:
                return  float('inf')

        def dJ(theta, x_b, y):  # 求导数
            # res = np.empty(len(theta))  # 开辟一个theta大小的空间 j对theta每个维度求导
        #             # res[0] = np.sum(x_b.dot(theta) - y)  # 第一维度 很好求
        #             # for i in range(1, len(theta)):
        #             #     res[i] = (x_b.dot(theta) - y).dot(x_b[:, i])  # 观察上面式子 x_{2}^{(i)} 第一维度都需要有结果 第二维度取指定的值
        #             # return res * 2 / len(x_b)
            return  X_b.T.dot(X_b.dot(theta) - y) * 2. / len(y)

        # 相对单个特征值 只需要修改 x_b y 以及histroy删除即可
        def gradient_descent(x_b, y, initial_theta, eta, n_iters=1e4, epsilon=1e-8):
            theta = initial_theta
            i_iter = 0
            while i_iter < n_iters:
                gradient = dJ(theta, x_b, y)
                last_theta = theta
                theta = theta - eta * gradient

                if (abs(J(theta, x_b, y) - abs(J(last_theta, x_b, y))) < epsilon):
                    break
                i_iter += 1
            return theta

        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])  # x_train + one column
        # hstack horizontal 水平多加一列
        initial_theta = np.zeros(X_b.shape[1])
        self._theta = gradient_descent(X_b,y_train,initial_theta,eta,n_iters=1e4)
        # np.linalg.inv 求逆 对应
        self.interception_ = self._theta[0]
        self.coef_ = self._theta[1:]

        return self


    def fit_sgd(self,X_train,y_train,n_iters = 1e4,t0=5,t1 = 50):
        #n_iters:在随机梯度中表示 测试对所有样本serveral times 在对应的代码需要更改 见problem
        """根据训练数据集X_train y_train 使用随机梯度下降法训练linear regression"""
        assert  X_train.shape[0] == y_train.shape[0] , \
            "the size of X_train must be equal to the size of y_train "
        #相应的随机梯度方向
        def dJ_sgd(theta,X_b_i,y_i):
            return X_b_i * (X_b_i.dot(theta) - y_i) * 2
        #
        def sgd(X_b,y,initial_theta,n_iters,t0=5,t1=50):

            def learning_rate(t):
                return t0 / (t + t1)

            theta = initial_theta
            m = len(X_b)
            #由于随机梯度 可能会有样本看不到 solution：iteration twice
            for cur_iter in range(n_iters):
                #对所有样本进行乱序排序
                indexs = np.random.permutation(m)
                X_b_new = X_b[indexs]
                y_new = X_b[indexs]
                for i in range(m):
                    gradient = dJ_sgd(theta,X_b_new[i],y_new[i])
                    theta = theta - learning_rate(cur_iter) * gradient

            return  theta

        X_b = np.hstack([np.ones((len(X_train),1)),X_train])




            

    def predict(self,X_predict):
        """给定待预测数据集X_predict,返回X_predict的结果向量"""
        assert  self.interception_ is  not None and self.coef_ is not None , \
            "must fit before predict!"
        #系数对应每一个预测样本的特征
        assert  X_predict.shape[1] == len(self.coef_) , \
            "the feature number of X_predict must be equal to X_train"
        #计算x_b
        X_b = np.hstack([np.ones((len(X_predict),1)),X_predict])
        return X_b.dot(self._theta)

    def score(self,X_test,y_test):
        """根据测试数据集 X_test y_test确定当前模型的准确度"""
        y_predict = self.predict(X_test)
        return r2_square(y_test,y_predict)

    def __repr__(self):
        return "LinearRegression()"
