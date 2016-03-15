import numpy as np
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
import time
import matplotlib.pyplot as plt
from numpy.random import multivariate_normal
import cvxopt.solvers
import warnings

warnings.filterwarnings('ignore', message='elementwise comparison failed')


__author__ = 'Stepan'


def plot_2d(X, Y, SV, desicion_function):
    """
    support function to plot samples, support vector and decision boundary
    """
    fignum = 1
    plt.figure(fignum, figsize=(8, 8))
    plt.clf()

    if SV.shape != (1, 0):
        plt.scatter(SV[:, 0], SV[:, 1], s=80,
                    facecolors='none', zorder=10)
    plt.scatter(X[:, 0], X[:, 1], c=Y, zorder=10, cmap=plt.cm.Paired)

    plt.axis('tight')
    x_min = np.min(X[:, 0]) - 4
    x_max = np.max(X[:, 0]) + 4
    y_min = np.min(X[:, 1]) - 4
    y_max = np.max(X[:, 1]) + 4

    XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
    Z = desicion_function(np.c_[XX.ravel(), YY.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(XX.shape)
    plt.figure(fignum, figsize=(6, 6))
    plt.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.Paired)
    plt.contour(XX, YY, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'],
                levels=[-.5, 0, .5])

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    plt.xticks(())
    plt.yticks(())
    plt.show()


def data_generate(size, rs=73):
    """
    generate model data
    """
    mean1 = np.array([2, 2])
    mean2 = np.array([4, 4])
    cov = np.eye(2)
    np.random.seed(rs)
    X1 = multivariate_normal(mean1, cov, size=size)
    X2 = multivariate_normal(mean2, cov, size=size)
    return X1, X2

def data_generate_N(d, m1 = 2, m2 = 5, rs=73):
    mean1 = np.ones(d[1]) * m1
    mean2 = np.ones(d[1]) * m2
    cov = np.eye(d[1])
    np.random.seed(rs)
    M1 = multivariate_normal(mean1, cov, size=d[0])
    M2 = multivariate_normal(mean2, cov, size=d[0])
    return M1, M2

def accuracy(y_real, y_recived):
    if y_real.shape[0] != y_recived.shape[0]:
        raise ValueError('Wrong vector length')
    return np.sum(y_real == y_recived)/float(y_real.shape[0])


class my_svm:
    def build_kernel_matrix(self, X1, X2, gamma):
        if gamma != 0:
            res = np.exp(-gamma * ((X1 ** 2).sum(axis=1)[:, np.newaxis]
                                    + (X2 ** 2).sum(axis=1)[np.newaxis, :] - 2 * X1.dot(X2.T)))
        else:
            res = np.sum((X1[np.newaxis, :, :] * X2[:, np.newaxis, :]), axis=2)
        return res

    def plot_2d_line(self, normal, bias, rng=np.array(range(0, 10))):
        y = -normal[0]/normal[1] * rng - bias/normal[1]
        return rng, y

    def __init__(self, solver_type='liblinear'):
        self.solver = solver_type
        self.fitted = False

    def compute_primal_objective(self, X, y, w, C):
        """
        :param X: numpy array (N x D), Training set
        :param y: target variable (N x 1)
        :param w: SVM weight
        :param C: l2 regularization param
        :return: value of primal objective
        """
        val = np.sum(np.abs(w[1:])) * np.sum(np.abs(w)[1:]) * 0.5
        s2 = 1 - y * (w[1:].dot(X.transpose()) + w[0])
        #print s2
        val += np.sum(s2 * (s2 > 0)) * C
        return val

    def compute_dual_objective(self, X, y, a, gamma):
        """
        :param X: numpy array (N x D), Training set
        :param y: target variable (N x 1)
        :param w: SVM weight
        :param C: l2 regularization param
        :param gamma: RBF kernel width
        :return: value of dual objective
        """
        if self.gamma != gamma:
            warnings.warn("IN DUAL OBJECTIVE COMPUTING: gamma not from this model!")
        kernel_matrix = self.build_kernel_matrix(X, X, gamma)
        return -(a.sum()) + 0.5 * ((a * y)[np.newaxis, :] * (a * y)[:, np.newaxis] * kernel_matrix).sum()

    def dual_to_primal_weights(self, X, y, a):
        """
        :param X: numpy array (N x D), Training set
        :param y: target variable (N x 1)
        :param a: dual weights
        :return: primal weights
        """
        w = (X * a[:, np.newaxis] * y[:, np.newaxis]).sum(axis=0)
        return np.hstack((np.array([0]), w))

    def subgradient_predict(self, X):
        """
        :param X: numpy array of objects (N x D)
        :return: predictions (N x 1)
        """
        res = X.dot(self.weight[1:]) + self.weight[0]
        return res

    def dual_predict(self, X):
        """
        :param X: numpy array of objects (N x D)
        :return: predictions (N x 1)
        """
        if self.gamma == 0:
            return X.dot(self.weight[1:]) + self.weight[0]
        new_kernel_matrix = self.build_kernel_matrix(X, self.train_X, self.gamma)
        res = ((self.a * self.y_train)[np.newaxis, :] * new_kernel_matrix).sum(axis=1) + self.weight[0]
        return res

    def subgradient_step(self, X, y, w, C, learning_rate):
        """
        make subgradient step
        :param X: numpy array (N x D), training set
        :param y: numpy array (N x 1), target variable
        :param w: model's weights (D x 1)
        :param C: l2 regularization weight
        :param learning_rate: learning rate
        :return: new vector of weights
        """
        X_extend = np.hstack((np.ones(X.shape[0])[:, np.newaxis], X))
        s2 = C * y[:, np.newaxis] * X_extend * (y * X_extend.dot(w) < 1)[:, np.newaxis]
        grad = w - np.sum(s2, axis=0)
        grad = grad/np.sqrt(np.sum(grad * grad))
        w = w - learning_rate * grad
        return w

    def fit_subgradient_solver(self, X, y, C, tol=1e-6, max_iter=100, verbose=False, it=0, every_it=0,
                               beta=1, alpha=1.0):
        """
        :param y: numpy array (N x 1), target variable
        :param X: numpy array (N x D), training set
        :param C: l2 regularization weight
        :param tol: Tolerance for stopping criteria
        :param max_iter: number of max iteration
        :param verbose: verbose
        :param it: current iteration number
        :param every_it: how often add value of objective
        :param beta: power in subgradient step
        :param alpha: coefficient in subgradient step
        :return: 0 if converged, else 1
        """
        w = self.weight
        self.status = 1

        prev_value = self.compute_primal_objective(X, y, w, C)
        objective_curve = self.objective_curve
        for i in range(max_iter):
            w = self.subgradient_step(X, y, w, C, float(alpha)/((it + i + 1)**beta))
            val = self.compute_primal_objective(X, y, w, C)
            if every_it != 0:
                if i % every_it == 0:
                    self.objective_curve.append(val)
            else:
                objective_curve.append(val)
            if (verbose):
                print 'Iteration #', i
                print 'Value of cost function = ', val
            if np.abs(prev_value - val) < tol:
                self.status = 0
                break
        self.decision_funtion = self.subgradient_predict
        self.weight = w
        self.fitted = True
        self.objective_curve = objective_curve
        return self.status

    def fit_batch_subgradient(self, X, y, C, tol=1e-6, max_iter=100, verbose=False, batch_size = 200):
        """
        :param y: numpy array (N x 1), target variable
        :param X: numpy array (N x D), training set
        :param C: l2 regularization weight
        :param tol: Tolerance for stopping criteria
        :param max_iter: number of max iteration
        :param verbose: verbose
        :param batch_size: size of batch in subgradient step
        :return: None
        """
        iteration_in_batch = 100
        for i in range(int(float(max_iter)/iteration_in_batch)):
            for j in range(X.shape[0]/batch_size):
                indx = range(X.shape[0])
                np.random.shuffle(indx)
                sub_X = X[indx[0:batch_size]]
                self.fit_subgradient_solver(sub_X, y[indx[0:batch_size]], C, tol, max_iter=100,
                                            verbose=verbose, it=(i + 1)*(j + 1),
                                            every_it=int(float(X.shape[0])/batch_size))

                if self.status == 0:
                    return
                if verbose:
                    print i * j, ' iteration, objective curve value: ', self.objective_curve[-1]

    def fit_svm_liblinear_solver(self, X, y, C, tol=1e-6, max_iter=100, verbose=False):
        """
        :param y: numpy array (N x 1), target variable
        :param X: numpy array (N x D), training set
        :param C: l2 regularization weight
        :param tol: Tolerance for stopping criteria
        :param max_iter: number of max iteration
        :param verbose: verbose
        :return fitted LinearSVC object
        """
        w = np.ones(X.shape[1] + 1)
        self.status = 1

        predictor = LinearSVC(C=C, tol=tol, max_iter=max_iter, verbose=verbose)
        predictor.fit(X, y)
        w[1:] = predictor.coef_[0]
        w[0] = predictor.intercept_

        self.weight = w
        self.fitted = True
        self.decision_funtion = predictor.decision_function
        return predictor

    def fit_svc_solver(self, X, y, C, tol=1e-6, max_iter=100, verbose=False, gamma=0):
        """
        :param y: numpy array (N x 1), target variable
        :param X: numpy array (N x D), training set
        :param C: l2 regularization weight
        :param tol: Tolerance for stopping criteria
        :param max_iter: number of max iteration
        :param verbose: verbose
        :return fitted SVC object
        """
        self.status = 1
        w = np.ones(X.shape[1] + 1)
        prev_value = self.compute_primal_objective(X, y, w, C)
        prev_sv = np.array([])
        prev_dual_coef = np.array([[]])
        objective_curve = [prev_value]
        if gamma == 0:
            for i in range(1):
                predictor = SVC(kernel='linear', verbose=verbose)
                predictor.support_ = prev_sv
                predictor.dual_coef_ = prev_dual_coef
                predictor.fit(X, y)
                w[1:] = predictor.coef_
                w[0] = predictor.intercept_
                val = self.compute_primal_objective(X, y, w, C)
                objective_curve.append(val)
                if np.abs(prev_value - val) < tol:
                    self.status = 0
                    break
                prev_value = val
                prev_sv = predictor.support_
                prev_dual_coef = predictor.dual_coef_
                self.weight = w
        else:
            objective_curve = [0]
            prev_value = 0
            for i in range(1):
                predictor = SVC(gamma=gamma, verbose=verbose)
                predictor.support_ = prev_sv
                predictor.dual_coef_ = prev_dual_coef
                predictor.fit(X, y)
                self.dual_coef = np.zeros(X.shape[0])
                self.dual_coef[predictor.support_] = predictor.dual_coef_ * y[predictor.support_]
                val = self.compute_dual_objective(X, y, self.dual_coef, gamma)
                objective_curve.append(val)
                if np.abs(prev_value - val) < tol:
                    self.status = 0
                    break
                prev_value = val
                prev_sv = predictor.support_
                prev_dual_coef = predictor.dual_coef_
                self.dual_coef = prev_dual_coef

        self.fitted = True
        self.predictor = predictor
        self.decision_funtion = self.predictor.decision_function
        self.objective_curve = np.array(objective_curve)
        return predictor

    def fit_qp_solver_primal(self, X, y, C, tol=1e-6, max_iter=100, verbose=False):
        """
        :param y: numpy array (N x 1), target variable
        :param X: numpy array (N x D), training set
        :param C: l2 regularization weight
        :param tol: Tolerance for stopping criteria
        :param max_iter: number of max iteration
        :param verbose: verbose
        :return None
        """
        cvxopt.solvers.options['maxiters'] = 1
        cvxopt.solvers.options['show_progress'] = verbose
        cvxopt.solvers.options['abstol'] = tol
        self.status = 1
        w = np.ones(X.shape[1] + 1)
        P = cvxopt.matrix(np.diag([1] * X.shape[1] + [0] * (X.shape[0] + 1)), tc='d')

        q = cvxopt.matrix(np.array([0] * (X.shape[1] + 1) + [C] * X.shape[0]), tc='d')

        G = np.zeros((2 * X.shape[0], X.shape[0] + X.shape[1] + 1))
        G[:X.shape[0], 0] = y
        G[:X.shape[0], 1:X.shape[1] + 1] = X * y[:, np.newaxis]
        G[X.shape[0]:, X.shape[1] + 1:] = np.eye(X.shape[0])
        G[:X.shape[0], X.shape[1] + 1:] = np.eye(X.shape[0])
        G = cvxopt.matrix(-G, tc='d')

        h = cvxopt.matrix(np.array([-1] * X.shape[0] + [0] * X.shape[0]), tc='d')

        objective_curve = []

        for i in range(max_iter):
            sol = cvxopt.solvers.qp(P, q, G, h, initvals=w)
            w = sol
            self.weight = np.array(w['x'])[:, 0]
            self.weight = self.weight[:X.shape[1] + 1]
            val = self.compute_primal_objective(X, y, self.weight, C)
            objective_curve.append(val)
            if sol['status'] == 'optimal':
                self.status = 0
                break

        self.decision_funtion = self.subgradient_predict
        self.fitted = True
        w = np.array(w['x'])[:, 0]
        w = w[:X.shape[1] + 1]
        self.weight = w
        self.objective_curve = np.array(objective_curve)


    def fit_qp_solver_dual(self, X, y, C, tol=1e-6, max_iter=100, verbose=False, gamma=0):
        """
        :param y: numpy array (N x 1), target variable
        :param X: numpy array (N x D), training set
        :param C: l2 regularization weight
        :param tol: Tolerance for stopping criteria
        :param max_iter: number of max iteration
        :param verbose: verbose
        :param gamma: RBF kernel width
        :return None
        """
        self.train_X = X
        self.gamma = gamma
        kernel_matrix = self.build_kernel_matrix(X, X, gamma)
        self.kernel_matrix = kernel_matrix

        P = cvxopt.matrix(kernel_matrix * y[np.newaxis, :] * y[:, np.newaxis], tc='d')
        q = cvxopt.matrix(- np.ones(X.shape[0]), tc='d')
        G = cvxopt.matrix(np.vstack((np.eye(X.shape[0]), -np.eye(X.shape[0]))), tc='d')
        h = cvxopt.matrix(np.hstack((np.ones(X.shape[0]) * C, np.zeros(X.shape[0]))), tc='d')
        A = cvxopt.matrix(y[np.newaxis, :], tc='d')
        b = cvxopt.matrix(0, tc='d')

        cvxopt.solvers.options['maxiters'] = 1
        cvxopt.solvers.options['abstol'] = tol
        cvxopt.solvers.options['show_progress'] = verbose

        eps = 0.1
        w = np.ones(X.shape[1] + 1)
        objective_curve = []
        for i in range(max_iter):
            sol = cvxopt.solvers.qp(P, q, G, h, A, b, initvals=w)
            w = sol
            self.weight = self.dual_to_primal_weights(X, y, np.array(w['x'])[:, 0][:X.shape[0]])
            self.weight[0] = np.array(w['y'])[:, 0]
            self.dual_coef = np.array(w['x'])[:, 0]
            objective_curve.append(self.compute_dual_objective(X, y, np.array(w['x'])[:, 0], gamma))
            if sol['status'] == 'optimal':
                self.status = 0
                break

        a = np.array(w['x'])[:, 0]
        self.y_train = y
        self.a = a[:X.shape[0]]
        self.weight = self.dual_to_primal_weights(X, y, a)
        self.weight[0] = np.array(w['y'])[:, 0]
        self.decision_funtion = self.dual_predict
        self.fitted = True

        self.support_vectors = X[(self.a > eps)]
        self.objective_curve = np.array(objective_curve)

    def fit(self, X, y, C, tol=1e-6, max_iter=100, verbose=False, gamma=0, plot=True, alpha=1.0, beta=1, batch_size=200):
        """
        :param X: numpy array (N x D), training set
        :param y: numpy array (N x 1), target variable
        :param C: regularization param
        :param tol: tolerance stop criteria
        :param max_iter: number of max iteration
        :param verbose: verbose
        :param gamma: width of RBF kernel (linear if 0)
        :param plot: do you want plot it all?
        :param beta: power in subgradient step
        :param alpha: coefficient in subgradient step
        :param batch_size: size of butch in subgradient
        :return: dict: {
                            'weights': numpy array (D x 1), array of SVM coefficient
                            'status': converged?
                            'time': time to execution
                            'objective': value of objective function
                        }
        """
        start_time = time.time()
        self.status = 1
        self.objective_curve = []
        if self.solver == 'liblinear':
            clf = self.fit_svm_liblinear_solver(X, y, C, tol, max_iter, verbose)
            end_time = time.time()
            if plot:
                plot_2d(X, y, np.array([[]]), clf.decision_function)

        if self.solver == 'SVC':
            self.gamma = gamma
            clf = self.fit_svc_solver(X, y, C, tol, max_iter, verbose, gamma=gamma)
            end_time = time.time()
            if plot:
                plot_2d(X, y, clf.support_vectors_, clf.decision_function)

        if self.solver == 'subgradient':
            self.objective_curve = []
            self.weight = np.ones(X.shape[1] + 1)
            self.fit_subgradient_solver(X, y, C, tol, max_iter, verbose, alpha=alpha, beta=beta)
            end_time = time.time()
            if plot:
                plot_2d(X, y, np.array([[]]), self.subgradient_predict)

        if self.solver == 'batchsubgradient':
            self.objective_curve = []
            self.weight = np.ones(X.shape[1] + 1)
            self.fit_batch_subgradient(X, y, C, tol, max_iter, verbose, batch_size=batch_size)
            end_time = time.time()
            if plot:
                plot_2d(X, y, np.array([[]]), self.subgradient_predict)

        if self.solver == 'qpprimal':
            self.fit_qp_solver_primal(X, y, C, tol, max_iter, verbose)
            end_time = time.time()
            if plot:
                plot_2d(X, y, np.array([[]]), self.subgradient_predict)

        if self.solver == 'qpdual':
            self.fit_qp_solver_dual(X, y, C, tol, max_iter, verbose, gamma=gamma)
            end_time = time.time()
            if plot:
                plot_2d(X, y, self.support_vectors, self.dual_predict)

        if type(self.objective_curve) == 'list':
            self.objective_curve = np.array(self.objective_curve)

        if not self.fitted:
            raise ValueError

        result = 0
        if (self.solver == 'subgradient') | (self.solver == 'qpprimal') | (self.solver == 'batchsubgradient') \
                | (self.solver == 'liblinear') | ((self.solver == 'SVC') & (gamma == 0)):
            result = {'w': self.weight,
                      'status': self.status,
                      'time': end_time - start_time,
                      'objective': self.objective_curve}
        else:
            result = {'A': self.dual_coef,
                      'status': self.status,
                      'time': end_time - start_time,
                      'objective': self.objective_curve}

        return result

    def predict(self, X):
        """
        :param X: numpy array (N x D) of sample
        :return: numpy array (N x 1), predictions
        """
        if not self.fitted:
            raise ValueError("You need fit me first")
        return (self.decision_funtion(X) > 0) * 1 - 1 * (self.decision_funtion(X) < 0)
