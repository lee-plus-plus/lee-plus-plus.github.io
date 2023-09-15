'''
demo implementation of Rank-SVM
author: leeython (leezhuoming@qq.com)

[1] Elisseeff, André and Jason Weston. “Kernel methods for Multi-labelled classification and Categorical regression problems.” NIPS 2001 (2001).
'''

import numpy as np
import scipy.sparse as sp

# 1. Initialization

n = 100
m = 20
q = 10
c = 100.0

X = np.random.rand(n, m-1)
X = np.hstack([X, np.ones((n, 1))])
# Y = np.random.randint(0, 2, [n, q])
Y = ((X @ np.random.randn(m, q) + 0.1 * np.random.randn(n, q) ) > 0.0).astype(int)

# tilde_Y[i,p,q] = y[i,p] * (1 - y[i,q])
#       C[k,p,q] = (p==k) - (q==k)
tilde_Y = np.einsum('ip,iq->ipq', Y, 1 - Y).astype(bool)
I = np.eye(q)
C = np.einsum('pk,qq->kpq', I, I) - np.einsum('qk,pp->kpq', I, I)

# Lambda[i, p, q] =  c / (|Y_i| * |bar_Y[i]|)
Y_size = np.einsum('ij->i', Y)
bar_Y_size = np.einsum('ij->i', 1 - Y)
Lambda = 1.0 / (Y_size * bar_Y_size)
Lambda[Lambda == np.inf] = 0.0
Lambda = np.einsum('i,pp,qq->ipq', Lambda, I, I)

alpha = np.zeros((n,q,q))

# vecterization
vec_tilde_Y = tilde_Y.reshape((n, -1))
vec_C = C.reshape((q, -1))
vec_Lambda = Lambda.reshape((-1,))
vec_alpha = alpha.reshape((-1,))

vec_C_sp = sp.csc_matrix(vec_C)
ker_X = X @ X.T
ker_C = vec_C_sp.T @ vec_C_sp

# nqq x nqq sparse matrix, need a extremely large memory! 
A = sp.kron(ker_X, ker_C, 'csc')


# 2. Optimization

import cvxopt

def to_cvxopt_spmatrix(X):
    assert isinstance(X, sp.spmatrix)
    X0 = X.tocoo()
    X1 = cvxopt.spmatrix(X0.data, X0.row, X0.col, X0.shape)
    return X1

def to_cvxopt_matrix(X):
    assert isinstance(X, np.ndarray)
    return cvxopt.matrix(X)

def to_cvxopt(X):
    if isinstance(X, np.ndarray):
        return cvxopt.matrix(X)
    elif isinstance(X, sp.spmatrix):
        X0 = X.tocoo()
        X1 = cvxopt.spmatrix(X0.data, X0.row, X0.col, X0.shape)
        return X1
    elif isinstance(X, cvxopt.base.matrix):
        return X
    elif isinstance(X, cvxopt.base.spmatrix):
        return X
    else:
        raise TypeError('unable to transform type {}'.format(type(X)))

def to_spmatrix(X, copy=False):
    if isinstance(X, cvxopt.base.spmatrix):
        X1 = sp.coo_matrix((list(X.V), (list(X.I), list(X.J))), X.size)
        return X1.tocsc()
    elif isinstance(X, cvxopt.base.matrix):
        return sp.csc_matrix(np.array(X))
    elif isinstance(X, np.ndarray):
        return sp.csc_matrix(X)
    elif isinstance(X, sp.spmatrix):
        return X.copy().tocsc() if copy else X.tocsc()
    else:
       raise TypeError('unable to transform type {} to scipy.spmatrix'.format(type(X)))

def to_ndarray(X, copy=False):
    if isinstance(X, cvxopt.base.spmatrix):
        X1 = sp.coo_matrix((list(X.V), (list(X.I), list(X.J))), X.size)
        return np.array(X1.tocsc())
    elif isinstance(X, cvxopt.base.matrix):
        return np.array(X)
    elif isinstance(X, sp.spmatrix):
        return X.toarray()
    elif isinstance(X, np.ndarray):
        return X.copy() if copy else X
    else:
        raise TypeError('unable to transform type {} to numpy.ndarray'.format(type(X)))

# filtering
imu_filter = tilde_Y.reshape((-1,))
A_filtered = A[imu_filter, :][:, imu_filter].tolil()
Lambda_filtered = vec_Lambda[imu_filter]
alpha_filtered = vec_alpha[imu_filter]

print(f"density = {A_filtered.size / A.size:.4f}")


# minimize    (1/2)*x'*P*x + q'*x
# subject to  G*x + s = h
#             A*x = b
#             s >= 0
n2 = alpha_filtered.shape[0]

vec_1 = np.ones((n2, 1))
vec_0 = np.zeros((n2, 1))
P_coff = to_cvxopt(A_filtered)
q_coff = to_cvxopt(-vec_1)
G_coff = to_cvxopt(sp.vstack([-sp.eye(n2), sp.eye(n2)]))
s_coff = to_cvxopt(np.vstack([vec_0, vec_0]))
h_coff = to_cvxopt(np.vstack([vec_0, Lambda_filtered.reshape((-1,1))]))

# solve Quadratic Programming by cvxopt
result = cvxopt.solvers.coneqp(P=P_coff, q=q_coff, G=G_coff, s=s_coff, h=h_coff)
alpha_filtered = to_ndarray(result['x']).flatten()

vec_alpha[imu_filter] = alpha_filtered
alpha = vec_alpha.reshape(alpha.shape)
# w[j,k] = c[k,p,q] * alpha[i,p,q] * x[i,j]
W = np.einsum('kpq,ipq,ij->jk', C, alpha, X)


# 3. Prediction
Y_pred_proba = X @ W

def get_threshold(y_true, y_score, *, margin_score=0.1):
    assert(set(np.unique(y_true)) == {0, 1})
    n, q = y_true.shape
    indices = np.arange(n).reshape((-1, 1))
    reversed_rank = y_score.argsort(axis=1)[:,::-1]
    
    y_true_reversed = y_true[indices, reversed_rank]
    correctness = np.hstack([
        np.zeros((n, 1)), 
        np.cumsum(y_true_reversed * 2 - 1, axis=1)
    ])

    y_score_reversed = y_score[indices, reversed_rank]
    threshold_candidates = np.hstack([
        y_score_reversed[:, 0:1] + margin_score, 
        (y_score_reversed[:, 1:] + y_score_reversed[:, :-1]) / 2,
        y_score_reversed[:, -1:] - margin_score
    ])
    
    optimal_size_small = correctness.argmax(axis=1, keepdims=True)
    optimal_size_large = q - correctness[:, ::-1].argmax(axis=1, keepdims=True)
    optimal_size = (optimal_size_small + optimal_size_large) // 2
    print(correctness)
    print(optimal_size_small.flatten())
    print(optimal_size_large.flatten())

    return threshold_candidates[indices, optimal_size]

threshold = get_threshold(Y, Y_pred_proba)
W_threshold = np.linalg.inv(X.T @ X + 0.5 * np.eye(m)) @ X.T @ threshold
threshold_pred = X @ W_threshold

Y_pred = (Y_pred_proba > threshold_pred).astype(int)

print(f"Y_true = \n{Y[5:10, :].astype(float)}")
print(f"Y_score = \n{np.vectorize(lambda x: round(x, 1))(Y_pred_proba[5:10, :])}")
print(f"Y_score = \n{Y_pred[5:10, :].astype(float)}")

import matplotlib.pyplot as plt

plt.subplot(1, 3, 1)
plt.imshow(Y[5:20, :])
plt.subplot(1, 3, 2)
plt.imshow(Y_pred_proba[5:20, :])
plt.subplot(1, 3, 3)
plt.imshow(Y_pred[5:20, :])
plt.tight_layout()
plt.show()
