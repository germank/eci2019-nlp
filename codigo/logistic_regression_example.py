import numpy as np
import tqdm
import timeit
np.random.seed(42)

d = 100
N = 1000
# gold value
w_star = np.random.randn(d)
b_star = np.random.randn(1)
# training data
X = np.random.randn(N, d)
y = (X.dot(w_star) + b_star >= 0)
#testing data
X_te = np.random.randn(N, d)
y_te = (X_te.dot(w_star) + b_star >= 0)


def main():
    w = np.zeros(d)
    b = np.zeros(1)
    theta = [w, b]
    lr = 0.01
    T=10000
    with tqdm.trange(T) as t:
        for i in t:
            t.set_postfix(accuracy=get_accuracy(*theta))
            w_grad, b_grad = get_grad_vec(*theta)
            w -= lr * w_grad
            b -= lr * b_grad

def get_accuracy(w, b):
    y_hat = X_te.dot(w) >= 0
    return (y_te == y_hat).sum()/y.shape[0]

def get_grad_loop(w, b):
    w_grad = np.zeros(d)
    b_grad = np.zeros(1)
    for i in range(X.shape[0]):
        y_hat = sigmoid(X[i].dot(w) + b)
        e = (y_hat - y[i])
        w_grad += e * X[i]
        b_grad += e
    w_grad /= X.shape[0]
    b_grad /= X.shape[0]
    return w_grad, b_grad

def get_grad_vec(w, b):
    y_hat = sigmoid(X.dot(w) + b)
    e = (y_hat - y)
    w_grad = np.matmul(X.T, e)/X.shape[0]
    b_grad = e.mean(0)
    return w_grad, b_grad


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def test_impl():
    w = np.zeros(d)
    b = np.zeros(1)
    w_grad_vec, b_grad_vec = get_grad_vec(w, b)
    w_grad_loop, b_grad_loop = get_grad_loop(w, b)
    assert np.isclose(w_grad_vec,w_grad_loop).all()
    assert np.isclose(b_grad_vec,b_grad_loop).all()

def time_impl():
    global get_grad_vec, get_grad_loop
    w = np.zeros(d)
    b = np.zeros(1)
    T = 1000
    print("Timing function performances...")
    tloop =timeit.timeit(lambda: get_grad_loop(w, b), number=T)/T
    print(f"Loop implementation time: {tloop:.2e} s.")
    tvec = timeit.timeit(lambda: get_grad_vec(w, b), number=T)/T
    print(f"Vectorial implementation time: {tvec:.2e} s.")
    print("Vectorial is {:.2f}x faster".format(tloop/tvec))

test_impl()
time_impl()
main()

