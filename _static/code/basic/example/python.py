a = 0
b = 0

def f(x):
    y_pred = a * x + b
    return y_pred

def loss(x, y):
    l = (a * x + b - y) ** 2
    return l

def gradient_loss(x, y):
    g_a = 2 * (a * x + b - y) * x
    g_b = 2 * (a * x + b - y)
    return g_a, g_b

X_raw = [2013, 2014, 2015, 2016, 2017]
Y_raw = [12000, 14000, 15000, 16500, 17500]
x_pred_raw = 2018
X = [(x - min(X_raw)) / (max(X_raw) - min(X_raw)) for x in X_raw]
Y = [(y - min(Y_raw)) / (max(Y_raw) - min(Y_raw)) for y in Y_raw]

num_epoch = 10000
learning_rate = 1e-3
for e in range(num_epoch):
    for i in range(len(X)):
        x, y = X[i], Y[i]
        g_a, g_b = gradient_loss(x, y)
        a = a - learning_rate * g_a
        b = b - learning_rate * g_b
print(a, b)
for i in range(len(X)):
    x, y = X[i], Y[i]
    print(f(x), y)
x_pred = (x_pred_raw - min(X_raw)) / (max(X_raw) - min(X_raw))
y_pred = f(x_pred)
y_pred_raw = y_pred * (max(Y_raw) - min(Y_raw)) + min(Y_raw)
print(x_pred_raw, y_pred_raw)