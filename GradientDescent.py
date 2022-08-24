import numpy as np
import matplotlib.pyplot as plt

xData = [1.0, 2.0, 3.0]
yData = [2.0, 4.0, 6.0]

w = 0


def forecast(x: float) -> float:
    return x * w


# calculate average cost
def cost(xs: list, ys: list) -> float:
    cost = 0
    for x, y in zip(xs, ys):
        yPred = forecast(x)
        cost += (yPred - y) ** 2

    return cost / len(xs)


# calculate average gradient
def gradient(xs: list, ys: list) -> float:
    grad = 0
    for x, y in zip(xs, ys):
        grad += 2 * x * (x * w - y)

    return grad / len(xs)


for epoch in range(100):
    costVal = cost(xData, yData)
    gradVal = gradient(xData, yData)

    # parameter = 0.01
    w -= 0.01 * gradVal

    print('Epoch:', epoch, 'w =', w, 'loss =', costVal)

print('Predict after training', 4, forecast(4))
