import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

xData = [1.0, 2.0, 3.0]
yData = [2.0, 4.0, 6.0]

# predict function
def predict(x : float, w : np.ndarray, b : np.ndarray) -> np.ndarray:
    return x * w + b

# loss function
def loss(predictY : np.ndarray, realY : float) -> float:
    return (predictY - realY) ** 2

wList = np.arange(0.0, 4.1, 0.1)
bList = np.arange(-2.0, 2.0, 0.1)
w, b = np.meshgrid(wList, bList)
mse = np.zeros(w.shape)

# enum w and b to predict
for x, y in zip(xData, yData):
    predictY = predict(x, w, b)
    mse += loss(predictY, y)

mse /= len(xData)


# draw
# add axes
fig = plt.figure()
ax = Axes3D(fig)

# After matplotlib 3.4, Axes3D will need to be explicitly added to the figure
fig.add_axes(ax)
plt.xlabel(r'w', fontsize=20, color='cyan')
plt.ylabel(r'b', fontsize=20, color='cyan')
ax.plot_surface(w, b, mse, rstride = 1, cstride = 1, cmap = 'rainbow')
plt.show()