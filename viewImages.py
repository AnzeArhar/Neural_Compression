from matplotlib import pyplot as plt
import numpy as np

lines = np.loadtxt("out.txt", delimiter=",")

for i in range(0, len(lines)):
    plt.subplot(4, int(len(lines)/4), i+1)
    plt.imshow(lines[i].reshape(28, 28))
plt.show()