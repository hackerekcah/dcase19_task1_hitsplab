import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt

x = np.linspace(0.01, 0.99, 100)

y = []
for a in [0.1, 0.3,  0.5, 1, 10, 100]:
    y.append(beta.pdf(x, a, a))

plt.plot(x, np.array(y).T)
plt.show()