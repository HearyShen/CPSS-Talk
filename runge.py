import matplotlib.pyplot as plo
import matplotlib.pyplot as plt
import numpy as np

x = np.arange(-1, 1, 0.01)

y = 1 / (1 + 25 * x**2)     # Runge Function

plt.title("Function")
plt.plot(x, y)
plt.show()