import numpy as np
import matplotlib.pyplot as plt

p = np.linspace(0, 1, 1000)
n_p = 0
a = 5
b = 6
facteur = 1/((a/(a+b))**a * (b/(a+b))**b)
cloche = facteur * p**a * (1 - p)**b
Y = 0.5*cloche + p

# Y = -20*p**7 + 70*p**6 - 84*p**5 + 35*p**4 + p**2
# plt.plot(p, Y)
# plt.plot(p, cloche, 'r')
Y = 10*p**3 - 15*p**4 + 6*p**5
# Y = 70*p**4 - 140*p**5 + 84*p**6 - 20*p**7
plt.plot(p, Y)


plt.show()