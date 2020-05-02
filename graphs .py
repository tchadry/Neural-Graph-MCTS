

import matplotlib.pyplot as plt
import numpy as np

x = np.arange(0,1000,100):
y1=[0.35, 0.55, 0.55, 0.6, 0.6, 0.65, 0.65, 0.7, 0.7]
y2=[0.8, 0.95, 0.95, 0.95, 1.0, 1.0, 1.0, 1.0, 1.0]
nnet_100_line = plt.plot(x, y1, label='MCTS with NNet)
mcts_100_line = plt.plot(x, y2,'MCTS without NNet')
plt.xlabel('MCTS iterations')
plt.ylabel('Percent Games within 1.1 of Optimal')
plt.legend()
plt.show()
