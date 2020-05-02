

import matplotlib.pyplot as plt
import numpy as np

x = [0,20,50,75,100,150,200,250,300,350,400,450,500]
y1=[0,0.56, 0.72, 0.81,0.9,0.94,0.93,0.92,0.95,0.96,0.96,0.96,0.96] 
y2=[0,0.26, 0.36,0.58,0.65,0.73,0.82,0.86,0.84,0.85,0.87,0.88,0.88]
nnet_100_line = plt.plot(x, y1, label='MCTS with NNet')
mcts_100_line = plt.plot(x, y2,label='MCTS without NNet')
plt.xlabel('MCTS iterations')
plt.ylabel('Percent 100 Games within 1.1 of Optimal')
plt.legend()
plt.title(" Plotting the % within 1.1 of optimal versus MCTS Iterations")
plt.show()





