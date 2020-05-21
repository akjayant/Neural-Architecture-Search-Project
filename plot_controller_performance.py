import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

f = open("controller_performance_tracker_run_2.txt",'r')
l = f.readlines()
f.close()
xx = np.array(l)
xx = [float(i.replace("\n","")) for i in xx]
solved = [200 for i in range(len(list(xx)))]
plt.figure(figsize=(20,10))
plt.plot(range(len(l)),xx,label='Average score of sampled architectures in that iteration')
#plt.axvline(15,label='behavior learned',color='green',ls=('dashed'))
plt.plot(pd.Series(xx).rolling(5).mean(),color='yellow',label = 'rolling mean of 5 iterations')
plt.plot(solved,color='orange',label='solved')
plt.title("Improvement of expected rewards of sampled architectures")
plt.xlabel('# of iterations')
plt.ylabel('Average score of sampled architectures')
plt.legend()
plt.savefig('controller_performance.png')

