import pickle
import matplotlib.pyplot as plt
import numpy as np

results = pickle._load(open( "results.pkl", "rb" ))[:200]

plt.semilogy(np.mean(results, axis=1))
plt.show()