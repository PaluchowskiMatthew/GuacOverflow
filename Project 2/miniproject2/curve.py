import pickle
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb

results = pickle._load(open( "tau_variations.pkl", "rb" ))

fig, axs = plt.subplots(4, sharex=True, sharey=True)

fig.set_size_inches(10, 20, forward=True)
fig.suptitle("Exploring the tau-parameter", fontsize=20)

for i,key in enumerate(results):
    axs[i].plot(results[key])
    axs[i].set_title(key)


plt.show()