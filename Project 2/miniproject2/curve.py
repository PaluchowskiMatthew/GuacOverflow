import pickle
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb


def tau_plots():
    results = pickle._load(open( "tau_variations.pkl", "rb" ))
    fig, axs = plt.subplots(4, sharex=True, sharey=True)

    fig.set_size_inches(10, 20, forward=True)
    fig.suptitle("Exploring the tau-parameter", fontsize=20)

    for i,key in enumerate(results):
        axs[i].plot(results[key])
        axs[i].set_title(key)


    plt.show()
    plt.savefig("tau_variations.png")

def lambda_plots():
    results = pickle._load(open("lambda_variations.pkl", "rb"))

    fig, axs = plt.subplots(3, sharex=True, sharey=True)

    fig.set_size_inches(10, 20, forward=True)
    fig.suptitle("Exploring the lambda-parameter", fontsize=20)

    for i, key in enumerate(results):
        axs[i].plot(results[key])
        axs[i].set_title(key)

    plt.show()
    plt.savefig("lambda_variations.png")


if __name__ == "__main__":
    # tau_plots()
    lambda_plots()