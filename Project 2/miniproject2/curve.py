import pickle
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb


def tau_plots():
    results = pickle._load(open( "tau_variations.pkl", "rb" ))
    fig_tau, axs_tau = plt.subplots(4, sharex=True, sharey=True)

    fig_tau.set_size_inches(10, 20, forward=True)
    fig_tau.suptitle("Exploring the tau-parameter", fontsize=20)

    for i,key in enumerate(sorted(results)):
        axs_tau[i].plot(results[key])
        axs_tau[i].set_title(key)

    fig_tau.savefig("tau_variations.png")

def lambda_plots():
    results = pickle._load(open("lambda_variations.pkl", "rb"))

    fig_lamb, axs_lamb = plt.subplots(3, sharex=True, sharey=True)

    fig_lamb.set_size_inches(10, 20, forward=True)
    fig_lamb.suptitle("Exploring the lambda-parameter", fontsize=20)

    for i, key in enumerate(sorted(results)):
        axs_lamb[i].plot(results[key])
        axs_lamb[i].set_title(key)

    fig_lamb.savefig("lambda_variations.png")


if __name__ == "__main__":
    tau_plots()
    lambda_plots()