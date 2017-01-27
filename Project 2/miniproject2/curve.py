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

def vector_field_plots():
    results = pickle._load(open("vector_fields2.pkl", "rb"))

    fig, axs = plt.subplots(3, sharex=True, sharey=True)

    fig.set_size_inches(10, 20, forward=True)
    fig.suptitle("Exploring the vector fields", fontsize=20)

    x = np.linspace(-150,30,20)
    dx  = np.linspace(-15,15,20)
    u,v = np.meshgrid(x,dx)

    for i, key in enumerate(results):
        dummy = np.zeros((results[key][0].shape[0],results[key][0].shape[1]))
        axs[0].quiver(u, v, results[key][0], dummy)
        axs[0].set_title("Trial no 1")

        axs[1].quiver(u, v, results[key][20], dummy)
        axs[1].set_title("Trial no 20")

        axs[2].quiver(u, v, results[key][99], dummy)
        axs[2].set_title("Trial no 100")

    plt.show()
    plt.savefig("lambda_variations.png")


if __name__ == "__main__":
    # tau_plots()
    # lambda_plots()
    vector_field_plots()
