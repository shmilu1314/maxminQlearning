import numpy as np
import tqdm
import datetime
import multiprocessing as mp
import matplotlib.pyplot as plt

from Qlearning_ex1 import Qlearning
from DoubleQlearning_ex1 import  DoubleQlearning
from MaxminQlearning_ex1 import  MaxminQlearning

epsilon = 0.1
discountFactor = 1
learningRate = 0.01
tryStep = 10000
times = 250

def Exec_Qlearning(u, N=1):
    pbar = tqdm.tqdm(total=times)
    pbar.set_description("Qlearning with " + str(u) + ":")
    count_choice = np.zeros(tryStep)
    for i in range(0, times):
        learning = Qlearning(u)
        learning.learn(count_choice)
        pbar.update(1)

    return np.array(count_choice)

def Exec_DoubleQlearning(u, N=1):
    pbar = tqdm.tqdm(total=times)
    pbar.set_description("DoubleQlearning with " + str(u) + " :")
    count_choice = np.zeros(tryStep)
    for i in range(0, times):
        learning = DoubleQlearning(u)
        learning.learn(count_choice)
        pbar.update(1)

    return np.array(count_choice)


def Exec_MaxminQlearning(u, N):
    pbar = tqdm.tqdm(total=times)
    pbar.set_description("MaxminQlearning with " + str(u) + " :")
    count_choice = np.zeros(tryStep)
    for i in range(0, times):
        learning = MaxminQlearning(u, N=N)
        learning.learn(count_choice)
        pbar.update(1)

    return np.array(count_choice)

def run(u):
    if (u > 0):
        target = epsilon / 2
    else:
        target = (-epsilon / 2 + 1)
    plt.figure()
    plt.title("u=%lf" % u)
    Qp = Exec_Qlearning(u)
    DoubleQ = Exec_DoubleQlearning(u)
    MaxminQ2 = Exec_MaxminQlearning(u,N=2)
    MaxminQ4 = Exec_MaxminQlearning(u,N=4)
    MaxminQ6 = Exec_MaxminQlearning(u,N=6)
    MaxminQ8 = Exec_MaxminQlearning(u,N=8)

    plt.plot(range(0, tryStep), np.abs(Qp / times - target), label="Qlearning")
    plt.plot(range(0, tryStep), np.abs(DoubleQ / times - target), label="DoubleQ")
    plt.plot(range(0, tryStep), np.abs(MaxminQ2 / times - target), label="Maxmin N=2")
    plt.plot(range(0, tryStep), np.abs(MaxminQ4 / times - target), label="Maxmin N=4")
    plt.plot(range(0, tryStep), np.abs(MaxminQ6 / times - target), label="Maxmin N=6")
    plt.plot(range(0, tryStep), np.abs(MaxminQ8 / times - target), label="Maxmin N=8")
    plt.legend()
    plt.savefig("figures/u=%lf.jpg"%u)
    plt.show()


if __name__ == '__main__':
    run(0.1)
    run(-0.1)
