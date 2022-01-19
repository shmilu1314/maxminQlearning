import math

import numpy as np
import gym
import tqdm
import datetime
import matplotlib.pyplot as plt
from Qlearning_ex2 import Qlearning
from DoubleQlearning_ex2 import  DoubleQlearning
from MaxminQlearning_ex2 import  MaxminQlearning

epsilon = 0.1
discountFactor = 1
learningRate = 0.15
tryEpisodes = 1000
maxStep = 1000
times = 10

def Exec_Qlearning(sigma):
    # pbar = tqdm.tqdm(total=times, postfix=i)
    pbar = tqdm.tqdm(total=times)
    pbar.set_description("Qlearning with " + str(sigma) + ":")
    count_choice = np.zeros(tryEpisodes)
    for i in range(0, times):
        learning = Qlearning(sigma)
        count_choice += learning.learn()
        pbar.update(1)

    return np.array(count_choice)

def Exec_DoubleQlearning(sigma):
    pbar = tqdm.tqdm(total=times)
    pbar.set_description("DoubleQlearning with " + str(sigma) + " :")
    count_choice = np.zeros(tryEpisodes)
    for i in range(0, times):
        learning = DoubleQlearning(sigma)
        count_choice += learning.learn()
        pbar.update(1)

    return np.array(count_choice)



def Exec_MaxminQlearning(sigma):
    pbar = tqdm.tqdm(total=times)
    pbar.set_description("MaxminQlearning with " + str(sigma) + " :")
    count_choice = np.zeros(tryEpisodes)
    for i in range(0, times):
        learning = MaxminQlearning(sigma)
        count_choice += learning.learn()
        pbar.update(1)

    return np.array(count_choice)

def run2_1():
    resQ=[]
    resMaxmin=[]
    resDouble=[]
    for i in range(6):
        sigma=math.sqrt(i*10)
        res=Exec_Qlearning(sigma=sigma)
        resQ.append(np.mean(res[-21:-1]) / times)
    for i in range(6):
        sigma=math.sqrt(i*10)
        res=Exec_MaxminQlearning(sigma=sigma)
        resMaxmin.append(np.mean(res[-21:-1]) / times)
    for i in range(6):
        sigma=math.sqrt(i*10)
        res=Exec_DoubleQlearning(sigma=sigma)
        resDouble.append(np.mean(res[-21:-1])/ times)

    plt.xlabel('Sigma^2')
    plt.ylabel('Steps')
    plt.plot(range(0, 60, 10), resQ, label="Qlearning")
    plt.plot(range(0, 60, 10), resMaxmin, label="Maxmin N=2")
    plt.plot(range(0, 60, 10), resDouble, label="DoubleQ")
    plt.legend()
    plt.savefig("2.1.jpg")
    plt.show()
    return

def run2_2():
    sigma=math.sqrt(10)
    # q = Qlearning(sigma=3)
    plt.figure()
    plt.title("sigma^2=%lf" % (sigma*sigma))
    plt.xlabel('Episodes')
    plt.ylabel('Steps')
    Q=Exec_Qlearning(sigma=sigma)
    MaxminQ=Exec_MaxminQlearning(sigma=sigma)
    DoubleQ=Exec_DoubleQlearning(sigma=sigma)
    plt.plot(range(0, tryEpisodes, 10), (Q / times)[::10], label="Qlearning")
    plt.plot(range(0, tryEpisodes, 10), (MaxminQ / times)[::10], label="Maxmin N=2")
    plt.plot(range(0, tryEpisodes, 10), (DoubleQ/ times)[::10], label="DoubleQ")
    plt.legend()
    plt.savefig("sigma^2=%lf.jpg"%(sigma*sigma))
    plt.show()

if __name__ == '__main__':
    run2_1()
    run2_2()