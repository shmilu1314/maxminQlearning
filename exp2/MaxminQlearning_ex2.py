import numpy as np
import gym

epsilon = 0.1
discountFactor = 1
learningRate = 0.15
tryEpisodes = 1000
maxStep = 1000
times = 10

class MaxminQlearning:
    def __init__(self, sigma, N=2):
        self.sigma = sigma
        self.N = N
        self.env = gym.make('MountainCar-v0').env
        self.actions = self.env.action_space
        self.num_pos = 20  # 将位置分为num_pos份
        self.num_vel = 15  # 将速度分为num_vel份
        self.Q = np.random.normal(loc=0, scale=0.01, size=(self.N, self.num_pos * self.num_vel, self.actions.n))
        self.pos_bins = self.toBins(-1.2, 0.6, self.num_pos)
        self.vel_bins = self.toBins(-0.07, 0.07, self.num_vel)

    def learn(self):
        countStep = np.zeros(tryEpisodes)
        for i in range(tryEpisodes):
            observation = self.env.reset()
            state = self.digitizeState(observation)
            for t in range(maxStep):
                minQ = self.selectQ(state)
                action = self.chooseAction(minQ, state)
                nextState, reward, done = self.stepAction(action)

                # 更新Q
                updateQ = np.random.randint(0, self.N)
                tdError = reward + discountFactor * self.Qmax(minQ, nextState) - self.Q[updateQ, state, action]
                self.Q[updateQ, state, action] += learningRate * tdError

                if done:
                    countStep[i] = t
                    break
                # 更新状态
                state = nextState
            if countStep[i] == 0:
                countStep[i] = maxStep
        return countStep

    def selectQ(self, state):
        return int(np.argmin(self.Q[:, state, :]) / self.actions.n)

    def chooseAction(self, curQ, state):
        if np.random.random() < epsilon:
            return self.actions.sample()
        else:
            return np.argmax(self.Q[curQ, state])

    def stepAction(self, action):
        observation, reward, done, info = self.env.step(action)
        reward = np.random.normal(loc=-1, scale=self.sigma)
        nextState = self.digitizeState(observation)
        return nextState, reward, done

    def Qmax(self, minQ, nextState):
        return np.max(self.Q[minQ, nextState])

    # 分箱处理函数，把[clip_min,clip_max]区间平均分为num段，
    def toBins(self, clip_min, clip_max, num):
        return np.linspace(clip_min, clip_max, num + 1)

    def digit(self, x, bin):
        n = np.digitize(x, bins=bin)
        if x == bin[-1]:
            n = n - 1
        return n

    # 将观测值observation离散化处理
    def digitizeState(self, observation):
        # 将矢量打散回连续特征值
        cart_pos, cart_v = observation
        # 分别对各个连续特征值进行离散化（分箱处理）
        digitized = [self.digit(cart_pos, self.pos_bins),
                     self.digit(cart_v, self.vel_bins), ]
        # 将4个离散值再组合为一个离散值，作为最终结果
        return (digitized[1] - 1) * self.num_pos + digitized[0] - 1

    def clear(self):
        self.Q = np.random.normal(loc=0, scale=0.01, size=(self.num_pos * self.num_vel, self.actions.n))
