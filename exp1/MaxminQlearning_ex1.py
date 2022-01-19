import numpy as np

epsilon = 0.1
discountFactor = 1
learningRate = 0.01
tryStep = 10000
run_time = 250

class MaxminQlearning:
    def __init__(self, u, N=1):
        self.Q = np.random.normal(loc=0, scale=0.01, size=(N, 2, 8))
        self.N = N
        self.u = u

    def learn(self, count_choice):
        for totalStep in range(tryStep):
            state = 1
            reachEnd = False
            while not reachEnd:
                minQ = self.selectQ(state)
                action = self.selectAction(minQ, state)
                nextState, reward, reachEnd = self.update_state(state, action)
                updateQ = np.random.randint(0, self.N)
                tdError = reward + discountFactor * self.getmaxQ(minQ, nextState) - self.Q[updateQ, state, action]
                self.Q[updateQ, state, action] += learningRate * tdError
                state = nextState

            if state == 2:
                count_choice[totalStep] += 1


    def selectQ(self, state):
        if state == 1:
            return int(np.argmin(self.Q[:, state, :2]) / 2)
        else:
            return int(np.argmin(self.Q[:, state, :]) / 8)

    def selectAction(self, curQ, state):
        if state == 1:
            if np.random.rand() < epsilon:
                return np.random.randint(0, 2)
            return np.argmax(self.Q[curQ, state, 0:2])
        else:
            if np.random.rand() < epsilon:
                return np.random.randint(0, 8)
            else:
                return np.argmax(self.Q[curQ, state])

    def update_state(self, state, action):
        if state == 1:
            if action == 0:
                return 0, 0, False
            return 2, 0, True
        else:
            reward = self.u + np.random.uniform(-1, 1)
            return -1, reward, True

    def getmaxQ(self, minQ, nextState):
        if nextState == 0:
            return np.max(self.Q[minQ, 0])
        return 0