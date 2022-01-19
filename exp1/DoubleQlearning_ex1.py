import numpy as np

epsilon = 0.1
discountFactor = 1
learningRate = 0.01
tryStep = 10000
run_time = 250


class DoubleQlearning:
    def __init__(self, u):
        self.QA = np.random.normal(loc=0, scale=0.1, size=(2, 8))
        self.QB = np.random.normal(loc=0, scale=0.1, size=(2, 8))
        self.u = u

    def learn(self, count_choice):
        for totalStep in range(tryStep):
            state = 1
            reachEnd = False
            while not reachEnd:
                action = self.selectAction(state)
                nextState, reward, reachEnd = self.update_state(state, action)
                if np.random.rand() < 0.5:
                    tdError = reward + discountFactor * self.getmaxQ('A', nextState) - self.QA[state, action]
                    self.QA[state, action] += learningRate * tdError
                else:
                    tdError = reward + discountFactor * self.getmaxQ('B', nextState) - self.QB[state, action]
                    self.QB[state, action] += learningRate * tdError
                state = nextState

            if state == 2:
                count_choice[totalStep] += 1

    def selectAction(self, state):
        if state == 1:
            if np.random.rand() < epsilon:
                return np.random.randint(0, 2)
            return np.argmax(self.QA[state, 0:2] + self.QB[state, 0:2])
        else:
            if np.random.rand() < epsilon:
                return np.random.randint(0, 8)
            else:
                return np.argmax(self.QA[state, 0:8] + self.QB[state, 0:8])

    def update_state(self, state, action):
        if state == 1:
            if action == 0:
                return 0, 0, False
            return 2, 0, True
        else:
            reward = self.u + np.random.uniform(-1, 1)
            return -1, reward, True

    def getmaxQ(self, choice, nextState):
        if nextState == 0:
            if choice == 'A':
                pos = np.argmax(self.QA[nextState])
                return self.QB[nextState, pos]
            else:
                pos = np.argmax(self.QB[nextState])
                return self.QA[nextState, pos]
        return 0