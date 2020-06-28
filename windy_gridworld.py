import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')


# World
WORLD_WIDTH = 7
WORLD_HEIGHT = 10

# wind strength
WIND = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]

# possible actions
ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3
# probability of exploration
EPSILON = 0.1
# 衰减引子
GAMMA = 1
# step size
ALPHA = 0.5
# reward for each step
REWARD = -1.0

START = [0, 3]
GOAL = [7, 3]
ACTIONS = [ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT]


# step:given current state and action, return new state
def step(state, action):
    i, j = state
    if action == ACTION_UP:
        return [max(i - 1 - WIND[j], 0), j]  # 网格最上面一行为0，最下面一行为WORLD_HEIGHT-1
    elif action == ACTION_DOWN:
        return [max(min(i + 1 - WIND[j], WORLD_HEIGHT - 1), 0), j]
    elif action == ACTION_LEFT:
        return [max(i - WIND[j], 0), max(j - 1, 0)]
    elif action == ACTION_RIGHT:
        return [max(i - WIND[j], 0), min(j + 1, WORLD_WIDTH - 1)]
    else:
        assert False


# play a episode
def episode(q_value):
    # track the total time steps in this episode
    time = 0

    # initial state
    state = START

    while state != GOAL:
        # choose a action based on epsilon
        if np.random.binomial(1, EPSILON) == 1:
            action = np.random.choice(ACTIONS)
        else:
            values_ = q_value[state[0], state[1], :]
            action = np.random.choice([action_ for action_, value_ in enumerate(values_) if value_ == np.max(values_)])

        # keep going until get to the goal state
        next_state = step(state, action)
        values_ = q_value[next_state[0], next_state[1], :]
        # choose next step by greedy search
        next_action = np.random.choice([action_ for action_, value_ in enumerate(values_) if value_ == np.max(values_)])

        # update q value
        q_value[state[0], state[1], action] += ALPHA * (REWARD + GAMMA * q_value[next_state[0], next_state[1], next_action] - q_value[state[0], state[1], action])
        state = next_state
        time += 1
    return time


def q_learning():
    q_value = np.zeros((WORLD_HEIGHT, WORLD_WIDTH, 4))
    episode_limit = 500

    # store q value for each episode totally 500
    steps = []
    ep = 0
    while ep < episode_limit:
        steps.append(episode(q_value))
        ep += 1

    steps = np.add.accumulate(steps)

    plt.plot(steps, np.arange(1, len(steps) + 1))
    plt.xlabel('Time steps')
    plt.ylabel('Episodes')
    plt.savefig('./q-learning.png')
    plt.close()
    # display the optimal policy
    optimal_policy = []
    for i in range(0, WORLD_HEIGHT):
        optimal_policy.append([])
        for j in range(0, WORLD_WIDTH):
            if [i, j] == GOAL:
                optimal_policy[-1].append('G')
                continue
            bestAction = np.argmax(q_value[i, j, :])
            if bestAction == ACTION_UP:
                optimal_policy[-1].append('U')
            elif bestAction == ACTION_DOWN:
                optimal_policy[-1].append('D')
            elif bestAction == ACTION_LEFT:
                optimal_policy[-1].append('L')
            elif bestAction == ACTION_RIGHT:
                optimal_policy[-1].append('R')

    print('Optimal policy is:')
    for row in optimal_policy:
        print(row)

    print('Wind strength for each column:\n{}'.format([str(w) for w in WIND]))


if __name__ == '__main__':
    q_learning()
