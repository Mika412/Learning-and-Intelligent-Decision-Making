import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

np.set_printoptions(threshold=100)

# Problem specific parameters
WIND = (0, 0, 0, 1, 1, 1, 2, 2, 1, 0)
nrows = 7
ncols = 10
init = [3, 0]
goal = [3, 7]

# States
X = [[x, y] for x in range(nrows) for y in range(ncols)]
nX = len(X)

# Actions
A = ['U', 'D', 'L', 'R']
nA = len(A)

# Transition probabilities
P = dict()
P['U'] = np.zeros((nX, nX))
P['D'] = np.zeros((nX, nX))
P['L'] = np.zeros((nX, nX))
P['R'] = np.zeros((nX, nX))

for i in range(len(X)):
    x = X[i]
    y = dict()

    y['U'] = [x[0] - WIND[x[1]] - 1, x[1]]
    y['D'] = [x[0] - WIND[x[1]] + 1, x[1]]
    y['L'] = [x[0] - WIND[x[1]], x[1] - 1]
    y['R'] = [x[0] - WIND[x[1]], x[1] + 1]

    for k in y:
        y[k][0] = max(min(y[k][0], nrows - 1), 0)
        y[k][1] = max(min(y[k][1], ncols - 1), 0)
        j = X.index(y[k])
        P[k][i, j] = 1

c = np.ones((nX, nA))
c[X.index(goal), :] = 0

gamma = 0.99



J = np.zeros((70, 1))
Q = np.zeros((70,4))
err = 1

while err > 1e-8:
    Q_up = c[:,0:1] + gamma*P['U'].dot(J)
    Q_down = c[:,1:2] + gamma*P['D'].dot(J)
    Q_left = c[:,2:3] + gamma*P['L'].dot(J)
    Q_right = c[:,3:4] + gamma*P['R'].dot(J)
    Q_new = np.column_stack((Q_up, Q_down, Q_left, Q_right))
    J_new = np.min((Q_up, Q_down, Q_left, Q_right), axis=0)
    err = np.linalg.norm(Q_new-Q)
    J = J_new
    Q = Q_new

Q_optimal = Q



# Q is a numpy matrix
# State is a list

def select_action(Q, state, eps=0.1):
    Q_values = Q[X.index(state)]
    greedy_indexes = np.where(Q_values == min(Q_values))[0]
    greedy_action = A[np.random.choice(greedy_indexes)]
    return np.random.choice([greedy_action, np.random.choice(A)], p=[1-eps, eps])


# Initializing our probabilities matrix
P_rl = dict()
P_rl['U'] = np.eye(nX, nX)
P_rl['D'] = np.eye(nX, nX)
P_rl['L'] = np.eye(nX, nX)
P_rl['R'] = np.eye(nX, nX)

# Initializing our cost function
c_rl = np.zeros((nX, nA))

# Initializing our number of visits matrix
N = np.zeros((nX, nA))

# Initializing our Q matrix
Q_MB = np.zeros((nX, nA))


def I(x_next, y):
    if x_next == y:
        return 1
    return 0


def alpha(x, a):
    return 1 / (N[x][a] + 1)


def probability_update(x, a, x_next):
    for y in range(len(X)):
        P_rl[a][x][y] = P_rl[a][x][y] + alpha(x, A.index(a)) * (I(x_next, y) - P_rl[a][x][y])


def cost_update(x, a, c_t):
    c_rl[x][a] = c_rl[x][a] + alpha(x, a) * (c_t - c_rl[x][a])


def Q_update(x_t, a_t, c_t, x_next):
    soma = 0
    for y in range(len(X)):
        soma = soma + P_rl[A[a_t]][x_t][y] * np.amin(Q_MB[y])
    Q_MB[x_t][a_t] = c_rl[x_t][a_t] + gamma * soma  # Note that c will already be updated


x_t = init  # Start state

iterations_MB = []
norms_MB = []

for iteration in range(100000):
    a_t = select_action(Q_MB, x_t)
    N[X.index(x_t)][A.index(a_t)] += 1
    c_t = c[X.index(x_t)][A.index(a_t)]
    x_next = X[np.random.choice(range(len(X)), p=P[a_t][X.index(x_t)])]
    probability_update(X.index(x_t), a_t, X.index(x_next))
    cost_update(X.index(x_t), A.index(a_t), c_t)
    Q_update(X.index(x_t), A.index(a_t), c_t, X.index(x_next))
    if x_t == goal:
        x_next = X[np.random.choice(range(len(X)))]
    x_t = x_next
    if (iteration % 500 == 0):
        iterations_MB += [iteration]
        norms_MB += [np.linalg.norm(Q_optimal - Q_MB)]

print(Q_MB)
plt.figure(1)
plt.scatter(iterations_MB, norms_MB)
plt.xlabel('Number of iterations')
plt.ylabel('||Q* - Q||')
plt.title("Model based learning - progress")
plt.show()