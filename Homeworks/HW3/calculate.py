# import numpy as np
# pA = np.array([[0,0.5,0.5,0,0,0,0],
#                 [0,0,0,0,0,1,0],
#                 [0,0,0,0,0,0,1],
#                 [0,1,0,0,0,0,0],
#                 [0,0,1,0,0,0,0],
#                 [1,0,0,0,0,0,0],
#                 [1,0,0,0,0,0,0]])
#
# pB = np.array([[0,0.5,0.5,0,0,0,0],
#             [0,0,0,0,0,0,1],
#             [0,0,0,0,0,1,0],
#             [0,1,0,0,0,0,0],
#             [0,0,1,0,0,0,0],
#             [1,0,0,0,0,0,0],
#             [1,0,0,0,0,0,0]])
#
# pC = np.array([[0,0.5,0.5,0,0,0,0],
#             [0,0,0,1,0,0,0],
#             [0,0,0,0,1,0,0],
#             [0,1,0,0,0,0,0],
#             [0,0,1,0,0,0,0],
#             [1,0,0,0,0,0,0],
#             [1,0,0,0,0,0,0]])
#
# observation_matrix = np.array([ [1,0,0,0,0,0],
#                                 [0,1,0,0,0,0],
#                                 [0,1,0,0,0,0],
#                                 [0,0,1,0,0,0],
#                                 [0,0,0,1,0,0],
#                                 [0,0,0,0,1,0],
#                                 [0,0,0,0,0,1]])
#
# belief = np.array([0.0,0.5,0.5,0.0,0.0,0.0,0.0])
#
# obsA = np.identity(7)
# obsB = obsA
# obsC = obsA
#
# def belief_update(belief, probability_action, observation):
#     sum = 0
#     t1 = np.dot(belief, probability_action)
#     t2 = observation
#     numerator= np.dot(t1, t2)
#     for x in np.nditer(numerator):
#         sum += x
#     return numerator/sum
#
# print("Belief at t+1 for action 'a':",belief_update(belief, pA, obsA))
# print("Belief at t+1 for action 'b':",belief_update(belief, pB, obsB))
# print("Belief at t+1 for action 'c':",belief_update(belief, pC, obsC))

import numpy as np

## The PO Markov Decision Process
X = ["Holding AC", "Holding AD"]
A = ["Guess AC", "Guess AD", "Peek"]
Z = ["Nothing", "AC", "AD"]
P_GAC = np.array([[0.5, 0.5], [0.5, 0.5]])
P_GAD = P_GAC
P_Peek = np.array([[1, 0], [0,1]])
O_GAC = np.array([[1,0,0], [1,0,0]])
O_GAD = O_GAC
O_Peek = np.array([[0,0.9,0.1], [0,0.1,0.9]])
C = np.array([[0,1,0.5], [1,0,0.5]])
gamma = 0.9

## Some useful mappings
action_matrixes = {"Guess AC":P_GAC, "Guess AD":P_GAD, "Peek":P_Peek}
observation_matrixes = {"Guess AC":O_GAC, "Guess AD":O_GAD, "Peek":O_Peek}


from random import randint

policy = np.array([[1/3, 1/3, 1/3], [1/3, 1/3, 1/3]]) #Uniform random policy

x_initial = X[randint(0, len(X)-1)] #Choose a random initial state

trajectory = []
actions = []
observations = []

currentState = x_initial
for i in range(10000):
    action = np.random.choice(A, p=policy[X.index(currentState)]) #Choosing an action from the policy
    observation = np.random.choice(Z, p=observation_matrixes[action][X.index(currentState)]) #Getting an observation
    actions += [action]
    observations += [observation]
    trajectory += [currentState]
    currentState = np.random.choice(X, p=action_matrixes[action][X.index(currentState)]) #Updating our current state



def belief_update(belief, action, observation):
    soma = 0
    numerator = np.dot(np.dot(belief, action_matrixes[action]), np.diag(observation_matrixes[action][:,Z.index(observation)]))

    print("----------", numerator)

    for x in np.nditer(numerator):
        soma += x
    return numerator/soma

def beliefDifferent(nextBelief, beliefSequence):
    for belief in beliefSequence:
        if (np.linalg.norm(nextBelief - belief) < 10**(-4)):
            return False
    return True

initial_belief = np.array([0.5, 0.5])

belief_sequence = [initial_belief]

currentBelief = initial_belief
for i in range(10000):
    nextBelief = belief_update(currentBelief, actions[i], observations[i])
    currentBelief = nextBelief
    if beliefDifferent(nextBelief, belief_sequence):
        belief_sequence += [nextBelief]
