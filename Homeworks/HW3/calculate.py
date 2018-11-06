import numpy as np
pA = np.array([[0,0.5,0.5,0,0,0,0],
                [0,0,0,0,0,1,0],
                [0,0,0,0,0,0,1],
                [0,1,0,0,0,0,0],
                [0,0,1,0,0,0,0],
                [1,0,0,0,0,0,0],
                [1,0,0,0,0,0,0]])

pB = np.array([[0,0.5,0.5,0,0,0,0],
            [0,0,0,0,0,0,1],
            [0,0,0,0,0,1,0],
            [0,1,0,0,0,0,0],
            [0,0,1,0,0,0,0],
            [1,0,0,0,0,0,0],
            [1,0,0,0,0,0,0]])
            
pC = np.array([[0,0.5,0.5,0,0,0,0],
            [0,0,0,1,0,0,0],
            [0,0,0,0,1,0,0],
            [0,1,0,0,0,0,0],
            [0,0,1,0,0,0,0],
            [1,0,0,0,0,0,0],
            [1,0,0,0,0,0,0]])

observation_matrix = np.array([ [1,0,0,0,0,0],
                                [0,1,0,0,0,0],
                                [0,1,0,0,0,0],
                                [0,0,1,0,0,0],
                                [0,0,0,1,0,0],
                                [0,0,0,0,1,0],
                                [0,0,0,0,0,1]])

belief = np.array([0.0,0.5,0.5,0.0,0.0,0.0,0.0])

obsA = np.identity(7)
obsB = obsA
obsC = obsA

def belief_update(belief, probability_action, observation):
    sum = 0
    t1 = np.dot(belief, probability_action)
    t2 = observation
    numerator= np.dot(t1, t2)
    for x in np.nditer(numerator):
        sum += x
    return numerator/sum

print("Belief at t+1 for action 'a':",belief_update(belief, pA, obsA))
print("Belief at t+1 for action 'b':",belief_update(belief, pB, obsB))
print("Belief at t+1 for action 'c':",belief_update(belief, pC, obsC))
