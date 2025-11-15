# Author: Carter Roberts
# Institution: Loyola University New Orleans
# Instructor: Dr. Omar EL Khatib
# Filename: roberts_qIterationDRL.py
# Description: Utilizes Q-Iteration Deterministic Reinforcement Learning
# on an arbitrarily-devised 3x4 labyrinth state
# Date Created: 11/12/2025
# Date Modified: 11/15/2025

states = {(0, 0), (1,0), (2, 0), (0, 1), (1, 1), (2, 1), (0, 2), (1, 2), (2, 2),
          (0, 3), (1, 3), (2, 3)}
terminal_states = {(1, 1), (1, 2), (2, 1), (2, 3)}
rewards = {(1, 1): -10, (2, 1): -20, (1, 2): 10, (2, 3): 20, (0, 0): 0, (1, 0): 0,
           (2, 0): 0, (0, 1): 0, (0, 2): 0, (0, 3): 0, (1, 3): 0, (2, 2): 0}
actions = {'L', 'R', 'U', 'D'}
gamma = 0.9
max_iterations = 20
countIter = 0

# function to apply action to state
def apply(s, a):
    # deconstruct state tuple
    x, y = s
    # make new variables for returned new state tuple
    newX, newY = x, y

    # change coords for new state tuple based on a
    if a == 'L':
        newX -= 1
    elif a == 'R':
        newX += 1
    elif a == 'U':
        newY += 1
    elif a == 'D':
        newY -= 1

    # check that new coords are valid
    if newX <= 2 and newX >= 0:
        if newY <= 3 and newY >= 0:
            # return new coords as new state since valid
            return (newX, newY)
    # keep the state the same, otherwise
    return s

# Empty Q for each action & state
Q = {}
for s in states:
    Q[s] = [0]*len(actions)
# Find Q[] for all actions/states
for k in range(max_iterations):
    # to be set to true if a Q[] changes in current loop iteration
    change = False
    # go through every state
    for s in states:
        if s in terminal_states: continue
        # go through every action for that state
        for i, a in enumerate(actions):
            # apply that action to that state and get s1 out of it
            s1 = apply(s, a)
            # get current max Q[] by taking max of Q[s1]
            maxQ = max(Q[s1])
            # get actual q with maxQ and Bellman equation
            q = rewards[s1] + gamma * maxQ
            # if this actual q doesn't match current q for state/action,
            if q != Q[s][i]:
                # redefine that current q to actual q
                Q[s][i] = q
                # don't activate break cond.
                change = True
    countIter += 1
    # break cond.
    if not change: break

# print Q[] for current gamma value
print("Q iterations =", countIter)
print("Q for gamma", gamma, "=")
print(Q)

policy = {}
for s in states:
    if s in terminal_states: continue
    policy[s] = None
# Find policy for all states
for s in states:
    # don't evaluate terminal states
    if s in terminal_states: continue
    # best Q[] and best action for current state, to update
    best_q, best_a = float('-inf'), None
    # go through every action for that state
    for i, a in enumerate(actions):
        # never evaluate an illegal (apply returns s) move
        if apply(s, a) == s: continue
	# if the current q for that state/action is better than best Q[],
        if Q[s][i] > best_q:
            # set best Q[] to that current q
            best_q = Q[s][i]
            # and best action to that current action
            best_a = a
    # if best action doesn't match action in that state's policy, change policy
    if best_a != policy[s]: policy[s] = best_a

# print optimal policy for current gamma value
print("Policy for gamma", gamma, "=")
print(policy)
