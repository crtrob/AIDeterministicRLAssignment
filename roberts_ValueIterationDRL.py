# Author: Carter Roberts
# Institution: Loyola University New Orleans
# Instructor: Dr. Omar EL Khatib
# Filename: roberts_ValueIterationDRL.py
# Description: Utilizes Value Iteration Deterministic Reinforcement Learning
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
    newX = x
    newY = y

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
    if newX < 3 and newX >= 0:
        if newY < 4 and newY >= 0:
            # return new coords as new state since valid
            return (newX, newY)
    # otherwise return same state
    return s

V = {}
for s in states: V[s] = 0
# Find V[] for all states
for k in range(max_iterations):
    # to be set to true if a V[] changes in current loop iteration
    change = False
    # go through every state
    for s in states:
        # don't evaluate terminal states
        if s in terminal_states: continue
        # go through every action for that state
        for a in actions:
            # apply that action to that state and get s1 out of it
            s1 = apply(s, a)
            # create holder for what will be past V[s]
            v = V[s]
            # redefine V[s] with Bellman equation
            V[s] = max(V[s], rewards[s1] + gamma * V[s1])
            # past V[s] doesn't match current V[s], don't activate break cond.
            if v != V[s]: change = True
    countIter += 1
    # break cond.
    if not change: break

# print V[] for current gamma value
print("Iterations =", countIter)
print("V for gamma", gamma, "=")
print(V)

policy = {}
for s in states:
    if s in terminal_states: continue
    policy[s] = None
# Find policy for all states
for s in states:
    # don't evaluate terminal states
    if s in terminal_states: continue
    # best V[] and best action for current state, to update
    best_v, best_a = float('-inf'), None
    # go through every action for that state
    for a in actions:
        # apply that action to that state and get s1 out of it
        s1 = apply(s, a)
        # prevents use of illegal (apply returns s) moves
        if s1 == s: continue
        # if s1 is a terminal state, V[s] will just be reward
        if s1 in terminal_states: v = rewards[s1]
        # otherwise create holder for what will be V[s]
        else: v = V[s1]
        # if V[s] is better than best V[], redefine best V[] and action
        if v > best_v: best_v = v; best_a = a;
    # best action for state is that state's policy
    policy[s] = best_a

# print policy for current gamma value
print("Policy for gamma", gamma, "=")
print(policy)
