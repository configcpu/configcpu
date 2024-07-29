import numpy as np
import pandas as pd
import os

def bfs(src, target):
    queue = []
    queue.append(src)
    exp = []
    while len(queue) > 0:
        source = queue.pop(0)
        exp.append(source)
        print(source)
        if source == target:
            print("Success")
            return
        poss_moves_to_do = possible_move(source, exp)
        for move in poss_moves_to_do:
            if move not in exp and move not in queue:
                queue.append(move)

def possible_move(state, visited_states):
    b = state.index(0)
    d = []
    if b not in [0, 1, 2]:
        d.append('U')  # Up
    if b not in [6, 7, 8]:
        d.append('D')  # Down
    if b not in [0, 3, 6]:
        d.append('L')  # Left
    if b not in [2, 5, 8]:
        d.append('R')  # Right

    pos_moves_it_can = []
    for i in d:
        pos_moves_it_can.append(gen(state, i, b))
    
    return [move_it_can for move_it_can in pos_moves_it_can if move_it_can not in visited_states]

def gen(state, m, b):
    temp = state.copy()
    if m == 'D':
        temp[b + 3], temp[b] = temp[b], temp[b + 3]
    if m == 'U':
        temp[b - 3], temp[b] = temp[b], temp[b - 3]
    if m == 'L':
        temp[b - 1], temp[b] = temp[b], temp[b - 1]
    if m == 'R':
        temp[b + 1], temp[b] = temp[b], temp[b + 1]
    return temp

src = [1, 2, 3, 4, 5, 6, 0, 7, 8]
target = [1, 2, 3, 4, 5, 6, 7, 8, 0]
bfs(src, target)

