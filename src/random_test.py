import random
from git import FileState
from typing import Callable, List

def make_remove(pos: int) -> Callable[[List[str]], List[str]]:
    def remove(txt: List[str]) -> List[str]:
        return txt[:pos] + txt[pos+1:]
    return remove

def make_modify(pos: int, char: str) -> Callable[[List[str]], List[str]]:
    def modify(txt: List[str]) -> List[str]:
        return txt[:pos] + [char] + txt[pos+1:]
    return modify

def make_insert(pos: int, char: str) -> Callable[[List[str]], List[str]]:
    def insert(txt: List[str]) -> List[str]:
        return txt[:pos] + [char] + txt[pos:]
    return insert

state = FileState('~/git_test', 'test.txt')
branches = ['master']

# fork some branches first
for i in range(4):
    branches.append('branch_' + str(i + 1))
    state.fork(branches[0], branches[i + 1])

n = 3
while n > 0:
    n = n - 1
    num_branches = len(branches)
    branch = random.randint(0, num_branches - 1)
    behavior = random.randint(0, 1)
    if behavior == 0:  # merge
        other = random.randint(0, num_branches - 1)
        if other == branch:
            continue
        state.merge(branches[branch], branches[other])
        print(branches[branch] + ' merge to ' + branches[other])
    else:  # commit
        txt = state.get(branches[branch])
        len_txt = len(txt)
        prob = random.random()
        if prob > 0.8:  # remove
            if len_txt == 0:
                continue
            pos = random.randint(0, len_txt - 1)
            msg = branches[branch] + ' remove ' + str(pos)
            state.commit(branches[branch], msg, make_remove(pos))
        elif prob > 0.4:  # modify
            if len_txt == 0:
                continue
            pos = random.randint(0, len_txt - 1)
            char = chr(ord('a') + random.randint(0, 25))
            msg = branches[branch] + ' modify ' + str(pos) + ' ' + char
            state.commit(branches[branch], msg, make_modify(pos, char))
        else:  # insert
            pos = random.randint(0, len_txt)
            char = chr(ord('a') + random.randint(0, 25))
            msg = branches[branch] + ' insert ' + str(pos) + ' ' + char
            state.commit(branches[branch], msg, make_insert(pos, char))
        print(msg)