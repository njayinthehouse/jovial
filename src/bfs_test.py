from collections import deque
from git import ClonableFileState
from lang import Result
from typing import Callable, List

def make_remove(pos: int) -> Callable[[List[str]], List[str]]:
    def remove(txt: List[str]) -> List[str]:
        return txt[:pos] + txt[pos+1:]
    return remove

def make_insert(pos: int, char: str) -> Callable[[List[str]], List[str]]:
    def insert(txt: List[str]) -> List[str]:
        return txt[:pos] + [char] + txt[pos:]
    return insert

def add_state(s: ClonableFileState) -> bool:
    global history
    global state_id
    global q
    global r
    q.append(s)
    r.append(s)
    state_id += 1

def filter(history: List[str]) -> List[str]:
    res = []
    for s in history:
        if s != '':
            t = s.split(' ')
            if t[1] != 'Merge':
                res.append(' '.join(t[1:]))
    return res

def check_state(s: ClonableFileState) -> None:
    global branches
    global r
    for o in r:
        for br_s in branches:
            for br_o in branches:
                if filter(s.history(br_s)) == filter(o.history(br_o)) and s.get(br_s) != o.get(br_o):
                    #raise Exception('Inconsistent between (%s, %s) and (%s, %s)' % (s.dir, br_s, o.dir, br_o))
                    print('Inconsistent between (%s, %s) and (%s, %s)' % (s.dir, br_s, o.dir, br_o))

prefix = '~/git_test/state'
filename = 'test.txt'
state = ClonableFileState(prefix + '0', filename)
state_id = 1
branches = ['master']
# fork some branches first
for i in range(1):
    branches.append('branch_' + str(i + 1))
    state.fork(branches[0], branches[i + 1])
# bfs
q = deque()
q.append(state)
r = []
r.append(state)

while len(q) > 0:
    if state_id > 10000000:
        break
    state = q.popleft()
    for branch in branches:
        # merge
        for other in branches:
            if branch == other:
                continue
            new_state = state.clone(prefix + str(state_id))
            if new_state.merge(branch, other) == Result.Ok:
                if new_state.history(other) != state.history(other):
                    add_state(new_state)
                    check_state(new_state)
        # commit
        txt = state.get(branch)
        len_txt = len(txt)
        if len_txt > 0:
            # commit-remove
            for pos in range(len_txt):
                msg = branch + ' remove ' + str(pos)
                new_state = state.clone(prefix + str(state_id))
                new_state.commit(branch, msg, make_remove(pos))
                add_state(new_state)
        # commit-insert
        for pos in range(len_txt + 1):
            for i in range(3):
                char = chr(ord('a') + i)
                msg = branch + ' insert ' + str(pos) + ' ' + char
                new_state = state.clone(prefix + str(state_id))
                new_state.commit(branch, msg, make_insert(pos, char))
                add_state(new_state)