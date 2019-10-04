from collections import deque
from git import FileState
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
while len(q) > 0:
    if state_id > 10000:
        break
    state = q.popleft()
    num_branches = len(branches)
    for branch in branches:
        # merge
        for other in branches:
            if branch == other:
                continue
            new_state = state.copy(prefix + str(state_id))
            if new_state.merge(branch, other) == Result.Ok:
                q.append(new_state)
                state_id += 1
        # commit
        txt = state.get(branch)
        len_txt = len(txt)
        if len_txt > 0:
            # commit-remove
            for pos in range(len_txt):
                msg = branch + ' remove ' + str(pos)
                new_state = state.copy(prefix + str(state_id))
                new_state.commit(branch, msg, make_remove(pos))
                q.append(new_state)
                state_id += 1
        # commit-insert
        for pos in range(len_txt + 1):
            for i in range(2):
                char = chr(ord('a') + i)
                msg = branch + ' insert ' + str(pos) + ' ' + char
                new_state = state.copy(prefix + str(state_id))
                new_state.commit(branch, msg, make_insert(pos, char))
                q.append(new_state)
                state_id += 1