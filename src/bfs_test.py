from collections import deque
from git import ClonableFileState
from lang import Result
from typing import Callable, List

def make_remove(pos: int) -> Callable[[List[str]], List[str]]:
    def remove(txt: List[str]) -> List[str]:
        return txt[:pos] + txt[pos+1:]
    return remove

def make_update(pos: int, char: str) -> Callable[[List[str]], List[str]]:
    def update(txt: List[str]) -> List[str]:
        return txt[:pos] + [char] + txt[pos+1:]
    return update

def make_insert(pos: int, char: str) -> Callable[[List[str]], List[str]]:
    def insert(txt: List[str]) -> List[str]:
        return txt[:pos] + [char] + txt[pos:]
    return insert

def filter(history: List[str]) -> List[str]:
    res = []
    for s in history:
        if s != '':
            t = s.split(' ')
            if t[1] != 'Merge':
                res.append(t[0])
    return res

prefix = '~/git_test2/state'
filename = 'test.txt'
state = ClonableFileState(prefix + '0', filename)
state_id = 1
branches = ['master']
num_chars = 2
# fork some branches first
for i in range(2):
    branches.append('branch_' + str(i + 1))
    state.fork(branches[0], branches[i + 1])
# bfs
q = deque()
q.append(state)
while len(q) > 0:
    if state_id > 10000:
        break
    state = q.popleft()
    for branch in branches:
        # merge
        for other in branches:
            if branch == other:
                continue
            new_state = state.clone(prefix + str(state_id))
            if new_state.merge(other, branch) == Result.Ok:
                if new_state.history(branch) != state.history(branch):
                    q.append(new_state)
                    state_id += 1
                    new_his = filter(new_state.history(branch))
                    new_txt = new_state.get(branch)
                    for another in branches:
                        if another != branch and \
                                new_his == filter(new_state.history(another)) and \
                                new_txt != new_state.get(another):
                            print('Inconsistent in %s between %s and %s' % (new_state.dir, branch, another))
        # commit
        txt = state.get(branch)
        len_txt = len(txt)
        if len_txt > 0:
            # commit-update
            for pos in range(len_txt):
                if txt[pos] == 'a':
                    char = 'b'
                    msg = branch + ' update ' + str(pos) + ' ' + char
                    new_state = state.clone(prefix + str(state_id))
                    new_state.commit(branch, msg, make_update(pos, char))
                    q.append(new_state)
                    state_id += 1
        # commit-insert
        for pos in range(len_txt + 1):
            char = 'a'
            msg = branch + ' insert ' + str(pos) + ' ' + char
            new_state = state.clone(prefix + str(state_id))
            new_state.commit(branch, msg, make_insert(pos, char))
            q.append(new_state)
            state_id += 1