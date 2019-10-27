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

def add_state(s: ClonableFileState, branch: str, branch_2: str = "") -> None:
    global branches
    global prefix
    global q
    global state_id
    for other in branches:
        if other == branch or other == branch_2:
            continue
        t = s.clone(prefix + str(state_id + 1))
        if t.merge(other, branch) != Result.Ok:
            return
    q.append(s)
    state_id += 1

prefix = '~/git_test/state'
filename = 'test.txt'
state = ClonableFileState(prefix + '0', filename)
state_id = 1
branches = ['master']
num_chars = 2
for pos in range(3):
    branch = branches[0]
    char = 'a'
    msg = branch + ' insert ' + str(pos) + ' ' + char
    state.commit(branch, msg, make_insert(pos, char))
# fork some branches first
for i in range(2):
    branches.append('branch_' + str(i + 1))
    state.fork(branches[0], branches[i + 1])
# bfs
q = deque()
q.append(state)
p = deque()
while len(q) > 0 or len(p) > 0:
    if len(q) > 0:
        state = q.popleft()
        p.append(state)
        for branch in branches:
            # merge
            for other in branches[:-1]:
                if branch == other:
                    continue
                new_state = state.clone(prefix + str(state_id))
                assert new_state.merge(other, branch) == Result.Ok, 'Fail %d %s %s' % (state_id, other, branch)
                if new_state.history(branch) != state.history(branch):
                    new_his = filter(new_state.history(branch))
                    new_txt = new_state.get(branch)
                    inconsistent = False
                    for another in branches:
                        if another != branch and \
                                new_his == filter(new_state.history(another)) and \
                                new_txt != new_state.get(another):
                            print('Inconsistent in %s between %s and %s' % (new_state.dir, branch, another))
                            inconsistent = True
                    if inconsistent:
                        state_id += 1
                        new_state = new_state.clone(prefix + str(state_id))
                    add_state(new_state, branch, other)
    else:
        state = p.popleft()
        for branch in branches[:-1]:
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
                        add_state(new_state, branch)
            # commit-insert
            txt.append('b')
            for pos in range(len_txt + 1):
                if txt[pos] == 'b':
                    char = 'a'
                    msg = branch + ' insert ' + str(pos) + ' ' + char
                    new_state = state.clone(prefix + str(state_id))
                    new_state.commit(branch, msg, make_insert(pos, char))
                    add_state(new_state, branch)