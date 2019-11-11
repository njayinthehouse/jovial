# pyre-strict
from abc import ABC
import subprocess
from typing import Callable, List, Optional, Dict, Set, Generic, Tuple, TypeVar
from enum import Enum
from random import randint
from copy import copy, deepcopy

class Action(ABC): pass

class Insert(Action):
    def __init__(self, i: int, x: str, br: str):
        self.i, self.x, self.br = i, x, br

class Replace(Action):
    def __init__(self, i: int, x: str, br: str):
        self.i, self.x, self.br = i, x, br

class Merge(Action):
    def __init__(self, br1: str, br2: str):
        self.br1, self.br2 = br1, br2

LANGUAGE = '/bin/bash'

def run_program(commands: List[str], path: Optional[str] = None) -> str:
    p = subprocess.Popen(LANGUAGE, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if path is not None:
        p.stdin.write(('cd %s > /dev/null\n' % path).encode())
    for command in commands:
        p.stdin.write(('%s\n' % command).encode())
    if path is not None:
        p.stdin.write(b'cd - > /dev/null\n')
    p.stdin.close()
    return p.stdout.read().decode('utf-8')

def make_replace(pos: int, char: str) -> Callable[[List[str]], List[str]]:
    def replace(txt: List[str]) -> List[str]:
        return txt[:pos] + [char] + txt[pos+1:]
    return replace

def make_insert(pos: int, char: str) -> Callable[[List[str]], List[str]]:
    def insert(txt: List[str]) -> List[str]:
        return txt[:pos] + [char] + txt[pos:]
    return insert

class RawFileState:
    def __init__(self, dir_path: str, fname: str) -> None:
        self.name: str = fname
        self.dir: str = dir_path
        self.run: Callable[[List[str]], str] = lambda cs: run_program(cs, self.dir)
        self.branches = ['master']
        program = ['mkdir -p %s' % dir_path]
        run_program(program)
        program = ['git init', 'touch %s' % fname, 'git add %s' % fname, 'git commit -m "First commit"']
        self.run(program)

    def get(self, br: str) -> List[str]:
        program = ['git checkout %s > /dev/null' % br, 'cat %s' % self.name]
        r = self.run(program)
        if r == '':
            return []
        else:
            return list(filter(lambda l: l != "", r.split('\n')))

    def set(self, br: str, local: List[str]) -> None:
        if len(local) == 0:
            program = ['git checkout %s' % br, 'echo -n "" > %s' % self.name]
        else:
            program = ['git checkout %s' % br, 'echo "%s" > %s' % (local[0], self.name)]
            for line in local[1:]:
                program += ['echo "%s" >> %s' % (line, self.name)]
        self.run(program)

    def merge(self, br1: str, br2: str) -> bool:
        program = ['git checkout %s > /dev/null' % br2, 'git merge %s > /dev/null' % br1, 'git ls-files -u']
        return self.run(program) == ''

    def fork(self, br: str, brn: str, add_to_branches = False) -> None:
        program = ['git checkout %s' % br, 'git checkout -b %s' % brn]
        self.run(program)
        if add_to_branches:
            self.branches += [brn]

    def commit(self, br: str, msg: str, update: Callable[[List[str]], List[str]]) -> None:
        lstate = self.get(br)
        lstate = update(lstate)
        self.set(br, lstate)
        program = ['git checkout %s' % br, 'git add .', 'git commit -m "%s"' % msg]
        self.run(program)

    def last_commit_id(self, br: str) -> str:
        program = ['git checkout %s > /dev/null' % br, 'git log --oneline | head -n 1']
        return self.run(program).split(' ')[0]

class FileState(RawFileState):
    def next(self, id: str, new_id: str, action: Action) -> Optional[str]:
        for br in self.branches:
            self.fork(id + br, new_id + br)
        if isinstance(action, Insert):
            self.commit(new_id + action.br, action.br + ' insert ' + str(action.i) + ' ' + action.x, make_insert(action.i, action.x))
            return self.last_commit_id(new_id + action.br)
        if isinstance(action, Replace):
            self.commit(new_id + action.br, action.br + ' replace ' + str(action.i) + ' ' + action.x, make_replace(action.i, action.x))
            return self.last_commit_id(new_id + action.br)
        if isinstance(action, Merge):
            if self.merge(new_id + action.br1, new_id + action.br2):
                return self.last_commit_id(new_id + action.br2)
            else:
                return None

    def value(self, id: str, br: str) -> List[str]:
        return self.get(id + br)

    def virtual_ancestor_value(self, commit_ids: List[str]) -> Optional[List[str]]:
        program = ['git checkout %s > /dev/null' % commit_ids[0], 'git checkout -b #temp > /dev/null']
        for commit_id in commit_ids:
            program.append('git merge %s > /dev/null' % commit_id)
        if self.run(program) != '':
            return None
        program = ['git checkout #temp > /dev/null', 'cat %s' % self.name, 'git checkout master > /dev/null', 'git branch -D #temp > /dev/null']
        r = self.run(program)
        if r == '':
            return []
        else:
            return list(filter(lambda l: l != "", r.split('\n')))

T = TypeVar('T')
class Graph(Generic[T]):
    def __init__(self, vs: List[T], es: List[Tuple[T, T]]) -> None:
        self.g: Dict[T, List[T]] = {v: [] for v in vs}
        for (v, u) in es:
            self.insert(v, [u])
        
    def insert(self, f: T, t: List[T]) -> None:
        self.g[f] = (self.g[f] if f in self.g.keys() else []) + t

    def _dfs1(self, v: T, acc: Dict[T, int]) -> None:
        acc[v] = acc[v] + 1 if v in acc.keys() else 1
        for w in self.g[v]:
            self._dfs1(w, acc)

    def _dfs2(self, v: T, acc: Dict[T, int], k: int) -> None:
        acc[v] = acc[v] + k if v in acc.keys() else 1
        for w in self.g[v]:
            self._dfs2(w, acc, acc[v])

    def lca(self, v: T, u: T) -> List[T]:
        acc = {}
        self._dfs1(v, acc)
        self._dfs2(u, acc, 1)
        r = []
        for (w, k) in acc.items():
            if k == 2:
                r.append(w)
        return r

class BranchInfo:
    def __init__(self, head: str, value: List[str], commit_history: Set[str]):
        self.head, self.value, self.commit_history = head, value, commit_history

    def __str__(self):
        return 'head: %s; value: %s; commit_history: %s' % (self.head, str(self.value), str(self.commit_history))

class Leaf:
    def __init__(self, id: str, action_set: List[Action],
            graph: Graph[str], branches_info: Dict[str, BranchInfo]) -> None:
        self.id, self.action_set = id, action_set
        self.graph, self.branches_info = graph, branches_info

def select(leaves: List[Leaf]):
    return leaves[-1]

fs = FileState('~/new_test', 't.txt')
branches = ['master']
num_branches = 3
init_file_len = 3
for pos in range(init_file_len):
    br = branches[0]
    char = chr(ord('a') + pos)
    msg = br + ' insert ' + str(pos) + ' ' + char
    fs.commit(br, msg, make_insert(pos, char))
for i in range(num_branches - 1):
    branches.append(chr(ord('A') + i))
    fs.fork(branches[0], branches[i + 1], True)
commit_id = fs.last_commit_id(branches[0])
branch_info = BranchInfo(commit_id, fs.value('', branches[0]), {commit_id})
branches_info = {br: branch_info for br in branches}
#for u in branches_info:
#    print('%s,,, %s' % (u, branches_info[u]))
leaves = [Leaf('', [Insert(0, 'x', branches[0])], Graph([commit_id], []), branches_info)]
id = 1
while len(leaves) < 5:
    leaf = select(leaves)
    (action_set, graph, branches_info) = leaf.action_set, leaf.graph, leaf.branches_info
    action = action_set[0]
    new_id = str(id)
    commit_id = fs.next(leaf.id, new_id, action)
    id += 1
    if commit_id is None:
        continue
    new_action_set = action_set
    new_graph = deepcopy(graph)
    new_branches_info = copy(branches_info)
    if isinstance(action, Insert) or isinstance(action, Replace):
        new_graph.insert(commit_id, [branches_info[action.br].head])
        new_branches_info[action.br] = BranchInfo(commit_id, fs.value(new_id, action.br), branches_info[action.br].commit_history.union({commit_id}))
    else:
        assert isinstance(action, Merge)
        br1, br2 = action.br1, action.br2
        if commit_id == branches_info[br2].head:
            continue
        if commit_id == branches_info[br1].head:
            new_branches_info[br2] = BranchInfo(commit_id, fs.value(new_id, br2), branches_info[br1].commit_history)
        else:
            new_graph.insert(commit_id, [branches_info[br1].head, branches_info[br2].head])
            new_branches_info[br2] = BranchInfo(commit_id, fs.value(new_id, br2), branches_info[br2].commit_history.union(branches_info[br1].commit_history).union({commit_id}))
            inconsistence = False
            for br in branches:
                if br != br2 and \
                        new_branches_info[br].commit_history == new_branches_info[br2].commit_history and \
                        new_branches_info[br].value != new_branches_info[br2].value:
                    print('Inconsistence: %s(%s, %s)', new_id, br, br2)
                    inconsistence = True
            if inconsistence:
                continue
    leaves.append(Leaf(new_id, new_action_set, new_graph, new_branches_info))

# test begins
#commit_id_1 = fs.next('', '1', Insert(3, branches[1]), 'e')
#commit_id_2 = fs.next('1', '2', Replace(1, branches[2]), 'd')
#print(commit_id_1)
#print(commit_id_2)
#print(fs.next('2', '3', Merge(branches[1], branches[2]), 'f'))
#print(fs.value('3', branches[2]))
#print(fs.virtual_ancestor_value([commit_id_1, commit_id_2]))
