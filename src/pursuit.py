# pyre-strict
from abc import ABC
import subprocess
from typing import Callable, List, Optional, Dict, Set, Generic, Tuple, TypeVar
from enum import Enum
from random import randint, choice
from copy import copy, deepcopy

class Action(ABC): pass

class Insert(Action):
    def __init__(self, i: int, x: str, br: str):
        self.i, self.x, self.br = i, x, br

    def __str__(self):
        return 'insert ' + str(self.i) + ' ' + self.x + ' ' + self.br

class Replace(Action):
    def __init__(self, i: int, x: str, br: str):
        self.i, self.x, self.br = i, x, br

    def __str__(self):
        return 'replace ' + str(self.i) + ' ' + self.x + ' ' + self.br

class Merge(Action):
    def __init__(self, br1: str, br2: str):
        self.br1, self.br2 = br1, br2

    def __str__(self):
        return 'merge ' + self.br1 + ' ' + self.br2

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
        if self.run(program) != '':
            self.run(['git reset --hard HEAD'])
            return False
        return True

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
    def __init__(self, dir_path: str, fname: str, count: int = 0) -> None:
        RawFileState.__init__(self, dir_path, fname)
        self.count = 0

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

    def virtual_ancestor(self, commit_ids: List[str]) -> Optional[str]:
        br = 'temp#' + str(self.count)
        self.count += 1
        program = ['git checkout %s > /dev/null' % commit_ids[0], 'git checkout -b %s > /dev/null' % br]
        for commit_id in commit_ids:
            program.append('git merge %s > /dev/null' % commit_id)
        program.append('git ls-files -u')
        if self.run(program) != '':
            return None
        return self.last_commit_id(br)

    def virtual_ancestor_value(self, commit_ids: List[str]) -> Optional[List[str]]:
        program = ['git checkout %s > /dev/null' % commit_ids[0], 'git checkout -b #temp > /dev/null']
        for commit_id in commit_ids:
            program.append('git merge %s > /dev/null' % commit_id)
        program.append('git ls-files -u')
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

    def __str__(self):
        return str(self.g)
        
    def insert(self, f: T, t: List[T]) -> None:
        self.g[f] = (self.g[f] if f in self.g.keys() else []) + t

    def _dfs1(self, v: T, acc: Dict[T, int]) -> None:
        if v not in acc:
            acc[v] = 1
            for w in self.g[v]:
                self._dfs1(w, acc)

    def _dfs2(self, v: T, acc: Dict[T, int], acc2: Dict[T, int], k: int) -> None:
        if v not in acc2:
            if v in acc:
                acc2[v] = acc[v] + k + 1
                for w in self.g[v]:
                    self._dfs2(w, acc, acc2, 1)
            else:
                for w in self.g[v]:
                    self._dfs2(w, acc, acc2, 0)

    def ancestor(self, v: T, u: T) -> bool:
        if v == u:
            return True
        else:
            for w in self.g[v]:
                if self.ancestor(w, u):
                    return True
            return False

    def lca(self, v: T, u: T) -> List[T]:
        acc = {}
        acc2 = {}
        self._dfs1(v, acc)
        self._dfs2(u, acc, acc2, 0)
        r = []
        for (w, k) in acc2.items():
            if k == 2:
                r.append(w)
        return r

class BranchInfo:
    def __init__(self, head: str, value: List[str], commit_history: Set[str]):
        self.head, self.value, self.commit_history = head, value, commit_history

    def __str__(self):
        return 'head: %s; value: %s; commit_history: %s' % (self.head, str(self.value), str(self.commit_history))

def flimit(q):
    def aux(f, xs):
        if xs == []:
            raise Exception('Empty list!')
        r = xs[0]
        for x in xs[1:]:
            if q(f(x), f(r)):
                r = x
        return r
    return aux

fmax = flimit(lambda x, y: x > y)
fmin = flimit(lambda x, y: x < y)

def lcs(xs, ys):
    n = len(xs)
    m = len(ys)
    opt = [[[] for i in range(m + 1)] for j in range(n + 1)]

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if xs[i - 1] == ys[j - 1]:
                opt[i][j] = fmax(len, [opt[i - 1][j], opt[i][j - 1], opt[i - 1][j - 1] + [xs[i - 1]]])
            else:
                opt[i][j] = fmax(len, [opt[i - 1][j], opt[i][j - 1]])
    return opt[n][m]

def delta(ys: List[str], zs: List[str]) -> List[int]:
    xs = lcs(ys, zs)
    i = j = k = q = 0
    nx = len(xs)
    ny = len(ys)
    nz = len(zs)
    r = []

    while i < nx:
        if xs[i] == ys[j] == zs[k]:
            r += [q]
            i += 1
            j += 1
            k += 1
            q += 1
        elif xs[i] != ys[j]:
            r += [q]
            j += 1
        else: # if ys[j] != zs[k]
            k += 1
            q += 1

    r += [nz] * (ny - j + 1)
    return r

def alpha(lca: List[str], i: int, vs: List[str], d: Optional[List[int]] = None) -> int:
    if d is None:
        d = delta(lca, vs)
    for (j, n) in enumerate(d):
        if n >= i:
            return j
    return len(d) - 1

def gamma(lca: List[str], i: int, vs: List[str], d: Optional[List[int]] = None):
    if d is None:
        d = delta(lca, vs)
    if i == 0:
        return set(range(d[i] + 1))
    else:
        return set(range(d[i - 1] + 1, d[i] + 1))
    
def beta(f: List[str], lca: List[str], t: List[str]) -> List[str]:
    j = alpha(lca, i, f)
    return gamma(lca, j, t)
    
def inserted(anc: List[str], vs: List[str]):
    r = set([])
    d = delta(anc, vs)
    for i in range(len(anc) + 1):
        if len(gamma(anc, i, vs, d)) > 1:
            r.add(i)
    return r

def replaced(anc: List[str], vs: List[str]):
    r = set([])
    xs = lcs(anc, vs)
    j = 0
    for x in xs:
        while x != anc[j]:
            r.add(j)
            j += 1
        j += 1
    r |= set(range(j, len(anc)))
    return r

class ActionSetGenerator:
    def __init__(self, fs: FileState, root_commit: str, root_value: List[str], branch_names: List[str], chars: List[str]):
        self.branches_info = {br: BranchInfo (root_commit, root_value, {root_commit}) for br in branch_names}
        br1, br2, br3 = branch_names
        self._lca = {brp: root_commit for brp in [(br1, br2), (br2, br3), (br3, br1)]}
        self.graph = Graph([root_commit], [])
        self.value = {root_commit: root_value}
        self.chars = chars
        self.brm = set([])
        self.fs = fs

    def get(self, cid: str) -> List[str]:
        if cid not in self.value:
            self.value[cid] = self.fs.get(cid)
        return self.value[cid]

    def get_br(self, br: str) -> List[str]:
        return self.get(self.head(br))

    def lca(self, br1: str, br2: str) -> str:
        if (br1, br2) in self._lca:
            return self._lca[(br1, br2)]
        else:
            return self._lca[(br2, br1)]

    def set_lca(self, br1: str, br2: str, cid: str):
        if (br1, br2) in self._lca:
            self._lca[(br1, br2)] = cid
        else:
            self._lca[(br2, br1)] = cid
        
    def _insertable(self, br1: str, br: str):
        c1, c = self.head(br1), self.head(br)
        xs, ys = self.get(c1), self.get(c)
        clca = self.lca(br1, br)
        lca_val = self.get(clca)

        taken_i = inserted(lca_val, xs)
        taken_r_0 = replaced(lca_val, xs)
        taken_r = taken_r_0 | set(map(lambda i: i + 1, taken_r_0)) 
        slots = set(range(len(lca_val) + 1)) - (taken_i | taken_r)
        r = set([])
        for i in slots:
            r |= gamma(lca_val, i, ys)
        return r

    def insertable(self, br: str):
        r = None
        for br1 in self.branches_info:
            if br != br1:
                if r is None:
                    r = self._insertable(br1, br)
                else:
                    r &= self._insertable(br1, br)
        return r

    def _replaceable(self, br1: str, br: str):
        c1, c = self.head(br1), self.head(br)
        xs, ys = self.get(c1), self.get(c)
        clca = self.lca(br1, br)
        lca_val = self.get(clca)

        taken_r = replaced(lca_val, xs)
        taken_i_0 = inserted(lca_val, xs)
        taken_i = taken_i_0 | set(map(lambda i: i - 1, taken_i_0))
        slots = set(range(len(lca_val))) - (taken_i | taken_r)
        r = set([])
        for i in slots:
            r |= gamma(lca_val, i, ys)
        r -= {len(ys)}
        return r

    def replaceable(self, br: str):
        r = None
        for br1 in self.branches_info:
            if br != br1:
                if r is None:
                    r = self._replaceable(br1, br)
                else:
                    r &= self._replaceable(br1, br)
        return r

    def head(self, br: str) -> str:
        return self.branches_info[br].head

    def on_insert(self, i: int, x: str, br: str, cid: str):
        prev = self.head(br)
        self.graph.insert(cid, [prev])
        self.value[cid] = copy(self.value[prev])
        self.value[cid].insert(i, x)
        self.branches_info[br] = BranchInfo(cid, self.value[cid], self.branches_info[br].commit_history.union({cid}))
        self.brm |= {(br, bri) for bri in self.branches_info if br != bri}

    def on_replace(self, i: int, x: str, br: str, cid: str):
        prev = self.head(br)
        self.graph.insert(cid, [prev])
        self.value[cid] = copy(self.value[prev])
        self.value[cid][i] = x
        self.branches_info[br] = BranchInfo(cid, self.value[cid], self.branches_info[br].commit_history.union({cid}))
        self.brm |= {(br, bri) for bri in self.branches_info if br != bri}

    def on_merge(self, f: str, t: str, cid: str):
        v = self.fs.get(cid)
        self.value[cid] = v
        if cid != self.head(t):
            if cid == self.head(f):
                self.branches_info[t] = BranchInfo(cid, v, self.branches_info[f].commit_history)
            else:
                self.graph.insert(cid, [self.branches_info[f].head, self.branches_info[t].head])
                self.branches_info[t] = BranchInfo(cid, v, self.branches_info[t].commit_history.union(self.branches_info[f].commit_history).union({cid}))
        self.brm -= {(f, t)}
        other = None
        for br in self.branches_info:
            if br != f and br != t:
                other = br
                break
        lcas = self.graph.lca(cid, self.head(other))
        print('f', f)
        print('t', t)
        if len(lcas) > 1:
            new_lca = fs.virtual_ancestor(lcas)
        else:
            new_lca = lcas[0]
        self.set_lca(t, other, new_lca)
        self.set_lca(t, f, self.head(f))

    def on_action(self, action: Action, cid: str):
        print(self.graph)
        if isinstance(action, Insert):
            self.on_insert(action.i, action.x, action.br, cid)
        elif isinstance(action, Replace):
            self.on_replace(action.i, action.x, action.br, cid)
        else:
            self.on_merge(action.br1, action.br2, cid)

    def build(self):
        ins = {br: self.insertable(br) for br in self.branches_info}
        rep = {br: self.replaceable(br) for br in self.branches_info}
        bvals = {br: self.branches_info[br].value for br in self.branches_info}
        return ActionSet(self.chars, ins, rep, self.brm, bvals)

class ActionSet:
    def __init__(self, chars, ins, rep, brm, bvals): 
        self.ins = ins
        self.rep = rep
        self.cs = chars
        self.brm = brm
        self.bvals = bvals

    def __str__(self):
        return 'Insertions:' + str(self.ins) + '\nReplacements:' + str(self.rep) + '\nMerges:' + str(self.brm)

    def pop(self):
        q1 = randint(1, 12)
        if q1 <= 2:
            x = choice(list(self.cs))
            q2 = randint(0, 2)
            br = list(self.ins.keys())[q2]
            if len(self.ins[br]) > 0:
                i = choice(list(self.ins[br]))
            else:
                return self.pop()
            return Insert(i, x, br)
        elif q1 <= 4:
            x = choice(list(self.cs))
            q2 = randint(0, 2)
            br = list(self.rep.keys())[q2]
            if len(self.rep[br]) > 0:
                i = choice(list(self.rep[br]))
                while self.bvals[br][i] == x:
                    x = choice(list(self.cs))
            else:
                return self.pop()
            return Replace(i, x, br)
        else:
            if len(self.brm) > 0:
                br1, br2 = choice(list(self.brm))
            else:
                return self.pop()
            return Merge(br1, br2)

class Leaf:
    def __init__(self, id: str, action_set_gen: ActionSetGenerator) -> None:
        self.id, self.action_set_gen = id, action_set_gen

def select(leaves: List[Leaf]):
    return leaves[-1]

if __name__ == '__main__':
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
    id = 1
    leaves = [Leaf('', ActionSetGenerator(fs, commit_id, fs.get(commit_id), ['master', 'A', 'B'], [chr(ord('a') + i) for i in range(26)]))]
    while True:
        leaf = select(leaves)
        asg = leaf.action_set_gen
        action_set = asg.build()
        print(action_set)
        action = action_set.pop()
        new_id = str(id)
        commit_id = fs.next(leaf.id, new_id, action)
        id += 1
        if commit_id is None:
            leaves.pop()
            print('pop')
            continue
        new_asg = deepcopy(asg)
        new_asg.on_action(action, commit_id)
        new_branches_info = new_asg.branches_info
        if isinstance(action, Merge):
            br1, br2 = action.br1, action.br2
            inconsistence = False
            for br in branches:
                if br != br2 and \
                        new_branches_info[br].commit_history == new_branches_info[br2].commit_history and \
                        new_branches_info[br].value != new_branches_info[br2].value:
                    print('Inconsistence: %s(%s, %s)', new_id, br, br2)
                    print('value of (%s): %s', br, str(new_branches_info[br].value))
                    print('value of (%s): %s', br2, str(new_branches_info[br2].value))
                    inconsistence = True
                    assert False
            if inconsistence:
                continue
        if len(leaves) < 20 and (randint(1, 10) > 3 or len(leaves) == 1):
            print(action)
            leaves.append(Leaf(new_id, new_asg))
        else:
            print('pop')
            leaves.pop()
