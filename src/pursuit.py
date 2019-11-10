# pyre-strict
from abc import ABC
import subprocess
from typing import Callable, List, Optional
from enum import Enum

class Action(ABC): pass

class Insert(Action):
    def __init__(self, i: int, br: str):
        self.i, self.br = i, br

class Replace(Action):
    def __init__(self, i: int, br: str):
        self.i, self.br = i, br

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

class FileState(RawFileState):
    def next(self, id: str, new_id: str, action: Action, char: str) -> Optional[str]:
        for br in self.branches:
            self.fork(id + br, new_id + br)
        if isinstance(action, Insert):
            self.commit(new_id + action.br, action.br + ' insert ' + str(action.i) + ' ' + char, make_insert(action.i, char))
        elif isinstance(action, Replace):
            self.commit(new_id + action.br, action.br + ' replace ' + str(action.i) + ' ' + char, make_replace(action.i, char))
        elif isinstance(action, Merge) and not self.merge(new_id + action.br1, new_id + action.br2):
            return None
        program = ['git log --oneline | head -n 1']
        return self.run(program).split(' ')[0]

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

fs = FileState('~/new_test', 't.txt')
branches = ['master']
num_branches = 3
init_file_len = 3
for pos in range(init_file_len):
    branch = branches[0]
    char = chr(ord('a') + pos)
    msg = branch + ' insert ' + str(pos) + ' ' + char
    fs.commit(branch, msg, make_insert(pos, char))
for i in range(num_branches - 1):
    branches.append(chr(ord('A') + i))
    fs.fork(branches[0], branches[i + 1], True)

# test begins
commit_id_1 = fs.next('', '1', Insert(3, branches[1]), 'e')
commit_id_2 = fs.next('1', '2', Replace(1, branches[2]), 'd')
print(commit_id_1)
print(commit_id_2)
print(fs.next('2', '3', Merge(branches[1], branches[2]), 'f'))
print(fs.value('3', branches[2]))
print(fs.virtual_ancestor_value([commit_id_1, commit_id_2]))