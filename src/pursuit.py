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

class FileState:
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

    def commit(self, br: str, msg: str, update: Callable[[List[str]], List[str]]) -> str:
        lstate = self.get(br)
        lstate = update(lstate)
        self.set(br, lstate)
        program = ['git checkout %s' % br, 'git add .', 'git commit -m "%s"' % msg]
        return self.run(program).split('\n')[1].split(' ')[1][:-1]

def next(fs: FileState, id: str, new_id: str, action: Action, char: str) -> Optional[str]:
    for br in fs.branches:
        fs.fork(id + br, new_id + br)
    if isinstance(action, Insert):
        return fs.commit(new_id + action.br, action.br + ' insert ' + str(action.i) + ' ' + char, make_insert(action.i, char))
    if isinstance(action, Replace):
        return fs.commit(new_id + action.br, action.br + ' replace ' + str(action.i) + ' ' + char, make_replace(action.i, char))
    if isinstance(action, Merge) and fs.merge(new_id + action.br1, new_id + action.br2):
        return 'success'
    return None

def value(fs: FileState, id: str, br: str) -> List[str]:
    return fs.get(id + br)

fs = FileState('~/new_test', 't.txt')
branches = ['master']
num_branches = 3
init_file_len = 3
for pos in range(init_file_len):
    branch = branches[0]
    char = chr(ord('a') + pos)
    msg = branch + ' insert ' + str(pos) + ' ' + char
    print(fs.commit(branch, msg, make_insert(pos, char)))
for i in range(num_branches - 1):
    branches.append(chr(ord('A') + i))
    fs.fork(branches[0], branches[i + 1], True)
print(next(fs, '', '1', Insert(3, branches[1]), 'e'))
print(next(fs, '1', '2', Replace(1, branches[2]), 'd'))
print(next(fs, '2', '3', Merge(branches[1], branches[2]), 'f'))
print(value(fs, '3', branches[2]))