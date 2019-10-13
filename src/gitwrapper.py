# pyre-strict
import subprocess

from enum   import Enum
from typing import List, Optional

LANGUAGE = '/bin/bash'

def run(commands: List[str], path: Optional[str] = None, 
        output: bool = False) -> str:
    p = subprocess.Popen(LANGUAGE, stdin=subprocess.PIPE, 
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if path is not None:
        print('Change dir to path %s' % path)
        p.stdin.write(('cd %s\n' % path).encode())
    for command in commands:
        print(command)
        p.stdin.write(('%s\n' % command).encode())
    p.stdin.close()
    r = p.stdout.read().decode('utf-8')
    print('R' + r)
    if output:
        return r
    else:
        return "Should not be read!"

def bash_check_scr(test: str) -> List[str]:
    return ['if [[ %s ]]' % test,
            'then',
            'echo "T"',
            'else',
            'echo "F"',
            'fi']

is_git_scr = bash_check_scr('! -d .git')    # pyre-ignore

def merge_scr(br1: str, br2: str) -> List[str]:
    return ['git checkout %s' % br2,
            'git merge %s' % br1]

def is_merge_conflict_scr(br: str) -> List[str]:
    return ['r=$(git ls-files -u)'] + bash_check_scr('r != ""')

def bash_check(scr: List[str], path: str) -> bool:
    r = run(scr, path, True)
    if len(r) != 1:
        raise Exception('Not suitable for checking using bash_check!')
    else:
        if r[0] == 'F':
            return False
        elif r[0] == 'T':
            return True
        else:
            raise Exception('Not suitable for checking using bash_check!')

mapl = lambda f, xs: list(map(f, xs))  # pyre-ignore

class MergeConflict:
    def __init__(self, br1: str, br2: str) -> None:
        self.br1: str = br1
        self.br2: str = br2

    def throw(self) -> None:
        raise Exception("Merge conflict when merging %s into %s" % (self.br1, self.br2))

class Repo:
    def __init__(self, path: str, init: bool = False) -> None:
        self.path: str = path
        self.name: str = path.split('/')[-1]
        if init:
            print('Repo initialized at %s' % path)
            run(['git init'], path)
        else:
            if not bash_check(is_git_scr, path):
                raise Exception('Directory is not git repo!')
        self.branches: List[str] = self._branches()
        
    def _branches(self) -> List[str]:
        bs = run(['git branch | cat'], self.path, output = True)
        r = mapl(lambda l: l[2:], bs)
        print(r)
        return r

    def clone(self, path: str):
        run(['git clone %s %s' % (self.path, path)])
        return Repo(path)

    def fork(self, br: str, nbr: str) -> None:
        if nbr in self.branches:
            raise Exception('Branch %s already exists!' % nbr)
        elif br not in self.branches:
            raise Exception('Branch %s does not exist!' % br)
        else:
            run(['git checkout %s' % br, 
                 'git checkout -b %s' % nbr], self.path)
            self.branches = self._branches()

    def merge(self, br1: str, br2: str) -> Optional[MergeConflict]:
        if br1 not in self.branches:
            raise Exception('Branch %s does not exist!' % br1)
        elif br2 not in self.branches:
            raise Exception('Branch %s does not exist!' % br2)
        else:
            run(merge_scr(br1, br2), self.path)
            if bash_check(is_merge_conflict_scr(br2), self.path):
                return MergeConflict(br1, br2)

    def history(self, br: str) -> List[str]:
        r = run(['git checkout %s' % br,
                 'git log --pretty=format:"%h" | cat'], self.path, True)
        return r.split('\n')

    
