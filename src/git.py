# pyre-strict
import subprocess
from   typing     import Callable, List, Optional

from .lang import State

LANGUAGE = '/bin/bash/'

def run_program(commands: List[str], path: str) -> str:
    p = subprocess.Popen(LANGUAGE, stdin=subprocess.PIPE, 
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    p.stdin.write(b'cd %s\n' % path)
    for command in commands:
        p.stdin.write(('%s\n' % command).encode())
    p.stdin.write(b'cd -\n')
    return p.stdout.read().decode('utf-8')

class FileState(State[str, List[str]]):
    def __init__(self, dir_path: str, fname: str) -> None:
        self.name: str = fname
        self.dir: str = dir_path
        self.run: Callable[[List[str]], str] = lambda cs: run_program(cs, self.dir)
        self.branches: List[str] = []

    def get(self, br: str) -> List[str]:
        program = ['git checkout %s' % br, 'cat %s' % self.name]
        return self.run(program).split('\n')

    def set(self, br: str, local: List[str]) -> None:
        program = ['git checkout %s' % br, 
                   'echo "%s" > %s' % (local[0], self.name)]
        for line in local[1:]:
            program += ['echo "%s" >> %s' % (line, self.name)]
        self.run(program)
        self

    def common_ancestor(self, br1: str, br2: str) -> str:
        program = ['git merge-base %s %s' % (br1, br2)]
        return self.run(program)

    def merge(self, br1: str, br2: str) -> None:
        program = ['git checkout %s' % br2, 'git merge %s' % br1]
        self.run(program)

    def fork(self, br: str, brn: str) -> None:
        program = ['git checkout %s' % br, 'git checkout -b %s' % brn]
        self.run(program)
        self.branches += [brn]

    def commit(self, br: str, msg: str, update: Callable[[List[str]], List[str]]) -> None:
        lstate = self.get(br)
        lstate = update(lstate)
        self.set(br, lstate)
        program = ['git checkout %s' % br, 'git add .',
                   'git commit -m "%s"' % msg]
        self.run(program)

    def history(self, br: str) -> List[str]:
        program = ['git checkout %s' % br, 'git log --pretty=oneline | cat']
        return self.run(program).split('\n')

    def alert(self) -> Optional[str]:
        for br1 in self.branches:
            for br2 in self.branches:
                if self.history(br1) == self.history(br2) and self.get(br1) != self.get(br2):
                    return "Interesting behaviour between %s and %s!" % (br1, br2)
        return None

    def record_state(self) -> None:
        pass
