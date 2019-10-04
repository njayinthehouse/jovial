# pyre-strict
import subprocess
from   typing import Callable, List, Optional
from   enum   import Enum

from lang import Result, State

LANGUAGE = '/bin/bash'

def run_program(commands: List[str], path: Optional[str] = None) -> str:
    p = subprocess.Popen(LANGUAGE, stdin=subprocess.PIPE, 
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if path is not None:
        p.stdin.write(('cd %s\n > /dev/null' % path).encode())
    for command in commands:
        p.stdin.write(('%s\n' % command).encode())
    if path is not None:
        p.stdin.write(b'cd -\n > /dev/null')
    p.stdin.close()
    return p.stdout.read().decode('utf-8')


class FileState(State[str, List[str]]):
    def __init__(self, dir_path: str, fname: str, created: bool = False) -> None:
        self.name: str = fname
        self.dir: str = dir_path
        self.run: Callable[[List[str]], str] = lambda cs: run_program(cs, self.dir)
        self.branches: List[str] = ['master']
        if not created:
            program = ['mkdir %s' % dir_path]
            run_program(program)
            program = ['git init', 'touch %s' % fname, 'git add %s' % fname, 'git commit -m "First commit"']
            self.run(program)

    def get(self, br: str) -> List[str]:
        program = ['git checkout %s > /dev/null' % br, 'cat %s' % self.name]
        r = self.run(program)
        if len(r) == 1 and r[0] == '':
            return []
        else:
            return r.split('\n')

    def set(self, br: str, local: List[str]) -> None:
        if len(local) == 0:
            program = ['git checkout %s' % br, 'echo "" > %s' % self.name]
        else:
            program = ['git checkout %s' % br, 'echo "%s" > %s' % (local[0], self.name)]
            for line in local[1:]:
                program += ['echo "%s" >> %s' % (line, self.name)]
        self.run(program)

    def common_ancestor(self, br1: str, br2: str) -> str:
        program = ['git merge-base %s %s' % (br1, br2)]
        return self.run(program)

    def merge(self, br1: str, br2: str) -> Result:
        program = ['git checkout %s > /dev/null' % br2, 'git merge %s > /dev/null' % br1, 'git ls-files -u']
        return Result.Ok if self.run(program) == [] else Result.Failure

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


class ClonableFileState(FileState):
    def __init__(self, dir_path: str, fname: str, created: bool = False) -> None:
        super().__init__(dir_path, fname, created)

    def clone(self, dir_path: str) -> ClonableFileState:
        program = ['rm -rf %s' % dir_path, 'cp -r . %s' % dir_path]
        self.run(program)
        return ClonableFileState(dir_path, self.name, True)

#if __name__ == '__main__':
#    fs = FileState('/tmp/git-test', 'state.txt', True)
#    #fs.commit('master', 'Second commit', lambda _: ["Committed"])
#    fs.fork('master', 'child')
#    fs.commit('child', 'Third commit', lambda _: ["Boom"])
#    fs.merge('child', 'master')
