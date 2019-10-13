# pyre-strict
from typing     import Callable, List

from gitwrapper import Repo, bash_check, run
from lang       import State

class FileState(Repo):
    def __init__(self, path: str, init: bool = False) -> None:
        self.name: str = path.split('/')[-1]
        super().__init__('/'.join(path.split('/')[:-1]), init)
        if init:
            run(['echo "%s" > %s' % ('a', self.name),
                 'git add %s' % self.name,
                 'git commit -m "First"'], self.path)

    def clone(self, path: str):
        repo = super().clone(path)
        return FileState(repo.path + self.name)

    def get(self, br: str) -> List[str]:
        if br not in self.branches:
            raise Exception('Branch %s does not exist!' % br)
        r = run(['git checkout %s' % br,
                 'cat %s' % self.name], self.path, True)
        return r.split('\n') if r != '' else []

    def set(self, br: str, st: List[str]) -> None:
        if br not in self.branches:
            raise Exception('Branch %s does not exist!' % br)
        run(['git checkout %s' % br,
             'echo -n "" > %s' % self.name], self.path)
        for line in st:
            run(['echo "%s" >> %s' % (line, self.name)], self.path)

    def commit(self, br: str, msg: str, 
            update: Callable[[List[str]], List[str]]) -> None:
        self.set(br, update(self.get(br)))
        run(['git add %s' % self.name,
             'git commit -m "%s"' % msg], self.path)

    def num_lines(self, br: str) -> int:
        return len(self.get(br))
