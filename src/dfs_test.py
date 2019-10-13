# pyre-strict
import sys

from abc       import ABC, abstractmethod
from enum      import Enum
from typing    import Any, List, Tuple

from filestate import FileState

class Command(ABC):
    @abstractmethod
    def to_tuple(self) -> Tuple[Any, ...]: pass  

class Update(ABC):
    @abstractmethod
    def apply(self, st: List[str]) -> List[str]: pass  

class Merge(Command):
    def __init__(self, br1: str, br2: str) -> None:
        self.br1: str = br1
        self.br2: str = br2

    def to_tuple(self) -> Tuple[str, str]:  
        return self.br1, self.br2

class Fork(Command):
    def __init__(self, br: str, brn: str) -> None:
        self.br: str = br
        self.brn: str = brn

    def to_tuple(self) -> Tuple[str, str]:
        return self.br, self.brn

class Commit(Command):
    def __init__(self, br: str, msg: str, update: Update) -> None:
        self.br: str = br
        self.msg: str = msg
        self.update = update

    def to_tuple(self) -> Tuple[str, str, Update]:
        return self.br, self.msg, self.update

class Insert(Update):
    def __init__(self, n: int, s: str) -> None:
        self.n: int = n
        self.s: str = s

    def apply(self, st: List[str]) -> List[str]:
        st.insert(self.n, self.s)
        return st

class Remove(Update):
    def __init__(self, n: int) -> None:
        self.n = n

    def apply(self, st: List[str]) -> List[str]:
        st.pop(self.n)
        return st

def update_gen(n: int, cs: List[str]) -> List[Update]:
    r = []
    for i in range(n):
        r.append(Remove(i))
        for c in cs:
            r.append(Insert(i, c))
    for c in cs:
        r.append(Insert(n, c))
    return r

def actions(fs: FileState, cs: List[str]) -> List[Command]:
    r = []
    for b1 in fs.branches:
        l = fs.num_lines(b1)
        updates = update_gen(l, cs)
        r += list(map(lambda f: Commit(b1, 'q', f), updates))
        for b2 in fs.branches:
            r.append(Merge(b1, b2))
    return r
    
def interesting(fs: FileState) -> bool:
    for br1 in fs.branches:
        for br2 in fs.branches:
            h1 = fs.history(br1)
            h2 = fs.history(br2)
            if h1 == h2:
                return fs.get(br1) == fs.get(br2)
    return False

i = 0

def fresh(name: str) -> str:
    i += 1
    return name + ('_%d' % i)

def do(fs: FileState, command: Command) -> FileState:
    nfs = fs.clone(fresh(fs.path))

    if type(command) == Commit:
        br, msg, update = command.to_tuple() 
        nfs.commit(br, msg, update)
    elif type(command) == Merge:
        br1, br2 = command.to_tuple()
        nfs.merge(br1, br2)
    else:
        raise Exception('Unwanted command!')

    return nfs
        

def pursuit(fs: FileState, depth: int, cs: List[str]) -> List[str]:
    def aux(fs: FileState, depth: int) -> List[str]:
        r = [fs.path] if interesting(fs) else []
        if depth != 0:
            for action in actions(fs, cs):
                r += aux(do(fs, action), depth - 1)
        return r
    return aux(fs, depth)

if __name__ == '__main__':
    fs = FileState(sys.argv[1], True)
    branches = ['b1', 'b2', 'b3', 'b4']
    characters = ['a', 'b', 'c', 'd']

    for b in branches:
        fs.fork('master', b)

    print(pursuit(fs, 10, characters))
