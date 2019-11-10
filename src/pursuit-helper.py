# pyre-strict
from typing import Callable, Dict, Generic, List, Tuple, TypeVar

CommitId = str
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

if __name__ == '__main__':
    g = Graph([0, 1, 2, 3], [(0, 1), (2, 1), (0, 3), (2, 3)])
    print(g.lca(0, 2))
