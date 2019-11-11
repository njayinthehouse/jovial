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

def delta(ys, zs):
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

def alpha(fs: FileState, g: Graph, br1: BranchInfo, i: int, br2: BranchInfo) -> int:
    lca = fs.virtual_ancestor_value(g.lca(br1.head, br2.head))
    d = delta(lca, br2.value)
    n = len(d)
    return flimit(lambda x, y: x >= i and x < y)(id, d)

def gamma(fs: FileState, g: Graph, br1: BranchInfo, i: int, br2: BranchInfo) -> set[int]:
    lca = fs.virtual_ancestor_value(g.lca(br1.head, br2.head))
    d = delta(lca, br2.value)
    return set(range(d[i - 1] + 1, d[i] + 1))

def beta(fs: FileState, g: Graph, br1: BranchInfo, i: int, br2: BranchInfo) -> set[int]:
    j = alpha(fs, g, br2, i, br1)
    return gamma(fs, g, br1, j, br2)

class ActionSet:
    def __init__(self, chars: List[str], branches: List[BranchInfo], n: int = 3): 
        self.n = {br: set(range(n)) for br in branches}
        self.cs = set(chars)
        self.brs = branches
        self.bri = set(bs)
        self.brr = set(bs)
        brm = []
        for br in bs:
            for br1 in bs:
                if br != br1:
                    brm += [(br, br1)]
        self.brm = set(brm)

    def on_insert(self, i, br):
        n = len(self.n[br])
        self.n[br].add(n)
        other_brs = self.branches.filter(lambda b: b != br) 



if __name__ == '__main__':
    g = Graph([0, 1, 2, 3], [(0, 1), (2, 1), (0, 3), (2, 3)])
    print(alpha([1, 2, 3], 6, [0, 1, 1, 4, 2, 5, 3, 6]))
