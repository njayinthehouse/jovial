# pyre-strict
from abc import ABC

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
