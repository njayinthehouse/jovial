from abc    import ABC, abstractmethod
from typing import Callable, Dict, Generic, List, NamedTuple, TypeVar

Branch = TypeVar("Branch")
Local  = TypeVar("Local")

class Frontend(ABC, Generic[Branch]):

    class Update(ABC):
        '''
        Abstract datatype for the AST of all kinds of local updates. 
        Add any update expressions to a subclass of Frontend as subclasses of this class.
        '''

    class Command(ABC): pass
    
    class Merge(Command, NamedTuple):
        br1: Branch
        br2: Branch

    class Fork(Command, NamedTuple):
        br: Branch
        br_new: Branch

    class Commit(Command, NamedTuple):
        br: Branch
        id: str
        msg: str
        update: Update

    def parseCommand(self, statement: str) -> Command:
        pass

    @abstractmethod
    def parseUpdate(self, update: str):
        pass

class State(ABC, Generic[Branch, Local]):
    @abstractmethod
    def get(self, br: Branch) -> Local: pass

    @abstractmethod 
    def set(self, br: Branch, local: Local): pass
    
    @abstractmethod
    def common_ancestor(self, br1: Branch, br2: Branch): pass
    
    @abstractmethod
    def merge(self, br1: Branch, br2: Branch): pass
    
    @abstractmethod
    def fork(self, br: Branch, brn: Branch): pass
    
    @abstractmethod
    def commit(self, br: Branch, cid: str, msg: str, update: Callable[[Local], Local]): pass
    
class Backend(ABC, Generic[Branch, Local]):

    def run(self, state: State[Branch, Local], program: List[Frontend.Command]):
        for command in program:
            if self.alert():
                self.record_state()
                break
            else:
                self.eval(state, command)

    def eval(self, state: State[Branch, Local], command: Frontend.Command) -> State[Branch, Local]:
        if type(command) == Frontend.Merge:
            br1, br2 = command                              # pyre-ignore
            return state.merge(br1, br2)
        elif type(command) == Frontend.Fork:
            br, brn = command                               # pyre-ignore
            return state.fork(br, brn)
        elif type(command) == Frontend.Commit:
            br, cid, msg, update = command                  # pyre-ignore
            return state.commit(br, cid, msg, update)
        else:
            raise Exception("Command not found!")

    @abstractmethod
    def alert(self, state) -> bool:
        '''For cases where we want to break execution and store the result'''
        pass

    @abstractmethod
    def record_state(self, state):
        '''To record interesting situations'''
        pass
    
    @abstractmethod
    def update(self, update: Frontend[Branch].Update): 
        '''To update the local state. Implements the Update AST node.'''
 
