# pyre-strict
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

    arity = {"merge": 2, "fork": 2, "commit": 4}

    def parseCommand(self, command: str) -> Command:
        args = command.split(" ")
        c = args[0]

        try:
            if c == "merge":
                return self.Merge(*args[1:])
            elif c == "fork":
                return self.Fork(*args[1:])
            elif c == "commit":
                tail = "".join(args[3:]).split("\"")[1:]
                msg = tail[0]
                update = self.parseUpdate(tail[1])
                return self.Commit(args[1], args[2], msg, update)
            else:
                raise Exception("Parse error: Unknown command %s" % args[0])
        except TypeError:
            raise Exception("Parse error: Arity error for command %s. %d arguments were expected, %d were given." % (c, self.arity[c], len(args[1:])))
        except IndexError:
            raise Exception("Parse error: Arity error for command %s. %d arguments were expected, %d were given." % (c, self.arity[c], len(args[1:])))

    @abstractmethod
    def parseUpdate(self, update: str) -> Update:
        pass

    def parse(self, program: List[str]) -> List[Command]:
        return list(map(self.parseCommand, program))

class State(ABC, Generic[Branch, Local]):
    @abstractmethod
    def get(self, br: Branch) -> Local: pass

    @abstractmethod 
    def set(self, br: Branch, local: Local) -> State[Branch, Local]: pass
    
    @abstractmethod
    def common_ancestor(self, br1: Branch, br2: Branch) -> Branch: pass
    
    @abstractmethod
    def merge(self, br1: Branch, br2: Branch) -> State[Branch, Local]: pass
    
    @abstractmethod
    def fork(self, br: Branch, brn: Branch) -> State[Branch, Local]: pass
    
    @abstractmethod
    def commit(self, br: Branch, cid: str, msg: str, update: Callable[[Local], Local]) -> State[Branch, Local]: pass
    
class Backend(ABC, Generic[Branch, Local]):

    def run(self, state: State[Branch, Local], program: List[Frontend.Command]) -> None:
        for command in program:
            if self.alert(state):
                self.record_state(state)
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
            br, cid, msg, updateNode = command                  # pyre-ignore
            return state.commit(br, cid, msg, self.update(updateNode))
        else:
            raise Exception("Command not found!")

    @abstractmethod
    def alert(self, state: State[Branch, Local]) -> bool:
        '''For cases where we want to break execution and store the result'''
        pass

    @abstractmethod
    def record_state(self, state: State[Branch, Local]) -> None:
        '''To record interesting situations'''
        pass
    
    @abstractmethod
    def update(self, update: Frontend.Update) -> Callable[[LocalState], LocalState]: 
        '''To update the local state. Implements the Update AST node.'''
 
