# pyre-strict
from abc    import ABC, abstractmethod
from typing import Any, Callable, Dict, Generic, List, Optional, Tuple, TypeVar

Branch = TypeVar("Branch")
Local  = TypeVar("Local")

class Frontend(ABC, Generic[Branch]):

    class Update(ABC):
        '''
        Abstract datatype for the AST of all kinds of local updates. 
        Add any update expressions to a subclass of Frontend as subclasses of this class.
        '''

    class Command(ABC):
        @abstractmethod
        def to_tuple(self) -> Tuple[Any, ...]: pass
    
    class Merge(Command):
        def __init__(self, br1: Branch, br2: Branch) -> None:
            self.br1: Branch = br1
            self.br2: Branch = br2

        def to_tuple(self) -> Tuple[Branch, Branch]:
            return self.br1, self.br2

    class Fork(Command):
        def __init__(self, br: Branch, brn: Branch) -> None:
            self.br: Branch = br
            self.brn: Branch = brn

        def to_tuple(self) -> Tuple[Branch, Branch]:
            return self.br, self.brn

    class Commit(Command):
        def __init__(self, br: Branch, msg: str, update) -> None:
            self.br: Branch = br
            self.msg: str = msg
            self.update = update

        def to_tuple(self):
            return self.br, self.msg, self.update

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
                tail = "".join(args[2:]).split("\"")[1:]
                msg = tail[0]
                update = self.parseUpdate(tail[1])
                return self.Commit(args[1], msg, update)
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
    def set(self, br: Branch, local: Local) -> None: pass
    
    @abstractmethod
    def common_ancestor(self, br1: Branch, br2: Branch) -> Branch: pass
    
    @abstractmethod
    def merge(self, br1: Branch, br2: Branch) -> None: pass
    
    @abstractmethod
    def fork(self, br: Branch, brn: Branch) -> None: pass
    
    @abstractmethod
    def commit(self, br: Branch, msg: str, update: Callable[[Local], Local]) -> None: pass
 
    @abstractmethod
    def alert(self) -> Optional[str]:
        '''For cases where we want to break execution and store the result'''
        pass

    @abstractmethod
    def record_state(self) -> None:
        '''To record interesting situations'''
        pass
    
class Backend(ABC, Generic[Branch, Local]):

    def run(self, state: State[Branch, Local], program: List[Frontend.Command]) -> None:
        for command in program:
            if state.alert() is not None:
                state.record_state()
                break
            else:
                self.eval(state, command)

    def eval(self, state: State[Branch, Local], command: Frontend.Command) -> None:
        if type(command) == Frontend.Merge:
            br1, br2 = command.to_tuple()
            state.merge(br1, br2)
        elif type(command) == Frontend.Fork:
            br, brn = command.to_tuple()
            state.fork(br, brn)
        elif type(command) == Frontend.Commit:
            br, msg, updateNode = command.to_tuple()
            state.commit(br, msg, self.update(updateNode))
        else:
            raise Exception("Command not found!")
   
    @abstractmethod
    def update(self, update: Frontend.Update) -> Callable[[Local], Local]: 
        '''To update the local state. Implements the Update AST node.'''
 
