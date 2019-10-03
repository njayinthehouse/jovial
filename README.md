# Jovial

Git is a grumpy old man with a lot of problems. For the happiness of future
generations, we want to find these problems to learn from them. Jovial is a
framework for trying to automatically detect these problems. It consists
of a language to model git, which allows us to talk about Git's behavior.
To achieve happiness (i.e. detect problems), we have a tweakable testing
harness, Pursuit (to be written).

## Contributing

### Getting Started

1. Clone the repo.
2. It's recommended to use a virtual environment. Install dependencies from 
`requirements.txt`. Don't forget to put your virtual environment directory
in the `.gitignore`, if you choose not to name it `venv`.
3. I've used [pyre][https://pyre-check.org] to typecheck my code. It's pretty
easy to use, and is in the requirements. But you don't have to use it if you
don't want to, of course. 

### Some Information

- `Branch` type variable for all branches.

- `Local` type variable for local state at a branch.

- The `Frontend` abstract class provides a framework for parsing commands.
  + Note the `parse` method, which converts strings into `Frontend.Command` nodes.
  + Also note the `Update` abstract class. Users must provide subtypes of this
  class to serve as AST nodes for the various kinds of updation that can be 
  performed at a `Local`

- The `State` abstract class is used to denote the global state of the 
concurrent system. Any implementation of it must provide a definition for 
merging, forking and committing to local state.

- The `Backend` abstract class is used to operate on the state. 
  + Its `update` method is of particular relevance. It converts `Frontend.Update`
  nodes into functions which can modify local states.
  + The `alert` method is used to check for interesting cases, which we might 
  record using `record_state`.

### Future Work

- FileState.alert is O(n^2). Need to be smarter about this in the future.
- For now, if we get a merge conflict, we just abandon the test. In the future, we want to undoing the merge conflict by reverting state instead.
