# Jovial

A testing framework for version control systems that support merge and
forks operations.

## Getting Started

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
