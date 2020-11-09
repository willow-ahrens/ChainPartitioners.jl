# ChainPartitioners

[![Travis](https://travis-ci.org/peterahrens/ChainPartitioners.jl.svg?branch=master)](https://travis-ci.org/peterahrens/ChainPartitioners.jl)
[![Codecov](http://codecov.io/github/peterahrens/ChainPartitioners.jl/coverage.svg?branch=master)](http://codecov.io/github/peterahrens/ChainPartitioners.jl?branch=master)

This package is a collection of matrix partitioning and reordering routines. ChainPartitioners provides functions to partition graphs with and without first reordering the vertices. Wrapper functions are provided to unify the interface over several external partitioning algorithms. More documentation is incoming, and I welcome your feedback, especially on interface decisions.

## Extras
Because several contiguous partitioners require complex statistics about matrix structure, this package also provides datastructures to lazily compute sparse prefix sums on sparse matrices.

This package has a pretty fast implementation of Reverse Cuthill-McKee reordering.