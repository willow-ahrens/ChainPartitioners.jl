# ChainPartitioners

<!---
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://peterahrens.github.io/ChainPartitioners.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://peterahrens.github.io/ChainPartitioners.jl/dev)
--->
[![Build Status](https://github.com/peterahrens/ChainPartitioners.jl/workflows/CI/badge.svg)](https://github.com/peterahrens/ChainPartitioners.jl/actions)
[![Coverage](https://codecov.io/gh/peterahrens/ChainPartitioners.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/peterahrens/ChainPartitioners.jl)

This package is a collection of matrix partitioning and reordering routines. ChainPartitioners provides functions to partition graphs with and without first reordering the vertices. Wrapper functions are provided to unify the interface over several external partitioning algorithms. More documentation is incoming, and I welcome your feedback, especially on interface decisions.

### Publications

[P. Ahrens, “Contiguous Graph Partitioning For Optimal Total Or Bottleneck Communication,” arXiv:2007.16192v4 [cs], Nov. 2020](http://arxiv.org/abs/2007.16192)

[P. Ahrens and E. Boman, “On Optimal Partitioning For Sparse Matrices In Variable Block Row Format,” arXiv:2005.12414v2 [cs], Nov. 2020](http://arxiv.org/abs/2005.12414)

### Extras

Because several contiguous partitioners require complex statistics about matrix structure, this package also provides datastructures to lazily compute sparse prefix sums on sparse matrices. There's also a pretty fast implementation of Reverse Cuthill-McKee reordering.
