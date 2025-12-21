Random Projection
========
Random projection is a dimensionality reduction technique that projects high-dimensional data onto a lower-dimensional subspace using a random matrix. It's based on the Johnson-Lindenstrauss lemma, which states that distances between points are approximately preserved when projected to a sufficiently high dimensional, but much lower than original, random subspace.

The computational demands of evaluating the full expected site frequency spectrum (SFS) increase substantially with both sample size and model complexity. To address these challenges, we implement random projection as an efficient, low-dimensional approximation method that preserves essential signals of the full SFS while dramatically reducing computational cost.

All random projection capabilities are seamlessly integrated into Momi3's core architecture, accessible through the same functional interfaces demonstrated in the Tutorial section. Users can activate these accelerated methods by simply providing an additional parameter to existing functions, maintaining the same intuitive workflow while gaining significant performance benefits.

