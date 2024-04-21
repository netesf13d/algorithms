# -*- coding: utf-8 -*-
"""
Miscellaneous algorithms. These are not `standard` algorithms but can typically
be asked for in quant interviews.

* Miscellaneous algorithms
    - <sqrt> : compute square root of a float using Newton-Raphson method.
* Dynamic programming algorithms
    - <part_array_permutation> : solution to a dynamic programming problem.
    - <largest_square> : solution to a dynamic programming problem.
"""

import numpy as np


# =============================================================================
# Misc
# =============================================================================

def sqrt(a: float, n: int)-> float:
    """
    Compute the square root of a with precision 1e-n using Newton-Raphson
    method.
    The max possible precision is actually that of a float64, ie, about 1e-17.
    
    Examples
    --------
    >>> sqrt(2., 2)
    1.41
    >>> sqrt(2., 10)
    1.4142135623
    >>> sqrt(8.45, 10)
    2.9068883707
    """
    prec = 1/10**n
    P = lambda x: x**2 - a
    dP = lambda x: 2*x
    
    x, dx = a, 1.
    while dx >= prec:
        dx = P(x) / dP(x)
        x -= dx
    return int(x / prec) * prec


# =============================================================================
# Dynamic programming
# =============================================================================

def part_array_permutation(a: list)-> tuple[float, list[tuple[int, int]]]:
    """
    Given an array a, you can switch the values of two adjacent elements and
    each element can be switched at most once. Maximise the sum a[i]*(i+1).
    
    Returns
    -------
    sum : float
        The sum a[i]*(i+1).
    permutations : list[tuple[int, int]]
        Indices of a combination of items that sums to the best weight.
    
    Examples
    --------
    >>> a = [3, 2, 1]
    >>> part_array_permutation(a)
    (11.0, [(0, 1)]) # [2, 3, 1] and [3, 1, 2] sum to 11
    >>> a = [3, -6, 5, -4, -6, 5, 2, 3, 9, -3, 6, -4, 1, 1]
    >>> part_array_permutation(a)
    (167.0, [(0, 1), (2, 3), (5, 6), (8, 9), (10, 11)])
    """
    L = np.empty((len(a),), dtype=float)
    perms = [0]*len(a)
    # init 0
    L[0], perms[0] = a[0], []
    # init 1
    M1, M2 = a[0] + 2*a[1], a[1] + 2*a[0]
    if M1 >= M2:
        L[1], perms[1] = M1, []
    else:
        L[1], perms[1] = M2, [(0, 1)]
    
    for i, val in enumerate(a[2:], start=2):
        M1, M2 = (i+1)*val + L[i-1], i*val + (i+1)*a[i-1] + L[i-2]
        if M1 >= M2:
            L[i], perms[i] = M1, perms[i-1]
        else:
            L[i], perms[i] = M2, perms[i-2] + [(i-1, i)]
    
    return L[-1], perms[-1]


def largest_square(matrix: np.ndarray)-> tuple[tuple[int, int], int]:
    """
    Find the largest square of '1' in a binary matrix.
    
    Returns
    -------
    max_pos : tuple[int, int]
        Index of the position (top left, minimal indices) of the largest square.
    max_size : int
        Size of the largest square.
    
    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(12345)
    >>> M = rng.binomial(1, 0.8, size=(16, 16))
    >>> largest_square(M)
    ((3, 7), 5)
    >>> M = np.array([[0, 1, 0], [0, 1, 1], [1, 1, 1]], dtype=bool)
    >>> largest_square(M)
    ((1, 1), 2)
    """
    if matrix.size == 0:
        return
    
    def find_contiguous(line):
        contig = []
        curr = 0
        for i, l in enumerate(line):
            if l == 1 and curr == 0:
                curr = 1
                start = i
            if l == 0 and curr == 1:
                curr = 0
                contig.append((start, i)) # (start, stop)
        if curr == 1:
            contig.append((start, i+1))
        return contig
    
    def update_squares(curr_sq, i, update, threshold):
        updated_sq = {}
        for j0, j1 in update:
            new_sq = (i, j0), (i+1, j1)
            for (i0_, j0_), (i1_, j1_) in curr_sq.items():
                if j0_ <= j0 and j1_ >= j1:
                    new_sq = None
                upd_j0, upd_j1 = max(j0, j0_), min(j1, j1_)
                if upd_j1 - upd_j0 > threshold:
                    updated_sq[(i0_, upd_j0)] = (i+1, upd_j1)
            if new_sq is not None:
                updated_sq[new_sq[0]] = new_sq[1]
        return updated_sq
    
    
    max_pos = (0, 0) # position of the largest square (min coords)
    max_size = 0 # size of the largest square
    
    curr_squares = {} # top left corner: bottom right corner; (i0, j0): (i1, j1)
    for i in range(matrix.shape[0]):
        contig = find_contiguous(matrix[i])
        contig = [elt for elt in contig if elt[1] > max_size] # filter small elements
        #
        curr_squares = update_squares(curr_squares, i, contig, max_size)
        # remove completed squares, update max square
        for (i0, j0), (i1, j1) in curr_squares.items():
            sz = min(i1 - i0, j1 - j0)
            if sz > max_size:
                max_pos = (i0, j0)
                max_size = sz
    
    return max_pos, max_size

rng = np.random.default_rng(12345)
M = rng.binomial(1, 0.8, size=(16, 16))
M = np.array([[0, 1, 0], [0, 1, 1], [1, 1, 1]], dtype=bool)
print(largest_square(M))
