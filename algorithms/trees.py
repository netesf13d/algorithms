# -*- coding: utf-8 -*-
"""
Tree traversal algorithms.

* Binary tree traversal
    - <breadth_first_search> : Breadth-first search in a binary tree.
    - <depth_first_search> : Depth-first search in a binary tree.
"""

# =============================================================================
# 
# =============================================================================

def breadth_first_search(bin_tree: list):
    """
    Breadth-first search on binary trees.
    
    The binary tree structure used is a list where
    - bin_tree[0] is the root
    - for a node bin_tree[i] the childs are
        left child: bin_tree[2*i + 1]
        right child: bin_tree[2*i + 2]
    
    With this structure, breadth-first search is almost trivial.
    """
    # iparent = lambda i: (i - 1) // 2
    ichildL = lambda i: 2*i + 1
    ichildR = lambda i: 2*i + 2
    
    bfs = [bin_tree[0]]
    i = 0
    while True:
        try:
            bfs.append(bin_tree[ichildL(i)])
            bfs.append(bin_tree[ichildR(i)])
            i += 1
        except IndexError:
            break
    
    return bfs
    


def depth_first_search(bin_tree: list, order: str):
    """
    Depth-first search in binary trees.
    See https://en.wikipedia.org/wiki/Tree_traversal
    
    bin_tree : list, binary tree structure
        - bin_tree[0] is the root
        - for a node bin_tree[i] the childs are
            left child: bin_tree[2*i + 1]
            right child: bin_tree[2*i + 2]
    order : str {'pre', 'in', 'post'}
        Tree traversal method: pre-order, in-order, post-order
    
    Examples
    --------
    >>> a = ['F', 'B', 'G', 'A', 'D', 'I', 'C', 'E', 'H']
    >>> depth_first_search(a, "pre")
    ['F', 'B', 'A', 'E', 'H', 'D', 'G', 'I', 'C']
    >>> depth_first_search(a, "in")
    ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
    >>> depth_first_search(a, "post")
    ['E', 'A', 'H', 'B', 'D', 'F', 'I', 'G', 'C']
    """
    iparent = lambda i: (i - 1) // 2
    ichildL = lambda i: 2*i + 1
    ichildR = lambda i: 2*i + 2
    
    visit = []
    
    if order == 'pre':
        stack = [0]
        while stack:
            i = stack.pop(-1)
            visit.append(i) # 1 visit node
            if (iR := ichildR(i)) < len(bin_tree):
                stack.append(iR) # 3 visit right
            if (iL := ichildL(i)) < len(bin_tree):
                stack.append(iL) # 2 visit left
    
    if order == 'in':
        i = 0
        while i >= 0:
            if (iL := ichildL(i)) < len(bin_tree) and iL not in visit:
                i = iL # 1 visit left
                continue
            if i not in visit:
                visit.append(i) # 2 visit node
            if (iR := ichildR(i)) < len(bin_tree) and iR not in visit:
                i = iR # 3 visit right
                continue
            i = iparent(i)
    
    if order == 'post':
        i = 0
        while i >= 0:
            if (iL := ichildL(i)) < len(bin_tree) and iL not in visit:
                i = iL # 1 visit left
                continue
            if (iR := ichildR(i)) < len(bin_tree) and iR not in visit:
                i = iR # 2 visit right
                continue
            visit.append(i) # 3 visit node
            i = iparent(i)

    return [bin_tree[i] for i in visit]
