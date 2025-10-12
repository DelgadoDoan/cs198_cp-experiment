import numpy as np
import random

# ideal cp (fully interconnected nodes with each peripheral node attached to each core)
def make_ideal_cp(blocks):
    n = sum(c + p for c, p in blocks)
    A = np.zeros((n, n), dtype=int)
    idx = 0

    for c, p in blocks:
        cores = list(range(idx, idx + c))
        periphery = list(range(idx + c, idx + c + p))

        # core-core connections (clique, no self-loops)
        for i in cores:
            for j in cores:
                if i != j:
                    A[i, j] = 1

        # periphery-core connections (both directions)
        for i in periphery:
            for j in cores:
                A[i, j] = 1
                A[j, i] = 1

        idx += c + p

    return A


def edit_cp(matrix, factor, random_state=42):
    np.random.seed(random_state)
    matrix = matrix.copy()
    n = matrix.shape[0]
    
    # get all upper-triangle indices (excluding diagonal)
    upper_indices = [(i, j) for i in range(n) for j in range(i+1, n)]
    num_edges = len(upper_indices)
    
    # determine number of edits
    num_edits = int(factor * num_edges)
    
    # randomly choose which edges to flip
    edit_indices = random.sample(upper_indices, num_edits)
    
    for i, j in edit_indices:
        matrix[i, j] = 1 - matrix[i, j]
        matrix[j, i] = matrix[i, j]  # maintain symmetry
    
    return matrix


def edit_pp(matrix, block, factor, random_state=42):
    np.random.seed(random_state)
    random.seed(random_state)
    matrix = matrix.copy()
    n = matrix.shape[0]
    
    # Determine p-node indices based on the block structure
    p_indices = []
    start = 0
    for c, p in block:
        # Correct upper-triangle p-node indices
        for i in range(start + c, start + c + p):
            for j in range(i + 1, start + c + p):  # only upper triangle
                p_indices.append((i, j))
        start += c + p
    
    # p-p edges are just the indices we collected
    pp_indices = p_indices
    
    # determine number of edits
    num_edits = int(factor * len(pp_indices))
    
    # randomly choose edges to flip
    if num_edits > 0:
        edit_indices = random.sample(pp_indices, num_edits)  # sample tuples directly
        for i, j in edit_indices:
            matrix[i, j] = 1 - matrix[i, j]
            matrix[j, i] = matrix[i, j]  # maintain symmetry
    
    return matrix


def edit_cc(matrix, block, factor, random_state=42):
    """
    Randomly flip edges between core nodes according to block structure,
    using logic similar to the provided edit_pp function.
    """
    np.random.seed(random_state)
    random.seed(random_state)
    matrix = matrix.copy()
    n = matrix.shape[0]
    
    # Determine core-node indices based on the block structure
    cc_indices = []
    start = 0
    for c, p in block:
        # Upper-triangle core-core edges within the block
        for i in range(start, start + c):
            for j in range(i + 1, start + c):
                cc_indices.append((i, j))
        start += c + p  # move to next block
    
    # Determine number of edits
    num_edits = int(factor * len(cc_indices))
    
    # Randomly choose edges to flip
    if num_edits > 0:
        edit_edges = random.sample(cc_indices, num_edits)  # sample tuples directly
        for i, j in edit_edges:
            matrix[i, j] = 1 - matrix[i, j]
            matrix[j, i] = matrix[i, j]  # maintain symmetry
    
    return matrix