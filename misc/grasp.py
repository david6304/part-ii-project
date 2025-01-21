import itertools
import numpy as np
from scipy.stats import pearsonr
from math import log, pi

########################################
# Utility Functions
########################################

def cov_matrix(data):
    # Compute covariance matrix of data.
    return np.cov(data, rowvar=False)

def partial_corr(i, j, cond_set, S):
    # Compute partial correlation using Schur complement if cond_set is not empty
    # i,j are indices, cond_set is a list of indices, S is covariance matrix
    if len(cond_set) == 0:
        # Direct correlation
        return S[i,j] / np.sqrt(S[i,i]*S[j,j])
    else:
        # Compute partial correlation via block inverse
        indices = [i, j] + cond_set
        subS = S[np.ix_(indices, indices)]
        # Invert submatrix
        inv_subS = np.linalg.inv(subS)
        # Partial correlation formula:
        # pcor(i,j|cond_set) = -inv_subS[0,1]/sqrt(inv_subS[0,0]*inv_subS[1,1])
        return -inv_subS[0,1]/np.sqrt(inv_subS[0,0]*inv_subS[1,1])

def ci_test_gaussian(i, j, cond_set, data):
    # Test conditional independence via partial correlation
    # For simplicity, just check magnitude of partial correlation
    # In practice, use a significance test or a threshold
    S = cov_matrix(data)
    pcor = partial_corr(i, j, cond_set, S)
    # Threshold could be chosen. Here we just treat near-zero pcor as independence:
    # This is a simplification. In practice, you'd apply a statistical test.
    return abs(pcor) < 1e-3  # threshold

def gaussian_log_likelihood(data, X, PaX):
    # Compute log-likelihood for variable X given parents PaX using linear Gaussian model
    # Fit a linear regression of X ~ PaX and compute residual variance
    y = data[:, X]
    if len(PaX) == 0:
        # Just model X as N(mean, var)
        mu = np.mean(y)
        resid = y - mu
    else:
        Xmat = data[:, PaX]
        # Fit linear regression via OLS
        # beta = (X'X)^(-1) X'y
        beta = np.linalg.lstsq(Xmat, y, rcond=None)[0]
        pred = Xmat @ beta
        resid = y - pred
    n = len(y)
    var = np.var(resid, ddof=len(PaX)+1)
    # Gaussian log-likelihood for residual
    ll = -0.5*n*(log(2*pi*var) + 1)
    return ll

def bic_score(data, dag):
    # Decomposable score: sum over variables X of BIC(X|PaX)
    n, m = data.shape
    score = 0
    for X in range(m):
        PaX = dag['parents'][X]
        ll = gaussian_log_likelihood(data, X, PaX)
        # Number of parameters = len(PaX)+1 (linear Gaussian)
        k = len(PaX)+1
        penalty = k*log(n)/2
        score += ll - penalty
    return score

########################################
# Grow-Shrink Markov Boundary
########################################

def grow(data, X, Z):
    # Grow phase: add variables that increase BIC until no improvement
    current_set = []
    current_score = gaussian_log_likelihood(data, X, current_set)
    improved = True
    while improved:
        improved = False
        best_score = current_score
        best_var = None
        for Y in Z:
            if Y not in current_set:
                test_set = current_set + [Y]
                new_score = gaussian_log_likelihood(data, X, test_set)
                if new_score > best_score:
                    best_score = new_score
                    best_var = Y
        if best_var is not None:
            current_set.append(best_var)
            current_score = best_score
            improved = True
    return current_set

def shrink(data, X, Z):
    # Shrink phase: remove variables that improve BIC if removed
    current_set = Z[:]
    current_score = gaussian_log_likelihood(data, X, current_set)
    improved = True
    while improved and len(current_set) > 0:
        improved = False
        best_score = current_score
        best_var = None
        for Y in current_set:
            test_set = [v for v in current_set if v != Y]
            new_score = gaussian_log_likelihood(data, X, test_set)
            if new_score > best_score:
                best_score = new_score
                best_var = Y
        if best_var is not None:
            current_set.remove(best_var)
            current_score = best_score
            improved = True
    return current_set

def markov_boundary(data, X, Z):
    # For compositional graphoid, MB = shrink(X, grow(X,Z))
    # Assuming Gaussian case here
    gr = grow(data, X, Z)
    sh = shrink(data, X, gr)
    return sh

########################################
# Induce DAG from Permutation using (VP)
########################################

def dag_from_permutation(data, permutation):
    m = len(permutation)
    parents = [[] for _ in range(m)]
    # For each variable in order, find Markov boundary relative to predecessors
    for idx, X in enumerate(permutation):
        pred = permutation[:idx]
        # MB(X, Pre(X,π))
        MBX = markov_boundary(data, X, pred)
        parents[X] = MBX
    return {'parents': parents}

########################################
# Tuck Operation
########################################

def ancestors(dag, node):
    # Return set of ancestors of 'node' in dag
    an = set()
    frontier = list(dag['parents'][node])
    while frontier:
        cur = frontier.pop()
        if cur not in an:
            an.add(cur)
            frontier.extend(dag['parents'][cur])
    return an

def edge_set(dag):
    E = []
    m = len(dag['parents'])
    for k in range(m):
        for p in dag['parents'][k]:
            E.append((p,k))
    return E

def covered_edge(dag, e):
    # e = (j,k)
    j,k = e
    # j->k is covered if Pa(j)=Pa(k)\{j}
    Pa_j = set(dag['parents'][j])
    Pa_k = set(dag['parents'][k])
    return Pa_j == (Pa_k - {j})

def is_singular_edge(dag, e):
    # e = (j,k)
    # edge is singular if no other unidirectional path from j to k except j->k
    # Check if after removing j->k, k is still reachable from j
    j,k = e
    # Temporarily remove edge j->k and check reachability
    parents_copy = [p[:] for p in dag['parents']]
    parents_copy[k].remove(j)
    new_dag = {'parents': parents_copy}
    # Check reachability of k from j in new_dag
    return k not in descendants(new_dag, j)

def descendants(dag, node):
    # Return set of descendants
    de = set()
    m = len(dag['parents'])
    children = [[] for _ in range(m)]
    for X in range(m):
        for P in dag['parents'][X]:
            children[P].append(X)
    frontier = children[node][:]
    while frontier:
        cur = frontier.pop()
        if cur not in de:
            de.add(cur)
            frontier.extend(children[cur])
    return de

def tuck(permutation, dag, j, k):
    # We have π = ... j ... k ...
    # Find δj∼k and split into ancestors_of_k and others
    pos_j = permutation.index(j)
    pos_k = permutation.index(k)
    if pos_j > pos_k:
        # If not in correct order, no tuck
        return permutation
    middle = permutation[pos_j+1:pos_k]
    an_k = ancestors(dag, k)
    gamma = [x for x in middle if x in an_k]
    gamma_c = [x for x in middle if x not in an_k]
    # If edge j->k in dag, tuck: hδ<j, j, δj∼k, k, δ>k i -> hδ<j, γ, k, j, γc, δ>k i
    if (j,k) in edge_set(dag):
        left = permutation[:pos_j]
        right = permutation[pos_k+1:]
        return left + gamma + [k, j] + gamma_c + right
    else:
        # no change
        return permutation

########################################
# DFS procedure (Algorithm 1)
########################################

def score_permutation(data, permutation):
    dag = dag_from_permutation(data, permutation)
    return bic_score(data, dag), dag

def dfs(data, permutation, d, dcur, t):
    # t in {0,1,2} determines which edges to consider
    cur_score, cur_dag = score_permutation(data, permutation)
    E = edge_set(cur_dag)
    # Edge type selection
    if t == 0:
        edges_to_consider = [e for e in E if covered_edge(cur_dag,e)]
    elif t == 1:
        edges_to_consider = [e for e in E if is_singular_edge(cur_dag, e)]
    else:
        edges_to_consider = E[:]
    for (j,k) in edges_to_consider:
        new_perm = tuck(permutation, cur_dag, j, k)
        if new_perm == permutation:
            continue
        new_score, new_dag = score_permutation(data, new_perm)
        if abs(new_score - cur_score)<1e-12:
            # Same score, try deeper DFS if allowed
            if dcur < d:
                deeper_perm = dfs(data, new_perm, d, dcur+1, t)
                deeper_score, _ = score_permutation(data, deeper_perm)
                if deeper_score > cur_score:
                    return deeper_perm
        elif new_score > cur_score:
            return new_perm
    return permutation

########################################
# GRaSP (Algorithm 2)
########################################

def grasp(data, permutation, d, t):
    # If t != 0, run previous tier first
    if t != 0:
        permutation = grasp(data, permutation, d, t-1)
    improved = True
    iteration = 0
    while improved:
        iteration += 1
        print(f"Iteration {iteration}, tier {t}")
        cur_score = score_permutation(data, permutation)
        new_perm = dfs(data, permutation, d, 1, t)
        new_score = score_permutation(data, new_perm)
        if new_score > cur_score:
            permutation = new_perm
        else:
            improved = False
    return permutation

########################################
# Example Usage
########################################

if __name__ == "__main__":
    # Generate synthetic linear Gaussian data
    np.random.seed(42)
    n = 10000
    m = 5
    # True model: X0->X2, X1->X2, X2->X3, X2->X4
    # X0,X1 exogenous normal(0,1)
    # X2 = X0+X1+noise
    # X3 = X2+noise
    # X4 = X2+noise
    X0 = np.random.randn(n)
    X1 = np.random.randn(n)
    X2 = X0 + X1 + 0.1*np.random.randn(n)
    X3 = X2 + 0.1*np.random.randn(n)
    X4 = X2 + 0.1*np.random.randn(n)
    data = np.column_stack([X0,X1,X2,X3,X4])

    permutation = list(range(m))
    np.random.shuffle(permutation)
    print("Initial permutation:", permutation)
    # Run GRaSP2 (t=2) with depth d=3, for example
    final_perm = grasp(data, permutation, d=3, t=2)
    print("Final permutation:", final_perm)
    final_score, final_dag = score_permutation(data, final_perm)
    print("Final DAG parents:", final_dag['parents'])
    print("True DAG parents:", [[], [], [0, 1], [2], [2]])
    print("Final BIC score:", final_score)
