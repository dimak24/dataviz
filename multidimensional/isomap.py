import numpy as np

def _distance(a, b):
    assert a.shape == b.shape
    assert len(a.shape) == 1
    return np.sqrt(np.sum((a - b) ** 2))


def _get_neighbours(data, k):
    n = len(data)
    d = np.zeros(shape=(n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d[i][j] = d[j][i] = _distance(data[i], data[j])
    not_nearest = d.argsort()[:, k+1:]
    for i, not_nearest_ in enumerate(not_nearest):
        d[i, not_nearest_] = np.inf
    return d
    
def floyd_warshall(d):
    n = len(d)
    assert d.shape == (n, n)
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if d[i, k] < np.inf and d[k, j] < np.inf:
                    d[i, j] = min(d[i, j], d[i, k] + d[k, j])
    return d

def _md_scaling(data, target):  
    n = len(data)
    D = data ** 2
    # centering matrix
    C = np.eye(n) - np.ones(shape=data.shape) / n
    # apply double centering
    B = -.5 * C @ D @ C
    eigval, eigvec = np.linalg.eig(B)
    # find @target largest eigen values
    indices = np.argsort(eigval)[::-1]   
    eigval = eigval[indices][:target]
    eigvec = eigvec[:, indices][:, :target]    
    E = eigvec
    L = np.diag(eigval)
    return np.real(E @ np.sqrt(L))

def isomap(data, n_neighbors=5, n_components=2):
    d = _get_neighbours(data, n_neighbors)
    d = floyd_warshall(d)
    return _md_scaling(d, n_components)
