import numpy as np

def gradiente_numerico(f, x):
    """Diferencias centrales. Cuesta 2d evaluaciones de f."""
    h = np.cbrt(np.finfo(float).eps)
    x = np.asarray(x, dtype=float)
    g = np.zeros_like(x)
    for i in range(x.size):
        e = np.zeros_like(x); e[i] = h
        g[i] = (f(x + e) - f(x - e)) / (2 * h)
    return g