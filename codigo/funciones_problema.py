import numpy as np


N=2
"""Ackley's function"""
def f1(x):
    x = np.asarray(x, dtype=float)
    n = x.shape[0]
    suma_cuad = np.sum(x**2, axis=0)
    suma_cos = np.sum(np.cos(2 * np.pi * x), axis=0)
    return (-20 * np.exp(-0.2 * np.sqrt(suma_cuad / n))
            - np.exp(suma_cos / n) + 20 + np.e)

def grad_f1(x):
    x = np.asarray(x, dtype=float)
    n = x.size
    r = np.sqrt(np.sum(x**2) / n)
    c = np.sum(np.cos(2 * np.pi * x)) / n
    termino1 = 4 * np.exp(-0.2 * r) * x / (n * r) if r > 0 else np.zeros(n)
    termino2 = (2 * np.pi / n) * np.exp(c) * np.sin(2 * np.pi * x)
    return termino1 + termino2



"""Damavandi"""

def f2(x):
    x1, x2 = x[0], x[1]
    u = np.sinc(x1 - 2) * np.sinc(x2 - 2)        # sinc(t) = sin(pi t) / (pi t)
    i = 1 - np.abs(u)**5
    d = 2 + (x1 - 7)**2 + 2 * (x2 - 7)**2
    return i * d


def dsinc(t):
    """Derivada de sinc(t) = sin(pi t)/(pi t); vale 0 en t = 0."""
    t = np.asarray(t, dtype=float)
    pequeno = np.abs(t) < 1e-8
    t_seguro = np.where(pequeno, 1.0, t)
    return np.where(pequeno, 0.0, (np.cos(np.pi * t) - np.sinc(t)) / t_seguro)


def grad_f2(x):
    x1, x2 = x[0], x[1]
    s1, s2 = np.sinc(x1 - 2), np.sinc(x2 - 2)
    u = s1 * s2
    A = 1 - np.abs(u)**5
    Q = 2 + (x1 - 7)**2 + 2 * (x2 - 7)**2
    dA = -5 * u * np.abs(u)**3
    df_dx1 = dA * dsinc(x1 - 2) * s2 * Q + A * 2 * (x1 - 7)
    df_dx2 = dA * s1 * dsinc(x2 - 2) * Q + A * 4 * (x2 - 7)
    return np.array([df_dx1, df_dx2])

"""HIPERELIPSOIDE"""

def f3(x):
    return sum((k + 1) * x[k]**2 for k in range(len(x)))


def grad_f3(x):
    x = np.asarray(x, dtype=float)
    i = np.arange(1, x.size + 1)
    return 2 * i * x

"""EASOM"""

def f4(x):
    x1, x2 = x[0], x[1]
    return -np.cos(x1) * np.cos(x2) * np.exp(-((x1 - np.pi)**2 + (x2 - np.pi)**2))


def grad_f4(x):
    x1, x2 = x[0], x[1]
    E = np.exp(-((x1 - np.pi)**2 + (x2 - np.pi)**2))
    c = np.cos(x1) * np.cos(x2)
    return np.array([
        E * (np.sin(x1) * np.cos(x2) + 2 * (x1 - np.pi) * c),
        E * (np.cos(x1) * np.sin(x2) + 2 * (x2 - np.pi) * c),
    ])



PROBLEMAS = {
    # Ackley: mínimo global en 0
    "f1": {"f": f1, "grad": grad_f1, "x_opt": np.zeros(N),
           "puntos": [np.full(N, 25.0),     # lejos
                      np.full(N, 0.3),      # cerca
                      np.full(N, 0.5)]},    # conflictivo: cresta entre cuencas

    # Damavandi: mínimo global en (2, 2)
    "f2": {"f": f2, "grad": grad_f2, "x_opt": np.array([2.0, 2.0]),
           "puntos": [[12.0, 12.0],     # lejos
                      [2.1, 1.9],       # cerca: dentro del pozo
                      [2.3, 2.3]]},     # conflictivo: en el borde del pozo

    # Hiperelipsoide: mínimo en 0
        "f3": {"f": f3, "grad": grad_f3, "x_opt": np.zeros(N),
           "puntos": [np.full(N, 5.0),       # lejos
                      np.full(N, 0.1),       # cerca
                      np.full(N, 1000.0)]},  # conflictivo: escala muy grande
    "f4": {"f": f4, "grad": grad_f4, "x_opt": np.array([np.pi, np.pi]),
           "puntos": [[20.0, 20.0],          # lejos: meseta
                      [2.5, 3.5],            # cerca
                      [1.0, 1.0]]},          # conflictivo: borde de la meseta
}