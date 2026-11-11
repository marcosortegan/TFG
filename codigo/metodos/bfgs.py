import sys
import time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # carpeta codigo/
from funciones import gradiente_numerico


def bfgs(f, x0, alpha=None, grad=None, tol=1e-5, max_iter=1000):
    """
    Método BFGS (actualización de la inversa de la hessiana).

    f        : función a minimizar, f(x) con x un vector
    x0       : punto de inicio
    alpha    : None -> backtracking (Armijo); un número -> paso fijo
    grad     : gradiente exacto (opcional); si no se da, se aproxima numéricamente
    tol      : se para cuando ||grad f(x_k)|| <= tol
    max_iter : número máximo de iteraciones
    """
    if grad is None:
        grad = lambda x: gradiente_numerico(f, x)

    t0 = time.perf_counter()
    x = np.array(x0, dtype=float)
    n = x.size
    H = np.eye(n)                               # H_0 = identidad
    I = np.eye(n)
    historial = [x.copy()]
    n_f, n_g = 1, 1
    fx, g = f(x), grad(x)
    k = 0

    while np.linalg.norm(g) > tol and k < max_iter:
        p = -H @ g                              # dirección de búsqueda

        if alpha is None:                       # backtracking (Armijo)
            a = 1.0
            while True:
                f_nuevo = f(x + a * p); n_f += 1
                if f_nuevo <= fx + 1e-4 * a * (g @ p) or a < 1e-12:
                    break
                a *= 0.5
        else:                                   # paso fijo
            a = alpha
            f_nuevo = f(x + a * p); n_f += 1

        s = a * p                               # s_k = x_{k+1} - x_k
        x_nuevo = x + s
        g_nuevo = grad(x_nuevo); n_g += 1
        y = g_nuevo - g                         # y_k = grad_{k+1} - grad_k

        sy = s @ y
        if sy > 1e-10:                          # condición de curvatura
            rho = 1.0 / sy
            H = (I - rho * np.outer(s, y)) @ H @ (I - rho * np.outer(y, s)) \
                + rho * np.outer(s, s)

        x, fx, g = x_nuevo, f_nuevo, g_nuevo
        historial.append(x.copy())
        k += 1

        if not np.all(np.isfinite(x)):          # divergencia
            break

    return {
        "x": x,
        "iteraciones": k,
        "historial": np.array(historial),
        "evals_f": n_f,
        "evals_grad": n_g,
        "evals_hess": 0,
        "tiempo": time.perf_counter() - t0,
        "convergio": bool(np.linalg.norm(g) <= tol),
    }


if __name__ == "__main__":
    # Prueba rápida: mínimo en (2, 1)
    f = lambda x: (x[0] - 2)**2 + 3 * (x[1] - 1)**2
    res = bfgs(f, x0=[0.0, 0.0])
    print(res["x"], res["iteraciones"], res["convergio"])