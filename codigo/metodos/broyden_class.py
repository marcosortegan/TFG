import sys
import time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # carpeta codigo/
from funciones import gradiente_numerico


def broyden(f, x0, phi=0.5, alpha=None, grad=None, tol=1e-5, max_iter=1000):
    """
    Clase de Broyden (6.32), Nocedal y Wright.

    f        : función a minimizar
    x0       : punto de inicio
    phi      : parámetro de la clase (0 = BFGS, 1 = DFP); en [0, 1] mantiene B def. positiva
    alpha    : None -> backtracking (Armijo); un número -> paso fijo
    grad     : gradiente exacto (opcional); si no, numérico
    tol      : se para cuando ||grad f(x_k)|| <= tol
    max_iter : número máximo de iteraciones
    """
    if grad is None:
        grad = lambda x: gradiente_numerico(f, x)

    t0 = time.perf_counter()
    x = np.array(x0, dtype=float)
    n = x.size
    B = np.eye(n)                               # B_0 = identidad
    historial = [x.copy()]
    n_f, n_g = 1, 1
    fx, g = f(x), grad(x)
    k = 0

    while np.linalg.norm(g) > tol and k < max_iter:
        p = np.linalg.solve(B, -g)              # B_k p_k = -grad f(x_k)
        if g @ p >= 0:                          # por seguridad: si no es de descenso
            B = np.eye(n)
            p = -g

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

        s = a * p                               # s_k
        x_nuevo = x + s
        g_nuevo = grad(x_nuevo); n_g += 1
        y = g_nuevo - g                         # y_k

        sy = s @ y
        if sy > 1e-10:                          # condición de curvatura
            Bs = B @ s
            sBs = s @ Bs
            v = y / sy - Bs / sBs               # v_k
            B = (B - np.outer(Bs, Bs) / sBs     # actualización (6.32)
                   + np.outer(y, y) / sy
                   + phi * sBs * np.outer(v, v))

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
    for phi in [0.0, 0.5, 1.0]:
        res = broyden(f, x0=[0.0, 0.0], phi=phi)
        print(f"phi = {phi}:", res["x"], res["iteraciones"], res["convergio"])