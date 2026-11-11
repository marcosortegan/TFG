import sys
import time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # carpeta codigo/
from funciones import gradiente_numerico


def subproblema_rc(g, B, delta):
    """
    Resuelve (6.27):  min  g^T s + 1/2 s^T B s   sujeto a  ||s|| <= delta.
    Solución exacta por descomposición espectral de B (válida aunque B sea indefinida).
    """
    lam, Q = np.linalg.eigh(B)                  # B = Q diag(lam) Q^T, lam ordenados
    gt = Q.T @ g
    s_de = lambda mu: -Q @ (gt / (lam + mu))    # s(mu) = -(B + mu I)^{-1} g

    # 1) B definida positiva y el paso de Newton cabe en la región
    if lam[0] > 0:
        s = s_de(0.0)
        if np.linalg.norm(s) <= delta:
            return s

    # 2) Solución en la frontera: buscar mu >= max(0, -lam_min) con ||s(mu)|| = delta
    mu_min = max(0.0, -lam[0])
    singular = np.abs(lam + mu_min) < 1e-12

    # Caso difícil: g ortogonal a los autovectores del menor autovalor
    if np.any(singular) and np.all(np.abs(gt[singular]) < 1e-12):
        coef = np.zeros_like(gt)
        coef[~singular] = -gt[~singular] / (lam[~singular] + mu_min)
        s = Q @ coef
        tau = np.sqrt(max(delta**2 - s @ s, 0.0))
        return s + tau * Q[:, 0]

    # Bisección en mu (||s(mu)|| decrece con mu)
    mu_lo = mu_min + 1e-14
    mu_hi = mu_min + np.linalg.norm(g) / delta + 1e-14
    for _ in range(200):
        mu = 0.5 * (mu_lo + mu_hi)
        if np.linalg.norm(s_de(mu)) > delta:
            mu_lo = mu
        else:
            mu_hi = mu
    return s_de(mu_hi)


def sr1(f, x0, grad=None, tol=1e-5, max_iter=1000,
        delta0=1.0, eta=1e-4, r=1e-8):
    """
    Algoritmo 6.2 (SR1 con región de confianza), Nocedal y Wright.

    f        : función a minimizar
    x0       : punto de inicio
    grad     : gradiente exacto (opcional); si no, numérico
    tol      : epsilon, se para cuando ||grad f(x_k)|| <= tol
    max_iter : número máximo de iteraciones
    delta0   : radio inicial de la región de confianza (Delta_0)
    eta      : umbral de aceptación del paso, en (0, 1e-3)
    r        : parámetro de la regla de salto (6.26), en (0, 1)
    """
    if grad is None:
        grad = lambda x: gradiente_numerico(f, x)

    t0 = time.perf_counter()
    x = np.array(x0, dtype=float)
    B = np.eye(x.size)                          # B_0
    delta = delta0
    historial = [x.copy()]
    n_f, n_g = 1, 1
    fx, g = f(x), grad(x)
    k = 0
    rechazados = 0

    while np.linalg.norm(g) > tol and k < max_iter:
        # Paso: resolver el subproblema (6.27)
        s = subproblema_rc(g, B, delta)

        # y_k, ared y pred
        x_prueba = x + s
        f_prueba = f(x_prueba); n_f += 1
        g_prueba = grad(x_prueba); n_g += 1
        y = g_prueba - g
        Bs = B @ s
        ared = fx - f_prueba                    # reducción real
        pred = -(g @ s + 0.5 * s @ Bs)          # reducción predicha
        rho = ared / pred if (pred > 0 and np.isfinite(f_prueba)) else -1.0

        # Actualizar el radio Delta
        if rho > 0.75:
            if np.linalg.norm(s) > 0.8 * delta:
                delta = 2 * delta
        elif 0.1 <= rho <= 0.75:
            pass
        else:
            delta = 0.5 * delta

        # Actualización SR1 de B (6.24) si se cumple (6.26), aunque se rechace el paso
        v = y - Bs
        sv = s @ v
        if abs(sv) >= r * np.linalg.norm(s) * np.linalg.norm(v) and sv != 0:
            B = B + np.outer(v, v) / sv

        # Aceptar o rechazar el paso
        if rho > eta:
            x, fx, g = x_prueba, f_prueba, g_prueba
        else:
            rechazados += 1

        historial.append(x.copy())
        k += 1

    return {
        "x": x,
        "iteraciones": k,
        "historial": np.array(historial),
        "evals_f": n_f,
        "evals_grad": n_g,
        "evals_hess": 0,
        "tiempo": time.perf_counter() - t0,
        "convergio": bool(np.linalg.norm(g) <= tol),
        "rechazados": rechazados,
    }


if __name__ == "__main__":
    # Prueba rápida: mínimo en (2, 1)
    f = lambda x: (x[0] - 2)**2 + 3 * (x[1] - 1)**2
    res = sr1(f, x0=[0.0, 0.0])
    print(res["x"], res["iteraciones"], res["convergio"])