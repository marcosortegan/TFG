"""
Análisis de f3 (hiperelipsoide) con BFGS, SR1 y Broyden.
Ejecuta todos los casos (tabla completa) y dibuja los seleccionados
agrupados de 3 en 3.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from funciones_problema import PROBLEMAS, N
from metodos.bfgs import bfgs
from metodos.sr1 import sr1
from metodos.broyden_class import broyden

# ======================= CONFIGURACIÓN =======================
NOMBRE = "f3"
TITULO = "el hiperelipsoide"
MAX_ITER = 1000
TOL = 1e-5
DOMINIO = (-5.12, 5.12)                       # dominio en cada coordenada (3D)
ZOOM = None                                   # sin zoom: es un paraboloide
ETIQUETAS_PUNTOS = ["Lejos", "Cerca", "Conflictivo"]

VARIANTES = {
    "BFGS":    [None, 0.1, 0.5],              # alpha (None = backtracking)
    "SR1":     [0.1, 1.0, 5.0],               # delta0 (radio inicial)
    "Broyden": [None, 0.1, 0.5],              # alpha (phi = 0.5)
}

# Gráficas agrupadas de 3 en 3: cada grupo es una figura con 3 paneles
# Cada caso es (punto, método, valor del parámetro)
GRUPOS = {
    "convergencia": [                       # los tres métodos desde el punto cercano
        ("Cerca",       "BFGS",    None),
        ("Cerca",       "SR1",     1.0),    # termina en n + 1 = 3 pasos (cuadrática)
        ("Cerca",       "Broyden", None),
    ],
    "paso": [                               # efecto de la longitud de paso en BFGS
        ("Lejos",       "BFGS",    None),   # backtracking: 7 iteraciones
        ("Lejos",       "BFGS",    0.1),    # paso fijo pequeño: 136 iteraciones
        ("Lejos",       "BFGS",    0.5),    # paso fijo medio: 22 iteraciones
    ],
    "escala": [                             # empezando muy lejos, en (1000, 1000)
        ("Conflictivo", "BFGS",    None),   # el primer paso se pasa de largo
        ("Conflictivo", "SR1",     0.1),    # la región de confianza crece desde 0.1
        ("Conflictivo", "Broyden", 0.1),    # paso fijo pequeño: 186 iteraciones
    ],
}
# =============================================================

if N != 2:
    raise SystemExit("Pon N = 2 en funciones_problema.py para poder dibujar las trayectorias.")

RAIZ = Path(__file__).resolve().parent.parent
CARPETA_FIG = RAIZ / "figuras" / NOMBRE
CARPETA_TAB = RAIZ / "tablas"
CARPETA_FIG.mkdir(parents=True, exist_ok=True)
CARPETA_TAB.mkdir(parents=True, exist_ok=True)

p = PROBLEMAS[NOMBRE]
f, grad, x_opt = p["f"], p["grad"], p["x_opt"]
f_opt = f(x_opt)


# ------------------------- Métodos ---------------------------
def ejecutar(metodo, x0, v):
    if metodo == "BFGS":
        return bfgs(f, x0, alpha=v, grad=grad, tol=TOL, max_iter=MAX_ITER)
    if metodo == "SR1":
        return sr1(f, x0, grad=grad, tol=TOL, max_iter=MAX_ITER, delta0=v)
    return broyden(f, x0, alpha=v, grad=grad, tol=TOL, max_iter=MAX_ITER)


def nombre_variante(metodo, v):
    """Devuelve (texto LaTeX, texto plano, etiqueta para el archivo)."""
    if metodo == "SR1":
        return rf"$\Delta_0 = {v:g}$", f"Delta0 = {v:g}", f"d{v:g}".replace(".", "p")
    if v is None:
        return "backtracking", "backtracking", "bt"
    return rf"$\alpha = {v:g}$", f"alpha = {v:g}", f"a{v:g}".replace(".", "p")


# ------------------------- Gráficas --------------------------
def grafica_3d(intervalo, archivo, puntos):
    t = np.linspace(*intervalo, puntos)
    X1, X2 = np.meshgrid(t, t)
    Z = f(np.array([X1, X2]))
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(projection="3d")
    ax.plot_surface(X1, X2, Z, cmap="viridis", linewidth=0, antialiased=True,
                    rcount=puntos, ccount=puntos)
    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$"); ax.set_zlabel("$f(x)$")
    fig.savefig(CARPETA_FIG / archivo, dpi=300, bbox_inches="tight")
    plt.close(fig)


def dibujar_trayectoria(fig, ax, tray, titulo):
    tray = tray[np.all(np.isfinite(tray), axis=1)]
    m = 1.0
    x_min, x_max = tray[:, 0].min() - m, tray[:, 0].max() + m
    y_min, y_max = tray[:, 1].min() - m, tray[:, 1].max() + m
    X1, X2 = np.meshgrid(np.linspace(x_min, x_max, 400), np.linspace(y_min, y_max, 400))
    Z = f(np.array([X1, X2]))

    if np.ptp(Z) < 1e-12:                    # función plana en toda la ventana
        ax.set_facecolor(plt.cm.viridis(1.0))
        ax.text(0.5, 0.06, r"$f(x) \approx 0$ en toda la zona (meseta)",
                transform=ax.transAxes, ha="center", fontsize=9)
    else:
        cs = ax.contourf(X1, X2, Z, 40, cmap="viridis")
        fig.colorbar(cs, ax=ax, fraction=0.046, pad=0.04, label="$f(x)$")
    ax.plot(tray[:, 0], tray[:, 1], "w.-", lw=0.8, ms=3, label="trayectoria")
    ax.plot(*tray[0], "o", color="white", mec="black", ms=8, label="inicio")
    ax.plot(*tray[-1], "*", color="red", mec="black", ms=13, label="final")
    if x_min <= x_opt[0] <= x_max and y_min <= x_opt[1] <= y_max:
        ax.plot(*x_opt, "X", color="orange", mec="black", ms=9, label="mínimo global")
    ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
    ax.set_title(titulo, fontsize=10)


def grafica_grupo(nombre_grupo, casos):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for letra, ax, caso in zip("abc", axes, casos):
        punto, metodo, v = caso
        v_latex, _, _ = nombre_variante(metodo, v)
        titulo = f"({letra}) {metodo}, {v_latex}, punto {punto.lower()}"
        dibujar_trayectoria(fig, ax, historiales[caso], titulo)

    # Leyenda común con todos los marcadores que aparezcan
    etiquetas = {}
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            etiquetas.setdefault(l, h)
    fig.legend(etiquetas.values(), etiquetas.keys(), loc="lower center",
               ncol=len(etiquetas), facecolor="lightgray", bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    fig.savefig(CARPETA_FIG / f"{NOMBRE}_{nombre_grupo}.pdf", bbox_inches="tight")
    plt.close(fig)


# -------------------------- Tabla ----------------------------
def sci(v):
    if not np.isfinite(v):
        return "---"
    if v == 0:
        return "$0$"
    m, e = f"{v:.2e}".split("e")
    return rf"${m}\cdot 10^{{{int(e)}}}$"


def vec(x):
    return "$(" + ", ".join(f"{xi:.4f}" for xi in x) + ")$"


def tabla_latex(filas):
    lineas = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lllrrrrlllc}",
        r"\toprule",
        r"Punto inicial & Método & Variante & Iter. & Ev. $f$ & Ev. $\nabla f$ & "
        r"Tiempo (ms) & $x_{\mathrm{final}}$ & $\|x - x^*\|$ & $|f - f^*|$ & Conv. \\",
        r"\midrule",
    ]
    for i, fila in enumerate(filas):
        if i > 0 and i % 9 == 0:
            lineas.append(r"\midrule")
        lineas.append(" & ".join(fila) + r" \\")
    lineas += [
        r"\bottomrule",
        r"\end{tabular}}",
        rf"\caption{{Resultados de BFGS, SR1 y Broyden sobre {TITULO} ($n = 2$).}}",
        rf"\label{{tab:{NOMBRE}}}",
        r"\end{table}",
    ]
    return "\n".join(lineas)


# -------------------------- Main -----------------------------
grafica_3d(DOMINIO, f"{NOMBRE}_3d.png", 300)
if ZOOM is not None:
    grafica_3d(ZOOM, f"{NOMBRE}_3d_zoom.png", 200)

filas = []
historiales = {}
for x0, etq_p in zip(p["puntos"], ETIQUETAS_PUNTOS):
    x0 = np.array(x0, dtype=float)
    print(f"\n=== Punto {etq_p.lower()}: {x0} ===")
    primera = True

    for metodo, valores in VARIANTES.items():
        for v in valores:
            res = ejecutar(metodo, x0, v)
            v_latex, v_plano, tag = nombre_variante(metodo, v)

            historiales[(etq_p, metodo, v)] = res["historial"]

            err_x = np.linalg.norm(res["x"] - x_opt)
            err_f = abs(f(res["x"]) - f_opt)
            conv = "Sí" if res["convergio"] else "No"

            filas.append([
                f"{etq_p} {vec(x0)}" if primera else "",
                metodo, v_latex,
                str(res["iteraciones"]), str(res["evals_f"]), str(res["evals_grad"]),
                f"{1000 * res['tiempo']:.2f}", vec(res["x"]),
                sci(err_x), sci(err_f), conv,
            ])
            primera = False

            print(f"  {metodo:8s} {v_plano:15s} iter = {res['iteraciones']:5d}   "
                  f"||x - x*|| = {err_x:.2e}   conv = {conv}")

for nombre_grupo, casos in GRUPOS.items():
    grafica_grupo(nombre_grupo, casos)

tabla = tabla_latex(filas)
(CARPETA_TAB / f"{NOMBRE}.tex").write_text(tabla, encoding="utf-8")
print("\n" + tabla)
print(f"\nGráficas en: {CARPETA_FIG}")
print(f"Tabla en:    {CARPETA_TAB / (NOMBRE + '.tex')}")