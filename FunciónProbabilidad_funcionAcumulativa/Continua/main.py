import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

"""
OBJETIVO: Análisis de Variables Aleatorias Continuas
1. Distribución Uniforme (0, 20)
2. Distribución Exponencial (lambda = 0.5)
"""

# --- DEFINICIÓN DE FUNCIONES DE DENSIDAD (PDF) ---

def pdf_uniforme(x):
    return 1/20 if 0 <= x <= 20 else 0

def pdf_exponencial(x, lambd=0.5):
    return lambd * np.exp(-lambd * x) if x >= 0 else 0

# --- CÁLCULO DE ESTADÍSTICAS MEDIANTE INTEGRACIÓN ---

def calcular_estadisticas_continuas(pdf_func, limite_inf, limite_sup):
    # Esperanza: Integral de x * f(x)
    esperanza, _ = quad(lambda x: x * pdf_func(x), limite_inf, limite_sup)
    
    # Momento de segundo orden: Integral de x^2 * f(x)
    segundo_momento, _ = quad(lambda x: (x**2) * pdf_func(x), limite_inf, limite_sup)
    
    varianza = segundo_momento - (esperanza**2)
    return esperanza, varianza

# --- VISUALIZACIÓN ---

def graficar_continua(x_range, pdf_values, cdf_values, title, filename):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Graficar PDF
    ax1.plot(x_range, pdf_values, color='blue', lw=2, label='f(x)')
    ax1.fill_between(x_range, pdf_values, color='blue', alpha=0.2)
    ax1.set_title(f'PDF: {title}')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Graficar CDF
    ax2.plot(x_range, cdf_values, color='green', lw=2, label='F(x)')
    ax2.set_title(f'CDF: {title}')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.savefig(filename)
    print(f"Gráfica guardada: {filename}")

def main():
    # 1. ANALIZAR UNIFORME
    x_uni = np.linspace(-5, 25, 500)
    pdf_uni = [pdf_uniforme(val) for val in x_uni]
    cdf_uni = [quad(pdf_uniforme, -np.inf, val)[0] for val in x_uni]
    
    e_uni, v_uni = calcular_estadisticas_continuas(pdf_uniforme, 0, 20)
    print(f"\nUNIFORME (0,20): Esperanza = {e_uni:.2f}, Varianza = {v_uni:.2f}")
    graficar_continua(x_uni, pdf_uni, cdf_uni, "Uniforme (Espera Bus)", "continuo_uniforme.png")

    # 2. ANALIZAR EXPONENCIAL
    x_exp = np.linspace(0, 10, 500)
    pdf_exp = [pdf_exponencial(val) for val in x_exp]
    cdf_exp = [1 - np.exp(-0.5 * val) for val in x_exp] # CDF analítica
    
    e_exp, v_exp = calcular_estadisticas_continuas(pdf_exponencial, 0, np.inf)
    print(f"EXPONENCIAL (L=0.5): Esperanza = {e_exp:.2f}, Varianza = {v_exp:.2f}")
    graficar_continua(x_exp, pdf_exp, cdf_exp, "Exponencial (Vida Chip)", "continuo_exponencial.png")

    # 3. CONJUNTA (Visualización 3D conceptual)
    from matplotlib import cm
    X = np.linspace(0, 20, 30)
    Y = np.linspace(0, 20, 30)
    X, Y = np.meshgrid(X, Y)
    Z = np.full(X.shape, 1/400) # f(x,y) = 1/20 * 1/20

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
    ax.set_title("PDF Conjunta: Dos esperas uniformes")
    ax.set_zlim(0, 0.01)
    plt.savefig("conjunta_continua.png")
    
    plt.show()

if __name__ == "__main__":
    main()