import numpy as np
import matplotlib.pyplot as plt
from fractions import Fraction
import seaborn as sns

"""
OBJETIVO: Análisis completo de PMF, CDF, Esperanza, Varianza y Conjunta.
"""

def calculate_stats(outcomes, probs):
    """Calcula esperanza y varianza de la distribución."""
    esperanza = sum(o * p for o, p in zip(outcomes, probs))
    esperanza_sq = sum((o**2) * p for o, p in zip(outcomes, probs))
    varianza = esperanza_sq - (esperanza**2)
    return esperanza, varianza

def process_probability_data(pmf_dict):
    """Ordena los datos y calcula la CDF."""
    outcomes = sorted(pmf_dict.keys())
    probs = [pmf_dict[o] for o in outcomes]
    cdf = np.cumsum(probs)
    return outcomes, probs, cdf

def print_table(name, outcomes, pmf, cdf, e, var):
    """Imprime tablas de PMF, CDF y estadísticas en consola."""
    print(f"\n--- Tablas para: {name} ---")
    print(f"ESTADÍSTICAS: E[X] = {e:.2f}, Var(X) = {var:.4f}")
    print(f"{'Valor':<8} | {'PMF (Frac)':<12} | {'PMF (Dec)':<10} | {'CDF (Dec)':<10}")
    print("-" * 55)
    for i in range(len(outcomes)):
        frac = Fraction(pmf[i]).limit_denominator(36)
        print(f"{outcomes[i]:<8} | {str(frac):<12} | {pmf[i]:<10.4f} | {cdf[i]:<10.4f}")

def plot_distributions(name, outcomes, pmf, cdf, e, filename):
    """Genera y guarda las gráficas de PMF y CDF con línea de media."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Análisis de Distribución: {name}", fontsize=16)

    # Gráfica PMF
    bars = ax1.bar(outcomes, pmf, color='skyblue', edgecolor='navy', alpha=0.7)
    ax1.axvline(e, color='red', linestyle='--', label=f'E[X]={e:.2f}')
    ax1.set_title("PMF con Valor Medio")
    ax1.set_xlabel("Resultado (X)")
    ax1.set_ylabel("P(X = x)")
    ax1.set_xticks(outcomes)
    ax1.legend()

    # Gráfica CDF
    x_step = np.array(outcomes)
    y_step = np.array(cdf)
    ax2.step(x_step, y_step, where='post', color='orange', label='F(x)', linewidth=2)
    ax2.scatter(x_step, y_step, color='red', zorder=3)
    ax2.set_title("Función de Distribución Acumulada (CDF)")
    ax2.set_ylim(0, 1.1)
    ax2.grid(linestyle='--', alpha=0.6)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(filename)

def plot_joint_distribution():
    """Genera un mapa de calor para la probabilidad conjunta de dos dados."""
    conjunta = np.full((6, 6), 1/36)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conjunta, annot=True, fmt=".3f", cmap="Blues", 
                xticklabels=range(1, 7), yticklabels=range(1, 7))
    plt.title("Probabilidad Conjunta P(Dado 1, Dado 2)")
    plt.xlabel("Resultado Dado 2")
    plt.ylabel("Resultado Dado 1")
    plt.savefig("probabilidad_conjunta_dados.png")

def main():
    # 1. Moneda
    pmf_moneda = {0: 0.5, 1: 0.5}
    # 2. Dado
    pmf_dado = {i: 1/6 for i in range(1, 7)}
    # 3. Suma de 2 dados
    conteo_suma = {}
    for d1 in range(1, 7):
        for d2 in range(1, 7):
            s = d1 + d2
            conteo_suma[s] = conteo_suma.get(s, 0) + 1
    pmf_2dados = {s: c/36 for s, c in conteo_suma.items()}

    datasets = [
        ("Moneda", pmf_moneda, "analisis_moneda.png"),
        ("Un Dado", pmf_dado, "analisis_dado.png"),
        ("Suma de Dos Dados", pmf_2dados, "analisis_suma_dados.png")
    ]

    for name, pmf_dict, fname in datasets:
        outcomes, pmf, cdf = process_probability_data(pmf_dict)
        e, var = calculate_stats(outcomes, pmf)
        print_table(name, outcomes, pmf, cdf, e, var)
        plot_distributions(name, outcomes, pmf, cdf, e, fname)

    # Gráfica extra de conjunta
    plot_joint_distribution()
    print("\nVisualizaciones generadas exitosamente.")
    plt.show()

if __name__ == "__main__":
    main()