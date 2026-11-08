"""
Fronteira xG × substituições: para cada limiar, quanto cada estratégia rende
e quantas trocas gasta. Permite comparar estratégias com o mesmo nº de trocas.

Usa sementes SEPARADAS (5000+) das da comparação final (1000+), para que a
escolha do limiar não seja avaliada nas mesmas partidas em que foi ajustada.

Uso:  python ajuste_limiar.py [--partidas 500]
"""

import argparse
from multiprocessing import Pool
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from modulo_fuzzy import TecnicoFuzzy
from simulacao import SistemaDeControle

PASTA = Path(__file__).parent / "graficos"
LIMIARES_FUZZY = [0.44, 0.46, 0.48, 0.50, 0.52, 0.54, 0.56, 0.58]
LIMIARES_CANSACO = [55, 60, 65, 70, 75, 80, 85, 90]
SEMENTE = 5000


def rodar(tarefa):
    estrategia, valor, n = tarefa
    chave = "limiar" if estrategia == "fuzzy" else "limiar_cansaco"
    sistema = SistemaDeControle(TecnicoFuzzy(), estrategia, **{chave: valor})
    res = [sistema.jogar(s) for s in range(SEMENTE, SEMENTE + n)]
    xg = np.array([r.xg for r in res])
    return estrategia, valor, xg.mean(), 1.96 * xg.std(ddof=1) / np.sqrt(n), \
        float(np.mean([len(r.substituicoes) for r in res]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--partidas", type=int, default=500)
    n = p.parse_args().partidas
    tarefas = [("fuzzy", v, n) for v in LIMIARES_FUZZY] + [("cansaco", v, n) for v in LIMIARES_CANSACO]
    with Pool() as pool:
        linhas = pool.map(rodar, tarefas)

    print(f"{'estratégia':<10}{'limiar':>7}{'xG':>8}{'±IC95':>8}{'trocas':>8}")
    for e, v, x, ic, s in linhas:
        print(f"{e:<10}{v:>7}{x:>8.3f}{ic:>8.3f}{s:>8.2f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    for e, cor, rotulo in (("fuzzy", "#2a78d6", "Técnico fuzzy"), ("cansaco", "#eb6834", "Só cansaço")):
        d = [l for l in linhas if l[0] == e]
        ax.errorbar([l[4] for l in d], [l[2] for l in d], yerr=[l[3] for l in d], fmt="o-",
                    color=cor, capsize=3, ms=8, lw=2, label=rotulo)
    ax.set(xlabel="Substituições por partida", ylabel="xG por partida (IC 95%)",
           title="Rendimento para o mesmo número de trocas")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    PASTA.mkdir(exist_ok=True)
    fig.savefig(PASTA / "fronteira.png", dpi=150)


if __name__ == "__main__":
    main()
