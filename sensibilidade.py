"""
Análise de sensibilidade: muda um parâmetro por vez e observa o resultado.

Uso:  python sensibilidade.py [--partidas 300]
"""

import argparse
from multiprocessing import Pool
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from modulo_fuzzy import TecnicoFuzzy
from simulacao import LIMIAR_PADRAO, SistemaDeControle

PASTA = Path(__file__).parent / "graficos"
AZUL = "#2a78d6"
BASE = {"limiar": LIMIAR_PADRAO, "suavizacao": 20, "intervalo": 5}
EXPERIMENTOS = {"Limiar": ("limiar", [0.45, 0.50, 0.55]),
                "Suavização (min)": ("suavizacao", [0, 10, 20, 30]),
                "Intervalo (min)": ("intervalo", [5, 10])}


def rodar(tarefa):
    n, params = tarefa
    sistema = SistemaDeControle(TecnicoFuzzy(), "fuzzy", **params)
    xg, subs, primeira = [], [], []
    for seed in range(1000, 1000 + n):
        r = sistema.jogar(seed)
        xg.append(r.xg)
        subs.append(len(r.substituicoes))
        if r.substituicoes:
            primeira.append(r.substituicoes[0][0])
    return np.array(xg), np.array(subs), primeira


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--partidas", type=int, default=300)
    n = p.parse_args().partidas
    tarefas = [(n, BASE)] + [(n, {**BASE, c: v}) for c, vs in EXPERIMENTOS.values() for v in vs]
    with Pool() as pool:
        resultados = iter(pool.map(rodar, tarefas))
    base_xg, _, _ = next(resultados)

    print(f"Um parâmetro por vez, {n} partidas. Base: {BASE}")
    print(f"{'parâmetro':<19}{'valor':>6}{'xG':>8}{'dif. pareada ± IC95':>23}"
          f"{'trocas':>8}{'sem troca':>11}{'1ª troca':>10}")
    fig, axs = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, (nome, (chave, valores)) in zip(axs, EXPERIMENTOS.items()):
        medias, erros = [], []
        for v in valores:
            xg, subs, primeira = next(resultados)
            dif = xg - base_xg
            ic = 1.96 * dif.std(ddof=1) / np.sqrt(n)
            medias.append(dif.mean())
            erros.append(ic)
            print(f"{nome:<19}{v:>6}{xg.mean():>8.3f}{dif.mean():>+15.4f} ±{ic:.4f}{subs.mean():>10.2f}"
                  f"{100 * np.mean(subs == 0):>10.1f}%{(f'{np.mean(primeira):.0f}' if primeira else '-'):>10}")
        ax.errorbar([str(v) for v in valores], medias, yerr=erros, fmt="o", color=AZUL, capsize=4)
        ax.axhline(0, color="#a8a7a0", lw=1)
        ax.set(title=nome)
        ax.spines[["top", "right"]].set_visible(False)
    axs[0].set_ylabel("xG em relação à base (IC 95%)")
    fig.tight_layout()
    PASTA.mkdir(exist_ok=True)
    fig.savefig(PASTA / "sensibilidade.png", dpi=150)
    print(f"\nGráfico salvo em: {PASTA / 'sensibilidade.png'}")


if __name__ == "__main__":
    main()
