"""
Protótipo: técnico virtual nebuloso para substituição de atacantes.

Uso:
    python main.py                 # partida de demonstração (seed 7) + comparação em 300 partidas
    python main.py --seed 12       # outra partida de demonstração
    python main.py --partidas 1000 # comparação com mais partidas
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from modulo_fuzzy import TecnicoFuzzy
from simulacao import SistemaDeControle

PASTA = Path(__file__).parent / "graficos"


def imprimir_partida(res):
    print(f"{'min':>4} {'jogador':<10} {'nota':>5} {'cansaço':>8} {'urgência':>9}  ação")
    for r in res.historico:
        if r.urgencia is not None or r.acao:
            u = f"{r.urgencia:.2f}" if r.urgencia is not None else "  -"
            print(f"{r.minuto:>4} {r.jogador:<10} {r.nota:>5.2f} {r.cansaco:>8.1f} {u:>9}  {r.acao}")
    print(f"\nGols da posição: {res.gols} | Finalizações no alvo: {res.no_alvo}")


def grafico_partida(res, seed, caminho):
    m = [r.minuto for r in res.historico]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(m, [r.nota for r in res.historico], label="Nota de atuação (0-1)", color="#1f77b4")
    ax.plot(m, [r.cansaco / 100 for r in res.historico], label="Cansaço (÷100)", color="#ff7f0e")
    pts = [(r.minuto, r.urgencia) for r in res.historico if r.urgencia is not None]
    if pts:
        ax.plot(*zip(*pts), "o--", label="Urgência de substituição", color="#d62728")
    ax.axhline(0.6, color="#d62728", lw=0.8, ls=":", label="Limiar de troca (0,6)")
    for minuto, sai, entra in res.substituicoes:
        ax.axvline(minuto, color="gray", lw=1.2)
        ax.text(minuto + 0.5, 1.02, f"{minuto}': {entra}", fontsize=9, color="gray")
    ax.set(xlim=(0, 90), ylim=(0, 1.1), xlabel="Minuto", ylabel="Valor",
           title=f"Partida simulada (seed {seed}) — decisões do módulo nebuloso")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(caminho, dpi=150)
    plt.close(fig)


def grafico_pertinencias(tecnico, caminho):
    variaveis = list(tecnico.vars_avaliacao) + list(tecnico.vars_decisao[1:])
    fig, axs = plt.subplots(2, 4, figsize=(15, 6))
    for ax, var in zip(axs.flat, variaveis):
        for nome, termo in var.terms.items():
            ax.plot(var.universe, termo.mf, label=nome)
        ax.set_title(var.label)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    axs.flat[-1].axis("off")
    fig.suptitle("Funções de pertinência (avaliação: valores do artigo; decisão: definidas pelo grupo)")
    fig.tight_layout()
    fig.savefig(caminho, dpi=150)
    plt.close(fig)


def comparar(tecnico, n):
    fuzzy = SistemaDeControle(tecnico, "fuzzy")
    fixa = SistemaDeControle(tecnico, "fixa")
    dados = {"fuzzy": [], "fixa": []}
    for seed in range(1000, 1000 + n):
        for nome, sist in (("fuzzy", fuzzy), ("fixa", fixa)):
            r = sist.jogar(seed)
            dados[nome].append((r.gols, r.no_alvo, len(r.substituicoes),
                                r.substituicoes[0][0] if r.substituicoes else None))
    print(f"\nComparação em {n} partidas (mesmos jogadores para as duas estratégias)")
    print(f"{'estratégia':<12} {'gols/jogo':>10} {'no alvo/jogo':>13} {'subs/jogo':>10} {'1ª troca (méd.)':>16}")
    resumo = {}
    for nome, linhas in dados.items():
        a = np.array([l[:3] for l in linhas], float)
        primeiras = [l[3] for l in linhas if l[3] is not None]
        resumo[nome] = a.mean(axis=0)
        print(f"{nome:<12} {a[:,0].mean():>10.3f} {a[:,1].mean():>13.3f} {a[:,2].mean():>10.2f} "
              f"{np.mean(primeiras) if primeiras else float('nan'):>15.1f}'")
    return dados, resumo


def grafico_comparacao(dados, caminho):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    nomes = ["fuzzy", "fixa"]
    rotulos = ["Técnico fuzzy", "Regra fixa (60'/75')"]
    medias = [np.mean([l[1] for l in dados[n]]) for n in nomes]
    ax1.bar(rotulos, medias, color=["#1f77b4", "#aaaaaa"])
    ax1.set_title("Finalizações no alvo por partida (média)")
    for i, v in enumerate(medias):
        ax1.text(i, v, f"{v:.2f}", ha="center", va="bottom")
    trocas = [l[3] for l in dados["fuzzy"] if l[3] is not None]
    ax2.hist(trocas, bins=range(10, 95, 5), color="#1f77b4", edgecolor="white")
    ax2.set(title="Minuto da 1ª substituição (técnico fuzzy)", xlabel="Minuto", ylabel="Partidas")
    fig.tight_layout()
    fig.savefig(caminho, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=7, help="semente da partida de demonstração")
    p.add_argument("--partidas", type=int, default=300, help="nº de partidas na comparação")
    args = p.parse_args()

    PASTA.mkdir(exist_ok=True)
    tecnico = TecnicoFuzzy()

    print(f"=== Partida de demonstração (seed {args.seed}) ===")
    res = SistemaDeControle(tecnico, "fuzzy", nota_todo_minuto=True).jogar(args.seed)
    imprimir_partida(res)
    grafico_partida(res, args.seed, PASTA / "partida.png")
    grafico_pertinencias(tecnico, PASTA / "funcoes_pertinencia.png")

    dados, _ = comparar(tecnico, args.partidas)
    grafico_comparacao(dados, PASTA / "comparacao.png")
    print(f"\nGráficos salvos em: {PASTA}")


if __name__ == "__main__":
    main()
