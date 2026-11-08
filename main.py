"""
Protótipo: técnico virtual nebuloso para substituição de atacantes.

Uso:
    python main.py                 # demonstração (seed 7), calibração e comparação em 300 partidas
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
from simulacao import LIMIAR_PADRAO, MEDIAS_ARTIGO, SistemaDeControle, calibracao

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
    ax.plot(m, [r.nota for r in res.historico], label="Nota de atuação (0-1)", color="#2a78d6")
    ax.plot(m, [r.cansaco / 100 for r in res.historico], label="Cansaço (÷100)", color="#eb6834")
    pts = [(r.minuto, r.urgencia) for r in res.historico if r.urgencia is not None]
    if pts:
        ax.plot(*zip(*pts), "o--", label="Urgência de substituição", color="#6250d6")
    ax.axhline(LIMIAR_PADRAO, color="#6250d6", lw=0.8, ls=":", label=f"Limiar de troca ({LIMIAR_PADRAO:.2f})")
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


# nome: (rótulo, estratégia, parâmetros)
ESTRATEGIAS = {"fuzzy": ("Técnico fuzzy", "fuzzy", {}),
               "cansaco80": ("Cansaço ≥ 80", "cansaco", {"limiar_cansaco": 80}),
               "cansaco55": ("Cansaço ≥ 55", "cansaco", {"limiar_cansaco": 55}),
               "fixa": ("Fixa (60'/75')", "fixa", {}),
               "sem_troca": ("Sem trocas", "sem_troca", {})}
AZUL, CINZA = "#2a78d6", "#a8a7a0"


def ic95(v):
    """Metade do intervalo de confiança de 95% da média (aprox. normal)."""
    return 1.96 * np.std(v, ddof=1) / np.sqrt(len(v))


def imprimir_calibracao():
    print("\nCalibração: titular em 90 min x médias do artigo (pico central das pertinências)")
    medido = calibracao()
    for campo, ref in MEDIAS_ARTIGO.items():
        print(f"  {campo:<13} simulado {medido[campo]:>6.2f}   artigo {ref:>4}")


def comparar(tecnico, n):
    dados = {e: {"gols": [], "alvo": [], "xg": [], "subs": [], "primeira": []} for e in ESTRATEGIAS}
    for nome, (_, estrategia, params) in ESTRATEGIAS.items():
        sistema = SistemaDeControle(tecnico, estrategia, **params)
        for seed in range(1000, 1000 + n):
            r = sistema.jogar(seed)
            d = dados[nome]
            d["gols"].append(r.gols)
            d["alvo"].append(r.no_alvo)
            d["xg"].append(r.xg)
            d["subs"].append(len(r.substituicoes))
            if r.substituicoes:
                d["primeira"].append(r.substituicoes[0][0])

    print(f"\nComparação em {n} partidas (mesmas sementes para todas as estratégias)")
    print(f"{'estratégia':<22}{'gols ± IC95':>15}{'dp':>6}{'xG ± IC95':>16}{'dp':>6}"
          f"{'alvo':>6}{'trocas':>8}{'sem troca':>11}{'1ª troca':>10}")
    for nome, (rotulo, *_) in ESTRATEGIAS.items():
        d = {k: np.array(v, float) for k, v in dados[nome].items()}
        primeira = f"{d['primeira'].mean():.0f}'" if len(d["primeira"]) else "-"
        print(f"{rotulo:<22}{d['gols'].mean():>8.3f} ±{ic95(d['gols']):.3f}{d['gols'].std(ddof=1):>6.2f}"
              f"{d['xg'].mean():>9.3f} ±{ic95(d['xg']):.3f}{d['xg'].std(ddof=1):>6.2f}"
              f"{d['alvo'].mean():>6.2f}{d['subs'].mean():>8.2f}"
              f"{100 * np.mean(d['subs'] == 0):>10.1f}%{primeira:>10}")

    print("\nComparação pareada: fuzzy menos cada estratégia, partida a partida")
    print(f"{'contra':<22}{'dif. gols ± IC95':>20}{'dif. xG ± IC95':>20}"
          f"{'fuzzy fez mais gols':>21}{'empate':>8}{'menos':>7}")
    fz_g, fz_x = np.array(dados["fuzzy"]["gols"]), np.array(dados["fuzzy"]["xg"])
    for nome, (rotulo, *_) in list(ESTRATEGIAS.items())[1:]:
        dg = fz_g - np.array(dados[nome]["gols"])
        dx = fz_x - np.array(dados[nome]["xg"])
        print(f"{rotulo:<22}{dg.mean():>+12.3f} ±{ic95(dg):.3f}{dx.mean():>+13.3f} ±{ic95(dx):.3f}"
              f"{100 * np.mean(dg > 0):>20.1f}%{100 * np.mean(dg == 0):>7.1f}%{100 * np.mean(dg < 0):>6.1f}%")
    return dados


def grafico_comparacao(dados, caminho):
    nomes = list(ESTRATEGIAS)
    rotulos = [ESTRATEGIAS[n][0] for n in nomes]
    cores = [AZUL if n == "fuzzy" else CINZA for n in nomes]
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.5))

    caixas = ax1.boxplot([dados[n]["xg"] for n in nomes], tick_labels=rotulos, patch_artist=True,
                         widths=0.55, flierprops={"markersize": 2, "alpha": 0.4},
                         medianprops={"color": "#0b0b0b"})
    for caixa, cor in zip(caixas["boxes"], cores):
        caixa.set_facecolor(cor)
        caixa.set_edgecolor(cor)
    ax1.set_title("Gols esperados (xG) por partida")
    ax1.tick_params(axis="x", labelsize=8)

    medias = [np.mean(dados[n]["subs"]) for n in nomes]
    ax2.bar(rotulos, medias, color=cores, width=0.55)
    for i, v in enumerate(medias):
        ax2.text(i, v, f"{v:.2f}", ha="center", va="bottom")
    ax2.set_title("Substituições por partida (média)")
    ax2.tick_params(axis="x", labelsize=8)

    ax3.hist(dados["fuzzy"]["primeira"], bins=range(10, 95, 5), color=AZUL, edgecolor="white")
    ax3.set(title="Minuto da 1ª substituição (técnico fuzzy)", xlabel="Minuto", ylabel="Partidas")

    for ax in (ax1, ax2, ax3):
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
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

    imprimir_calibracao()
    dados = comparar(tecnico, args.partidas)
    grafico_comparacao(dados, PASTA / "comparacao.png")
    print(f"\nGráficos salvos em: {PASTA}")


if __name__ == "__main__":
    main()
