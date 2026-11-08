"""
Mostra o funcionamento interno do módulo nebuloso:

  1. Exemplo completo de inferência Mamdani (pertinências, regras, agregação, centroide)
  2. Cenários: mesma nota, decisões diferentes
  3. Superfície de controle da urgência
  4. Verificação de cobertura das regras, continuidade e coerência

Uso:  python analise_fuzzy.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import skfuzzy as fuzz

from modulo_fuzzy import REGRAS_DECISAO, TecnicoFuzzy, regras_avaliacao
from simulacao import LIMIAR_PADRAO as LIMIAR

PASTA = Path(__file__).parent / "graficos"
AZUL, LARANJA = "#2a78d6", "#eb6834"
NOTAS = ["baixa", "media", "alta"]


def pertinencias(var, valor):
    return {t: float(fuzz.interp_membership(var.universe, m.mf, valor)) for t, m in var.terms.items()}


def inferir(regras, entradas, saida):
    """Mamdani à mão: E = mínimo; implicação = corte; agregação = máximo; centroide.
    regras: lista de (termos das entradas..., termo da saída)."""
    u = saida.universe
    agregado, ativadas = np.zeros_like(u), []
    for *termos, consequente in regras:
        forca = min(g[t] for g, t in zip(entradas, termos))
        if forca > 0:
            ativadas.append((termos, consequente, forca))
            agregado = np.maximum(agregado, np.minimum(forca, saida[consequente].mf))
    return ativadas, agregado, float(fuzz.defuzz(u, agregado, "centroid"))


def regras_decisao():
    return [(tm, tc, tn, tu) for tm, linhas in REGRAS_DECISAO.items()
            for tc, saidas in linhas.items() for tn, tu in zip(NOTAS, saidas)]


def mostrar_graus(nome, graus):
    print(f"  {nome:<20}" + "  ".join(f"{t}={g:.2f}" for t, g in graus.items() if g > 0))


def mostrar_regras(ativadas, nomes):
    for termos, cons, forca in sorted(ativadas, key=lambda a: -a[2]):
        cond = " E ".join(f"{n} é {t}" for n, t in zip(nomes, termos))
        print(f"  [{forca:.2f}] SE {cond} ENTÃO {cons}")


def exemplo(tecnico, alvo=3, passe=75, perdas=2, cansaco=65, minuto=70):
    print(f"=== Exemplo: {alvo} no alvo, {passe}% de passe, {perdas} perdas, "
          f"cansaço {cansaco}, minuto {minuto} ===")
    v_alvo, v_passe, v_perdas, v_nota = tecnico.vars_avaliacao
    g = [pertinencias(v_alvo, alvo), pertinencias(v_passe, passe), pertinencias(v_perdas, perdas)]
    print("\n1. Fuzzificação da avaliação")
    for nome, graus in zip(["finalizações alvo", "precisão passe", "perdas de bola"], g):
        mostrar_graus(nome, graus)
    ativ1, agr1, nota = inferir(regras_avaliacao(), g, v_nota)
    print("\n2. Regras ativadas (força = mínimo dos graus)")
    mostrar_regras(ativ1, ["alvo", "passe", "perdas"])
    print(f"\n3. Agregação + centroide: nota = {nota:.3f} "
          f"(scikit-fuzzy: {tecnico.avaliar(alvo, passe, perdas):.3f})")

    v_nota2, v_cans, v_min, v_urg = tecnico.vars_decisao
    g2 = [pertinencias(v_min, minuto), pertinencias(v_cans, cansaco), pertinencias(v_nota2, nota)]
    print("\n4. Fuzzificação da decisão")
    for nome, graus in zip(["minuto", "cansaço", "nota"], g2):
        mostrar_graus(nome, graus)
    ativ2, agr2, urg = inferir(regras_decisao(), g2, v_urg)
    print("\n5. Regras ativadas")
    mostrar_regras(ativ2, ["minuto", "cansaço", "nota"])
    print(f"\n6. Agregação + centroide: urgência = {urg:.3f} "
          f"(scikit-fuzzy: {tecnico.decidir(nota, cansaco, minuto):.3f})")
    print(f"   Decisão do sistema externo: {'SUBSTITUIR' if urg >= LIMIAR else 'MANTER'} (limiar {LIMIAR:.2f})\n")

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    for ax, var, agr, c, titulo in ((axs[0], v_nota, agr1, nota, "Sistema 1: nota"),
                                    (axs[1], v_urg, agr2, urg, "Sistema 2: urgência")):
        for termo in var.terms.values():
            ax.plot(var.universe, termo.mf, color="#a8a7a0", lw=1)
        ax.fill_between(var.universe, agr, color=AZUL, alpha=0.45, label="Saída agregada")
        ax.axvline(c, color=LARANJA, lw=2, label=f"Centroide = {c:.2f}")
        ax.set(title=titulo, xlabel="Valor de saída", ylabel="Pertinência", ylim=(0, 1.05))
        ax.legend(loc="upper right", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(PASTA / "exemplo_inferencia.png", dpi=150)
    plt.close(fig)


def cenarios(tecnico):
    print("=== Mesma variável, decisões diferentes (nota, cansaço, minuto -> urgência) ===")
    casos = [("Nota baixa, descansado, início", 0.3, 15, 15),
             ("Nota baixa, descansado, meio", 0.3, 30, 50),
             ("Nota baixa, cansado, meio", 0.3, 80, 60),
             ("Nota baixa, cansado, últimos min.", 0.3, 80, 88),
             ("Nota alta, cansado, meio", 0.8, 80, 60),
             ("Nota alta, cansado, últimos min.", 0.8, 80, 88),
             ("Nota alta, descansado, meio", 0.8, 30, 50)]
    for nome, n, c, m in casos:
        u = tecnico.decidir(n, c, m)
        print(f"  {nome:<36}{n:>4} {c:>4} {m:>4}'  ->  {u:.2f}  {'SUBSTITUIR' if u >= LIMIAR else 'manter'}")
    print()


def superficie(tecnico, minutos=(20, 50, 80)):
    notas = np.linspace(0, 1, 26)
    cansacos = np.linspace(0, 100, 26)
    X, Y = np.meshgrid(notas, cansacos)
    fig = plt.figure(figsize=(15, 8.5))
    grades = {}
    for i, m in enumerate(minutos):
        Z = np.array([[tecnico.decidir(n, c, m) for n in notas] for c in cansacos])
        grades[m] = Z
        ax = fig.add_subplot(2, 3, i + 1, projection="3d")
        ax.plot_surface(X, Y, Z, cmap="Blues", vmin=0, vmax=1, linewidth=0, antialiased=True)
        ax.set(title=f"Minuto {m}", xlabel="Nota", ylabel="Cansaço", zlabel="Urgência", zlim=(0, 1))
        ax.view_init(28, -130)
        ax2 = fig.add_subplot(2, 3, i + 4)
        im = ax2.pcolormesh(X, Y, Z, cmap="Blues", vmin=0, vmax=1, shading="auto")
        if Z.max() >= LIMIAR:
            ax2.contour(X, Y, Z, levels=[LIMIAR], colors=LARANJA, linewidths=2)
        ax2.set(xlabel="Nota", ylabel="Cansaço",
                title=f"Linha laranja: limiar de troca ({LIMIAR:.2f})")
        fig.colorbar(im, ax=ax2, label="Urgência")
    fig.tight_layout()
    fig.savefig(PASTA / "superficie_controle.png", dpi=150)
    plt.close(fig)


def verificar(tecnico):
    """Cobertura: toda entrada ativa alguma regra? Continuidade: há saltos bruscos?
    Coerência: mais cansaço nunca reduz a urgência; nota maior nunca a aumenta?"""
    print("=== Verificação do sistema de decisão ===")
    v_nota, v_cans, v_min, _ = tecnico.vars_decisao
    notas, cansacos, minutos = np.linspace(0, 1, 21), np.linspace(0, 100, 21), np.arange(0, 91, 10)

    menor = 1.0
    for n in notas:
        gn = max(pertinencias(v_nota, n).values())
        for c in cansacos:
            gc = max(pertinencias(v_cans, c).values())
            for m in minutos:
                menor = min(menor, gn, gc, max(pertinencias(v_min, m).values()))
    print(f"  Cobertura: menor força da regra mais ativada = {menor:.2f} (0 seria um buraco)")

    U = np.array([[[tecnico.decidir(n, c, m) for m in minutos] for c in cansacos] for n in notas])
    print(f"  Continuidade: maior salto entre vizinhos -> nota (Δ0,05) {np.abs(np.diff(U, axis=0)).max():.3f}; "
          f"cansaço (Δ5) {np.abs(np.diff(U, axis=1)).max():.3f}; minuto (Δ10) {np.abs(np.diff(U, axis=2)).max():.3f}")
    d_cans, d_nota = np.diff(U, axis=1), np.diff(U, axis=0)
    tol = 0.02
    print(f"  Coerência (tolerância {tol}): cansaço sobe e urgência cai em "
          f"{100 * np.mean(d_cans < -tol):.1f}% dos passos (pior {d_cans.min():+.3f}); "
          f"nota sobe e urgência sobe em {100 * np.mean(d_nota > tol):.1f}% (pior {d_nota.max():+.3f})")

    v_alvo, v_passe, v_perdas, _ = tecnico.vars_avaliacao
    N = np.array([[[tecnico.avaliar(a, p, l) for l in np.linspace(0, 5, 11)]
                   for p in np.linspace(0, 100, 11)] for a in np.linspace(0, 8, 9)])
    print(f"  Nota: faixa alcançável {N.min():.2f} a {N.max():.2f}")


def main():
    PASTA.mkdir(exist_ok=True)
    tecnico = TecnicoFuzzy()
    exemplo(tecnico)
    cenarios(tecnico)
    superficie(tecnico)
    verificar(tecnico)
    print(f"\nGráficos salvos em: {PASTA}")


if __name__ == "__main__":
    main()
