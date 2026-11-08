"""
Módulo nebuloso do técnico virtual.

Dois sistemas de inferência Mamdani (scikit-fuzzy), em hierarquia, como no artigo
de Sałabun et al. (2020):

  1. AVALIAÇÃO  : finalizações no alvo, precisão de passe, perdas de bola -> nota (0 a 1)
  2. DECISÃO    : nota, cansaço, minuto do jogo                         -> urgência de substituição (0 a 1)

As funções de pertinência das entradas da avaliação usam os valores característicos
publicados no artigo (figuras 5, 6 e 7):
  - finalizações no alvo (C10): 0, 2, 8
  - precisão de passe   (C5) : 0, 70, 100
  - perdas de bola      (C12): 0, 2, 5
"""

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

TERMOS = ["baixo", "medio", "alto"]


def _tres_triangulos(var, a, b, c, nomes=TERMOS):
    """Cria três triângulos com picos em a, b e c (como os números fuzzy do artigo)."""
    u = var.universe
    var[nomes[0]] = fuzz.trimf(u, [a, a, b])
    var[nomes[1]] = fuzz.trimf(u, [a, b, c])
    var[nomes[2]] = fuzz.trimf(u, [b, c, c])


# ---------------------------------------------------------------------------
# 1. Sistema de AVALIAÇÃO (nota de atuação)
# ---------------------------------------------------------------------------
def regras_avaliacao():
    """Base de regras: cada termo dá "pontos"; a soma escolhe o termo da nota.
    Finalizações pesam mais (é atacante); perdas de bola contam negativamente."""
    pts_alvo = {"baixo": 0, "medio": 2, "alto": 4}
    pts_passe = {"baixo": 0, "medio": 1, "alto": 2}
    pts_perdas = {"baixo": 2, "medio": 1, "alto": 0}
    saida_por_pontos = {0: "ruim", 1: "ruim", 2: "fraca", 3: "regular",
                        4: "regular", 5: "boa", 6: "boa", 7: "otima", 8: "otima"}
    return [(ta, tp, tl, saida_por_pontos[pts_alvo[ta] + pts_passe[tp] + pts_perdas[tl]])
            for ta in TERMOS for tp in TERMOS for tl in TERMOS]


def _criar_avaliacao():
    alvo = ctrl.Antecedent(np.linspace(0, 8, 161), "finalizacoes_alvo")
    passe = ctrl.Antecedent(np.linspace(0, 100, 201), "precisao_passe")
    perdas = ctrl.Antecedent(np.linspace(0, 5, 101), "perdas_bola")
    nota = ctrl.Consequent(np.linspace(0, 1, 101), "nota")

    _tres_triangulos(alvo, 0, 2, 8)
    _tres_triangulos(passe, 0, 70, 100)
    _tres_triangulos(perdas, 0, 2, 5)

    nota["ruim"] = fuzz.trimf(nota.universe, [0, 0, 0.4])
    nota["fraca"] = fuzz.trimf(nota.universe, [0.15, 0.35, 0.55])
    nota["regular"] = fuzz.trimf(nota.universe, [0.35, 0.5, 0.65])
    nota["boa"] = fuzz.trimf(nota.universe, [0.45, 0.65, 0.85])
    nota["otima"] = fuzz.trimf(nota.universe, [0.6, 1, 1])

    regras = [ctrl.Rule(alvo[ta] & passe[tp] & perdas[tl], nota[saida])
              for ta, tp, tl, saida in regras_avaliacao()]
    return ctrl.ControlSystem(regras), (alvo, passe, perdas, nota)


# ---------------------------------------------------------------------------
# 2. Sistema de DECISÃO (urgência de substituição)
# ---------------------------------------------------------------------------
# Tabela de regras lida como: REGRAS[minuto][cansaco][nota] -> urgência
# Estratégia que a tabela codifica (detalhes no README):
#   - início: evita trocar; só nota baixa com cansaço relevante justifica.
#   - meio: é quando a troca mais compensa; nota baixa ou cansaço alto pesam.
#   - final: restam poucos minutos, então a urgência cai um nível.
REGRAS_DECISAO = {
    #            nota:  baixa     media     alta
    "inicio": {"baixo": ("baixa", "baixa", "baixa"),
               "medio": ("media", "baixa", "baixa"),
               "alto":  ("alta",  "media", "baixa")},
    "meio":   {"baixo": ("media", "baixa", "baixa"),
               "medio": ("alta",  "media", "baixa"),
               "alto":  ("alta",  "alta",  "media")},
    # No final, o reserva teria poucos minutos para render: trocar compensa menos.
    "final":  {"baixo": ("baixa", "baixa", "baixa"),
               "medio": ("media", "baixa", "baixa"),
               "alto":  ("alta",  "media", "baixa")},
}


def _criar_decisao():
    nota = ctrl.Antecedent(np.linspace(0, 1, 101), "nota")
    cansaco = ctrl.Antecedent(np.linspace(0, 100, 201), "cansaco")
    minuto = ctrl.Antecedent(np.linspace(0, 90, 181), "minuto")
    urg = ctrl.Consequent(np.linspace(0, 1, 101), "urgencia")

    _tres_triangulos(nota, 0, 0.5, 1, ["baixa", "media", "alta"])
    cansaco["baixo"] = fuzz.trapmf(cansaco.universe, [0, 0, 25, 50])
    cansaco["medio"] = fuzz.trimf(cansaco.universe, [30, 55, 80])
    cansaco["alto"] = fuzz.trapmf(cansaco.universe, [60, 85, 100, 100])
    minuto["inicio"] = fuzz.trapmf(minuto.universe, [0, 0, 20, 45])
    minuto["meio"] = fuzz.trimf(minuto.universe, [25, 50, 75])
    minuto["final"] = fuzz.trapmf(minuto.universe, [60, 80, 90, 90])
    _tres_triangulos(urg, 0, 0.5, 1, ["baixa", "media", "alta"])

    regras = []
    for tm, linhas in REGRAS_DECISAO.items():
        for tc, saidas in linhas.items():
            for tn, tu in zip(["baixa", "media", "alta"], saidas):
                regras.append(ctrl.Rule(minuto[tm] & cansaco[tc] & nota[tn], urg[tu]))
    return ctrl.ControlSystem(regras), (nota, cansaco, minuto, urg)


class TecnicoFuzzy:
    """Interface usada pelo sistema de controle externo."""

    def __init__(self):
        sis_av, self.vars_avaliacao = _criar_avaliacao()
        sis_de, self.vars_decisao = _criar_decisao()
        self._av = ctrl.ControlSystemSimulation(sis_av)
        self._de = ctrl.ControlSystemSimulation(sis_de)

    def avaliar(self, finalizacoes_alvo_90, precisao_passe, perdas_bola_90):
        self._av.input["finalizacoes_alvo"] = float(np.clip(finalizacoes_alvo_90, 0, 8))
        self._av.input["precisao_passe"] = float(np.clip(precisao_passe, 0, 100))
        self._av.input["perdas_bola"] = float(np.clip(perdas_bola_90, 0, 5))
        self._av.compute()
        return float(self._av.output["nota"])

    def decidir(self, nota, cansaco, minuto):
        self._de.input["nota"] = float(np.clip(nota, 0, 1))
        self._de.input["cansaco"] = float(np.clip(cansaco, 0, 100))
        self._de.input["minuto"] = float(np.clip(minuto, 0, 90))
        self._de.compute()
        return float(self._de.output["urgencia"])
