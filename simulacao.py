"""
Sistema de controle EXTERNO.

Simula, minuto a minuto, a atuação do atacante de um time (a "planta" do controle)
e, a cada intervalo, consulta o módulo nebuloso. Quando o módulo indica urgência
alta, o sistema aplica a decisão (substitui o jogador), o que muda o estado
das decisões seguintes: laço fechado.

As probabilidades foram calibradas para reproduzir, em média, os números de
atacantes das tabelas 2 a 6 do artigo (≈ 40-55 passes, 70-80% de precisão,
3-4 finalizações e ~2-3 perdas de bola por partida).
"""

from dataclasses import dataclass, field
import numpy as np

# Médias "a priori" por 90 min, usadas para suavizar estatísticas com poucos minutos
PRIOR_ALVO_90 = 2.0
PRIOR_PASSE = 72.0
PRIOR_PERDAS_90 = 2.0
PESO_PRIOR_MIN = 20  # equivale a 20 minutos de jogo "médio"


@dataclass
class Jogador:
    nome: str
    qualidade: float           # 0 a 1
    taxa_cansaco: float        # pontos de cansaço por minuto
    rng: np.random.Generator
    forma_dia: float = 0.0     # "dia bom/ruim", sorteado na entrada
    cansaco: float = 0.0
    minutos: int = 0
    passes: int = 0
    passes_certos: int = 0
    finalizacoes: int = 0
    no_alvo: int = 0
    gols: int = 0
    perdas: int = 0

    def __post_init__(self):
        self.forma_dia = float(self.rng.normal(0, 0.15))

    @property
    def habilidade(self):
        return float(np.clip(self.qualidade + self.forma_dia, 0.05, 1.0))

    def jogar_minuto(self):
        """Sorteia os eventos de um minuto. O cansaço piora o rendimento."""
        r, s, f = self.rng, self.habilidade, self.cansaco / 100
        self.minutos += 1
        self.cansaco = min(100.0, self.cansaco + self.taxa_cansaco * r.uniform(0.7, 1.3))

        n = r.poisson(0.6)
        p_acerto = 0.62 + 0.20 * s - 0.20 * f
        certos = r.binomial(n, np.clip(p_acerto, 0.3, 0.95))
        self.passes += n
        self.passes_certos += certos

        if r.random() < 0.045 * (0.5 + s) * (1 - 0.5 * f):
            self.finalizacoes += 1
            if r.random() < 0.40 + 0.25 * s:
                self.no_alvo += 1
                if r.random() < 0.30:
                    self.gols += 1

        if r.random() < 0.025 * (1.5 - s) * (1 + 1.5 * f):
            self.perdas += 1

    def estatisticas_projetadas(self):
        """Projeta para 90 min, suavizando com a média a priori (evita extremos no início)."""
        m, k = self.minutos, PESO_PRIOR_MIN
        alvo_90 = (self.no_alvo + PRIOR_ALVO_90 * k / 90) / (m + k) * 90
        perdas_90 = (self.perdas + PRIOR_PERDAS_90 * k / 90) / (m + k) * 90
        k_passes = 12  # ~12 passes "médios"
        precisao = (self.passes_certos + PRIOR_PASSE / 100 * k_passes) / (self.passes + k_passes) * 100
        return alvo_90, precisao, perdas_90


def criar_elenco(seed):
    """Titular + 2 reservas. Cada jogador tem seu próprio gerador aleatório,
    então estratégias diferentes enfrentam exatamente os mesmos jogadores."""
    r = np.random.default_rng(seed)
    elenco = [Jogador("Titular", r.uniform(0.55, 0.8), r.uniform(0.8, 1.25),
                      np.random.default_rng(r.integers(1 << 32)))]
    for i in (1, 2):
        elenco.append(Jogador(f"Reserva {i}", r.uniform(0.45, 0.7), r.uniform(0.8, 1.25),
                              np.random.default_rng(r.integers(1 << 32))))
    return elenco


@dataclass
class Registro:
    minuto: int
    jogador: str
    nota: float
    cansaco: float
    urgencia: float | None
    acao: str = ""


@dataclass
class ResultadoPartida:
    historico: list = field(default_factory=list)
    substituicoes: list = field(default_factory=list)
    gols: int = 0
    no_alvo: int = 0


class SistemaDeControle:
    """
    estrategia = "fuzzy" : consulta o TecnicoFuzzy a cada `intervalo` minutos
    estrategia = "fixa"  : regra de referência, troca aos 60' e aos 75'
    """

    def __init__(self, tecnico, estrategia="fuzzy", intervalo=5, limiar=0.6,
                 max_subs=2, minutos_minimos=10, nota_todo_minuto=False):
        self.nota_todo_minuto = nota_todo_minuto  # só para o gráfico (é mais lento)
        self.tecnico = tecnico
        self.estrategia = estrategia
        self.intervalo = intervalo
        self.limiar = limiar
        self.max_subs = max_subs
        self.minutos_minimos = minutos_minimos

    def jogar(self, seed):
        elenco = criar_elenco(seed)
        em_campo, banco = elenco[0], elenco[1:]
        res = ResultadoPartida()

        for minuto in range(1, 91):
            em_campo.jogar_minuto()
            urgencia, acao, nota = None, "", None
            if self.nota_todo_minuto or minuto % self.intervalo == 0:
                nota = self.tecnico.avaliar(*em_campo.estatisticas_projetadas())

            if minuto % self.intervalo == 0 and banco and len(res.substituicoes) < self.max_subs:
                if self.estrategia == "fuzzy":
                    if em_campo.minutos >= self.minutos_minimos:
                        urgencia = self.tecnico.decidir(nota, em_campo.cansaco, minuto)
                        trocar = urgencia >= self.limiar
                    else:
                        trocar = False
                else:
                    trocar = minuto in (60, 75)

                if trocar:
                    sai, entra = em_campo, banco.pop(0)
                    acao = f"SAI {sai.nome} / ENTRA {entra.nome}"
                    res.substituicoes.append((minuto, sai.nome, entra.nome))
                    res.gols += sai.gols
                    res.no_alvo += sai.no_alvo
                    em_campo = entra

            if nota is not None:
                res.historico.append(Registro(minuto, em_campo.nome if not acao else sai.nome,
                                              nota, sai.cansaco if acao else em_campo.cansaco,
                                              urgencia, acao))

        res.gols += em_campo.gols
        res.no_alvo += em_campo.no_alvo
        return res
