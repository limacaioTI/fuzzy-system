"""
Sistema de controle EXTERNO.

Simula, minuto a minuto, a atuação do atacante de um time (a "planta" do controle)
e, a cada intervalo, consulta o módulo nebuloso. Quando o módulo indica urgência
alta, o sistema aplica a decisão (substitui o jogador), o que muda o estado
das decisões seguintes: laço fechado.

As probabilidades foram calibradas para reproduzir, em média, os números de
atacantes publicadas no artigo. A comparação medida é impressa por main.py.
"""

from dataclasses import dataclass, field
import numpy as np

# Médias "a priori" por 90 min, usadas para suavizar estatísticas com poucos minutos
LIMIAR_PADRAO = 0.50     # escolhido em ajuste_limiar.py, com sementes separadas

PRIOR_ALVO_90 = 2.0
PRIOR_PASSE = 72.0
PRIOR_PERDAS_90 = 2.0
PESO_PRIOR_MIN = 20  # equivale a 20 minutos de jogo "médio"

# Médias reais de atacantes: o artigo define o pico central de cada função de
# pertinência como a média dos dados de 2017 (figuras 5, 6 e 7).
MEDIAS_ARTIGO = {"passes": 25, "precisao": 70, "finalizacoes": 3,
                 "no_alvo": 2, "gols": 1, "perdas": 2}

# Parâmetros da planta, calibrados para ficar perto dessas médias
TAXA_PASSES = 0.33       # passes por minuto
TAXA_FINALIZACAO = 0.038
CONVERSAO_ALVO = 0.35    # chance de gol de uma finalização no alvo
TAXA_PERDAS = 0.017


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
    xg: float = 0.0            # gols esperados: soma das chances de gol de cada minuto

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

        n = r.poisson(TAXA_PASSES)
        p_acerto = 0.66 + 0.20 * s - 0.20 * f
        certos = r.binomial(n, np.clip(p_acerto, 0.3, 0.95))
        self.passes += n
        self.passes_certos += certos

        p_fin = TAXA_FINALIZACAO * (0.5 + s) * (1 - 0.5 * f)
        p_alvo = 0.45 + 0.25 * s
        self.xg += p_fin * p_alvo * CONVERSAO_ALVO
        if r.random() < p_fin:
            self.finalizacoes += 1
            if r.random() < p_alvo:
                self.no_alvo += 1
                if r.random() < CONVERSAO_ALVO:
                    self.gols += 1

        if r.random() < TAXA_PERDAS * (1.5 - s) * (1 + 1.5 * f):
            self.perdas += 1

    def estatisticas_projetadas(self, k=PESO_PRIOR_MIN):
        """Projeta para 90 min, suavizando com a média a priori (evita extremos no início).
        k = quantos minutos de jogo "médio" são somados aos dados reais (0 = sem suavização)."""
        m = self.minutos
        alvo_90 = (self.no_alvo + PRIOR_ALVO_90 * k / 90) / (m + k) * 90
        perdas_90 = (self.perdas + PRIOR_PERDAS_90 * k / 90) / (m + k) * 90
        k_passes = TAXA_PASSES * k
        if self.passes + k_passes == 0:
            return alvo_90, PRIOR_PASSE, perdas_90
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
    xg: float = 0.0


class SistemaDeControle:
    """
    Estratégias:
      "fuzzy"     : consulta o TecnicoFuzzy a cada `intervalo` minutos
      "fixa"      : troca aos 60' e aos 75'
      "cansaco"   : troca quando o cansaço chega a `limiar_cansaco` (só uma variável)
      "sem_troca" : nunca substitui (linha de base)
    """

    def __init__(self, tecnico, estrategia="fuzzy", intervalo=5, limiar=LIMIAR_PADRAO,
                 max_subs=2, minutos_minimos=10, nota_todo_minuto=False,
                 limiar_cansaco=75, suavizacao=PESO_PRIOR_MIN):
        self.nota_todo_minuto = nota_todo_minuto  # só para o gráfico (é mais lento)
        self.tecnico = tecnico
        self.estrategia = estrategia
        self.intervalo = intervalo
        self.limiar = limiar
        self.max_subs = max_subs
        self.minutos_minimos = minutos_minimos
        self.limiar_cansaco = limiar_cansaco
        self.suavizacao = suavizacao

    def _deve_trocar(self, jogador, nota, minuto):
        """Devolve (trocar, urgencia). Só a estratégia fuzzy produz urgência."""
        if self.estrategia == "fixa":
            return minuto in (60, 75), None
        if self.estrategia == "sem_troca" or jogador.minutos < self.minutos_minimos:
            return False, None
        if self.estrategia == "cansaco":
            return jogador.cansaco >= self.limiar_cansaco, None
        urgencia = self.tecnico.decidir(nota, jogador.cansaco, minuto)
        return urgencia >= self.limiar, urgencia

    def jogar(self, seed):
        elenco = criar_elenco(seed)
        em_campo, banco = elenco[0], elenco[1:]
        usados = [em_campo]
        res = ResultadoPartida()
        k = self.suavizacao

        for minuto in range(1, 91):
            em_campo.jogar_minuto()
            urgencia, acao, nota, sai = None, "", None, em_campo
            hora_de_decidir = minuto % self.intervalo == 0
            if self.nota_todo_minuto or (hora_de_decidir and self.estrategia == "fuzzy"):
                nota = self.tecnico.avaliar(*em_campo.estatisticas_projetadas(k))

            if hora_de_decidir and banco and len(res.substituicoes) < self.max_subs:
                trocar, urgencia = self._deve_trocar(em_campo, nota, minuto)
                if trocar:
                    entra = banco.pop(0)
                    acao = f"SAI {sai.nome} / ENTRA {entra.nome}"
                    res.substituicoes.append((minuto, sai.nome, entra.nome))
                    em_campo = entra
                    usados.append(entra)

            if nota is not None:
                res.historico.append(Registro(minuto, sai.nome, nota, sai.cansaco, urgencia, acao))

        res.gols = sum(j.gols for j in usados)
        res.no_alvo = sum(j.no_alvo for j in usados)
        res.xg = sum(j.xg for j in usados)
        return res


def calibracao(n=2000):
    """Médias de um titular que joga os 90 min, para comparar com o artigo."""
    campos = {c: [] for c in MEDIAS_ARTIGO}
    for seed in range(n):
        j = criar_elenco(seed)[0]
        for _ in range(90):
            j.jogar_minuto()
        campos["passes"].append(j.passes)
        campos["precisao"].append(100 * j.passes_certos / max(j.passes, 1))
        campos["finalizacoes"].append(j.finalizacoes)
        campos["no_alvo"].append(j.no_alvo)
        campos["gols"].append(j.gols)
        campos["perdas"].append(j.perdas)
    return {c: float(np.mean(v)) for c, v in campos.items()}
