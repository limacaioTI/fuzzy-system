# Técnico Virtual Nebuloso — Projeto Final de Lógica Nebulosa

**Grupo 01:** Caio Lima, Gabriel Vieira, Pedro Camargos, Yuri Moura, Erick Bessa

Protótipo de um **sistema de controle nebuloso** que decide, ao longo de uma partida,
quando substituir o atacante de um time. Ele parte do artigo estudado na primeira apresentação:

> SAŁABUN, W. et al. *A Fuzzy Inference System for Players Evaluation in Multi-Player
> Sports: The Football Study Case.* Symmetry, 12(12), 2029, 2020.

O artigo avalia um atacante **uma vez**, depois do jogo. Aqui essa avaliação entra num
**laço de controle**: a cada 5 minutos o módulo nebuloso toma uma nova decisão, e cada
decisão altera o estado do jogo para as decisões seguintes.

---

## O que está sendo feito

### Arquitetura

```
┌──────────────────────── SISTEMA DE CONTROLE (externo) ───────────────────────┐
│  simulacao.py                                                                 │
│                                                                               │
│   simula 1 minuto ──► estatísticas + cansaço + minuto ──► MÓDULO NEBULOSO      │
│        ▲                                                       │              │
│        │                                                       ▼              │
│   novo estado ◄──── aplica decisão (troca se urgência ≥ 0,6) ◄── urgência      │
└───────────────────────────────────────────────────────────────────────────────┘
```

- **Sistema de controle externo** ([simulacao.py](simulacao.py)): é a "planta". Ele simula a
  partida minuto a minuto, sorteando passes, finalizações, gols e perdas de bola. Também
  controla o cansaço do jogador, o banco (2 reservas) e o limite de substituições. O
  cansaço piora o rendimento, e uma substituição faz entrar um jogador descansado. É isso
  que fecha o laço.
- **Módulo nebuloso** ([modulo_fuzzy.py](modulo_fuzzy.py)): tem dois sistemas Mamdani em
  hierarquia (a mesma ideia do artigo), implementados com `scikit-fuzzy`. A
  defuzzificação é feita pelo centroide.

### Sistema 1: Avaliação (nota de atuação)

| Entrada | Universo | Valores característicos (picos dos triângulos) | Origem |
|---|---|---|---|
| Finalizações no alvo (projetadas p/ 90 min) | 0 a 8 | 0, 2, 8 | Artigo, C10 (fig. 6d) |
| Precisão de passe (%) | 0 a 100 | 0, 70, 100 | Artigo, C5 (fig. 5b) |
| Perdas de bola (projetadas p/ 90 min) | 0 a 5 | 0, 2, 5 | Artigo, C12 (fig. 7b) |

**Saída:** a nota, de 0 a 1, com 5 termos (ruim, fraca, regular, boa, ótima).
**Regras:** são 27, todas as combinações de 3 × 3 × 3. Cada termo das entradas vale pontos
(finalizações no alvo: 0/2/4; passe: 0/1/2; perdas: 2/1/0) e a soma dos pontos escolhe o
termo da saída. Exemplo:
*SE finalizações no alvo é alto E precisão é média E perdas é baixo ENTÃO nota é ótima.*

### Sistema 2: Decisão (urgência de substituição)

| Entrada | Termos |
|---|---|
| Nota (saída do sistema 1) | baixa, média, alta |
| Cansaço (0 a 100) | baixo, médio, alto |
| Minuto do jogo (0 a 90) | início, meio, final |

**Saída:** a urgência de substituição, de 0 a 1 (baixa, média, alta). **São 27 regras**,
na tabela `REGRAS_DECISAO` do código. Exemplos:
- *SE minuto é início E cansaço é baixo ENTÃO urgência é baixa* (seja qual for a nota)
- *SE nota é baixa E cansaço é médio E minuto é meio ENTÃO urgência é alta*
- *SE nota é alta E cansaço é alto E minuto é final ENTÃO urgência é média*

O sistema externo substitui o jogador quando a urgência chega a **0,6** ou mais. O jogador
precisa ter pelo menos 10 minutos em campo, e há no máximo 2 trocas.

### De onde vêm os dados

Os dados são **simulados**. As probabilidades de cada evento foram calibradas para
reproduzir, em média, os números de atacantes das tabelas 2 a 6 do artigo: cerca de 40 a
55 passes por jogo, 70% a 80% de precisão, 3 a 4 finalizações e 2 a 3 perdas de bola.
Cada jogador tem uma qualidade, uma taxa de cansaço e uma "forma do dia" sorteadas.

Com poucos minutos em campo, as estatísticas projetadas para 90 min oscilam muito: um
chute no alvo aos 5 minutos viraria 18 por jogo. Por isso elas são suavizadas com uma
média "a priori" equivalente a 20 minutos de um jogo médio.

### Comparação

O programa joga as mesmas partidas, com os mesmos jogadores e as mesmas sementes, usando
duas estratégias:
- **Técnico fuzzy:** decide a cada 5 minutos.
- **Regra fixa:** sempre troca aos 60' e aos 75'.

Resultado com 300 partidas:

| Estratégia | Gols/jogo | No alvo/jogo | Trocas/jogo | 1ª troca (média) |
|---|---|---|---|---|
| Técnico fuzzy | ≈ 0,68 | ≈ 2,22 | ≈ 0,96 | ≈ 76' |
| Regra fixa | ≈ 0,67 | ≈ 2,22 | 2,00 | 60' |

O técnico fuzzy alcança **o mesmo rendimento usando metade das substituições**. Ele só
troca quando a combinação de nota, cansaço e minuto justifica, e às vezes não troca
ninguém.

---

## Como rodar

Requisitos: **Python 3.10 ou mais recente**.

```bash
cd prototipo
python3 -m venv .venv
source .venv/bin/activate          # no Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

Opções:

```bash
python main.py --seed 12           # outra partida de demonstração
python main.py --partidas 1000     # comparação com mais partidas (mais lento)
```

A execução padrão leva cerca de 30 a 40 segundos (o `scikit-fuzzy` não é rápido).

### Saídas

No terminal aparecem as decisões a cada 5 minutos da partida de demonstração (nota,
cansaço, urgência e ação) e a tabela da comparação. Na pasta `graficos/` ficam:

| Arquivo | Conteúdo |
|---|---|
| `partida.png` | Nota, cansaço e urgência ao longo da partida, com as substituições marcadas |
| `funcoes_pertinencia.png` | Todas as funções de pertinência dos dois sistemas |
| `comparacao.png` | Técnico fuzzy × regra fixa e o histograma do minuto da 1ª troca |

## Estrutura

```
prototipo/
├── modulo_fuzzy.py    # módulo nebuloso (2 sistemas Mamdani, 54 regras)
├── simulacao.py       # sistema de controle externo (simulação + laço)
├── main.py            # executa a demonstração, a comparação e gera os gráficos
├── requirements.txt
└── graficos/          # gerado ao rodar
```

## Limitações

- Os dados são simulados, e o modelo de jogo é simples: um só atacante, sem placar nem adversário.
- As regras de decisão e o limiar de 0,6 foram definidos pelo grupo, e não por especialistas
  como no COMET do artigo.
- Usamos só 3 dos 17 critérios do artigo, para manter o protótipo simples.
