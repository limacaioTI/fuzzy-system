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
│   novo estado ◄──── aplica decisão (troca se urgência ≥ 0,5) ◄── urgência      │
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
na tabela `REGRAS_DECISAO` do código. Elas não foram preenchidas caso a caso: seguem esta estratégia.

| Condição | Comportamento esperado |
|---|---|
| Início do jogo | Evitar trocar; só nota baixa com cansaço relevante justifica |
| Meio do jogo | É quando a troca mais compensa: o reserva ainda tem tempo para render |
| Final do jogo | A urgência cai um nível, porque restam poucos minutos |
| Nota alta + cansaço baixo | Manter sempre |
| Nota baixa + cansaço alto | Substituir (exceto nos minutos finais, com urgência já menor) |
| Nota alta + cansaço alto | Depende do momento: média no meio, baixa no final |
| Nota baixa + cansaço baixo | Considerar a troca só no meio do jogo |

Exemplos:
- *SE minuto é início E cansaço é baixo ENTÃO urgência é baixa* (seja qual for a nota)
- *SE minuto é meio E cansaço é médio E nota é baixa ENTÃO urgência é alta*
- *SE minuto é final E cansaço é alto E nota é alta ENTÃO urgência é baixa*

O sistema externo substitui o jogador quando a urgência chega a **0,50** ou mais. O jogador
precisa ter pelo menos 10 minutos em campo, e há no máximo 2 trocas. O limiar foi escolhido
por experimento, em partidas separadas (veja **Resultados**).

### De onde vêm os dados

Os dados são **simulados**. As probabilidades foram calibradas pelas médias reais de
atacantes: o artigo define o pico central de cada função de pertinência como a média dos
dados de 2017. Medição de um titular em 90 minutos (2.000 simulações):

| Estatística | Simulado | Artigo |
|---|---|---|
| Passes | 29,8 | 25 |
| Precisão de passe | 70,4% | 70% |
| Finalizações | 3,06 | 3 |
| Finalizações no alvo | 1,93 | 2 |
| Gols | 0,68 | 1 |
| Perdas de bola | 2,09 | 2 |

Cada jogador tem uma qualidade, uma taxa de cansaço e uma "forma do dia" sorteadas. Os
reservas têm qualidade menor em média (0,45 a 0,70, contra 0,55 a 0,80 do titular), então
a troca compara um jogador cansado com um descansado, mas possivelmente inferior.

As estatísticas projetadas para 90 min são suavizadas com uma média "a priori" equivalente a
20 minutos de jogo, porque um chute no alvo aos 5 minutos viraria 18 por jogo.

---

## Resultados

Métricas: gols e **xG** (gols esperados, a soma das chances de gol de cada minuto). O xG
varia bem menos que os gols, por isso detecta diferenças pequenas. Todas as estratégias
jogam com as mesmas sementes, o que permite uma comparação **pareada**.

### Como o limiar foi escolhido

`ajuste_limiar.py` mede xG e número de trocas para vários limiares, em 500 partidas de
sementes separadas (`graficos/fronteira.png`). Com menos de 1 troca por jogo, o fuzzy
rende mais que a regra de cansaço para o mesmo número de trocas (0,726 com 0,76 troca,
contra 0,701 com 0,80). Com cerca de 1 troca por jogo, as duas empatam. O limiar 0,50 foi
escolhido por esse equilíbrio.

### Comparação final (1.000 partidas que não participaram do ajuste)

| Estratégia | Gols ± IC95 | xG ± IC95 | Trocas/jogo | Jogos sem troca | 1ª troca |
|---|---|---|---|---|---|
| Técnico fuzzy | 0,733 ± 0,051 | 0,721 ± 0,007 | 0,74 | 26,4% | 55' |
| Cansaço ≥ 80 | 0,735 ± 0,052 | 0,701 ± 0,008 | 0,79 | 20,8% | 78' |
| Cansaço ≥ 55 | 0,757 ± 0,053 | 0,730 ± 0,007 | 1,01 | 0,0% | 57' |
| Fixa (60'/75') | 0,761 ± 0,054 | 0,740 ± 0,007 | 2,00 | 0,0% | 60' |
| Sem trocas | 0,700 ± 0,051 | 0,672 ± 0,009 | 0,00 | 100% | - |

Diferença pareada (fuzzy menos a outra estratégia):

| Contra | Dif. gols ± IC95 | Dif. xG ± IC95 | Fuzzy fez mais gols | Empate | Fez menos |
|---|---|---|---|---|---|
| Cansaço ≥ 80 | -0,002 ± 0,031 | +0,020 ± 0,003 | 9,2% | 80,7% | 10,1% |
| Cansaço ≥ 55 | -0,024 ± 0,029 | -0,010 ± 0,002 | 6,8% | 84,1% | 9,1% |
| Fixa (60'/75') | -0,028 ± 0,038 | -0,019 ± 0,003 | 11,4% | 74,2% | 14,4% |
| Sem trocas | +0,033 ± 0,037 | +0,049 ± 0,005 | 13,8% | 74,2% | 12,0% |

**Como interpretar:**
- Em **gols**, nenhuma diferença é conclusiva, porque todos os intervalos incluem zero.
- Em **xG**, com número parecido de trocas, o fuzzy supera a regra só de cansaço
  (+0,020, intervalo que não inclui zero).
- Quem usa mais trocas rende mais em xG. Isso é esperado: na simulação, trocar **não tem
  custo**. A regra fixa vence em xG, mas gasta 2,7 vezes mais trocas.
- Os resultados valem só para esta simulação. Eles não mostram equivalência formal nem
  superioridade em jogos reais.

### Sensibilidade (500 partidas, um parâmetro por vez, base: limiar 0,50)

| Parâmetro | Valor | xG | Dif. pareada ± IC95 | Trocas/jogo |
|---|---|---|---|---|
| Limiar | 0,45 | 0,724 | +0,002 ± 0,005 | 1,06 |
| Limiar | 0,55 | 0,691 | -0,031 ± 0,005 | 0,27 |
| Suavização | 0 min | 0,722 | -0,000 ± 0,001 | 0,76 |
| Suavização | 10 min | 0,722 | -0,000 ± 0,000 | 0,76 |
| Suavização | 30 min | 0,722 | -0,000 ± 0,001 | 0,76 |
| Intervalo | 10 min | 0,717 | -0,005 ± 0,002 | 0,69 |

- **O limiar é o parâmetro crítico.** Entre 0,50 e 0,55 as trocas caem de 0,76 para 0,27,
  porque há um patamar de urgência perto de 0,50: o termo "média" tem centroide 0,5.
- **A suavização quase não muda o resultado.** Isso sugere que as decisões dependem mais do
  cansaço e do minuto do que da nota.

### Verificação do módulo nebuloso

- **Exemplo de inferência:** feito à mão em `analise_fuzzy.py`, chega aos mesmos valores do
  scikit-fuzzy (nota 0,593 e urgência 0,390).
- **Cobertura:** toda a grade testada ativa ao menos uma regra (a menor força foi 0,40).
- **Continuidade:** o maior salto foi de 0,18 para 5 pontos de cansaço e de 0,12 para 0,05
  de nota. Não há quedas bruscas.
- **Coerência:** a urgência nunca sobe de forma relevante quando a nota sobe. Ao subir o
  cansaço, ela cai em 3,0% dos passos, no máximo 0,045. É um pequeno efeito do centroide.
- **Superfície de controle:** a região de troca é maior no minuto 50 do que nos minutos 20 e 80.

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

Opções e outros experimentos:

```bash
python main.py --seed 12           # outra partida de demonstração
```

```bash
python main.py --partidas 1000     # números do README, cerca de 3 minutos
```

```bash
python analise_fuzzy.py            # exemplo de inferência, superfície e verificações, cerca de 1 minuto
```

```bash
python ajuste_limiar.py            # fronteira xG × trocas, cerca de 3,5 minutos em 10 núcleos
```

```bash
python sensibilidade.py --partidas 500   # testes de sensibilidade
```

`python main.py` sozinho compara 300 partidas (cerca de 1 minuto), e os números variam um
pouco em relação às 1.000 do README.

### Saídas

| Arquivo em `graficos/` | Conteúdo |
|---|---|
| `partida.png` | Nota, cansaço e urgência ao longo da partida, com as substituições marcadas |
| `funcoes_pertinencia.png` | Todas as funções de pertinência dos dois sistemas |
| `comparacao.png` | Distribuição de xG, trocas por estratégia e minuto da 1ª troca |
| `exemplo_inferencia.png` | Saída agregada e centroide do exemplo |
| `superficie_controle.png` | Urgência por nota e cansaço nos minutos 20, 50 e 80 |
| `fronteira.png` | xG × número de trocas, fuzzy contra cansaço |
| `sensibilidade.png` | Efeito de cada parâmetro no xG |

## Estrutura

```
prototipo/
├── modulo_fuzzy.py    # módulo nebuloso (2 sistemas Mamdani, 54 regras)
├── simulacao.py       # sistema de controle externo (simulação + laço)
├── main.py            # demonstração, calibração e comparação
├── analise_fuzzy.py   # exemplo de inferência, superfície e verificações
├── ajuste_limiar.py   # fronteira xG × trocas, para escolher o limiar
├── sensibilidade.py   # testes de sensibilidade
├── requirements.txt
└── graficos/          # gerado ao rodar
```

## Validação e limitações experimentais

1. **Natureza dos dados:** são simulados e calibrados com médias agregadas do artigo. Não
   representam partidas reais. Passes e gols ficam a alguma distância da referência.
2. **Regras heurísticas:** regras e parâmetros foram definidos pelo grupo, sem validação com
   técnicos, ao contrário do COMET do artigo. Este protótipo usa Mamdani, não COMET.
3. **Escopo:** há um só atacante, sem placar, adversário ou contexto tático, e trocar não tem custo.
4. **Critérios:** usamos só 3 dos 17 critérios do artigo. A nota varia pouco e parece pesar
   menos na decisão do que o cansaço e o minuto.
5. **Validade:** os resultados descrevem o protótipo nas condições simuladas. Eles não
   comprovam superioridade em partidas reais.
