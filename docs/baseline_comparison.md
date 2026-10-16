# Comparativo Completo: Baselines Clássicos vs. Autoencoders Neurais — Set 2 (`2nd_test`)

Resultados consolidados em 3 seeds independentes (`[42, 123, 456]`), seguindo rigorosamente o protocolo de avaliação não supervisionada em PHM (*Prognostics and Health Management*).

---

## 1. Tabela Comparativa Global (Média ± Desvio Padrão nas 3 Seeds)

| Métrica | PCA (Linear, $L=1$) | Isolation Forest (Árvores, $L=1$) | **MLP Autoencoder (Denso, $L=1$)** | **Conv1D Autoencoder (Seq., $L=5$)** | **LSTM Autoencoder (Recorrente, $L=5$)** |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1º Alarme Sustentado (`first_alarm_idx`)** | `601,0 ± 0,0` | `636,0 ± 14,1` | **`532,0 ± 0,0`** | `535,0 ± 0,0` *(1ª janela)* | **`535,0 ± 0,0` *(1ª janela)*** |
| **RUL no 1º Alarme (`detection_lead_hours`)** | `63,33 ± 0,00 h` (`380` arq.) | `57,50 ± 2,36 h` (`345` arq.) | **`74,83 ± 0,00 h` (`449` arq.)** | `74,33 ± 0,00 h` (`446` arq.) | **`74,33 ± 0,00 h` (`446` arq.)** |
| **Recall na Degradação (`test`)** | `86,25% ± 0,00%` | `79,08% ± 1,50%` | **`99,78% ± 0,00%`** | **`100,0% ± 0,0%`** | **`100,0% ± 0,0%`** |
| **Taxa de Falso Alarme (`val`)** | **`1,25% ± 0,00%`** | `2,08% ± 0,59%` | `5,42% ± 1,56%` | `22,80% ± 1,24%` | **`3,51% ± 2,24%`** |
| **ROC-AUC (Val vs Teste)** | `0,9838 ± 0,0000` | `0,9839 ± 0,0016` | **`0,9989 ± 0,0005`** | **`1,0000 ± 0,0000`** | **`1,0000 ± 0,0000`** |
| **Monotonicidade com RUL (`spearman`)** | `0,8291 ± 0,0000` | **`0,9001 ± 0,0104`** | `0,7877 ± 0,0218` | `0,8020 ± 0,0060` | **`0,8598 ± 0,0010`** |
| **Contraste Teste/Val (`separation_ratio`)** | **`47.920×`** | `1,48×` | `15.398×` | `7.273×` | `8.836×` |
| **Top Features no Alarme (Explicabilidade)** | `b1_rms` (35,3%), `b1_energy_high` (27,1%), `b1_energy_bpfi` (9,3%) | `b1_kurtosis` (18,5%), `b1_rms` (18,5%), `b1_energy_high` (16,4%), `b1_energy_bpfo` (7,9%) | `b1_energy_high` (57,4%), `b1_rms` (32,1%), `b1_kurtosis` (4,1%) | `b1_energy_high` (59,3%), `b1_rms` (27,6%), `b1_energy_mid` (3,9%) | `b1_energy_high` (53,1%), `b1_rms` (33,0%), `b1_energy_bsf2x` (3,8%), `b1_kurtosis` (2,2%) |

---

## 2. Principais Conclusões da Comparação Experimental

### 2.1 Por que os Autoencoders Neurais (`MLP` e `LSTM`) superaram o `PCA` e o `Isolation Forest` em mais de 11 horas de antecedência (RUL)?
1. **Detecção exata no surgimento físico da microtrinca (`idx=532` / `535`):**
   - Enquanto o **PCA** (linear) só sustenta o alarme a partir do arquivo `601` (`63,33 h` de RUL) e o **Isolation Forest** no arquivo `636` (`57,50 h` de RUL), o **MLP Autoencoder ($L=1$)** dispara o alarme sustentado no **arquivo `532` (`74,83 h` de RUL)** — literalmente o ponto exato documentado por Hai Qiu et al. (2006) para o surgimento da falha incipiente na pista externa do Rolamento 1.
   - O **LSTM Autoencoder ($L=5$)** e o **Conv1D Autoencoder ($L=5$)** disparam no **arquivo `535` (`74,33 h` de RUL)**, que corresponde à **primeira janela completa de 5 arquivos** do conjunto de teste (`531..535`), atingindo **Recall de `100%`** e **`ROC-AUC = 1,0000`** entre validação e teste!
2. **Compressão Não-Linear no Gargalo ($11 \to 4$ dimensões latentes):**
   - Com camadas de ativação não-lineares (`GELU`), os Autoencoders aprendem uma variedade (*manifold*) curva muito mais justa em torno do regime saudável (`0..400`) do que o subespaço plano de 8 componentes do PCA. Assim, a mudança sutil na relação entre `b1_energy_high` e `b1_rms` entre os arquivos `532` e `600` já é suficiente para produzir um erro de reconstrução consistentemente acima do percentil 99 do treino.

---

### 2.2 Comparação entre as Arquiteturas de Autoencoders (`MLP` vs. `Conv1D` vs. `LSTM`)
- **Melhor Equilíbrio Global — `LSTM Autoencoder` ($L=5$):**
  - Combina **detecção imediata** (`idx=535`, `74,33 h` de RUL, `100%` Recall, `ROC-AUC = 1,0000`), **baixo falso alarme na validação** (`3,51%`, chegando a `1,32%` na seed 123) e a **maior monotonicidade de Spearman entre todas as redes de reconstrução (`0,8598 ± 0,0010`)**, pois a memória recorrente de 5 arquivos suaviza as oscilações do período de alisamento (*healing* em `750..880`).
- **Mais Rápido e Direto ($L=1$) — `MLP Autoencoder`:**
  - Com apenas `1.967` parâmetros e sem necessidade de janela temporal ($L=1$), crava o alarme no arquivo `532` (`74,83 h` de RUL) nas 3 seeds com `99,78%` de Recall e `ROC-AUC = 0,9989`.
- **Sensibilidade Excessiva na Validação — `Conv1D Autoencoder` ($L=5$):**
  - Embora atinja `ROC-AUC = 1,0000` (isto é, todas as janelas de degradação têm erro maior que todas as janelas saudáveis da validação!), o limiar automático fixado no percentil 99 do treino gerou `22,80%` de falso alarme na validação (`451..530`), pois as convoluções locais capturaram micro-padrões temporais específicos do início do teste (`0..400`). Isso demonstra na prática a utilidade do conjunto de validação para diagnosticar a calibração do limiar.

---

### 2.3 Trade-off: Detecção Precoce vs. Monotonicidade de RUL (Spearman)
1. **Fenômeno de *Healing* (Alisamento Temporário da Trinca) vs. Erro Quadrático (MSE):**
   Na mecânica de fadiga de rolamentos, após a abertura inicial da trinca (`idx=700..720`, onde a curtose e o erro de reconstrução dão um salto abrupto), a passagem repetida dos rolos chega a "lixar" e arredondar temporariamente as bordas vivas da micro-cratera entre os arquivos `750` e `880`, reduzindo momentaneamente os picos de vibração antes do lascamento severo (*spalling*) no `900+`. Como o score do PCA e dos Autoencoders é o **erro quadrático médio (MSE)** $(x - \hat{x})^2$, ele amplifica quadraticamente o pico em `705` e reflete a queda no trecho `750..880`. O **LSTM** atenua esse efeito via contexto temporal ($L=5$), elevando o Spearman de `0,7877` (MLP) para **`0,8598`** (LSTM).
2. **Suavidade Assintótica do Isolation Forest:**
   O score do Isolation Forest é baseado na profundidade média de isolamento nas árvores ($2^{-E(h(x))/c(n)} \in [0, 1]$), comprimindo variações extremas (`0,9001` de Spearman), porém ao custo de atrasar o 1º alarme em **~17 horas** em relação ao MLP e ao LSTM.

---

### 2.4 Metodologia da "Contribuição Relativa" e Por que `b1_energy_high` e `b1_rms` Dominam sobre `b1_energy_bpfo` no Alarme Inicial

#### A. Como a Contribuição Relativa foi Calculada (`src/models/anomaly_detector.py` e `src/models/autoencoder.py`)
- **No PCA e nos Autoencoders (`MLP`, `Conv1D`, `LSTM`) — Decomposição Exata do MSE:**
  A contribuição percentual da feature $j$ é a sua fração exata na soma dos erros quadráticos de reconstrução (no caso de janelas $L=5$, calculada sobre a média temporal dos $L$ passos da janela):
  $$\text{Contrib}_{\text{Recon}}(j) = 100 \times \frac{(x_j - \hat{x}_j)^2}{\sum_{k=1}^D (x_k - \hat{x}_k)^2}$$
- **No Isolation Forest — Ablação Marginal para o Baseline Saudável:**
  Mede a redução no score de anomalia ao substituir a feature $j$ pelo centro saudável ($0,0$ no espaço padronizado pelo `StandardScaler`):
  $$\Delta_j = \max\big(0,\; S(x) - S(x_{\text{com } x_j = 0})\big), \quad \text{Contrib}_{\text{IF}}(j) = 100 \times \frac{\Delta_j}{\sum_{k=1}^D \Delta_k}$$

#### B. Por que `b1_energy_high` (`53%–59%`) e `b1_rms` (`27%–35%`) dominam no 1º alarme?
Em todos os 3 Autoencoders (`MLP`, `Conv1D`, `LSTM`), **`b1_energy_high` (`2.000–5.000 Hz`) responde por `53% a 59%` do erro de reconstrução no instante do 1º alarme (`idx=532..535`)**, seguida por **`b1_rms` (`27% a 33%`)**.

| Feature (`Canal 1`) | Z-Score no Alarme do PCA (`idx=601`) | Z-Score no Alarme do IF (`idx=646`) | Comportamento Físico |
| :--- | :---: | :---: | :--- |
| **`b1_energy_high` (`2000–5000 Hz`)** | **`+19,62 σ`** | **`+26,62 σ`** | **1ª a disparar:** micro-impactos excitam a ressonância estrutural de alta frequência. |
| **`b1_rms` (Energia Global)** | **`+14,86 σ`** | **`+19,73 σ`** | Baseline saudável tinha $\sigma$ minúsculo (`0,0010 g`), tornando o z-score hipersensível. |
| **`b1_p2p` / `b1_peak`** | `+3,63 σ` / `+3,61 σ` | `+1,40 σ` / `+1,51 σ` | Picos transitórios dos impactos iniciais. |
| **`b1_energy_mid` (`500–2000 Hz`)** | `+2,11 σ` | `+2,38 σ` | Excitação de frequências intermediárias. |
| **`b1_energy_bpfo` (`216–256 Hz`)** | **`+0,04 σ`** | **`+2,95 σ`** | Na FFT direta bruta, só supera $+3\sigma$ quando a trinca se alarga (`idx >= 646`). |

Conforme estabelecido por Qiu et al. (2006) e Randall & Antoni (2011), um defeito incipiente na pista externa gera **pulsos curtos de choque** a cada passagem de esfera ($236,4\text{ Hz}$) que excitam primeiramente a **ressonância estrutural de alta frequência do mancal (`2.000–5.000 Hz` = `b1_energy_high`)**. Na FFT direta (sem demodulação por envelope de Hilbert), a energia fundamental em `216–256 Hz` (`b1_energy_bpfo`) cresce de `1,39×` no alarme do MLP (`idx=532`) para `2,93×` no alarme do IF (`idx=646`) e atinge **`5,15×`** no estágio severo (`idx=951`).
