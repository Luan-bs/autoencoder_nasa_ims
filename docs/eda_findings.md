# Análise Exploratória (EDA) — Features Inferidas e Mapeamento de Zonas Temporais

Este documento registra os resultados, formulações matemáticas e decisões tomadas durante a etapa de **Análise Exploratória de Dados (EDA)** do projeto `autoencoder_nasa_ims`, focando no **Set 2 (`2nd_test`)**.

---

## 1. Setup Experimental e Integridade dos Dados

- **Rolamentos:** 4× Rexnord ZA-2115 montados no mesmo eixo.
- **Carga radial:** 6.000 lbs.
- **Velocidade de rotação:** 2.000 RPM constantes ($f_r \approx 33,33\text{ Hz}$).
- **Taxa de amostragem:** 20.000 Hz ($20.480$ pontos por canal por arquivo $\approx 1,024\text{ s}$).
- **Total de arquivos no Set 2:** **984 snapshots** (coletados de 12/02/2004 10:32:39 a 19/02/2004 06:22:39).
- **Intervalo nominal:** 10 minutos contínuos.
- **Detecção de lacunas (gaps > 15 min):** **0 lacunas**. O Set 2 foi coletado ininterruptamente por 163,8 horas (~6,8 dias).
- **Mapeamento dos canais:**
  - **Canal 1 (Rolamento 1):** Sofreu falha na **pista externa (Outer Race)** ao final da vida útil. Foco primário da detecção.
  - **Canais 2, 3 e 4 (Rolamentos 2, 3 e 4):** Controles negativos que permaneceram íntegros durante todo o teste.

---

## 2. Formulações Matemáticas das Features Inferidas

Cada arquivo ASCII contendo $N = 20.480$ amostras por canal é processado para extrair 11 features escalares por rolamento (44 features no total por snapshot).

### A. Domínio do Tempo

1. **RMS (Root Mean Square):**
   $$\text{RMS} = \sqrt{\frac{1}{N} \sum_{i=1}^N x_i^2}$$
   *Interpretação mecânica:* Medida direta da energia contínua e magnitude vibratória global. É a métrica clássica de monitoramento de condição, porém reage de forma mais pronunciada em estágios médios a avançados de degradação.

2. **Pico Máximo Absoluto (Peak):**
   $$\text{Peak} = \max_{i=1..N} |x_i|$$
   *Interpretação mecânica:* Excursão máxima instantânea registrada pelo acelerômetro durante o choque das esferas contra imperfeições na pista.

3. **Pico a Pico (Peak-to-Peak / P2P):**
   $$\text{P2P} = \max(x) - \min(x)$$
   *Interpretação mecânica:* Amplitude total dinâmica do ciclo de vibração no snapshot de 1 segundo.

4. **Fator de Crista (Crest Factor):**
   $$\text{CF} = \frac{\text{Peak}}{\text{RMS}}$$
   *Interpretação mecânica:* Relação entre a amplitude de pico e o nível médio contínuo de energia. Indicador altamente sensível ao início da formação de microtrincas, pois choques isolados elevam o pico antes do aumento global do RMS.

5. **Curtose (Pearson Kurtosis):**
   $$\text{Kurtosis} = \frac{\frac{1}{N} \sum_{i=1}^N (x_i - \bar{x})^4}{\left(\frac{1}{N} \sum_{i=1}^N (x_i - \bar{x})^2\right)^2}$$
   *Interpretação mecânica:* Grau de achatamento da distribuição de amplitudes. Para um rolamento perfeitamente saudável (distribuição normal), $\text{Kurtosis} \approx 3,0$. O surgimento de impactos periódicos gera caudas pesadas na distribuição, fazendo a curtose saltar rapidamente para $> 4,5$.

---

### B. Domínio da Frequência (FFT)

Com amostragem $f_s = 20.000\text{ Hz}$ e $N = 20.480$ pontos, a resolução espectral é:
$$\Delta f = \frac{f_s}{N} = \frac{20000}{20480} \approx 0,97656\text{ Hz}$$

O espectro unilateral de magnitude é dado por $|X(f_k)| = |\text{rfft}(x)|$. A energia integrada em uma faixa $[f_{\min}, f_{\max}]$ é calculada como:

$$\text{Energy}_{\text{band}} = \sum_{f_k \in [f_{\min}, f_{\max}]} |X(f_k)|^2$$

As bandas canônicas configuradas em `configs/data.yaml` são:
- **`low`** $[0, 500]\text{ Hz}$ — Dinâmica de rotação do eixo e harmônicos fundamentais ($1\times, 2\times, 3\times$ RPM $\approx 33,33\text{ Hz}$).
- **`mid`** $[500, 2000]\text{ Hz}$ — Faixa média de vibração e ressonâncias de carcaça.
- **`high`** $[2000, 5000]\text{ Hz}$ — Frequências estruturais de alta velocidade e atrito de contato.
- **`bpfo`** $[216, 256]\text{ Hz}$ — Frequência teórica de passagem de esferas na pista externa (**$\text{BPFO} \approx 236,4\text{ Hz} \pm 8,5\%$**).
- **`bpfi`** $[272, 322]\text{ Hz}$ — Frequência teórica da pista interna ($\approx 296,9\text{ Hz} \pm 8,5\%$).
- **`bsf2x`** $[255, 302]\text{ Hz}$ — Duplo da rotação dos corpos rolantes ($\approx 278,4\text{ Hz} \pm 8,5\%$).

---

## 3. Mapeamento das Zonas Temporais (`configs/labeling.yaml`)

O protocolo experimental adotado é **estritamente temporal**, proibindo embaralhamento aleatório para evitar vazamento de dados (*data leakage*).

Diagrama de divisão das partições temporais ao longo dos 984 arquivos do Set 2:

```
[       TREINO       ][ CINZA ][     VAL     ][             TESTE              ][OFF]
 0 ────────────── 400   401─450   451 ─── 530   531 ───────────────────────── 981 982─983
  Saudável estável      Buffer    Calibração    Degradação Incipiente e Falha    Desligado
```

### Detalhamento e Justificativa de Cada Zona:

1. **Zona de Treino (`[0, train_end_idx = 400]`):**
   - **401 arquivos** (~66,8 horas de corrida contínua).
   - O sinal de todos os 4 rolamentos opera em regime estacionário saudável ($\text{RMS}_{\text{B1}} \approx 0,077\text{ g}$, $\text{Kurtosis}_{\text{B1}} \approx 3,5$).
   - O modelo autoencoder é treinado **exclusivamente com esses dados** para aprender a reconstrução do estado saudável.
   - O `StandardScaler` do scikit-learn é ajustado (*fit*) apenas nessa partição.

2. **Zona Cinza (`[grey_start_idx = 401, grey_end_idx = 450]`):**
   - **50 arquivos** (~8,3 horas).
   - Zona de segurança / resguardo temporal entre treino e validação.
   - **Totalmente excluída** de treino, validação e teste. Impede qualquer contaminação temporal ou descontinuidade de fronteira.

3. **Zona de Validação (`[val_start_idx = 451, val_end_idx = 530]`):**
   - **80 arquivos** (~13,3 horas).
   - Precede imediatamente o surgimento das primeiras microtrincas na pista externa.
   - Utilizada **estritamente para calibrar o limiar de anomalia** (ex.: percentil 99 do erro de reconstrução no conjunto de treino avaliado na validação).
   - **Nunca** utilizada para ajustar hiperparâmetros ou arquitetura do modelo.

4. **Zona de Teste / Degradação / Falha (`[test_start_idx = 531, 981]`):**
   - **451 arquivos** (~75,1 horas).
   - **Arquivo 532 (~88,6 h de teste):** A literatura canônica (Hai Qiu et al., 2006) e a análise espectral confirmam o surgimento da assinatura de falha na pista externa (aumento sutil de energia em 236,4 Hz).
   - **Arquivo 700 em diante:** A degradação acelera fortemente; a curtose ultrapassa $5,5$ e o RMS de B1 sobe para $> 0,17\text{ g}$.
   - **Arquivo 981:** Colapso físico final do rolamento 1 com $\text{RMS} > 0,72\text{ g}$ e vibração descontrolada.
   - Esta partição avalia as métricas cegas do modelo (**Recall, Precision, F1, PR-AUC, ROC-AUC e antecedência de detecção**).

5. **Parada Experimental / Motor Desligado (`[shutdown_idx = 982, 983]`):**
   - Nos dois últimos arquivos do dataset, o motor elétrico da bancada foi desligado após a destruição da peça.
   - Os 4 canais caem abruptamente para $\text{RMS} < 0,002\text{ g}$ (ruído de fundo elétrico).
   - Esses arquivos devem ser descartados da avaliação de performance do modelo.

---

## 4. Estrutura de Código Criada

1. **`src/data/loader.py`:**
   - `list_ims_files`: localiza e ordena cronologicamente os arquivos ASCII.
   - `parse_timestamp`: conversão precisa de `YYYY.MM.DD.HH.MM.SS`.
   - `detect_gaps`: monitoramento automático de lacunas temporais.
   - `load_snapshot`: carregamento em matriz NumPy $(20480, 4)$.
   - `load_timeline_metadata`: geração da tabela de metadados temporais.

2. **`src/data/features.py`:**
   - `compute_time_features`: implementação das fórmulas de RMS, Peak, P2P, Crest Factor e Curtose.
   - `compute_fft_features`: cálculo espectral via `rfft` e integração de energia nas bandas configuradas.
   - `extract_features_df`: extração vetorizada de todos os snapshots com sistema de cache automático (`data/processed/2nd_test_features.csv`).

3. **`src/utils/visualization.py`:**
   - `plot_dataset_timeline_zones`: gera o gráfico multi-eixo com todas as features dos 4 canais e as áreas coloridas sombreadas das zonas temporais.
   - `plot_raw_signal_comparison`: exibe as formas de onda brutas (Saudável, Degradação e Falha).
   - `plot_fft_spectrum_comparison`: plota os espectros de frequência destacando a BPFO (~236,4 Hz) e seus harmônicos $2\times$ e $3\times$.

4. **Notebooks:**
   - `notebooks/01_EDA.ipynb` e `notebooks/01_EDA_01.ipynb`: notebooks completos, executados e com todas as saídas e visualizações salvas.

---

## 5. Artefatos Gráficos Salvos

As imagens em alta resolução estão armazenadas em `reports/` (na raiz do projeto):
- `reports/eda_zones_timeline.png` — Visualização completa da timeline com as 4 features e marcação visual das zonas.
- `reports/eda_waveform_comparison.png` — Comparação de forma de onda temporal (Snapshots 100, 600 e 975).
- `reports/eda_fft_spectrum.png` — Comparação de espectro FFT evidenciando a BPFO (~236,4 Hz).
- `reports/evolucao_completa_rolamento_1.png` — Painel evolutivo das métricas físicas do Rolamento 1 ao longo do teste.

---

## 6. Configuração Aprovada em `configs/labeling.yaml`

```yaml
sets:
  "2nd_test":
    train_end_idx: 400
    grey_start_idx: 401
    grey_end_idx: 450
    val_start_idx: 451
    val_end_idx: 530
    test_start_idx: 531
    test_end_idx: 981
    shutdown_idx: 982
    note: "Defined via 01_EDA.ipynb. Incipient outer race defect (BPFO ~236.4 Hz) starts around file 532; motor turned off at 982."
    approved_by: "Owner + EDA (2026-10-16)"
    frozen: true
```

---

## 7. Investigação de Comportamentos Físicos e Anomalias de Canal (Notas Metodológicas)

### 7.1 Por que o Rolamento 3 (Controle Negativo) apresenta maior ruído de base que o Rolamento 1?
Ao observar os gráficos da EDA no período de treino saudável (`0..400`), nota-se que o **Rolamento 3 (`b3`)** apresenta métricas absolutas consistentemente superiores às do **Rolamento 1 (`b1`)**:

| Canal | RMS Médio (`0..400`) | Curtose Média / Máx (`0..400`) | Crest Factor Médio / Máx | Energia Banda BPFO Média / Máx |
| :--- | :---: | :---: | :---: | :---: |
| **Bearing 1 (`b1` — Falha no final)** | `0,0772 g` | `3,43` / `4,16` | `5,23` / `7,72` | `7.835` / `22.624` |
| **Bearing 2 (`b2` — Controle)** | `0,0946 g` | `3,19` / `3,51` | `4,99` / `7,93` | `21.642` / `77.860` |
| **Bearing 3 (`b3` — Controle)** | **`0,1019 g`** | **`4,53` / `6,70`** | **`7,56` / `12,05`** | **`75.908` / `238.475`** |
| **Bearing 4 (`b4` — Controle)** | `0,0555 g` | `3,13` / `3,81` | `4,70` / `9,24` | `4.175` / `8.199` |

A análise espectral e temporal dos dados esclarece as três hipóteses levantadas:
1. **Descarte de erro de mapeamento de canal:** O sinal de `b3` é **estritamente estacionário** ao longo de quase toda a vida do teste: seu RMS médio no treino (`0..400`) é `0,1019 ± 0,0028 g`, na validação (`451..530`) é `0,0994 ± 0,0013 g` e no teste (`531..900`) é `0,1017 ± 0,0037 g`. Ele não degrada no `2nd_test`. Apenas nos últimos ~80 arquivos (`901..981`), quando o Rolamento 1 entra em colapso severo (`RMS > 0,72 g`), a vibração extrema se propaga mecanicamente pelo eixo compartilhado (*cross-talk* estrutural), elevando o RMS de todos os demais canais (`b2`, `b3`, `b4`) e derrubando a curtose de `b3` para `3,12`.
2. **Causa física da energia alta na banda `bpfo` (`[216, 256] Hz`) em `b3`:** Inspecionando a FFT do Canal 3 no início saudável (`idx=100`), as frequências dominantes são `[985,4 Hz, 231,4 Hz, 492,2 Hz, 984,4 Hz]`. O pico estacionário intenso em **`231,4 Hz`** corresponde exatamente ao **7º harmônico da rotação do eixo** ($7 \times 33,06\text{ Hz} = 231,45\text{ Hz}$, acompanhado de $15\times = 492,2\text{ Hz}$ e $30\times = 985,4\text{ Hz}$). Como a banda `bpfo` compreende `216–256 Hz`, esse harmônico rotacional/estrutural da posição do mancal 3 (onde o mecanismo de carga radial de 6.000 lbs atua diretamente na bancada) cai dentro da janela e infla a energia absoluta desde o instante zero.
3. **Folga/impulsividade mecânica nativa de `b3`:** A posição do mancal 3 na bancada IMS sofre maior solicitação mecânica e desalinhamento residual (não por acaso, o Rolamento 3 foi justamente o que falhou no `1st_test` e no `3rd_test`).
4. **Consequência prática para modelagem:** Esse fenômeno demonstra por que **regras de limiar absoluto fixo** (ex.: *"Curtose > 4,5 = falha"*) falham na indústria. Em nosso pipeline não supervisionado, cada canal é padronizado pelo seu próprio `StandardScaler` ajustado no respectivo baseline saudável (`0..400`), aprendendo o que é normal para aquela posição específica de sensor.

---

### 7.2 Curtose Saudável Empírica (~3,43) vs. Valor Teórico Gaussiano (3,00)
- Embora uma distribuição normal teórica possua Curtose de Pearson igual a $3,00$, o baseline saudável real do Rolamento 1 apresenta média **$3,43$** (variando até $4,16$ no treino).
- *Nota terminológica:* Valores de curtose de Pearson **acima de $3,0$** caracterizam uma distribuição **leptocúrtica** (caudas mais pesadas e pico mais agudo que a Gaussiana; valores $< 3,0$ seriam platicúrticos). Em rolamentos reais sob carga radial elevada (6.000 lbs), a passagem contínua dos corpos rolantes e micro-imperfeições superficiais geram impulsividade de fundo mesmo em peças novas.
- Por esse motivo, **nenhum detector do projeto utiliza `Kurtosis = 3.0` como referência**: toda a detecção baseia-se na distribuição empírica aprendida dos arquivos `0..400` e no percentil 99 do erro de reconstrução/isolamento.


