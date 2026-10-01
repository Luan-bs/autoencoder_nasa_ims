# Contexto do Projeto — Autoencoder NASA IMS Bearings

> **Uso:** Este documento é o ponto de entrada para qualquer IA, colaborador ou
> ferramenta (NotebookLM, RAG, etc.) que precise entender o projeto sem contexto
> prévio. Ele descreve o **objetivo, os dados, o protocolo experimental e a
> estrutura de código** de forma autossuficiente.

---

## 1. O que este projeto faz

Detecção **não-supervisionada** de falhas em rolamentos industriais usando
**autoencoders** (redes neurais). A ideia central:

1. Treinar o autoencoder **somente com dados saudáveis** (início da vida do
   rolamento).
2. Na inferência, o autoencoder tenta reconstruir o sinal de entrada.
3. Quando o rolamento começa a se degradar, o sinal muda e o **erro de
   reconstrução aumenta** — esse erro é o sinal de anomalia.

O projeto **não é supervisionado**: não há rótulo por arquivo indicando
"saudável" ou "com falha". Os rótulos surgem de um protocolo temporal
definido manualmente após análise exploratória (EDA) e são congelados antes
de qualquer treino.

---

## 2. Dataset — NASA IMS Bearings

### Origem
- **Fonte:** [NASA Open Data Portal](https://data.nasa.gov/dataset/ims-bearings)
- **Provedor:** Center for Intelligent Maintenance Systems (IMS),
  University of Cincinnati / Rexnord Corp.
- **Referência canônica:** Hai Qiu, Jay Lee, Jing Lin. *"Wavelet Filter-based
  Weak Signature Detection Method and its Application on Roller Bearing
  Prognostics."* Journal of Sound and Vibration 289 (2006) 1066–1090.
- **Fonte de verdade local:**
  `data/raw/IMS/IMS/Readme Document for IMS Bearing Data.pdf`
  Em qualquer conflito de informação, o PDF vence.

### Setup experimental (igual nos 3 testes)
| Parâmetro | Valor |
|---|---|
| Rolamentos | 4× Rexnord ZA-2115 (dupla fileira) |
| Velocidade | 2.000 RPM constantes |
| Carga radial | 6.000 lbs constante |
| Lubrificação | Forçada |
| Acelerômetros | PCB 353B33 High Sensitivity Quartz ICP |
| DAQ | NI DAQ Card 6062E |
| Taxa de amostragem | **20.000 Hz** (declarado no PDF; valor canônico adotado) |
| Pontos por arquivo | 20.480 |
| Duração por arquivo | ~1,024 s (snapshot) |
| Formato | ASCII — cada linha é um ponto de vibração |
| Nome do arquivo | Timestamp da coleta: `YYYY.MM.DD.HH.MM.SS` |

> **Nota taxa:** Literatura secundária cita 20.480 Hz (20.480 pts / 1,024 s).
> O PDF diz 20 kHz. Adotado: **20.000 Hz** em todos os cálculos de frequência.
> Registrado em `configs/data.yaml` com nota explicativa.

### Os três testes (run-to-failure)

| Set | Arquivo RAR | Tamanho | Arquivos esperados | Canais | Falha ao final |
|---|---|---|---|---|---|
| **Set 1** | `1st_test.rar` | ~350 MB | **2.156** | **8** (x+y por rolamento) | B3 pista interna · B4 elemento rolante |
| **Set 2** | `2nd_test.rar` | ~82 MB | **984** | **4** (1 por rolamento) | **B1 pista externa** ← foco inicial |
| **Set 3** | `3rd_test.rar` | ~581 MB | **4.448** | **4** (1 por rolamento) | B3 pista externa |

#### Mapeamento de canais

**Set 1 (8 canais):**
```
Ch1, Ch2 → Bearing 1 (eixo x, eixo y)
Ch3, Ch4 → Bearing 2
Ch5, Ch6 → Bearing 3  ← falha pista interna
Ch7, Ch8 → Bearing 4  ← falha elemento rolante
```

**Sets 2 e 3 (4 canais):**
```
Ch1 → Bearing 1   (Set 2: FALHA outer race)
Ch2 → Bearing 2   (controle negativo)
Ch3 → Bearing 3   (Set 3: FALHA outer race)
Ch4 → Bearing 4   (controle negativo)
```

### Frequências teóricas de falha
Shaft speed = 33,33 Hz (2.000 RPM ÷ 60)

| Modo | Frequência |
|---|---|
| BPFO — Ball Pass Frequency Outer race | ~236 Hz |
| BPFI — Ball Pass Frequency Inner race | ~297 Hz |
| BSF×2 — Ball Spin Frequency × 2 | ~278 Hz |

### Lacunas de timestamp
Intervalos nominais entre arquivos: 10 min (5 min nos primeiros 43 do Set 1).
Gaps maiores que o nominal indicam **retomada no dia seguinte** (experimento
parado à noite). Janelas e sequências LSTM **nunca cruzam essas lacunas**.

---

## 3. Protocolo experimental (FIXO — igual para todos os modelos)

### 3.1 Modelos comparados
| Tipo | Arquitetura | Input |
|---|---|---|
| **MLP** | Autoencoder denso | janela de L arquivos achatada (vetor 1D) |
| **Conv1D** | Autoencoder convolucional 1D | janela de L arquivos como sequência (L × n_features) |
| **LSTM** | Autoencoder recorrente | mesmo que Conv1D |
| **PCA** | Baseline sklearn | mesmo pipeline de features |
| **Isolation Forest** | Baseline sklearn | mesmo pipeline de features |

### 3.2 Features (extração por arquivo, por canal)
Cada arquivo vira um vetor de features escalares:

| Feature | Domínio |
|---|---|
| RMS | tempo |
| Kurtosis | tempo |
| Peak | tempo |
| Crest Factor (peak/RMS) | tempo |
| Peak-to-peak | tempo |
| Energia em bandas FFT | frequência |

Bandas FFT configuradas em `configs/data.yaml`:
`low` (0–500 Hz), `mid` (500–2000 Hz), `high` (2000–5000 Hz),
`bpfo` (±8,5% de 236 Hz), `bpfi` (±8,5% de 297 Hz), `bsf2x` (±8,5% de 278 Hz).

**Fase 1:** 1 amostra = 1 arquivo (sem sub-janelas do sinal bruto).
Janela de entrada do modelo = L arquivos consecutivos (L configurável; L=1 como baseline).

### 3.3 Split temporal
```
[  TREINO  ][CINZA][  VAL  ][      TESTE      ]
  saudável   excl.  limiar   degradação/falha
```
- **Treino:** apenas dados saudáveis (início do teste).
- **Zona cinza:** excluída de tudo (transição indefinida).
- **Validação:** usada **só** para escolher o limiar de anomalia. Nunca para
  ajustar hiperparâmetros ou early stopping.
- **Teste:** avaliação final. Nunca olhado antes da inferência final.
- Cortes definidos em `configs/labeling.yaml` após EDA aprovado.
- **Congelado antes de qualquer treino.**

### 3.4 Normalização
- `StandardScaler` ajustado **somente no conjunto de treino**.
- Aplicado sem re-ajuste em val e teste.
- Salvo em `experiments/<run_id>/scaler.pkl`.

### 3.5 Limiar de anomalia
- Escolhido **somente na validação**, nunca olhando o teste.
- Estratégia padrão: percentil 99 do erro de reconstrução no treino.
- Configurável em `configs/eval.yaml`.

### 3.6 Seeds e reprodutibilidade
- Seeds: [42, 123, 456]. Todos os modelos rodam com todas as seeds.
- Resultado final: **média ± desvio padrão** de cada métrica.

### 3.7 Métricas reportadas (sempre todas)
| Métrica | Uso |
|---|---|
| **Recall** | Principal — fração de arquivos de falha corretamente detectados |
| Precision | Reportar |
| F1 | Reportar |
| PR-AUC | Independente de limiar |
| ROC-AUC | Independente de limiar |
| Taxa de falso alarme | Em dados saudáveis |
| Antecedência de detecção | Arquivos/horas antes do fim de vida no primeiro alarme sustentado |
| Accuracy | Reportar apenas — **NUNCA** usar para selecionar modelo |
| Balanced Accuracy | Reportar apenas — **NUNCA** usar para selecionar modelo |

> **ALERTA automático:** se recall < accuracy em qualquer run, marcar
> "ALERTA" no relatório. Isso indica desbalanceamento mascarando a performance.

---

## 4. Estrutura de pastas

```
autoencoder_nasa_ims/
│
├── configs/
│   ├── data.yaml          ← parâmetros do dataset (FONTE: PDF README)
│   ├── labeling.yaml      ← split temporal — CONGELADO antes do treino
│   └── eval.yaml          ← protocolo de avaliação fixo para todos os modelos
│
├── data/
│   ├── raw/               ← IMUTÁVEL — nunca modificar
│   │   └── IMS/IMS/
│   │       ├── 1st_test.rar
│   │       ├── 2nd_test.rar
│   │       ├── 3rd_test.rar
│   │       └── Readme Document for IMS Bearing Data.pdf
│   ├── interim/           ← arquivos ASCII extraídos + logs de extração/validação
│   │   ├── 2nd_test/      ← arquivos nomeados YYYY.MM.DD.HH.MM.SS
│   │   └── 2nd_test_extraction.log
│   └── processed/
│       ├── train/         ← features prontas, split treino
│       ├── val/
│       └── test/
│
├── docs/
│   ├── data.md            ← documentação do dataset
│   └── project_context.md ← este arquivo (contexto completo do projeto)
│
├── experiments/
│   ├── registry.csv       ← uma linha por run (tracking central)
│   └── <run_id>/          ← por run: config.yaml, metrics.json, pesos, figuras
│       ├── config.yaml    ← snapshot de todos os configs no momento do treino
│       ├── metrics.json
│       ├── scaler.pkl
│       ├── model_weights.pt
│       ├── confusion_matrix.png
│       ├── pr_curve.png
│       ├── roc_curve.png
│       └── reconstruction_error_timeline.png
│
├── notebooks/
│   ├── 01_EDA.ipynb               ← exploração, visualização de sinais, lacunas
│   ├── 02_feature_engineering.ipynb
│   ├── 03_autoencoder_training.ipynb
│   └── 04_anomaly_detection.ipynb
│
├── src/
│   ├── data/
│   │   ├── extractor.py   ← extrai RARs, valida contagem, gera log
│   │   ├── loader.py      ← lê ASCII, parseia timestamps, detecta lacunas
│   │   ├── preprocessor.py← normalização, windowing, split
│   │   └── features.py    ← RMS, kurtosis, FFT bands
│   ├── models/
│   │   ├── autoencoder.py ← MLP, Conv1D, LSTM (mesma interface)
│   │   └── anomaly_detector.py
│   ├── training/
│   │   ├── trainer.py
│   │   └── callbacks.py
│   └── utils/
│       ├── metrics.py
│       ├── visualization.py
│       └── seed.py
│
├── tests/
├── conftest.py            ← adiciona src/ ao sys.path para pytest
├── pyproject.toml         ← dependências + PyTorch CUDA 12.8
└── uv.lock                ← versões exatas de todas as dependências
```

---

## 5. Ambiente de desenvolvimento

### Máquinas
| Ambiente | GPU | Uso |
|---|---|---|
| Notebook | Intel Graphics (CPU only) | Desenvolvimento, EDA, testes unitários |
| PC Desktop | **NVIDIA RTX 3060 12 GB VRAM** | Treino dos modelos |

### Stack
| Componente | Versão mínima |
|---|---|
| Python | 3.12 |
| Gerenciador de pacotes | uv 0.12+ |
| PyTorch | 2.7.1 — wheel CUDA 12.8 |
| scikit-learn | 1.7.0 (PCA, IsolationForest, métricas) |
| numpy, pandas, scipy | 2.5+, 3.0+, 1.15+ |
| pyyaml | 6.0+ (leitura de configs/) |

### Instalação
```bash
# Único comando — instala tudo, inclusive PyTorch CUDA 12.8
uv sync

# Verificar device
uv run python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

O `device` é detectado em runtime: `cuda` no PC com 3060, `cpu` no notebook.
Nenhuma mudança de código necessária entre os dois ambientes.

---

## 6. Gestão de experimentos

Cada run de treino tem um **run_id único**: `YYYYMMDD-HHMM_{modelo}_{set}_seed{seed}`

Exemplo: `20241028-1430_mlp_2nd_test_seed42`

### Regras obrigatórias
1. Rodar `git status` antes de treinar — **abortar se houver mudanças não commitadas**.
2. Cada run salva em `experiments/<run_id>/` um snapshot completo:
   config, seed, commit, hash dos dados, versões das libs, métricas, figuras,
   scaler, pesos do modelo.
3. `experiments/registry.csv` tem uma linha por run com as métricas principais.
4. **Nunca modificar runs antigos.** Correção = novo run.
5. **Uma mudança por experimento.** Nunca alterar vários fatores de uma vez.
6. Hiperparâmetros sempre em `configs/*.yaml`, nunca hardcoded.

---

## 7. Regras invioláveis do protocolo

| Regra | Detalhes |
|---|---|
| Split sempre temporal | Nunca aleatório |
| Scaler fit só no treino | Nunca toca val/teste |
| Limiar escolhido só na val | Nunca olhar o teste para isso |
| labeling.yaml congelado | Antes do primeiro treino |
| Mesmo protocolo para todos | MLP, Conv1D, LSTM, PCA, IF — comparáveis |
| data/raw imutável | Nunca extrair para dentro de raw/ |
| Não remover outliers de degradação | São o sinal de interesse |
| Recall é a métrica principal | Accuracy não seleciona nada |

---

## 8. Fase atual do projeto

| Etapa | Status |
|---|---|
| Estrutura de pastas e configs | ✅ Concluído |
| pyproject.toml + dependências | ✅ Concluído (aguardando `uv sync`) |
| `src/data/extractor.py` | ✅ Implementado — aguardando execução aprovada |
| Extração do `2nd_test.rar` | ⏳ Aguardando aprovação para rodar |
| `src/data/loader.py` | 🔲 A implementar |
| `01_EDA.ipynb` | 🔲 A implementar |
| `configs/labeling.yaml` preenchido | 🔲 Aguarda EDA |
| Feature extraction | 🔲 A implementar |
| Modelos (MLP, Conv1D, LSTM) | 🔲 A implementar |
| Baselines (PCA, IF) | 🔲 A implementar |
| Treino e avaliação | 🔲 A implementar |

---

## 9. Glossário

| Termo | Significado |
|---|---|
| **Autoencoder** | Rede neural que comprime e reconstrói o input. Erro de reconstrução alto = anomalia |
| **Run-to-failure** | Experimento que roda até o rolamento falhar naturalmente |
| **BPFO** | Ball Pass Frequency Outer race — frequência de defeito na pista externa |
| **BPFI** | Ball Pass Frequency Inner race — pista interna |
| **BSF** | Ball Spin Frequency — elemento rolante |
| **RUL** | Remaining Useful Life — vida útil restante |
| **PHM** | Prognostics and Health Management |
| **run_id** | Identificador único de um experimento de treino |
| **Zona cinza** | Período entre dados claramente saudáveis e claramente degradados; excluído da avaliação |
| **Alarme sustentado** | N arquivos consecutivos acima do limiar (N configurável em eval.yaml) |
| **Scaler** | StandardScaler do scikit-learn; ajustado só no treino |
