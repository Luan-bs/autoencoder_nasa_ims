# Autoencoder NASA IMS — Bearing Anomaly Detection

Detecção não-supervisionada de anomalias em rolamentos industriais via autoencoders neurais e baselines clássicos.  
Dataset: [NASA IMS Bearings](https://data.nasa.gov/dataset/ims-bearings) — Center for Intelligent Maintenance Systems (IMS), University of Cincinnati.

> **Projeto de pesquisa em PHM (*Prognostics and Health Management*).** Prioridade: reprodutibilidade estrita, zero vazamento de dados (*data leakage*), explicabilidade física (FFT/BPFO + RUL) e comparabilidade justa entre modelos.

---

## Objetivo

1. Treinar modelos de detecção de anomalias **exclusivamente com dados saudáveis** do início de vida útil do rolamento.
2. Detectar o início da degradação pelo aumento do erro de reconstrução / score de anomalia e medir a **antecedência de detecção (RUL no 1º alarme sustentado)**.
3. Comparar três arquiteturas de Autoencoders (**MLP denso**, **Conv1D**, **LSTM**) e dois baselines (**PCA**, **Isolation Forest**) sob o mesmo protocolo experimental.

---

## Estrutura do Projeto

```text
autoencoder_nasa_ims/
├── configs/
│   ├── data.yaml                  # Parâmetros físicos do dataset e bandas FFT (fonte: PDF README)
│   ├── labeling.yaml              # Split temporal treino/cinza/val/teste — congelado pós-EDA
│   ├── eval.yaml                  # Protocolo de avaliação PHM e definição de métricas/seeds
│   └── models.yaml                # Hiperparâmetros dos baselines e Autoencoders (MLP, Conv1D, LSTM)
│
├── data/
│   ├── raw/                       # IMUTÁVEL — arquivos RAR originais e PDF README da NASA
│   ├── interim/                   # Snapshots ASCII extraídos (20.480 pontos/arquivo) + logs
│   └── processed/                 # Features extraídas e splits temporais materializados
│       ├── train/                 # 2nd_test_train.csv (saudável: 0..400)
│       ├── val/                   # 2nd_test_val.csv   (calibração de limiar: 451..530)
│       └── test/                  # 2nd_test_test.csv  (degradação/falha: 531..981)
│
├── docs/
│   ├── project_context.md         # Contexto geral, regras invioláveis e arquitetura do projeto
│   ├── data.md                    # Documentação técnica do dataset NASA IMS
│   ├── glossario.md               # Glossário de mecânica/vibração (FFT, BPFO, RUL) e Autoencoders
│   ├── eda_findings.md            # Relatório da EDA, fórmulas das features e justificativa de zonas
│   ├── baseline_comparison.md     # Comparativo global dos 5 modelos (PCA, IF, MLP, Conv1D, LSTM)
│   └── figures/                   # Gráficos da Análise Exploratória (EDA)
│
├── experiments/
│   ├── registry.csv               # Tabela central com histórico e métricas de todas as runs
│   ├── pca/                       # Runs por seed + summary_metrics.json (PCA)
│   ├── isolation_forest/          # Runs por seed + summary_metrics.json (Isolation Forest)
│   ├── mlp/                       # Runs por seed + summary_metrics.json (MLP Autoencoder)
│   ├── conv1d/                    # Runs por seed + summary_metrics.json (Conv1D Autoencoder)
│   └── lstm/                      # Runs por seed + summary_metrics.json (LSTM Autoencoder)
│
├── notebooks/
│   ├── 01_EDA.ipynb               # Notebook interativo de exploração de sinais, FFT e zonas
│   └── 03_deepod_benchmark.ipynb  # Benchmark de Deep SVDD, DIF e TranAD (deepod) vs. MLP Autoencoder
│
├── src/
│   ├── data/                      # extractor.py, loader.py, features.py, preprocessor.py
│   ├── models/                    # anomaly_detector.py (PCA, IF) + autoencoder.py (MLP, Conv1D, LSTM)
│   ├── training/                  # trainer.py (runner unificado) + callbacks.py
│   └── utils/                     # metrics.py, visualization.py, seed.py
│
├── pyproject.toml
└── uv.lock
```

---

## Setup e Execução

```powershell
# 1. Instalar dependências (PyTorch CUDA 12.8 — roda em GPU RTX 3060 ou CPU automaticamente)
uv sync

# 2. Executar os baselines clássicos (PCA e Isolation Forest) nas 3 seeds [42, 123, 456]
.venv\Scripts\python.exe -m src.training.trainer --model all_baselines --allow-dirty

# 3. Executar os Autoencoders neurais (MLP, Conv1D e LSTM) nas 3 seeds [42, 123, 456]
.venv\Scripts\python.exe -m src.training.trainer --model all_ae --allow-dirty

# 4. Executar todos os 5 modelos de uma vez
.venv\Scripts\python.exe -m src.training.trainer --model all --allow-dirty
```

---

## Protocolo Experimental

- **Split estritamente temporal** (definido em [`configs/labeling.yaml`](configs/labeling.yaml)):
  - `Treino (0..400)` → `Zona Cinza excluída (401..450)` → `Validação (451..530)` → `Teste/Degradação (531..981)` → `Desligamento excluído (982..983)`.
- **Normalização sem vazamento:** `StandardScaler` ajustado (`fit`) exclusivamente no treino e salvo em cada run (`scaler.pkl`).
- **Limiar de anomalia:** Percentil 99 do erro de reconstrução/isolamento saudável, sem jamais observar o conjunto de teste.
- **Reprodutibilidade:** 3 seeds fixas (`42, 123, 456`); resultados reportados como **média ± desvio padrão**.
- **Explicabilidade Física (FFT + RUL):** Cada run salva automaticamente o painel `explainability_fft_rul.png` (confirmando o crescimento de energia na frequência de defeito da pista externa **BPFO ≈ 236,4 Hz** e o ranking de contribuição das features) e `reconstruction_error_timeline.png` (com a contagem regressiva do **RUL** no instante do 1º alarme sustentado).

---

## Documentação e Resultados

- **Dataset:** [`docs/data.md`](docs/data.md)
- **Análise Exploratória e Fórmulas das Features:** [`docs/eda_findings.md`](docs/eda_findings.md)
- **Comparativo Completo dos 5 Modelos:** [`docs/baseline_comparison.md`](docs/baseline_comparison.md)
- **Protocolo de Avaliação:** [`configs/eval.yaml`](configs/eval.yaml)
