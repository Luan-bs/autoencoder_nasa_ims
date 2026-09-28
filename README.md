# Autoencoder NASA IMS — Bearing Anomaly Detection

Detecção não-supervisionada de anomalias em rolamentos via autoencoder.  
Dataset: [NASA IMS Bearings](https://data.nasa.gov/dataset/ims-bearings) — Center for Intelligent Maintenance Systems (IMS), University of Cincinnati.

> **Projeto de pesquisa.** Prioridade: reprodutibilidade e comparabilidade entre modelos.

---

## Objetivo

Treinar autoencoders **apenas com dados saudáveis** e detectar falha pelo erro de reconstrução.  
Comparar três arquiteturas (MLP denso, Conv1D, LSTM) e dois baselines (PCA, Isolation Forest) sob protocolo experimental idêntico.

## Estrutura do projeto

```
autoencoder_nasa_ims/
├── configs/
│   ├── data.yaml          # Parâmetros do dataset (fonte: PDF README)
│   ├── labeling.yaml      # Split temporal treino/val/teste — CONGELADO antes do treino
│   └── eval.yaml          # Protocolo de avaliação fixo para todos os modelos
│
├── data/
│   ├── raw/               # IMUTÁVEL — nunca modificar
│   ├── interim/           # Arquivos ASCII extraídos + logs de validação
│   └── processed/         # Features por arquivo prontas para treino
│
├── experiments/
│   ├── registry.csv       # Uma linha por run (tracking central)
│   └── <run_id>/          # Artefatos de cada run (config, pesos, métricas, figuras)
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_autoencoder_training.ipynb
│   └── 04_anomaly_detection.ipynb
│
├── src/autoencoder_nasa_ims/
│   ├── data/              # loader, preprocessor, features
│   ├── models/            # MLP, Conv1D, LSTM autoencoders + anomaly_detector
│   ├── training/          # trainer, callbacks, experiment logging
│   └── utils/             # visualisation, metrics, seed management
│
├── tests/
├── pyproject.toml
└── uv.lock
```

## Setup

```bash
# Instalar dependências (PyTorch CUDA 12.8 — funciona em CPU e RTX 3060)
uv sync

# Verificar device disponível
uv run python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## Protocolo experimental

- Split **temporal** (nunca aleatório): treino (saudável) → val → teste (degradação/falha)
- Normalização: `fit` apenas no treino; scaler salvo junto ao modelo
- Limiar de anomalia: escolhido **só na validação**
- Seeds fixas (42, 123, 456); reportar média ± desvio
- Cada run tem `run_id` único; resultados em `experiments/<run_id>/`
- `experiments/registry.csv` centraliza todas as runs

Ver [`configs/eval.yaml`](configs/eval.yaml) para o protocolo completo.

## Dataset

Ver [`data/data.md`](data/data.md) para documentação completa do dataset IMS.  
Foco inicial: **2nd_test** (984 arquivos, 4 canais, falha no rolamento 1 — pista externa).
