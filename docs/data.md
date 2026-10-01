# Dataset — NASA IMS Bearings

**Fonte:** [NASA Open Data Portal](https://data.nasa.gov/dataset/ims-bearings)  
**Provedor:** Center for Intelligent Maintenance Systems (IMS), University of Cincinnati / Rexnord Corp.  
**Referência:** Hai Qiu, Jay Lee, Jing Lin. *"Wavelet Filter-based Weak Signature Detection Method and its Application on Roller Bearing Prognostics."* Journal of Sound and Vibration 289 (2006) 1066–1090.

---

## Setup experimental

| Parâmetro | Valor |
|---|---|
| Rolamentos | 4× Rexnord ZA-2115 (dupla fileira) |
| Velocidade do eixo | 2.000 RPM constantes |
| Carga radial | 6.000 lbs constante |
| Lubrificação | Forçada |
| Acelerômetros | PCB 353B33 High Sensitivity Quartz ICP |
| DAQ | NI DAQ Card 6062E |
| Taxa de amostragem | **20.000 Hz** (declarado no README PDF) |
| Pontos por arquivo | 20.480 (snapshot de 1 segundo) |
| Formato | ASCII (cada linha = 1 ponto) |

> **Nota sobre taxa de amostragem:** a literatura secundária cita 20.480 Hz
> (20.480 pts ÷ 1,024 s). O README PDF declara explicitamente 20 kHz.
> Valor canônico adotado: **20.000 Hz**. Registrado em `configs/data.yaml`.

---

## Sets disponíveis

### Set 1 — `1st_test.rar` (~350 MB)
| Campo | Valor |
|---|---|
| Duração | 22 Out 2003 12:06:24 → 25 Nov 2003 23:39:56 |
| Arquivos esperados | **2.156** |
| Canais | **8** (x e y por rolamento) |
| Mapeamento | B1→Ch1+2 · B2→Ch3+4 · B3→Ch5+6 · B4→Ch7+8 |
| Intervalo | 10 min (primeiros **43 arquivos**: 5 min) |
| Falha ao final | B3 pista interna · B4 elemento rolante |

### Set 2 — `2nd_test.rar` (~82 MB) ← **FOCO INICIAL**
| Campo | Valor |
|---|---|
| Duração | 12 Fev 2004 10:32:39 → 19 Fev 2004 06:22:39 |
| Arquivos esperados | **984** |
| Canais | **4** (1 por rolamento) |
| Mapeamento | B1→Ch1 · B2→Ch2 · B3→Ch3 · B4→Ch4 |
| Intervalo | 10 min |
| Falha ao final | **B1 pista externa** (Ch1 = canal do defeito) |

### Set 3 — `3rd_test.rar` (~581 MB)
| Campo | Valor |
|---|---|
| Duração | 04 Mar 2004 09:27:46 → 04 Abr 2004 19:01:57 |
| Arquivos esperados | **4.448** |
| Canais | **4** (1 por rolamento) |
| Mapeamento | B1→Ch1 · B2→Ch2 · B3→Ch3 · B4→Ch4 |
| Intervalo | 10 min |
| Falha ao final | B3 pista externa |

---

## Frequências teóricas de falha

Velocidade do eixo: 33,33 Hz (2.000 RPM ÷ 60)

| Modo | Símbolo | Frequência |
|---|---|---|
| Ball Pass Frequency Outer race | BPFO | ~236 Hz |
| Ball Pass Frequency Inner race | BPFI | ~297 Hz |
| Ball Spin Frequency × 2 | BSF×2 | ~278 Hz |

---

## Estrutura de arquivos

```
data/
├── raw/          # IMUTÁVEL — nunca modificar
│   └── IMS/IMS/
│       ├── 1st_test.rar
│       ├── 2nd_test.rar
│       ├── 3rd_test.rar
│       └── Readme Document for IMS Bearing Data.pdf  ← fonte de verdade
├── interim/      # Arquivos ASCII extraídos + logs de validação
│   ├── 2nd_test/
│   ├── 1st_test/
│   └── 3rd_test/
└── processed/    # Features prontas para treino (geradas por src/pipeline)
    ├── train/
    ├── val/
    └── test/
```

---

## Regras de uso

- `data/raw` é **somente leitura**. Nunca extrair para dentro de `raw/`.
- Lacunas de timestamp > intervalo nominal indicam retomada no dia seguinte.  
  Janelas (e sequências LSTM) **nunca cruzam lacunas**.
- Labeling (split treino/val/teste) definido em `configs/labeling.yaml`  
  e **congelado antes de qualquer treino**.
- Verificar contagem real de arquivos após extração vs. valores esperados acima.