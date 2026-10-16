# Comparativo dos Baselines de Detecção de Anomalias — Set 2 (`2nd_test`)

Resultados consolidados em 3 seeds independentes (`[42, 123, 456]`), seguindo o protocolo de avaliação não supervisionada em PHM (*Prognostics and Health Management*).

---

## Tabela Comparativa de Métricas (Média ± Desvio Padrão)

| Métrica | PCA (Reconstrução Linear) | Isolation Forest (Árvores) | Interpretação PHM |
| :--- | :---: | :---: | :--- |
| **1º Alarme Sustentado (`first_alarm_idx`)** | **Arquivo `601,0 ± 0,0`** | Arquivo `636,0 ± 14,1` | **PCA detectou 35 arquivos (~5,8h) mais cedo** que o Isolation Forest. |
| **RUL no 1º Alarme (`detection_lead_hours`)** | **`63,33 ± 0,00 h` (`380` arq.)** | `57,50 ± 2,36 h` (`345` arq.) | O PCA avisa **2,64 dias antes** do colapso final do rolamento (`idx=981`). |
| **Taxa de Falso Alarme (`val`)** | **`1,25% ± 0,00%`** | `2,08% ± 0,59%` | Ambos mantêm baixíssimo falso alarme na validação saudável. |
| **Recall na Degradação (`test`)** | **`86,25% ± 0,00%`** | `79,08% ± 1,50%` | O PCA captura quase toda a fase de degradação após a transição inicial. |
| **Contraste Teste/Val (`separation_ratio`)** | **`47.920×`** | `1,48×` | O erro quadrático de reconstrução do PCA explode na falha; o score de árvore satura perto de 1.0. |
| **Monotonicidade com RUL (`spearman`)** | `0,8291 ± 0,0000` | **`0,9001 ± 0,0104`** | Ambos apresentam forte correlação monotônica com o desgaste progressivo. |
| **ROC-AUC (Val vs Teste)** | `0,9838 ± 0,0000` | `0,9839 ± 0,0016` | Excelente separabilidade global entre estado saudável e degradado. |
| **Top Features no Alarme (Explicabilidade)** | `b1_rms` (35,3%), `b1_energy_high` (27,1%), `b1_energy_bpfi` (9,3%) | `b1_kurtosis` (18,5%), `b1_rms` (18,5%), `b1_energy_high` (16,4%), `b1_energy_bpfo` (7,9%) | Confirmação física: no alarme do IF (`idx=646`), a energia na frequência teórica BPFO (~236,4 Hz) já é 2,93× maior que no estado saudável, chegando a 5,15× no estágio severo (`idx=951`). |

---

## Detalhes Metodológicos dos Modelos

- **PCA:** Ajustado exclusivamente no conjunto de treino saudável com variância explicada de 95% (8 componentes principais retidos). Score baseado no erro quadrático médio (MSE) de reconstrução.
- **Isolation Forest:** Ajustado com 200 estimadores e contaminação em modo automático. Score baseado no comprimento médio de isolamento invertido.
- **Limiar de Decisão:** 99º percentil do erro/score no conjunto de treino saudável, sem nunca observar os dados da partição de teste.

