# Glossário — Mecânica/Vibração e Autoencoder

Material de apoio pessoal (não é para a IA executora). Objetivo: entender o "porquê" de cada termo usado no projeto `autoencoder_nasa_ims`.

---

## Parte 1 — Mecânica e vibração

### SHM — Simple Harmonic Motion (Movimento Harmônico Simples)
Movimento oscilatório periódico em que a força restauradora é proporcional ao deslocamento e aponta na direção oposta a ele. É o modelo matemático mais simples de vibração, descrito por uma função seno/cosseno.

**Relação com o projeto:** a vibração "saudável" de um rolamento se aproxima de uma soma de componentes quase senoidais, ligadas à rotação do eixo e dos componentes internos. Quando aparece uma falha, o sinal deixa de ser suave e ganha **impactos** (picos curtos e repetidos), que não se explicam mais por SHM puro. Entender SHM ajuda a entender o que é o "normal" que o autoencoder vai aprender.

### RPM e frequência do eixo
O eixo gira a 2000 RPM, ou seja, 2000/60 ≈ 33,33 Hz. Essa é a frequência fundamental de rotação.

**Relação com o projeto:** todas as frequências de falha (BPFO, BPFI, BSF) são múltiplos dessa frequência, calculados pela geometria do rolamento. Sem ela, os cálculos de BPFO/BPFI não fazem sentido.

### Rolamento (bearing), pista interna e pista externa (inner/outer race)
O rolamento tem um anel preso ao eixo (pista interna, gira), um anel preso à carcaça (pista externa, fixo), e esferas/rolos entre eles. Cada peça pode desenvolver defeito.

**Relação com o projeto:** o tipo de defeito muda a frequência do impacto gerado. É por isso que existe BPFO (defeito na pista externa) e BPFI (defeito na pista interna), com valores diferentes.

### BPFO / BPFI / BSF — frequências características de falha
São frequências teóricas, calculadas a partir da geometria do rolamento e da rotação, em que aparecem impactos repetidos quando existe defeito em cada parte específica (pista externa, pista interna ou elemento rolante).

**Relação com o projeto:** servem como **verificação física** dos resultados do modelo. Se o autoencoder disparar um alarme e, ao olhar o espectro (FFT) daquele trecho, aparecer um pico crescendo perto de 236 Hz (BPFO), isso confirma que a detecção faz sentido mecânico, e não é um artefato do modelo.

### RMS — Root Mean Square
Raiz quadrada da média dos quadrados do sinal. É uma medida de energia/amplitude geral da vibração ao longo do tempo.

**Relação com o projeto:** é uma das features mais simples e mais usadas em manutenção preditiva. RMS tende a subir conforme o rolamento degrada, mas sobe relativamente tarde (quando o dano já é considerável).

### Curtose (kurtosis)
Mede o quão "pontudo" é o sinal, ou seja, o quanto ele tem picos isolados em relação a um sinal gaussiano comum (curtose ≈ 3 para distribuição normal).

**Relação com o projeto:** impactos de defeito aparecem como picos raros e fortes no meio de um sinal mais calmo, o que eleva bastante a curtose. Ela costuma reagir **antes** do RMS, sendo um indicador mais sensível a falhas iniciais.

### Fator de crista (crest factor)
Razão entre o valor de pico do sinal e o RMS.

**Relação com o projeto:** outro indicador sensível a impactos isolados, parecido em propósito com a curtose, mas calculado de forma diferente. Costuma ser usado junto, não no lugar.

### FFT — Fast Fourier Transform / espectro de frequência
Transforma um sinal do domínio do tempo (amplitude por instante) para o domínio da frequência (energia por frequência).

**Relação com o projeto:** é a ferramenta que permite checar se o pico de energia está exatamente nas frequências BPFO/BPFI/BSF esperadas, ligando o resultado do modelo a uma causa física, e não só a um número de erro de reconstrução.

### Run-to-failure
Tipo de experimento em que a máquina roda continuamente até a peça falhar de verdade, sem interrupção ou troca preventiva.

**Relação com o projeto:** é por isso que o dataset IMS tem degradação gradual registrada, e não só dois estados (novo / quebrado). É o que possibilita treinar com o início saudável e testar no restante.

### RUL — Remaining Useful Life (vida útil restante)
Estimativa de quanto tempo (ou quantos ciclos) faltam até a peça falhar, a partir do estado atual medido.

**Relação com o projeto:** é um objetivo mais avançado que detecção de anomalia. Detecção de anomalia responde "está falhando?"; RUL responde "quanto tempo falta?". O projeto atual é de detecção, mas RUL é o próximo passo natural, caso queira estender depois.

### PHM — Prognostics and Health Management
É a área de pesquisa que reúne monitoramento de condição, detecção de falha e previsão de vida útil de equipamentos.

**Relação com o projeto:** é o nome do campo onde este trabalho se encaixa academicamente, útil para buscar mais artigos e situar a introdução do trabalho.

---

## Parte 2 — Autoencoder e aprendizado não supervisionado

### Autoencoder
Rede neural treinada para reconstruir sua própria entrada, passando por um "gargalo" (camada menor que a entrada) no meio.

**Relação com o projeto:** ao forçar os dados por um gargalo, o modelo é obrigado a aprender só os padrões mais importantes do sinal saudável. Dados fora desse padrão (falha) ficam mal reconstruídos.

### Encoder / decoder / bottleneck (gargalo) / espaço latente
Encoder comprime a entrada até o gargalo; decoder tenta reconstruir a entrada original a partir dele. O gargalo é a representação comprimida, também chamada de espaço latente.

**Relação com o projeto:** o tamanho do gargalo é uma decisão de projeto: pequeno demais perde informação até de dados saudáveis; grande demais deixa o modelo "decorar" tudo, inclusive falhas, o que destrói a detecção.

### Erro de reconstrução (reconstruction error)
Diferença entre a entrada original e a saída reconstruída pelo autoencoder, normalmente medida por MSE ou MAE.

**Relação com o projeto:** é a métrica central do método. Erro baixo = parecido com o que o modelo viu no treino (saudável). Erro alto = fora do padrão aprendido (possível falha).

### Aprendizado não supervisionado
Treino sem rótulos: o modelo aprende só a partir da estrutura dos dados, sem alguém dizer "isso é falha" e "isso é saudável" exemplo a exemplo.

**Relação com o projeto:** o autoencoder nunca vê exemplo de falha no treino. Ele só aprende "normal", por isso os rótulos existem apenas para avaliação (no `labeling.yaml`), nunca para treinar.

### Limiar (threshold) de anomalia
Valor de corte no erro de reconstrução, acima do qual uma amostra é classificada como anomalia.

**Relação com o projeto:** é a peça que transforma um número contínuo (erro) em uma decisão binária (saudável/falha). Escolhido só na validação, nunca olhando o teste, para não "trapacear" o resultado.

### Vazamento de dados (data leakage)
Quando informação do conjunto de teste (ou do futuro) influencia, de forma indevida, o treino ou a escolha de hiperparâmetros/limiar.

**Relação com o projeto:** é o erro mais comum e mais grave nesse tipo de estudo. Por isso o protocolo exige split temporal, normalização ajustada só no treino, e limiar escolhido só na validação.

### PCA — Principal Component Analysis
Técnica clássica (não é rede neural) que também comprime os dados em menos dimensões, de forma linear, e pode medir erro de reconstrução do mesmo jeito.

**Relação com o projeto:** é um dos baselines do projeto. Se o autoencoder (não linear) não superar o PCA (linear), isso é um resultado importante a discutir, não motivo para descartar.

### Isolation Forest
Algoritmo de detecção de anomalia baseado em árvores de decisão, que isola pontos "fora do padrão" com menos divisões do que pontos "normais".

**Relação com o projeto:** é o segundo baseline, de uma família de método bem diferente (não é reconstrução), o que dá mais força à comparação.

### Overfitting (no contexto de autoencoder saudável)
Quando o modelo aprende até os ruídos específicos dos dados de treino, em vez do padrão geral.

**Relação com o projeto:** aqui tem uma armadilha ao contrário do usual: se o gargalo for grande demais, o autoencoder pode aprender a reconstruir **qualquer coisa** bem, inclusive falha, zerando a capacidade de detecção. Validação ajuda a flagrar isso.