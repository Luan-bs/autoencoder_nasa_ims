# Glossário Técnico - Mecânica, Processamento de Sinais e Autoencoders

**Projeto:** Detecção Não-Supervisionada de Falhas em Rolamentos (NASA IMS Bearings)  
**Autor:** Luan B. Sarmento  
**Objetivo:** Estabelecer a fundamentação conceitual, física e matemática dos termos técnicos empregados no pipeline de dados, na extração de características e na modelagem por autoencoders.

---

## 1. Fundamentos Mecânicos e o Sistema Físico

Antes da modelagem de dados, é necessário compreender a bancada física de onde os sinais se originam.

### Eixo (*Shaft*)
Barra metálica cilíndrica acoplada ao motor elétrico, responsável por transmitir torque e movimento rotativo aos demais componentes do sistema.
* **No contexto do projeto:** É a parte móvel central da bancada de testes da NASA. Os rolamentos sob monitoramento são montados ao redor desse eixo para sustentá-lo enquanto uma carga radial de 6.000 lbs(2.721,55 kg) é aplicada.

### Rolamento (*Bearing*), Pista Interna e Pista Externa
Componente mecânico projetado para suportar cargas e permitir a rotação suave do eixo com atrito reduzido. É constituído por quatro elementos principais:
* **Pista interna (*inner race*):** Anel metálico montado diretamente sobre o eixo, girando na mesma velocidade que ele.
* **Pista externa (*outer race*):** Anel metálico externo montado na carcaça fixa da máquina, permanecendo estático.
* **Elementos rolantes (*rolling elements*):** Esferas ou rolos cilíndricos posicionados entre as duas pistas que rolam durante o movimento.
* **Gaiola (*cage*):** Estrutura que mantém o espaçamento uniforme entre os elementos rolantes.
* **No contexto do projeto:** Cada região do rolamento está sujeita a tensões de fadiga distintas. Quando surge uma falha (como descamação ou trinca), a localização dessa falha define a frequência dos impactos gerados.

### Vibração
Movimento oscilatório contínuo de um corpo mecânico em torno de sua posição de equilíbrio estático.
* **No contexto do projeto:** Máquinas rotativas geram vibrações contínuas durante a operação. Acelerômetros piezelétricos instalados nas carcaças dos rolamentos medem essa oscilação sob a forma de uma série temporal de aceleração ($g$). Em estado saudável, a vibração apresenta amplitude baixa e comportamento estacionário; à medida que surgem defeitos estruturais, surgem impactos mecânicos repetitivos.

### RPM (Revolutions Per Minute - Rotações Por Minuto) e Frequência do Eixo (Hz - Hertz)
* **RPM:** Quantidade de rotações completas ($360^\circ$) que o eixo realiza no intervalo de um minuto. No banco de testes da NASA, o motor opera em regime permanente a $2000\text{ RPM}$.
* **Hertz (Hz):** Unidade do Sistema Internacional para frequência, equivalente a ciclos ou eventos por segundo ($1\text{ Hz} = 1\text{ ciclo/s}$).
* **Conversão e Frequência Fundamental ($f_r$):** Para converter a velocidade angular de minutos para segundos, divide-se o valor de RPM por 60:
  $$f_r = \frac{2000\text{ RPM}}{60} \approx 33{,}33\text{ Hz}$$
  O eixo completa cerca de $33{,}33$ voltas por segundo. Esse valor é denominado **frequência fundamental de rotação**.
* **No contexto do projeto:** Esta frequência é a grandeza base de todo o sistema cinemático. Qualquer assimetria ou rotação no eixo gera perturbações em $33{,}33\text{ Hz}$ e seus harmônicos.

### Frequências Características de Falha de Rolamentos (BPFO, BPFI, BSF)
Frequências cinemáticas teóricas calculadas a partir da velocidade do eixo ($f_r$) e da geometria interna do rolamento (diâmetro de pistas, diâmetro dos rolos e quantidade de elementos rolantes).

> **Atenção conceitual importante:** Essas frequências **não** indicam a rotação das peças. A pista externa, por exemplo, é fixa na carcaça e não gira ($0\text{ RPM}$). A unidade Hertz ($\text{Hz}$) aqui mede a **taxa de repetição dos impactos mecânicos por segundo**: cada vez que um elemento rolante passa por uma imperfeição na superfície metálica, gera-se um microimpacto brusco registrado pelo acelerômetro.

* **BPFO (*Ball Pass Frequency Outer Race*):** Taxa de repetição dos impactos quando os elementos rolantes passam por um defeito situado na pista externa fixa ($\approx 236{,}4\text{ Hz}$ no rolamento Rexnord ZA-2115 a 2000 RPM).
  * *Significado físico:* Havendo uma trinca na pista externa, passam elementos rolantes por cima dela gerando aproximadamente **236 impactos por segundo**.
* **BPFI (*Ball Pass Frequency Inner Race*):** Taxa de impactos quando os elementos passam por um defeito na pista interna giratória ($\approx 296{,}9\text{ Hz}$).
* **BSF (*Ball Spin Frequency*):** Frequência de rotação do próprio elemento rolante em torno de seu próprio centro geométrico ($\approx 139{,}2\text{ Hz}$). Quando o defeito está na esfera ou rolo, ele colide alternadamente contra a pista interna e externa ($2 \times \text{BSF} \approx 278{,}4\text{ Hz}$).
* **No contexto do projeto:** Servem como **validação física** dos modelos. Se o detector acusa uma anomalia em determinado instante, a análise espectral (FFT) deve revelar o surgimento de energia concentrada exatamente nessas frequências teóricas.

#### Como se calcula a BPFO ($236{,}4\text{ Hz}$)?
A frequência de passagem de esferas pela pista externa é obtida pela equação clássica da cinemática de rolamentos:

$$\text{BPFO} = \frac{N_b}{2} \cdot f_r \cdot \left(1 - \frac{d}{D} \cos\theta\right)$$

Onde os parâmetros físicos do rolamento **Rexnord ZA-2115** sob rotação de $2000\text{ RPM}$ são:
* $f_r$: Frequência fundamental de rotação do eixo ($\frac{2000}{60} \approx 33{,}333\text{ Hz}$).
* $N_b$: Quantidade de elementos rolantes por fileira ($N_b = 16$).
* $d$: Diâmetro do elemento rolante ($\approx 8{,}41\text{ mm}$ ou $0{,}331\text{ in}$).
* $D$: Diâmetro médio primitivo do rolamento (*pitch diameter*, $\approx 71{,}50\text{ mm}$ ou $2{,}815\text{ in}$).
* $\theta$: Ângulo de contato da carga ($\theta \approx 0^\circ \implies \cos(0^\circ) = 1$).

Substituindo os valores na equação:
1. Termo da fileira e rotação: $\frac{N_b}{2} \cdot f_r = \frac{16}{2} \cdot 33{,}333 = 8 \cdot 33{,}333 \approx 266{,}67$
2. Razão geométrica dos diâmetros: $\frac{d}{D} \cos\theta \approx \frac{8{,}41}{71{,}50} \approx 0{,}1135$
3. Cálculo final:
   $$\text{BPFO} = 266{,}67 \cdot (1 - 0{,}1135) = 266{,}67 \cdot 0{,}8865 \approx 236{,}4\text{ Hz}$$

Portanto, o valor de $\approx 236{,}4\text{ Hz}$ é uma propriedade mecânica exata: a cada segundo de funcionamento a $2000\text{ RPM}$, exatamente $236$ rolos passam pela mesma coordenada da pista externa.

---

## 2. Processamento de Sinais e Métricas Estatísticas

As medições contínuas de aceleração são analisadas no domínio do tempo e no domínio da frequência para rastrear a evolução da degradação.

### SHM - *Simple Harmonic Motion* (Movimento Harmônico Simples)
Modelo físico ideal de oscilação periódica no qual a força de restauração é diretamente proporcional ao deslocamento, resultando em uma função senoidal pura.
* **No contexto do projeto:** O sinal de vibração de um rolamento perfeitamente equilibrado e sem defeitos é modelado aproximadamente como a sobreposição de ondas senoidais regulares. O aparecimento de falhas quebra essa suavidade, introduzindo pulsos impulsivos e transitórios não-lineares.

### RMS (*Root Mean Square* - Valor Eficaz)
Medida estatística da magnitude média de uma quantidade que varia com o tempo, calculada como a raiz quadrada da média aritmética dos quadrados dos valores instantâneos:
$$\text{RMS} = \sqrt{\frac{1}{N}\sum_{i=1}^{N} x_i^2}$$
* **No contexto do projeto:** O RMS quantifica a energia global da vibração. É um indicador robusto do nível geral de severidade mecânica, embora costume se elevar apenas nos estágios intermediários e finais da degradação.

### Curtose (*Kurtosis*)
Momento estatístico de quarta ordem normalizado, que mede o grau de achatamento da distribuição de probabilidade do sinal em relação a uma distribuição normal (gaussiana, onde curtose $\approx 3$):
$$\text{Curtose} = \frac{\frac{1}{N}\sum_{i=1}^{N}(x_i - \bar{x})^4}{\left(\frac{1}{N}\sum_{i=1}^{N}(x_i - \bar{x})^2\right)^2}$$
* **No contexto do projeto:** Impactos incipientes de falha manifestam-se como picos pontuais de alta amplitude intercalados por períodos calmos. Essa presença de "caudas pesadas" eleva acentuadamente a curtose antes de haver alteração perceptível no RMS, tornando-a um indicador precoce de avaria.

### Fator de Crista (*Crest Factor*)
Razão matemática entre a amplitude de pico máxima ($|x|_{\max}$) e o valor eficaz ($\text{RMS}$) da série temporal:
$$\text{CF} = \frac{|x|_{\max}}{\text{RMS}}$$
* **No contexto do projeto:** Permite identificar se o sinal possui impactos isolados intensos em relação à sua energia média. É amplamente utilizado de forma complementar à curtose.

### FFT (*Fast Fourier Transform* - Transformada Rápida de Fourier)
Algoritmo eficiente para computar a Transformada Discreta de Fourier (DFT), decompondo um sinal registrado no domínio do tempo em seus componentes espectrais no domínio da frequência.
* **No contexto do projeto:** Converte o sinal de aceleração contínuo em um gráfico de amplitude por frequência. Permite inspecionar se a energia vibratória está concentrada na rotação fundamental ($33{,}33\text{ Hz}$) ou nas bandas associadas às falhas cinemáticas ($236\text{ Hz}$, $297\text{ Hz}$).

---

## 3. Filosofia Experimental de Manutenção e Prognóstico

### *Run-to-Failure* (Ensaio até a Falha)
Metodologia experimental em que o equipamento é mantido em funcionamento sob regime contínuo de operação até que a integridade mecânica seja completamente perdida, sem intervenção de manutenção corretiva ou preventiva.
* **No contexto do projeto:** O conjunto de dados IMS da NASA foi gerado seguindo esse protocolo. Isso fornece o histórico completo da vida do rolamento - desde a fase inicial de acomodação mecânica (*run-in*), passando por semanas de operação saudável, até a fadiga acelerada e quebra final.

### PHM (*Prognostics and Health Management*)
Disciplina da engenharia que combina sensoriamento, processamento de sinais e aprendizado de máquina para diagnosticar o estado de integridade de ativos industriais e prognosticar sua degradação futura.
* **No contexto do projeto:** Área acadêmica e industrial onde este estudo está circunscrito.

### RUL (*Remaining Useful Life* - Vida Útil Restante)
Tempo residual ou ciclos operacionais esperados até que um componente atinja um limiar intolerável de risco operacional ou quebra funcional.
* **No contexto do projeto:** Representa a etapa subsequente à detecção. Este projeto foca primordialmente na detecção de anomalia (responder "quando a degradação começou?"); a estimativa de RUL responde "quanto tempo a máquina ainda opera até o colapso?".

---

## 4. Aprendizado Não-Supervisionado e Arquiteturas de Autoencoders

A detecção de falha é formulada como uma tarefa de modelagem de densidade do estado saudável sem rótulos supervisionados.

### Aprendizado Não-Supervisionado (*Unsupervised Learning*)
Paradigma de aprendizado de máquina no qual o algoritmo infere representações e padrões intrínsecos a partir de dados de entrada sem depender de classes ou rótulos prévios fornecidos por especialistas.
* **No contexto do projeto:** Os modelos são ajustados exclusivamente com amostras extraídas dos primeiros dias de ensaio, quando se tem certeza física de que o rolamento está intacto. O modelo nunca é exposto a dados de defeito durante a etapa de treinamento.

### Autoencoder
Arquitetura de rede neural artificial composta por duas seções funcionais principais - o codificador (*encoder*) e o decodificador (*decoder*) - cujo objetivo é reconstruir o vetor de entrada $\mathbf{x}$ minimizando a divergência com a saída reconstruída $\hat{\mathbf{x}}$.

```
Entrada x (D dimensões) ───► [ Encoder ] ───► Espaço Latente z (d dimensões, d << D) ───► [ Decoder ] ───► Saída Reconstruída x̂
```

### Espaço Latente (*Latent Space* / *Bottleneck*)
Camada intermediária estreita da rede neural que atua como um gargalo de informação. 
* **Conceito Intuitivo:** O espaço latente é um conjunto compacto de números onde as características essenciais do sinal são resumidas. Trata-se de uma representação de dimensionalidade reduzida que preserva os padrões fundamentais dos dados, de modo que, ao serem alimentados em equações subsequentes (as camadas do decodificador), o sinal original é restaurado com alta fidelidade.
* **No contexto do projeto:** Se a entrada possui $D$ dimensões (ex.: 2048 amostras brutas ou $K$ métricas de engenharia) e o gargalo possui dimensão $d \ll D$, o modelo é forçado a reter somente as dinâmicas do sinal saudável. A escolha da dimensão latente é crítica: um gargalo excessivamente largo pode memorizar ruídos e sinais anômalos; um gargalo excessivamente estreito não preserva a variabilidade operacional saudável.

### Erro de Reconstrução (*Reconstruction Error*)
Diferença calculada ponto a ponto entre o vetor original $\mathbf{x}$ e a saída reconstruída $\hat{\mathbf{x}}$, frequentemente expressa pelo Erro Quadrático Médio (MSE) ou Erro Absoluto Médio (MAE):
$$\text{MSE}(\mathbf{x}, \hat{\mathbf{x}}) = \frac{1}{D}\sum_{j=1}^{D} (x_j - \hat{x}_j)^2$$
* **No contexto do projeto:** Constitui o escore de anomalia contínuo. Enquanto o rolamento opera em condições idênticas às de treinamento, a reconstrução é precisa e o erro permanece reduzido. No momento em que anomalias mecânicas se manifestam no sinal, a rede não dispõe de parâmetros treinados para sintetizá-las, resultando na elevação proporcional do erro.

### Limiar de Decisão (*Anomaly Threshold*)
Valor de corte escalar aplicado sobre a série temporal do erro de reconstrução, determinando a transição do estado classificado entre "saudável" e "anômalo".
* **No contexto do projeto:** O limiar é estabelecido exclusivamente a partir do conjunto de validação saudável (por critérios estatísticos como média $+ 3\sigma$ ou percentis pré-definidos). O conjunto de teste nunca é consultado para calibrar esse parâmetro, resguardando a integridade metodológica.

### Vazamento de Informação (*Data Leakage*)
Contaminação indevida do processo de modelagem pela inclusão acidental de dados provenientes do conjunto de validação, teste ou do futuro temporal durante o ajuste de hiperparâmetros, escalonamento ou treinamento.
* **No contexto do projeto:** Como séries temporais de degradação possuem forte autocorrelação temporal, o particionamento de dados respeita rigorosamente a divisão cronológica (*temporal split*). Os parâmetros de normalização (como média e desvio padrão do `StandardScaler`) são aprendidos estritamente na partição de treino.

### *Overfitting* em Modelos Não-Supervisionados
Fenômeno em que o modelo ajusta parâmetros para memorizar peculiaridades pontuais e ruídos específicos dos dados de treinamento, em detrimento do padrão global subjacente.
* **No contexto de autoencoders:** Apresenta uma consequência particular: caso a rede possua capacidade representativa exagerada (ou um espaço latente amplo demais), ela pode se tornar uma função identidade trivial. Nesse cenário, o autoencoder aprenderá a reconstruir qualquer sinal arbitrário - inclusive dados com falha -, anulando sua capacidade de gerar erro elevado na presença de anomalias.

---

## 5. Modelos de Referência (*Baselines*)

Para demonstrar a utilidade e o valor agregado das arquiteturas neurais profundas, o pipeline compara o desempenho dos autoencoders com algoritmos clássicos consagrados na literatura.

### PCA (*Principal Component Analysis* - Análise de Componentes Principais)
Método estatístico linear não-paramétrico que projeta os dados originais sobre um conjunto ortogonal de direções ordenadas pela máxima variância (componentes principais).
* **No contexto do projeto:** O PCA é utilizado como um autoencoder linear de referência. Se um autoencoder neural profundo não apresentar ganho expressivo de detecção em relação à projeção linear do PCA, evidencia-se que a dinâmica da falha sob análise pode ser satisfatoriamente capturada por relações lineares de menor custo computacional.

### *Isolation Forest*
Algoritmo de detecção de anomalias baseado em conjuntos de árvores de decisão aleatórias que parte do princípio de que observações anômalas necessitam de menos divisões para serem isoladas do restante dos dados.
* **No contexto do projeto:** Atua como um *baseline* representativo de uma família metodológica distinta (baseada em isolamento e densidade espacial, e não em reconstrução residual), enriquecendo o rigor comparativo do estudo.
