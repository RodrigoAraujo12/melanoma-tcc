# Documentação Técnica — TCC

**Diagnóstico Multimodal de Lesões Cutâneas com MedGemma**

Aluno: Rodrigo Araújo
Orientador: Prof. Clauirton Albuquerque Siebra
Curso: Engenharia da Computação — UFPB
Período: Maio–Junho 2026

---

## Sumário

1. [Objetivo do Trabalho](#1-objetivo-do-trabalho)
2. [Datasets Considerados](#2-datasets-considerados)
3. [Stack Técnica](#3-stack-técnica)
4. [Experimentos Realizados](#4-experimentos-realizados)
   - [v1 — Fine-tuning binário com ISIC 2020](#v1--fine-tuning-binário-com-isic-2020)
   - [v2 — Fine-tuning multiclasse com Derm7pt](#v2--fine-tuning-multiclasse-com-derm7pt)
   - [v3 — Classification head com MedSigLIP + Focal Loss](#v3--classification-head-com-medsiglip--focal-loss)
   - [v4 — Class weights agressivos sem undersample](#v4--class-weights-agressivos-sem-undersample)
   - [v5 — Merge Derm7pt + HAM10000 com schema unificado](#v5--merge-derm7pt--ham10000-com-schema-unificado)
   - [v6 — Balanceamento por corte de NEV + cache de embeddings](#v6--balanceamento-por-corte-de-nev--cache-de-embeddings)
   - [v7 — Corte profundo de NEV + threshold conservador para MEL](#v7--corte-profundo-de-nev--threshold-conservador-para-mel)
   - [v8 — Unfreeze dos 2 últimos blocos do SigLIP (resultado negativo)](#v8--unfreeze-dos-2-últimos-blocos-do-siglip-resultado-negativo)
   - [Validação cruzada (5-fold) + TTA — caracterização final do v7](#validação-cruzada-5-fold--tta--caracterização-final-do-v7)
5. [Comparação com a Literatura](#5-comparação-com-a-literatura)
6. [Problemas Técnicos Enfrentados](#6-problemas-técnicos-enfrentados)
7. [Decisões Arquiteturais Importantes](#7-decisões-arquiteturais-importantes)
8. [Estrutura do Código](#8-estrutura-do-código)
9. [Trabalhos Futuros](#9-trabalhos-futuros)
10. [Referências](#10-referências)

---

## 1. Objetivo do Trabalho

Desenvolver e avaliar um modelo de inteligência artificial multimodal capaz de auxiliar médicos no diagnóstico de lesões cutâneas, em especial melanoma, integrando análise de imagens dermatoscópicas com metadados clínicos textuais.

**Modelo base:** MedGemma 4B (Google DeepMind, 2025) — Vision-Language Model treinado em dados médicos.

**Tarefa:** Classificação multiclasse de lesões cutâneas.

**Ambiente de execução:** Kaggle Notebooks com GPU NVIDIA T4 (16 GB VRAM). Execução local foi descartada devido à incompatibilidade do bitsandbytes/QLoRA com Intel Arc B580 (sem suporte CUDA).

---

## 2. Datasets Considerados

### 2.1 ISIC 2020 (utilizado em v1)

- **Fonte:** Rotemberg et al., 2021 — SIIM-ISIC Melanoma Classification Challenge
- **Imagens:** ~33.000 imagens dermatoscópicas
- **Classes:** Binário (melanoma vs benign)
- **Metadados:** age_approx, sex, anatom_site_general_challenge
- **Limitação:** altamente desbalanceado (~98% benign vs ~2% melanoma); tarefa binária categórica

### 2.2 Derm7pt (utilizado em v2 e v3)

- **Fonte:** Kawahara et al., 2019 — Simon Fraser University
- **Imagens:** 1.011 casos com dermatoscopia + clínica + 7-point checklist
- **Splits oficiais:** 413 train / 203 validation / 395 test
- **Classes originais:** 16 subtipos diagnósticos
- **Agrupamento adotado:** 5 classes seguindo Kawahara (2019):
  - **BCC** (Basal Cell Carcinoma): 42 casos
  - **NEV** (Nevus — agrupa blue/clark/combined/congenital/dermal/recurrent/Reed-Spitz): 575 casos
  - **MEL** (Melanoma — agrupa in situ/<0.76mm/0.76-1.5mm/>1.5mm/metastasis): 252 casos
  - **SK** (Seborrheic Keratosis): 45 casos
  - **MISC** (dermatofibroma/lentigo/melanosis/vascular/miscellaneous): 97 casos
- **Metadados disponíveis:** sex, location, elevation, 7-point checklist (pigment_network, streaks, pigmentation, regression_structures, dots_and_globules, blue_whitish_veil, vascular_structures)
- **Licença:** CC BY-NC-ND 4.0 (uso acadêmico, sem redistribuição)

### 2.3 PAD-UFES-20 (considerado para trabalhos futuros)

- **Fonte:** Pacheco et al., 2020 — UFES (Brasil)
- **Imagens:** 2.298 imagens clínicas (smartphone, não dermatoscopia)
- **Classes:** 6 (BCC, SCC, ACK, SEK, MEL, NEV)
- **Diferencial:** 22 features clínicas estruturadas (idade, fototipo, diâmetro, sintomas, histórico)
- **Por que não foi usado:** apenas 52 amostras de melanoma; experimentos foram conduzidos com Derm7pt primeiro

### 2.4 HAM10000 (utilizado em v5 e v6)

- **Fonte:** Tschandl, Rosendahl & Kittler, 2018 — versão `kmader` no Kaggle
- **Imagens:** 10.015 dermatoscópicas, com `lesion_id` (múltiplas imagens por lesão)
- **Classes originais (dx):** nv, mel, bkl, bcc, akiec, vasc, df → mapeadas para os 5 grupos do projeto
- **Metadados:** sex, age, localization, dx, lesion_id
- **Por que foi escolhido:** grande volume de classes minoritárias (BCC, MEL, SK) para complementar o Derm7pt; mesma modalidade (dermatoscopia), permitindo merge com schema unificado
- **Cuidado adotado:** uso apenas de features compartilhadas com o Derm7pt e split por `lesion_id` para evitar domain bias e leakage (ver v5)

---

## 3. Stack Técnica

### Linguagem e bibliotecas principais
- Python 3.12
- PyTorch 2.x
- HuggingFace Transformers
- PEFT (Parameter-Efficient Fine-Tuning) — para QLoRA
- TRL (Transformer Reinforcement Learning) — SFTTrainer
- bitsandbytes — quantização 4-bit
- pandas, numpy, PIL, scikit-learn

### Ferramentas
- VS Code (desenvolvimento local em Windows)
- Git + GitHub (versionamento)
- Kaggle Notebooks (execução em GPU)

### Modelo base
- `google/medgemma-4b-it` — MedGemma 4B Instruction-Tuned
- Vision encoder: MedSigLIP (variante medical-tuned do SigLIP)
- Language model: variante do Gemma2

---

## 4. Experimentos Realizados

### Visão geral dos experimentos

| Experimento | Dataset | Paradigma | Accuracy final |
|---|---|---|---|
| **v1** | ISIC 2020 | VLM generativo binário | 51% (mode collapse) |
| **v2** | Derm7pt | VLM generativo multiclasse | 30% (80% inválidas → 6% real) |
| **v3** | Derm7pt | Classification head + Focal Loss | **69%** ✅ |
| **v4** | Derm7pt | Idem v3, class weights `inverse` sem undersample | 69,6% (MEL recall 50%) |
| **v5** | Derm7pt + HAM10000 | Classifier + schema unificado (10-dim) | val_acc 78,1% (timeout no test) |
| **v6** | Derm7pt + HAM10000 | Corte de 2k NEV + cache de embeddings | 72,9% (MEL recall 61%) |
| **v7** | Derm7pt + HAM10000 | Corte profundo NEV (1,4k) + threshold MEL | **73,7%** (MEL recall 66%) ✅ |
| **v8** | Derm7pt + HAM10000 | Unfreeze de 2 blocos do encoder + sampler | val macro-F1 0,56 (overfit, ✗ pior que v7) |

> Accuracy reportada no **test do Derm7pt** (395 amostras), comparável entre todas as versões. A partir do v5 os números refletem treino mesclado com HAM10000, mas o test permanece só Derm7pt. O **v7 é o modelo final**; o v8 (unfreeze) foi um experimento com resultado negativo. O v7 foi ainda validado por **5-fold CV** (76,2% ± 2,0%) — ver seção final.

---

### v1 — Fine-tuning binário com ISIC 2020

#### Plano original

Aplicar QLoRA ao MedGemma 4B para classificação binária melanoma vs benign no ISIC Archive, mantendo o paradigma generativo do VLM (modelo gera texto explicando o diagnóstico).

#### Metodologia

- **Dataset balanceado:** 500 melanoma + 500 benign (sample aleatório do ISIC 2020 train set)
- **Split:** 900 train / 100 val (10% val)
- **Modelo:** MedGemma 4B com quantização 4-bit (NF4) via bitsandbytes
- **Fine-tuning:** QLoRA com `r=16, lora_alpha=32, lora_dropout=0.05`
- **Target modules:** `q_proj, v_proj, k_proj, o_proj`
- **Learning rate:** 2e-4 com cosine decay
- **Warmup:** 50 steps
- **Otimizador:** paged_adamw_8bit
- **Gradient checkpointing:** `use_reentrant=False`
- **Epochs:** 3 (mas interrompido na epoch 2 por OOM)
- **Prompt template:**
  ```
  Patient: {sex}, approximately {age_approx} years old.
  Lesion location: {anatom_site_general_challenge}.
  Analyze the dermoscopy image of this skin lesion.
  Is this lesion melanoma or benign?
  Describe the key visual features that support your assessment.
  ```
- **Resposta de treino:** uma palavra (`"melanoma"` ou `"benign"`)

#### Resultados

| Métrica | Valor |
|---|---|
| Accuracy | 51% |
| Recall melanoma | 100% |
| Recall benign | 2% |
| Token accuracy | 98.5% |

**Análise:** O modelo colapsou — passou a responder `"melanoma"` para 99 de 100 amostras (mode collapse). Token accuracy alta porque o modelo aprendeu o atalho de chutar sempre a classe mais comum (no balanceado, 50/50, qualquer escolha dá ~50%).

#### Problemas enfrentados

1. **CheckpointError com gradient checkpointing** — corrigido com `use_reentrant=False`
2. **OOM (CUDA Out of Memory)** — T4 com 16GB ficou apertada com `bnb_4bit_compute_dtype=bfloat16`
3. **NotImplementedError com BFloat16 e AMP GradScaler** — corrigido desabilitando `fp16=False, bf16=False`
4. **Mode collapse** — problema fundamental, não recuperável dentro da config

#### Lições aprendidas

- Fine-tuning de VLM com respostas categóricas de 1 token destrói capacidade generativa do modelo base
- LoRA agressivo (`alpha=32`) + LR alto (`2e-4`) + classes binárias = recipe para colapso
- Token accuracy NÃO é classification accuracy em tarefas generativas
- Necessário aumentar diversidade das respostas e suavizar LoRA para evitar colapso

---

### v2 — Fine-tuning multiclasse com Derm7pt

#### Plano

Migrar para tarefa multiclasse (5 classes) com respostas descritivas em frase completa, dataset menor e mais estruturado, e config de LoRA mais conservadora para evitar mode collapse.

#### Metodologia

- **Dataset:** Derm7pt com agrupamento em 5 classes
- **Splits oficiais:** 413 train / 203 val / 395 test
- **Balanceamento:** undersample NEV (256→80) + oversample minoritárias (com cap de 4x):
  - NEV: 80, MEL: 80, MISC: 80, BCC: 76, SK: 64 (total: 380)
- **Image augmentation:** rotação ±10°, flip horizontal, jitter de brilho e contraste
- **LoRA suave:** `r=8, lora_alpha=16` (scaling=1.0)
- **Learning rate:** 5e-5 (4x menor que v1)
- **Warmup:** 10% das steps totais
- **Epochs:** 4 (vs 3 em v1)
- **Prompt:**
  ```
  Patient: {sex}, lesion on {location}, {elevation}.
  Analyze the dermoscopic image and identify the most likely diagnosis
  among: melanoma, nevus, basal cell carcinoma, seborrheic keratosis, or other lesion.
  Describe the key visual features that support your assessment.
  ```
- **Resposta rica:** `"Atypical pigment network and irregular streaks suggest melanoma."`
- **Otimizações de memória adicionais:**
  - `per_device_eval_batch_size=1`
  - `eval_accumulation_steps=4` (move predições do eval para CPU)
  - `MemoryCleanupCallback` (gc.collect + empty_cache antes do eval)

#### Resultados

**Métricas de treino (todas as 4 epochs completas):**

| Epoch | Train Loss | Val Loss | Token Accuracy |
|---|---|---|---|
| 1 | 9.74 | 7.36 | 10.3% |
| 2 | 3.64 | 3.35 | 93.1% |
| 3 | 2.94 | 2.91 | 96.2% |
| 4 | 2.87 | 2.88 | 96.3% |

**Métricas no test set (395 amostras):**

| Classe | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| BCC | 0.10 | 0.33 | 0.15 | 16 |
| NEV | 0.58 | 0.46 | 0.51 | 219 |
| MEL | 0.00 | 0.00 | 0.00 | 101 |
| SK | 0.06 | 0.74 | 0.12 | 19 |
| MISC | 0.00 | 0.00 | 0.00 | 40 |

**Accuracy global: 30%** — mas com problema crítico: **315 das 395 predições (80%) foram INVÁLIDAS**.

#### Análise

- O modelo passou a gerar **apenas a parte descritiva** das respostas ("Key visual features: regular, pigmented structures...") **sem incluir o diagnóstico** ao final
- Token accuracy alta (96%) era enganosa: modelo aprendeu o template inicial mas truncava antes da conclusão
- Hipótese principal: a image augmentation introduziu variabilidade que enfraqueceu a associação features→diagnóstico, fazendo o modelo "fugir" do compromisso final
- Accuracy efetiva considerando todas as 395 amostras: ~6% (24 acertos)

#### Problemas enfrentados

1. **Crescimento de memória do eval** — GPU foi de 6.9 GB → 13.5 GB após eval da epoch 1, depois estabilizou
2. **Perda do modelo treinado** — sessão Kaggle resetou durante a noite após inferência, perdendo as predictions in-memory
3. **Bug na extract_multiclass_label** — função usava "melanoma anywhere" como detector, gerando falsos positivos
4. **Geração truncada** — modelo aprendeu features mas não diagnósticos

#### Lições aprendidas

- Image augmentation pode prejudicar VLMs em tarefas de classificação ao introduzir incerteza nos pares features→diagnóstico
- Save incremental durante inferência longa é essencial
- Token accuracy é métrica enganosa para classificação via geração
- Paradigma generativo não é ideal para classificação multiclasse desbalanceada
- O extract de label precisa de regex específica (negrito, padrões "diagnosis is X"), não busca livre

---

### v3 — Classification head com MedSigLIP + Focal Loss

#### Plano

Abandonar o paradigma generativo. Usar apenas o **vision encoder** do MedGemma (MedSigLIP) como feature extractor congelado, combinado com uma branch tabular para metadados demográficos, e treinar uma cabeça de classificação dedicada com **Focal Loss + class weights**.

#### Arquitetura

```
┌────────────────────────────────┐    ┌──────────────────────────┐
│ Dermoscopia (224×224 RGB)      │    │ Metadata (13-dim one-hot)│
│      Visual modality           │    │     Tabular modality     │
└──────────────┬─────────────────┘    └──────────────┬───────────┘
               │                                     │
               ▼                                     ▼
       MedSigLIP encoder                   MLP encoder
       (vision_tower do MedGemma)          (13→128→256, GELU+Dropout)
       [CONGELADO, ~400M params]
               │                                     │
               ▼                                     ▼
       LayerNorm + Linear (1152→256)                 │
       + GELU + Dropout                              │
               │                                     │
               └─────► concat 512-dim ◄──────────────┘
                              │
                              ▼
                  Classifier head (512→256→5)
                  + GELU + Dropout
                              │
                              ▼
                  Logits (5 classes)
```

#### Metodologia

- **Vision encoder:** Extraído de `google/medgemma-4b-it.vision_tower` (~400M params), congelado
- **Metadata encoder:** MLP simples (13→128→256) com GELU + Dropout
- **Classification head:** Linear(512→256→5) com Dropout 0.2
- **Total trainable params:** ~330k (de ~400M totais → ~0.08%)
- **Metadata features (13 dim one-hot):**
  - sex: 2 categorias (female, male)
  - location: 8 categorias (abdomen, back, head neck, upper limbs, lower limbs, acral, buttocks, chest)
  - elevation: 3 categorias (flat, palpable, nodular)
- **Loss:** Focal Loss com class weights inverso-raiz e label smoothing
  ```python
  focal_loss = α * (1-p)^γ * log(p)
  γ = 2.0
  label_smoothing = 0.05
  α (class weights) = compute_class_weights(counts, mode="inverse_sqrt")
  ```
- **Optimizador:** AdamW (lr=5e-4, weight_decay=0.01)
- **Scheduler:** CosineAnnealingLR (T_max=15)
- **Batch size:** 16
- **Epochs:** 15
- **Image augmentation:** Sim (mesmo do v2, mas agora o modelo não sofre com ela porque a tarefa é direta)
- **Balanceamento de dados:** Mesmo do v2 (target=80 por classe, cap oversample 4x)

#### Por que essa escolha funciona

1. **Loss apropriada:** Focal Loss penaliza exemplos difíceis (MEL heterogêneo) mais que fáceis (NEV)
2. **Class weights:** Compensa desbalanceamento de forma suave (inverse_sqrt, não inverso direto)
3. **Encoder congelado:** Aproveita conhecimento médico do MedSigLIP sem risco de catastrophic forgetting
4. **Tabular branch:** Metadados estruturados são processados como categóricos, não interpretados como linguagem
5. **Sem geração de texto:** Elimina mode collapse e respostas truncadas
6. **Apenas 330k params treináveis:** Treino rápido (~30 min) e baixo risco de overfit

#### Resultados

**Métricas no test set (395 amostras):**

```
Test loss: 0.4837
Test accuracy: 0.6861 (68.6%)

              precision    recall  f1-score   support

         BCC       0.29      0.69      0.41        16
         NEV       0.85      0.78      0.81       219
         MEL       0.70      0.54      0.61       101
          SK       0.41      0.47      0.44        19
        MISC       0.46      0.62      0.53        40

    accuracy                           0.69       395
   macro avg       0.54      0.62      0.56       395
weighted avg       0.73      0.69      0.70       395
```

**Destaques:**
- ✅ Accuracy **68.6%** (vs 30% v2 e 51% v1)
- ✅ Recall MEL **54%** (vs 0% v2 e n/a v1) — clinicamente significativo
- ✅ Recall BCC **69%** (11/16 detectados)
- ✅ NEV com F1=0.81 (classe dominante bem aprendida)
- ✅ Sem mode collapse
- ✅ 100% das predições válidas

**Distribuição das predições vs. verdade:**

| Classe | Verdade | Predição | Diferença |
|---|---|---|---|
| NEV | 219 | 202 | -17 |
| MEL | 101 | 79 | -22 |
| MISC | 40 | 54 | +14 |
| SK | 19 | 22 | +3 |
| BCC | 16 | 38 | +22 |

O modelo distribui predições de forma alinhada à distribuição real (não há classe artificialmente inflada).

#### Comparação com v1 e v2

| Métrica | v1 (ISIC) | v2 (Derm7pt VLM) | **v3 (Classifier)** |
|---|---|---|---|
| Accuracy global | 51% (binário) | 30% (real 6%) | **69%** |
| Recall MEL | n/a (binário) | 0% | **54%** |
| Recall BCC | n/a | 0% | **69%** |
| Predições válidas | 100% | 20% | **100%** |
| Mode collapse | Total | Parcial | **Não** |
| Tempo de treino | ~2h | ~2h | **~30 min** |
| Params treináveis | ~10M (LoRA) | ~5M (LoRA leve) | **330k (head)** |

---

### v4 — Class weights agressivos sem undersample

#### Plano

Ajustar o v3 para reduzir falsos negativos de melanoma (clinicamente o erro mais grave) e melhorar BCC/SK, sem mudar arquitetura. A hipótese era que o undersample do v3 (NEV 256→80) jogava fora informação útil e que class weights mais agressivos compensariam melhor o desbalanceamento.

#### Mudanças vs v3

| Parâmetro | v3 | v4 |
|---|---|---|
| `undersample` | `True` (NEV 256→80) | **`False`** (mantém 256 NEV + 90 MEL) |
| `class_weights mode` | `inverse_sqrt` | **`inverse`** (pesos mais agressivos p/ minoritárias) |
| Resto (arquitetura, LR, epochs, augment) | — | inalterado |

#### Resultados

| Métrica | v3 | **v4** |
|---|---|---|
| Accuracy global | 68,6% | **69,6%** |
| Recall MEL | 54% | **50%** |
| Recall BCC | 69% | mantido na faixa |

#### Análise

- Accuracy subiu marginalmente (+1 ponto), mas o **recall de MEL caiu** (54%→50%), efeito contrário ao objetivo.
- Conclusão: nem `inverse` agressivo nem manter todos os NEV resolveram o teto de ~70%. O gargalo real é **volume de dados das classes minoritárias** (Derm7pt tem só 42 BCC, 45 SK no total), não a estratégia de pesos.
- Essa conclusão motivou o v5: trazer mais dados das minoritárias de um segundo dataset.

---

### v5 — Merge Derm7pt + HAM10000 com schema unificado

#### Plano

Aumentar drasticamente o volume das classes minoritárias mesclando o Derm7pt com o **HAM10000** (10.015 imagens dermatoscópicas, 7 classes). O desafio central: **evitar domain bias** — se o modelo conseguir identificar de qual dataset cada imagem veio (por uma feature que só existe em um deles), ele aprende a origem em vez do diagnóstico.

#### Dataset HAM10000

- **Fonte:** Tschandl et al., 2018 — versão kmader no Kaggle
- **Imagens:** 10.015 dermatoscópicas, com `lesion_id` (múltiplas imagens por lesão)
- **Mapeamento de classes (dx → 5 grupos):** `nv→NEV`, `mel→MEL`, `bkl→SK`, `bcc→BCC`, `akiec/vasc/df→MISC`

#### Schema de metadados unificado (decisão crítica)

Apenas features presentes **em ambos** os datasets, e apenas categorias que aparecem **nos dois** (uma categoria exclusiva de um dataset carrega sinal de origem):

- `sex` (2): female, male — **sem `unknown`** (só existia no HAM)
- `location` (8): abdomen, back, head_neck, upper_limbs, lower_limbs, acral, chest, genital — **sem `unknown`**
- **Total: 10 dim** (vs 13 do v3/v4 — `elevation` foi removido por ser exclusivo do Derm7pt)
- Mapeamentos `DERM7PT_LOCATION_MAP` e `HAM_LOCATION_MAP` harmonizam nomenclaturas (ex.: `trunk`/`back`→back, `face`/`scalp`/`ear`/`neck`→head_neck, `buttocks`→lower_limbs).
- **244 amostras do HAM (2,4%) foram descartadas** por terem sex/location `unknown`. Verificado que o impacto nas minoritárias é mínimo (BCC perde ~1%, MEL ~0,9%, SK ~2,1%).

> ⚠️ Durante a construção do schema descobriu-se um bug latente em v3/v4: a categoria `genital areas` do Derm7pt nunca esteve em `LOCATION_CATEGORIES` e era silenciosamente codificada como tudo-zero. Corrigido no schema unificado.

#### Splits

- **Train:** Derm7pt train (413) + HAM train (~8.300) = ~8.716
- **Val:** Derm7pt val (203) + HAM val (~1.468) = ~1.671
- **Test:** **apenas Derm7pt test (395)** — comparação justa com v3/v4.
- HAM dividido **por `lesion_id`** (não por linha) para evitar leakage de lesão entre train/val.

#### Hiperparâmetros (vs v4)

| Parâmetro | v4 | v5 |
|---|---|---|
| Batch size | 16 | **32** |
| Epochs | 15 | **8** |
| `class_weights mode` | `inverse` | **`inverse_sqrt`** |
| metadata_dim | 13 | **10** |

#### Resultados

- **Val accuracy atingiu 78,1%** (epoch 7) — salto claro vs o teto de ~70% do Derm7pt isolado, confirmando que o volume extra de minoritárias ajuda.
- **Timeout do Kaggle no epoch 8** (cada epoch ~95 min porque o encoder congelado reprocessava ~8.700 imagens por época). A execução não chegou a gerar matriz de confusão nem `classification_report` no test.
- Diagnóstico: o gargalo era **reprocessar embeddings idênticos a cada época** — o que motivou o cache do v6.

---

### v6 — Balanceamento por corte de NEV + cache de embeddings

#### Plano

Duas otimizações sobre o v5, sem tocar na arquitetura nem em LR/batch/FocalLoss:

**1. Balanceamento por corte de NEV.** O train do v5 ainda era ~66% NEV (razão NEV/BCC ≈ 12,8). Cortar 2.000 NEV do HAM (seed=42) reduz a dominância da majoritária. Os class weights `inverse_sqrt` são **mantidos** — o undersample complementa, não substitui.

**2. Cache de embeddings (mudança principal).** Como o encoder MedSigLIP é 100% congelado, o embedding de cada imagem é **idêntico em toda época**. Em vez de reprocessar a cada época (gargalo do v5), pré-computa-se o embedding de cada imagem **uma única vez** e treina-se apenas a cabeça (`vision_proj` + `metadata_encoder` + `classifier`) sobre os tensores cacheados em RAM.

#### Detalhes técnicos

- `augment=False` no train (obrigatório: com augment a imagem mudaria a cada época e o embedding cacheado ficaria inválido).
- `precompute_embeddings()` roda o encoder uma vez por conjunto (train, derm_val, ham_val, derm_test) e devolve `TensorDataset(embedding, metadata, label)`.
- `head_forward(model, emb, md)` aplica `vision_proj(emb)` → matematicamente idêntico ao forward completo do v5, já que `encode_image()` retorna o `pooled` **antes** do `vision_proj`.
- **Estratégia de épocas:** `EPOCHS=40` (teto de segurança) + `ReduceLROnPlateau(factor=0.5, patience=3)` + **early stopping (patience=6)**. Hierarquia intencional: o scheduler baixa o LR primeiro; o early stop só desiste depois que baixar o LR também não resolveu.
- **Seleção do melhor checkpoint por macro-F1 no Derm7pt val** (não no val combinado, nem por accuracy) — macro-F1 dá peso igual às minoritárias.
- **Avaliação final em dois domínios separados:** Derm7pt test e HAM test reportados isoladamente, nunca combinados num único número (cada um mede uma coisa diferente).

#### Ganho de tempo obtido

No v5 cada época custava ~95 min (reprocessar ~8.700 imagens). No v6 o encoder rodou **uma vez só**; a execução completa (cache + 40 épocas com early stopping) levou **~78 min** — contra os ~13h projetados do v5. O timeout foi eliminado.

#### Resultados

Treino parou por early stopping na **época 30** (melhor checkpoint na época ~24). Avaliação em dois domínios (argmax):

**Derm7pt test (395):** accuracy **72,9%** | macro-F1 **0,608**

| Classe | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| BCC | 0,556 | 0,625 | 0,588 | 16 |
| NEV | 0,819 | 0,845 | 0,832 | 219 |
| MEL | 0,738 | 0,614 | 0,670 | 101 |
| SK | 0,444 | 0,421 | 0,432 | 19 |
| MISC | 0,469 | 0,575 | 0,517 | 40 |

**HAM test (1.468, in-domain):** accuracy **83,1%** | macro-F1 **0,737**

#### Análise

- **Melhor modelo até então:** +3,3pt de accuracy sobre o v4 e **MEL recall recuperado para 61%** (vs 50% no v4), revertendo a queda — o corte de NEV + seleção por macro-F1 funcionaram.
- **BCC precision quase dobrou** (0,29 no v3 → 0,556): o corte reduziu o "chute" de classes minoritárias; a distribuição de predições passou a casar com a verdade.
- Erro residual principal (matriz de confusão): **24 melanomas classificados como NEV** (23,8% dos MEL) — o falso-negativo clinicamente mais grave, alvo do v7.
- O gap de ~10pt entre HAM (in-domain, 83%) e Derm7pt (cross-dataset, 73%) é honesto de reportar: mede generalização entre fontes.

---

### v7 — Corte profundo de NEV + threshold conservador para MEL

#### Plano

Duas mudanças sobre o v6, mantendo arquitetura, LR, batch, FocalLoss e cache de embeddings:

**1. Corte mais agressivo de NEV (`NEV_TARGET`).** O v6 deixava ~3.700 NEV no train (ainda ~3,6× o MEL). O v7 reduz o NEV do HAM a **1.400 imagens**, atacando o viés residual NEV→ que causava os 24 MEL→NEV.

**2. Threshold conservador para MEL (regra de decisão pós-treino).** Após o softmax, prediz MEL se `P(MEL) ≥ τ` mesmo que não seja o argmax. "Conservador" = τ baixo → captura mais melanomas ao custo de mais falsos positivos. O τ é **calibrado no Derm7pt val** (nunca no test, para não vazar) e o sweep vira uma **análise de sensibilidade**.

#### Resultados — Derm7pt test (argmax)

accuracy **73,7%** | macro-F1 **0,612** — **melhor modelo do trabalho**.

| Classe | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| BCC | 0,500 | 0,688 | 0,579 | 16 |
| NEV | 0,841 | 0,845 | 0,843 | 219 |
| MEL | 0,691 | 0,663 | 0,677 | 101 |
| SK | 0,450 | 0,474 | 0,462 | 19 |
| MISC | 0,528 | 0,475 | 0,500 | 40 |

**Progressão do corte de NEV (MEL recall, mesmo test):** v4 sem HAM 50% → v6 corte 2k 61% → **v7 corte 1,4k 66%**. Cada aperto no NEV subiu o recall de melanoma, **sem derrubar o NEV** (F1 NEV 0,832 → 0,843).

#### Análise de sensibilidade do threshold (calibração no val)

| τ | MEL recall | MEL precision | macro-F1 | accuracy |
|---|---|---|---|---|
| **argmax (~0,50)** | 0,607 | 0,787 | **0,7185** | 0,744 |
| 0,45 | 0,639 | 0,750 | 0,7175 | 0,739 |
| 0,40 | 0,639 | 0,709 | 0,694 | 0,724 |
| 0,35 | 0,689 | 0,646 | 0,670 | 0,704 |
| 0,30 | 0,803 | 0,583 | 0,661 | 0,680 |
| 0,25 | 0,836 | 0,520 | 0,643 | 0,635 |
| 0,20 | 0,902 | 0,466 | 0,615 | 0,586 |

- O macro-F1 é **máximo em argmax / τ≈0,45**. Abaixo disso, troca-se precisão e accuracy globais por sensibilidade a melanoma.
- No test, τ=0,30 elevou o MEL recall a **79%** (apenas 8 melanomas perdidos como NEV), mas a precisão do MEL caiu para 53% (60 nevos benignos flagados como melanoma) e o macro-F1 para 0,574.
- **Interpretação clínica:** o threshold é um **dial de sensibilidade × especificidade**, não um ganho líquido. O ponto de operação deve refletir a tolerância a falsos negativos (rastreio prioriza sensibilidade) vs falsos positivos (biópsias desnecessárias).

#### Decisão

- **Modelo final reportado: v7 argmax** (73,7% accuracy, MEL recall 66%, macro-F1 0,612).
- O sweep de threshold é apresentado como **análise de sensibilidade**, demonstrando que o ponto de operação é ajustável conforme a prioridade clínica.
- Cortar NEV abaixo de 1.400 não compensa: o ganho de macro-F1 já estava platôando (v6 0,608 → v7 0,612). O lever seguinte seria qualidade de features — testado no v8 (unfreeze), com resultado negativo.

---

### v8 — Unfreeze dos 2 últimos blocos do SigLIP (resultado negativo)

#### Plano

Testar se descongelar os **2 últimos blocos transformer** do encoder MedSigLIP (treinando-os com LR baixo) melhora a separação MEL/NEV — atacando os falsos-negativos de melanoma que o threshold só desloca, mas não elimina. Hipótese: adaptar as features ao domínio de lesões cutâneas reduziria os MEL→NEV de forma genuína.

#### Configuração

- **Unfreeze:** apenas os 2 últimos de 27 blocos (`encoder.layers[-2:]` do `SiglipVisionModel`) — 30,5M params treináveis (7,4% do total), vs ~330k da cabeça isolada.
- **LR discriminativo:** cabeça `5e-4`, blocos do encoder `1e-5` (LR alto destruiria o pré-treino médico). Param groups separados no AdamW.
- **Sem cache de embeddings:** descongelar invalida o cache (o encoder muda a cada época). Augmentation reativada no train.
- **`WeightedRandomSampler`:** mantém todos os dados (corte NEV=1400) mas balanceia os batches, em vez de descartar mais NEV. Class weights da loss mantidos.
- **EPOCHS=15** + `ReduceLROnPlateau(patience=2)` + early stopping (patience=4).

#### Resultados (interrompido na época 6)

| Época | train_acc | val_acc | val macro-F1 |
|---|---|---|---|
| 1 | 0,489 | 0,635 | 0,514 |
| 6 | **0,810** | 0,645 | 0,555 |

- **Overfitting claro:** o train_acc disparou (+0,32 em 5 épocas) enquanto o val ficou estagnado (+0,01). O modelo decorou o treino sem generalizar.
- **Pior que o baseline congelado:** no mesmo `derm_val`, o v8 atingiu macro-F1 **0,555** vs **0,7185** do v7 — uma queda de **0,16**. O unfreeze não recuperaria essa diferença enquanto já estava overfittando.
- Run interrompido: o val macro-F1 subia devagar mas de forma monótona, o que (a) impedia o early stopping de disparar e (b) levava ao risco de timeout (~50 min/época × 15 ≈ 12,5 h).

#### Conclusão

**Descongelar não compensou.** Com ~4,6k amostras de treino, ajustar 30,5M params do encoder leva a overfitting e **degrada** o macro-F1 vs o encoder congelado. O resultado confirma a decisão arquitetural central do trabalho: **usar o MedSigLIP como feature extractor congelado (v3–v7) foi a escolha correta** para o regime de dados disponível. Para descongelar valer a pena seria necessário muito mais dados de treino (ordem de dezenas de milhares) ou regularização bem mais forte.

> Nota: o v8 alterou vários fatores simultaneamente (unfreeze + `WeightedRandomSampler` + augmentation), então não é uma ablação isolada. O duplo balanceamento (sampler + class weights) pode ter contribuído para a queda. Mas a magnitude do gap (0,16 no val) e o overfitting evidente tornam improvável que um unfreeze isolado superasse o v7.

#### Percalços técnicos (documentados na seção 6)

O v8 exigiu resolver instabilidade numérica do fine-tuning: fp16 puro gerou `NaN` (overflow na atenção ao fazer backward pelo encoder), fp32 puro foi ~8× mais lento (T4 sem tensor cores em fp32), e a solução final foi **AMP** (master weights fp32 + `autocast` + `GradScaler`). Ver itens 6.11–6.14.

---

### Validação cruzada (5-fold) + TTA — caracterização final do v7

#### Motivação

Os conjuntos do Derm7pt são pequenos (val 203, test 395) e uma métrica de split único é ruidosa. Para reportar um número **robusto com variância** e testar um ganho barato de inferência, aplicou-se validação cruzada 5-fold estratificada + **test-time augmentation (TTA)** sobre o pipeline v7 (encoder congelado, corte NEV=1400).

#### Metodologia

- **5-fold estratificado** sobre os 1011 casos do Derm7pt (cada caso testado exatamente 1×; estratificado por classe).
- Cada fold: cabeça treinada em **(4/5 Derm7pt + HAM cortado)**, testada no 1/5 de fora.
- Encoder congelado + cache → 5 treinos de cabeça em segundos.
- **TTA:** original + 4 views aumentadas (flip/rotação/brilho/contraste), média das probabilidades softmax.
- Épocas fixas (25, onde o v7 convergia) — reprodutível, sem seleção no test.

#### Resultados (média ± desvio, 5 folds)

| Métrica | PLAIN | TTA |
|---|---|---|
| **Accuracy** | **0,762 ± 0,020** | 0,745 ± 0,020 |
| Macro-F1 | 0,636 ± 0,053 | 0,646 ± 0,033 |
| MEL recall | 0,667 ± 0,056 | **0,723 ± 0,043** |
| MEL precision | 0,712 ± 0,048 | 0,659 ± 0,055 |

**F1 por classe (PLAIN → TTA):**

| Classe | PLAIN | TTA |
|---|---|---|
| BCC | 0,552 ± **0,202** | 0,661 ± 0,103 |
| NEV | 0,850 ± 0,014 | 0,832 ± 0,017 |
| MEL | 0,687 ± 0,036 | 0,688 ± 0,035 |
| SK | 0,482 ± 0,067 | 0,485 ± 0,081 |
| MISC | 0,610 ± 0,068 | 0,566 ± 0,047 |

#### Análise

- **Número robusto:** o v7 atinge **76,2% ± 2,0%** de accuracy em 5-fold. Fica acima do single-split oficial (73,7%) porque cada fold treina em ~808 casos (4/5) contra 413+203 do split oficial — **mais dados de treino por fold**. Ambos são válidos: o **73,7% é o comparável à literatura** (mesmo split oficial); o **76,2% ± 2,0% é a estimativa robusta**.
- **NEV é estável** (0,850 ± 0,014); **BCC é altamente variável** (0,552 ± **0,202**) — cada fold de test tem só ~8 BCC, então um erro move muito o F1. É limitação intrínseca ao **suporte pequeno**, não instabilidade do modelo.
- **TTA não é ganho líquido de accuracy** (−1,7pt; macro-F1 +0,01 está dentro do ruído), mas é um **lever de sensibilidade a melanoma**: MEL recall +5,6pt (66,7 → 72,3%), ao custo de precisão (−5,3pt). Também **estabiliza o BCC** (F1 0,552→0,661; desvio 0,20→0,10).

#### Conclusão

O v7 é robusto (**76,2% ± 2,0%** em 5-fold). O trabalho oferece **dois levers de sensibilidade a melanoma** — threshold conservador e TTA — ambos trocando precisão por recall, com o ponto de operação a critério clínico. A alta variância do BCC entre folds é reportada honestamente como limitação de suporte amostral, não como falha do modelo.

---

## 5. Comparação com a Literatura

| Modelo | Tipo | Accuracy no Derm7pt | Comentário |
|---|---|---|---|
| **GPT-4V** (OpenAI) | VLM proprietário ~1T params | 85% | Outro patamar (modelo gigante) |
| **SkinM2Former** (Yan et al., 2024) | Arquitetura especializada (Swin tri-modal) | 77% | Custom para Derm7pt, 200 epochs |
| **Este trabalho v7** | MedSigLIP + head + merge HAM10000 | **73,7%** (5-fold: 76,2% ± 2,0%) | Open-source, single image + tabular |
| **Este trabalho v3** | MedSigLIP + classification head | 69% | Só Derm7pt, sem merge |
| **LLaVA-13B** (Heinlein et al., 2024) | VLM open-source ~13B | 45% | Maior que MedGemma mas sem fine-tuning |
| Random baseline (5 classes) | — | ~20% | — |

**Posicionamento:** o melhor modelo (v7) supera modelos VLM open-source maiores (LLaVA-13B) e fica a ~3 pontos do estado-da-arte especializado (SkinM2Former), com fração da complexidade e tempo de treino — e usando apenas a imagem dermatoscópica + metadados demográficos básicos (sem o 7-point checklist que o SkinM2Former consome). O **73,7%** é no split oficial do Derm7pt (comparável às linhas acima); a validação 5-fold dá uma estimativa robusta de **76,2% ± 2,0%** (treina em mais dados por fold, protocolo diferente).

---

## 6. Problemas Técnicos Enfrentados

### 6.1 Incompatibilidade Intel Arc B580 com CUDA

A GPU local (Intel Arc B580 12GB VRAM) **não suporta CUDA**, apenas XPU via Intel Extension for PyTorch. O ecossistema de QLoRA/bitsandbytes requer CUDA, tornando execução local inviável. **Solução:** todos os treinos foram executados no Kaggle (NVIDIA T4 16GB, 30h grátis por semana).

### 6.2 OOM (Out of Memory) recorrente em v1

GPU T4 com 16GB ficava no limite com `bnb_4bit_compute_dtype=bfloat16` porque o T4 (arquitetura Turing) **não suporta bfloat16 em hardware** — simula em float32, dobrando memória. **Solução:** `bnb_4bit_compute_dtype=float16` + `fp16=False, bf16=False` (desabilita AMP GradScaler que era incompatível).

### 6.3 CheckpointError com gradient checkpointing + PEFT

Erro `"A different number of tensors saved during forward (196) and recomputation (62)"`. **Solução:** `use_reentrant=False` no gradient_checkpointing_kwargs.

### 6.4 Mode collapse em v1

Modelo passou a chutar "melanoma" para tudo. **Causa raiz:** LoRA agressivo (`alpha=32, lr=2e-4`) + resposta categórica de 1 token + 1 epoch incompleta + dataset balanceado 50/50 (modelo aprende que chutar a mesma classe dá ~50% accuracy).

### 6.5 Perda de dados durante inferência longa

Sessão Kaggle desconectou durante a noite, perdendo `predictions, labels, responses` em memória após inferência de 30+ min. **Solução implementada em v3:** save incremental do CSV a cada 20 predições durante o loop de inferência.

### 6.6 "No module named melanoma_tcc" no Save Version

`pip install -e` em Jupyter requer kernel restart para o pacote ficar visível, mas Save Version executa em kernel único sem restart. **Solução:**
```python
sys.path.insert(0, REPO_DIR)
importlib.invalidate_caches()
site.main()
```

### 6.7 Gemma3 processor exige texto

`processor(images=image)` falha com `'NoneType' object is not subscriptable` em Gemma3 porque o processor exige texto além de imagens. **Solução:** usar `processor.image_processor(images=image)` diretamente.

### 6.8 Gemma3 vision_tower em path diferente

`model.vision_tower` não funcionou em `Gemma3ForConditionalGeneration`. **Solução:** função `_find_vision_tower` que tenta múltiplos paths (`model.vision_tower`, `model.model.vision_tower`, `model.vision_model`, etc.).

### 6.9 Geração truncada em v2

Modelo produzia features mas truncava antes do diagnóstico final. **Hipótese:** image augmentation introduziu incerteza nos pares features→diagnóstico. **Solução:** migrar para paradigma classificatório (v3), eliminando geração de texto.

### 6.10 Extract function over-permissiva

Função inicial buscava "melanoma" em qualquer lugar do texto, classificando como MEL respostas como "...not melanoma" ou "rule out melanoma". **Solução:** regex em ordem de prioridade: (1) negrito `**X**`, (2) padrões "diagnosis is X", (3) busca livre como último recurso.

### 6.11 NaN ao descongelar o encoder em float16 (v8)

O MedGemma é carregado em `float16`. Enquanto o encoder ficou **congelado** (v3–v7), nenhum gradiente fluía por ele e o fp16 não causava problema. No v8, ao fazer **backward pelos 2 blocos descongelados**, a atenção em fp16 estourava (overflow → `inf` → `NaN`) por volta do batch 30. **Causa:** treinar pesos fp16 puros é numericamente instável. **Solução:** ver 6.13.

### 6.12 float32 puro é ~8× mais lento no T4 (v8)

Primeira tentativa de corrigir o NaN: castar tudo para `float32` (`model.float()`). Resolveu o NaN, mas o tempo por época saltou para ~3,3 h (≈20 s/batch). **Causa:** o T4 (Turing) só acelera matmuls com **tensor cores em fp16/bf16**; em fp32 cai para os CUDA cores, ~8× mais lento. Inviável (15 épocas ≈ 48 h).

### 6.13 Solução: AMP (autocast + GradScaler) (v8)

Combinação que dá estabilidade **e** velocidade: **master weights em fp32** (`model.float()`, updates estáveis) + **`autocast`** (compute em fp16, usa tensor cores) + **`GradScaler`** (escala a loss antes do backward, evitando o underflow que gerava NaN). Também é preciso `scaler.unscale_(optimizer)` antes do `clip_grad_norm_`. Resultado: ~50 min/época (≈4× mais rápido que fp32) e treino estável. Observação adicional: como `encode_image` usa `torch.enable_grad()` quando `_freeze_vision=False`, isso **sobrepõe** um `torch.no_grad()` externo — por isso o `evaluate()` força `_freeze_vision=True` temporariamente, evitando construir grafo (e consumir memória) durante a avaliação.

### 6.14 OOM ao descongelar com batch 32 (v8)

Backward pelo encoder (mesmo só nos 2 últimos blocos) retém ativações que o caminho congelado (`no_grad`) não retinha. Batch 32 estourou os 14,5 GB do T4 por ~1 GB. Diagnóstico confirmou que o language model **estava** sendo liberado (apenas 2,18 GB pós-build), então o gargalo era ativação pura. **Solução:** `BATCH_SIZE=16` (corta ativações pela metade) + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` contra fragmentação.

---

## 7. Decisões Arquiteturais Importantes

### 7.1 Por que QLoRA (em v1 e v2)

Fine-tuning completo de modelo 4B em T4 16GB é inviável. QLoRA permite:
- Quantização 4-bit do modelo base (8GB → 2.5GB)
- Treinamento apenas de adapters de baixa rank (~0.1-1% dos params)
- Estado de otimizador em 8-bit com paging CPU↔GPU (`paged_adamw_8bit`)
- Gradient checkpointing para reduzir ativações

### 7.2 Por que abandonar QLoRA em v3

O paradigma generativo (que QLoRA fine-tuna) provou ser inadequado para classificação multiclasse desbalanceada — mode collapse e respostas truncadas. Em v3, é melhor usar o vision encoder como feature extractor congelado e treinar uma head classificatória pequena.

### 7.3 Por que Focal Loss

Combina três vantagens:
1. **Class weights** (α): compensa desbalanceamento
2. **Focusing factor** (γ=2): foca em exemplos difíceis (especificamente MEL heterogêneo)
3. **Label smoothing** (ε=0.05): previne overconfidence (foi parte do problema em v1)

### 7.4 Por que congelar o vision encoder (v3–v7)

- MedSigLIP já foi pré-treinado em imagens médicas
- Treinar com poucos dados (de 413 a ~4,6k amostras) risca catastrophic forgetting
- Reduz drasticamente params treináveis (~330k vs ~400M)
- Treino estável em 15 epochs
- **Confirmado empiricamente no v8:** descongelar 2 blocos (30,5M params) overfittou e piorou o macro-F1 (0,56 vs 0,72 no val). A decisão de congelar não foi só conveniência — é a escolha correta para o regime de dados.

### 7.5 Por que separar branches multimodais (vision + tabular)

Em v1/v2, metadados eram **embutidos no prompt como texto natural** ("Patient: female, lesion on back..."). O modelo precisava aprender a parsear texto. Em v3, metadados são **one-hot vetors** passados diretamente — informação estruturada disponível imediatamente, sem custo de interpretação. Princípio: **inductive bias** apropriado ao tipo de dado.

### 7.6 Por que image augmentation foi mantida em v3 mas prejudicou v2

Em v2 (geração), augmentation introduzia incerteza sobre "qual diagnóstico verbalizar" para cada variação visual, levando o modelo a evitar comprometimento. Em v3 (classificação direta), o modelo só precisa associar features visuais ao argmax dos logits — augmentation atua como regularização sem confundir o output.

### 7.7 Por que 5 classes (e não 16) em Derm7pt

Seguindo Kawahara et al. (2019), agrupar reduz desbalanceamento extremo. Algumas subclasses originais tinham apenas 4-13 amostras, inviabilizando aprendizado. Agrupamento captura categorias diagnósticas clinicamente relevantes (BCC, NEV, MEL, SK, MISC) sem perder informação útil.

---

## 8. Estrutura do Código

```
e:\Projeto_TCC\
├── melanoma_tcc/                    # Pacote Python instalável
│   ├── data/
│   │   ├── __init__.py
│   │   └── preprocessing.py         # Datasets, augmentation, balanceamento, metadata encoding
│   ├── model/
│   │   ├── __init__.py
│   │   ├── finetuning.py            # QLoRA pipeline (v1, v2)
│   │   ├── inference.py             # Predict + label extraction
│   │   ├── classifier.py            # DermClassifier (v3)
│   │   └── losses.py                # FocalLoss + class weights
│   └── utils/
│       ├── __init__.py
│       └── metrics.py               # compute_metrics, plot_confusion_matrix
├── notebooks/
│   ├── 01_data_exploration.ipynb    # EDA inicial
│   ├── 02_baseline_inference.ipynb  # MedGemma sem fine-tuning (baseline binário)
│   ├── 03_finetuning.ipynb          # v2: VLM fine-tuning Derm7pt
│   ├── 04_classification.ipynb      # v3 + v4: Classification head
│   ├── 05_combined.ipynb            # v5: Derm7pt + HAM10000 (schema unificado)
│   ├── 06_combined_balanced.ipynb   # v6: corte de NEV + cache de embeddings
│   ├── 07_combined_threshold.ipynb  # v7: corte profundo NEV + threshold MEL
│   ├── 08_unfreeze_2blocks.ipynb    # v8: unfreeze de 2 blocos + AMP (negativo)
│   └── 09_kfold_tta.ipynb           # v7: validação cruzada 5-fold + TTA
├── release_v0/                      # Derm7pt dataset (não comitado, gitignored)
│   ├── images/                      # 34 subpastas, 2022 imagens (1011 dermatoscópicas + 1011 clínicas)
│   └── meta/
│       ├── meta.csv                 # 1011 linhas com metadados + diagnósticos
│       ├── train_indexes.csv        # 413 indices
│       ├── valid_indexes.csv        # 203 indices
│       └── test_indexes.csv         # 395 indices
├── docs/
│   └── DOCUMENTACAO_TECNICA.md      # Este arquivo
├── pyproject.toml                   # Definição do pacote
├── requirements.txt
└── .gitignore
```

### Módulos principais

#### `melanoma_tcc/data/preprocessing.py`
- `DIAGNOSIS_GROUPS`: mapeamento 16→5 classes
- `Derm7ptDataset`: dataset para v2 (geração de texto)
- `Derm7ptClassificationDataset`: dataset para v3/v4 (classificação, 13-dim)
- `encode_metadata`: encoding one-hot de sex/location/elevation (13-dim, v3/v4)
- `balance_dataframe`: undersample + oversample com cap
- `augment_image`: rotação/flip/brilho/contraste
- `classification_collate_fn`: collate específico (pixel_values + metadata + labels)
- **Schema unificado v5/v6** (mescla Derm7pt + HAM10000):
  - `SEX_CATEGORIES_V5`, `LOCATION_CATEGORIES_V5`, `METADATA_DIM_V5` (10-dim, sem domain leak)
  - `DERM7PT_LOCATION_MAP`, `HAM_LOCATION_MAP`, `HAM_DX_TO_GROUP`: harmonização de nomenclaturas
  - `encode_metadata_unified`: one-hot 10-dim a partir de sex + location unificada
  - `Derm7ptUnifiedDataset`, `HAM10000Dataset`: datasets com mesmo schema
  - `CombinedDermDataset`: concatena datasets de mesmo schema
  - `_filter_ham_unknowns`: descarta HAM com sex/location `unknown`
  - `ham10000_train_val_split`: split por `lesion_id` (evita leakage de lesão)

#### `melanoma_tcc/model/finetuning.py`
- `load_model_for_finetuning`: QLoRA com bnb 4-bit
- `apply_lora`: prepare_model_for_kbit_training + LoraConfig + cast float32
- `MemoryCleanupCallback`: limpa cache antes do eval e fim de epoch
- `get_trainer`: SFTConfig + SFTTrainer

#### `melanoma_tcc/model/classifier.py`
- `_find_vision_tower`: busca vision encoder em múltiplos paths possíveis
- `load_medgemma_vision`: extrai vision tower e libera language model
- `DermClassifier`: modelo multimodal (vision branch + metadata branch + classifier)
- `build_dermclassifier`: factory function

#### `melanoma_tcc/model/losses.py`
- `FocalLoss`: implementação completa com α (class weights), γ (focusing), label smoothing
- `compute_class_weights`: três modos (inverse, inverse_sqrt, effective)

#### `melanoma_tcc/model/inference.py`
- `load_model`: MedGemma sem fine-tuning
- `load_finetuned_model`: MedGemma com PEFT adapters
- `predict`: gera resposta dado imagem + prompt
- `extract_label_from_response`: binário (legado)
- `extract_multiclass_label`: regex em camadas (bold > pattern > free)

---

## 9. Trabalhos Futuros

### 9.1 Validação cruzada com PAD-UFES-20

Replicar a arquitetura v3 no dataset PAD-UFES-20 (2.298 imagens smartphone, 22 features clínicas, dataset brasileiro). Possíveis ganhos:
- Mais imagens (2.298 vs 1.011)
- Features tabulares muito mais ricas (idade, fototipo, sintomas como itch/bleed/grew)
- Relevância nacional (UFES, Brasil)

Desafio: apenas 52 amostras de melanoma exigem oversampling agressivo ou augmentation forte.

### 9.2 Multi-image fusion

Derm7pt fornece imagem dermatoscópica + clínica do mesmo caso. Atualmente usamos apenas dermatoscopia. Adicionar uma segunda branch para imagem clínica poderia capturar informações complementares (à la SkinM2Former).

### 9.3 Features ABCDE computadas da imagem

Pré-computar features clássicas de melanoma (asymmetry, border irregularity, color variance, diameter) via processamento clássico de imagem e adicionar à branch tabular. Pode capturar conhecimento médico estruturado que o modelo aprenderia mais lentamente sozinho.

### 9.4 Unfreeze parcial do MedSigLIP — testado (v8), resultado negativo

Descongelar os 2 últimos blocos com LR baixo (1e-5) **foi testado no v8** e levou a overfitting, piorando o macro-F1 vs o encoder congelado (0,56 vs 0,72 no val). Com ~4,6k amostras, ajustar 30,5M params do encoder é demais. Para retomar essa direção seria necessário: (a) **muito mais dados** de treino (dezenas de milhares — ex.: ISIC Archive completo), (b) regularização bem mais forte (dropout/weight decay altos, layer-wise LR decay), ou (c) descongelar **só 1 bloco** com LR ainda menor. Ver [seção 4 — v8](#v8--unfreeze-dos-2-últimos-blocos-do-siglip-resultado-negativo).

### 9.5 Hierarchical classification

Treinar dois classificadores em cascata:
1. Benigno (NEV, SK, MISC) vs Maligno (BCC, MEL)
2. Dentro de maligno: BCC vs MEL
3. Dentro de benigno: NEV vs SK vs MISC

Aproveita o fato de que classificação grosseira (maligno/benigno) é mais fácil que multiclasse direta.

### 9.6 Avaliação clínica qualitativa

Submeter as predições do modelo a dermatologistas para avaliação qualitativa: falsos negativos de melanoma são clinicamente graves; falsos positivos geram biópsias desnecessárias. Análise custo-benefício clínico.

### 9.7 Calibração de probabilidades

Aplicar temperature scaling ou Platt scaling para calibrar a confiança das predições. Útil para sistemas de apoio à decisão (médico precisa saber "quão certo" o modelo está).

---

## 10. Referências

### Datasets

- **Kawahara, J., Daneshvar, S., Argenziano, G., & Hamarneh, G.** (2019). Seven-point checklist and skin lesion classification using multitask multimodal neural nets. *IEEE Journal of Biomedical and Health Informatics*, 23(2), 538-546.

- **Rotemberg, V., Kurtansky, N., Betz-Stablein, B., Caffery, L., Chousakos, E., Codella, N., ... & Halpern, A.** (2021). A patient-centric dataset of images and metadata for identifying melanomas using clinical context. *Scientific Data*, 8(1), 34. (ISIC 2020)

- **Pacheco, A. G., Lima, G. R., Salomão, A. S., Krohling, B., Biral, I. P., de Angelo, G. G., ... & Krohling, R. A.** (2020). PAD-UFES-20: a skin lesion dataset composed of patient data and clinical images collected from smartphones. *Data in Brief*, 32, 106221.

- **Tschandl, P., Rosendahl, C., & Kittler, H.** (2018). The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. *Scientific Data*, 5, 180161.

### Modelos

- **Google DeepMind** (2025). MedGemma Technical Report. arxiv:2507.05201

- **Sengupta, A., et al.** (2025). Fine-Tuning MedGemma for Clinical Captioning. arxiv:2510.15418

### Métodos

- **Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., ... & Chen, W.** (2021). LoRA: Low-rank adaptation of large language models. *arxiv:2106.09685*

- **Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L.** (2023). QLoRA: Efficient finetuning of quantized LLMs. *NeurIPS 2023*. arxiv:2305.14314

- **Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P.** (2017). Focal loss for dense object detection. *ICCV 2017*. (Focal Loss)

- **Cui, Y., Jia, M., Lin, T. Y., Song, Y., & Belongie, S.** (2019). Class-balanced loss based on effective number of samples. *CVPR 2019*.

### Trabalhos comparativos

- **Yan, S., et al.** (2024). A novel perspective for multi-modal multi-label skin lesion classification (SkinM2Former). arxiv:2409.12390

- **Heinlein, L., et al.** (2024). Assessing the utility of multimodal LLMs in identifying melanoma. PMC10973960

---

## Apêndice A — Histórico de Commits Relevantes

| Commit | Mudança |
|---|---|
| Inicial | Estrutura do projeto (src/) |
| `83dbcb6` | Rename src/ → melanoma_tcc/ para evitar namespace conflict |
| `dca0d89` | Adiciona Derm7ptDataset multiclasse + config LoRA suave |
| `084bae0` | Fix Cell 1 para Save Version (sys.path.insert + site.main) |
| `9b70af5` | Adiciona Opção B: classification head com MedSigLIP + Focal Loss |
| `94f5510` | Fix vision_tower lookup com fallback robusto |
| `69019c3` | Fix Derm7ptClassificationDataset usa image_processor diretamente |
| `cd09574` | v4: remove undersample do train + class weights `inverse` |
| `79d6a42` | v5: pipeline mesclado Derm7pt + HAM10000 com schema unificado |
| `71c16ba` | v5: schema 10-dim sem unknown + filtro de HAM unknowns (244 samples) |
| `f52b27a` | v6: corte de 2k NEV + cache de embeddings (encoder congelado) |
| `e755fa7` | v7: corte profundo de NEV (NEV_TARGET) + threshold conservador para MEL |
| `b9769ac` | v8: unfreeze dos 2 últimos blocos do SigLIP com LR discriminativo |
| `7165448` | v8: AMP (autocast + GradScaler) para corrigir NaN/lentidão do unfreeze |
| `60bf30c` | v7 kfold+tta: validação cruzada 5-fold estratificada + TTA |

---

## Apêndice B — Configurações de Hiperparâmetros Finais (v3)

```python
# Arquitetura
metadata_dim = 13         # sex(2) + location(8) + elevation(3)
num_classes = 5           # BCC, NEV, MEL, SK, MISC
hidden_dim = 256          # projeção comum

# Treino
batch_size = 16
epochs = 15
learning_rate = 5e-4      # mais alto que VLM porque só treina head
weight_decay = 0.01
optimizer = AdamW
scheduler = CosineAnnealingLR (T_max=epochs)
gradient_clip = 1.0

# Loss
focal_gamma = 2.0
label_smoothing = 0.05
class_weights_mode = "inverse_sqrt"

# Dataset
balance = True
target_per_class = 80
max_oversample = 4
augment = True (train only)
seed = 42
```

---

*Documento gerado em junho de 2026.*
