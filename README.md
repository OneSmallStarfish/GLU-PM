# GLU-PM: A Pre-trained Model Towards Generalized Log Understanding

## abstract
Log understanding is essential for maintaining system reliability and security. Although pre-trained models have improved log analysis, they still face limitations in generalizing to diverse log-related tasks due to structural regularities and domain-specific challenges. To address these issues, this paper proposes GLU-PM, a Generalized Log Understanding Pre-trained Model. First, GLU-PM applies the Hilbert-Schmidt Independence Criterion (HSIC) to reduce statistical correlations among features, which arise from the standardized and semi-structured design of log data. It alleviates the tendency of the model to overfit superficial patterns and enhances the ability to capture deeper semantic relationships. Moreover, the model is pre-trained on knowledge-enriched log corpora with two domain-specific objectives—contrastive learning and abbreviation prediction—to better capture contextual semantics and specialized terminology. Experimental results on multiple log analysis tasks demonstrate the strong generalization ability, domain adaptability, and practical value of GLU-PM.

![Framework of GLU-PM](https://github.com/OneSmallStarfish/GLU-PM/blob/main/fig/fig1.png)

## Installation
### Requirements
- torch==2.1.0
- transformers==4.38.2
- huggingface-hub==0.21.3


## Quick Start
### Pre-training

Our code employs the **“roberta-base”** model as the backbone.

```
python pretrain_all.py
```

### Downstream Tasks
```
python finetune_MC.py
```

### External validation
```
python finetune_external.py
```

## Acknowledgements
Our code is developed with reference to [KnowLog](https://github.com/LeaperOvO/KnowLog) and [StableNet](https://github.com/xxgege/StableNet).
