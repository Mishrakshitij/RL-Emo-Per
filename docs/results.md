# Results and evaluation

The following values are reported in
[Samad et al. (2022)](https://aclanthology.org/2022.findings-naacl.63/).

## Automatic evaluation

| Model | PerStr (%) ↑ | EmoPr (%) ↑ | Perplexity ↓ | Response length |
| --- | ---: | ---: | ---: | ---: |
| ARDM | 49.20 | — | 12.45 | 15.03 |
| RFI | 51.20 | — | 12.38 | 19.36 |
| RL-Emo-Per | **55.42** | **58.10** | **11.25** | 16.75 |

PerStr measures the percentage of responses with a persuasive strategy. EmoPr
is the paper's emotion-probability metric. Length is the average number of
generated tokens. These results come from the paper's training and evaluation
setting over PersuasionForGood; the downloadable 385-dialogue dataset provides
the emotion-labeled subset and its conversation context.

## Human evaluation

| Model | Persuasiveness ↑ | Empathy ↑ | Donation rate ↑ | Consistency ↑ | Fluency ↑ | Non-repetition ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ARDM | 2.33 | — | 0.50 | 3.95 | 4.17 | 3.17 |
| RFI | 2.98 | — | 0.61 | 4.17 | 4.41 | 3.50 |
| RL-Emo-Per | **3.91** | **3.51** | **0.68** | **4.59** | **4.62** | **3.89** |

Ratings use a 1–5 scale, except the donation rate, which is a proportion.

## Evaluating this codebase

`python -m rl_emo_per evaluate --model-path CHECKPOINT` computes held-out,
response-token-weighted NLL and perplexity. It excludes prompt and padding
tokens. Classifier training reports validation loss, accuracy and macro F1.
These are measured outputs of the supplied commands.

Unit tests check analytical loss values, gradients, clipping, padding and
reward formulas. Tiny local model tests exercise classifier training,
speaker-model training, checkpoint loading, generation and RL updates without
downloading a pretrained model. They validate the software; the table above
is not a claim that those tests reproduce the paper's benchmark scores.
