# RL-Emo-Per · Empathetic Persuasion

Code and dialogue data for **[Empathetic Persuasion: Reinforcing Empathy and Persuasiveness in Dialogue Systems](https://aclanthology.org/2022.findings-naacl.63/)**.

**Azlaan Mustafa Samad · Kshitij Mishra · Mauajama Firdaus · Asif Ekbal**

Indian Institute of Technology Patna

*Findings of NAACL 2022*, pages 844–856

[Paper](https://aclanthology.org/2022.findings-naacl.63.pdf) · [Dataset](datasets/README.md) · [Method](docs/method.md) · [Results](docs/results.md) · [BibTeX](CITATION.bib)

RL-Emo-Per studies how a dialogue system can express empathy while persuading a
conversation partner to support a charitable cause. The system starts from
speaker-specific language models and uses emotion, persuasion, consistency and
repetition feedback to improve the persuader's responses.

## Dataset

The dataset contains **385 conversations and 7,906 utterances** about donations
to Save the Children. Its **3,998 persuader turns** each have an emotion label
and a persuasion-strategy label. The **3,908 persuadee turns** supply the other
side of the conversation. The task predicts labels for persuader responses;
persuadee target fields are empty.

| Property | Value |
| --- | ---: |
| Conversations | 385 |
| Total utterances | 7,906 |
| Persuader responses with both targets | 3,998 |
| Emotion classes | 23 |
| Persuasion classes | 11 |
| Training / validation conversations | 346 / 39 |

The 23-class emotion distribution matches the paper. Ten persuasion strategies
and `no_strategy` form the 11-class persuasion task. Full label names, integer
IDs, per-class counts and split statistics are in the [dataset guide](datasets/README.md).

![Emotion distribution across the 3,998 persuader responses](docs/assets/emotion-distribution.png)

![Persuasion-strategy distribution across the 3,998 persuader responses](docs/assets/strategy-distribution.png)

Files are ordinary UTF-8 CSVs, ready to load without extracting an archive:

| File | Contents |
| --- | --- |
| [`datasets/dialogues.csv`](datasets/dialogues.csv) | Complete conversations and target labels |
| [`datasets/train.csv`](datasets/train.csv) | Training conversations |
| [`datasets/validation.csv`](datasets/validation.csv) | Held-out validation conversations |
| [`datasets/labels.json`](datasets/labels.json) | Emotion and persuasion class IDs |
| [`datasets/manifest.json`](datasets/manifest.json) | Dataset statistics and SHA-256 checksums |

Splits are made by complete dialogue, with seed 42. Every training class is
represented, and no conversation appears in both splits.

```python
from rl_emo_per.data import load_records, dialogue_examples

rows = load_records("datasets/train.csv")
examples = dialogue_examples(rows, speaker="persuader")
print(examples[0]["prompt"])
print(examples[0]["text"], examples[0]["emotion"], examples[0]["strategy"])
```

## Quick start

Use Python 3.10 or newer; Python 3.11 is used in CI.

```bash
git clone git@github.com:Mishrakshitij/RL-Emo-Per.git
cd RL-Emo-Per
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m rl_emo_per verify-data
```

Dataset inspection uses only the Python standard library. Install the training
dependencies to run the models:

```bash
python -m pip install -e '.[train]'
python -m nltk.downloader wordnet omw-1.4
```

Training downloads the selected Hugging Face base models on first use. Pass a
local checkpoint directory to `--model-name` to use an existing model. A CUDA
GPU is recommended for GPT-2 medium and RoBERTa-large;
`--device cpu` is available for small runs.

## Training and inference

### 1. Train the reward classifiers

```bash
python -m rl_emo_per train-classifier --task emotion \
  --output-dir checkpoints/emotion
python -m rl_emo_per train-classifier --task strategy \
  --output-dir checkpoints/strategy
```

Both classifiers use RoBERTa-large, learning rate `2e-5`, and batch size `32` by
default. They save class mappings, validation loss, accuracy and macro F1 with
the checkpoint. For a smaller model or GPU, set `--model-name roberta-base`
and `--batch-size 8`. The optional `--task binary` trains a two-class head where
`strategy_id != 0` means persuasive.

### 2. Train the speaker models

```bash
python -m rl_emo_per train-mle --speaker persuader \
  --output-dir checkpoints/mle-persuader
python -m rl_emo_per train-mle --speaker persuadee \
  --output-dir checkpoints/mle-persuadee
```

Each GPT-2 medium model predicts its speaker's response from the preceding
dialogue. Prompts use `A:` for the persuader and `B:` for the persuadee. Only
response tokens contribute to the loss. The persuader checkpoint initializes
the RL stage; the persuadee checkpoint provides the second speaker model.

### 3. Fine-tune the persuader with reinforcement learning

```bash
python -m rl_emo_per train-rl \
  --model-name checkpoints/mle-persuader \
  --emotion-model checkpoints/emotion \
  --strategy-model checkpoints/strategy \
  --output-dir checkpoints/rl-emo-per
```

The trainer samples two candidates per context and adds the dataset response.
It scores responses, normalizes rewards, and optimizes a clipped sequence policy
objective. **Cross-entropy, sequence log probabilities and PPO clipping are
implemented directly in PyTorch; TRL is not used.** The reward module combines
classifier probabilities, Jaccard overlap and NLTK METEOR.

| Setting | Default |
| --- | --- |
| Repetition / consistency / emotion / strategy weights | `0.1 / 0.1 / 0.55 / 0.25` |
| Emotion and strategy penalty, β | `2.0` |
| PPO clip range, ε | `0.2` |
| Nucleus sampling / temperature | `0.9 / 0.8` |
| RL learning rate | `2e-5` |
| Dataset-response reward | `10.0` |

The [method notes](docs/method.md) describe the exact objectives, sampling
convention, context handling and configuration choices. `--max-steps 2` runs a
short training check. All training commands evaluate on the validation split
and save the best checkpoint; `--help` lists the available settings.

### 4. Generate and evaluate

```bash
python -m rl_emo_per generate \
  --model-path checkpoints/rl-emo-per \
  --history examples/history.json

python -m rl_emo_per evaluate \
  --model-path checkpoints/rl-emo-per
```

The history file is a JSON list of objects with `speaker` (`persuader` or
`persuadee`) and `text` in conversation order. Generation ends at the turn
delimiter, EOS token or length limit. Evaluation reports response-token NLL and
perplexity on `datasets/validation.csv`.

## Results reported in the paper

| Model | Persuasive strategy (%) ↑ | Emotion probability (%) ↑ | Perplexity ↓ | Response length |
| --- | ---: | ---: | ---: | ---: |
| ARDM | 49.20 | — | 12.45 | 15.03 |
| RFI | 51.20 | — | 12.38 | 19.36 |
| **RL-Emo-Per** | **55.42** | **58.10** | **11.25** | 16.75 |

These are the paper's experimental results. See [results and evaluation](docs/results.md)
for human evaluations and the checks of this implementation.

## Repository layout

```text
RL-Emo-Per/
├── datasets/             # Dialogue CSVs, label maps, counts and checksums
├── src/rl_emo_per/       # Data loading, models, custom losses, rewards and CLI
├── tests/                # Numerical tests and small model integration tests
├── scripts/              # Dataset figure generation
├── examples/             # Input conversation for inference
├── docs/                 # Method notes, results and figures
├── licenses/             # PersuasionForGood license and attribution
├── CITATION.bib
└── CITATION.cff
```

Run the tests and regenerate the dataset figures:

```bash
python -m pip install -e '.[test,figures]'
python -m unittest discover -s tests -v
python scripts/plot_dataset.py
```

## Citation

```bibtex
@inproceedings{samad-etal-2022-empathetic,
  title = {Empathetic Persuasion: Reinforcing Empathy and Persuasiveness in Dialogue Systems},
  author = {Samad, Azlaan Mustafa and Mishra, Kshitij and Firdaus, Mauajama and Ekbal, Asif},
  booktitle = {Findings of the Association for Computational Linguistics: NAACL 2022},
  year = {2022},
  pages = {844--856},
  doi = {10.18653/v1/2022.findings-naacl.63},
  url = {https://aclanthology.org/2022.findings-naacl.63/}
}
```

The conversations build on [PersuasionForGood](https://gitlab.com/ucdavisnlp/persuasionforgood).
Please also cite its dataset paper from [CITATION.bib](CITATION.bib).
Its source terms are included under [licenses/](licenses/README.md).
