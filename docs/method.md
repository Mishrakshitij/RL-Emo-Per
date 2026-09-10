# Implementation notes

RL-Emo-Per combines speaker-conditioned maximum-likelihood training, two reward
classifiers and a sequence-level clipped policy objective. The paper is
[Samad et al., Findings of NAACL 2022](https://aclanthology.org/2022.findings-naacl.63/).

## Dialogue models

Two GPT-2 models are trained separately, one for each speaker. A target's prompt
contains all earlier turns, with `A:` and `B:` role prefixes; three newlines end
a turn. Each model encodes the textual history using its own parameters. This
implements the two conditional speaker likelihoods without transferring a KV
cache computed by another model.

The response mask excludes the history and padding. With target tokens
`y[1:L]`, the sequence log probability is the sum of next-token log probabilities:

```text
log p(y | x) = Σ[t=1..L] log p(y[t] | x, y[:t])
L_NLL = −sum(mask × token_log_probability) / sum(mask)
```

MLE training uses label smoothing of `0.02`: each target distribution is mixed
with the uniform vocabulary distribution before computing cross-entropy.
Validation NLL and perplexity use unsmoothed targets. The Python
`train_mle(..., label_smoothing=0.0)` API selects the unsmoothed training
objective shown above.

Supervised responses include the turn delimiter and EOS. Token packing retains
the most recent history and limits sequences to the model's context window.
Overlong supervised responses are truncated to fit. Generation reserves room
for its configured response length and retains the exact sampled token IDs.

## Rewards

The four components are computed per generated response:

```text
R_repetition = −Jaccard(previous persuader response, candidate)
R_consistency = METEOR(dataset response, candidate)
R_emotion = p(gold emotion | candidate) − β × (1 − p(gold emotion | candidate))
R_strategy = p(gold strategy | candidate) − β × (1 − p(gold strategy | candidate))
R = 0.1 R_repetition + 0.1 R_consistency + 0.55 R_emotion + 0.25 R_strategy
```

`β = 2` by default. The repetition component is negative overlap, matching the
penalty used in the released implementation. The paper displays unsigned
Jaccard in its equation; `repetition_reward(..., positive_overlap=True)` exposes that convention
for direct experiments. The training CLI uses the penalty.

Jaccard defaults to lowercased word sets. To use spaCy lemma normalization,
install spaCy and an English pipeline, then pass `--spacy-model en_core_web_sm`:

```bash
python -m pip install 'spacy>=3.7,<4'
python -m spacy download en_core_web_sm
```

METEOR uses NLTK's implementation with WordNet synonym matching. Missing resources
produce an explicit error. Reward classifiers are frozen and evaluated with
dropout disabled. Their checkpoint class names must match the dataset label map.

## Policy objective

By default, each context contributes two sampled candidates and the dataset
response, which has a fixed reward of 10. Advantages are reward values normalized across the rollout
batch; a constant-reward batch produces zero advantages. No value network is used.

```text
A = (R − mean(R)) / population_std(R)
r = exp(log p_active(response | context) − log p_old(response | context))
L_PPO = −mean(min(r × A, clip(r, 1 − ε, 1 + ε) × A))
```

The old policy is a frozen copy of the active model at collection time. It stays
fixed during the configured PPO epochs and is synchronized before the next
rollout. Old log probabilities and advantages are detached from gradients.
Both models disable dropout when computing the policy ratio. Only the active
model receives optimizer updates.

Nucleus sampling (`top_p=0.9`, `temperature=0.8`) proposes candidates; the ratio
uses the underlying language-model probabilities before this sampling
transformation. Gold responses are also supplied externally. This is an offline
clipped surrogate following the paper's sampling convention, rather than an
unbiased on-policy estimator of the nucleus sampling distribution. Set
`--top-p 1 --temperature 1` for unwarped candidate sampling; the dataset-response
term remains a demonstration term.

The implementation corrects the loss sign in the original prototype, uses
summed response log probabilities, normalizes advantages and refreshes the old
policy. It uses the paper's reward weights. These choices are explicit so
experiments can be interpreted consistently.

## Classifiers and checkpoints

RoBERTa-large is the default for both the 23-class emotion head and the 11-class
strategy head. An optional binary head uses `strategy_id != 0`. Cross-entropy
is implemented in `losses.py`. Evaluation computes example-weighted loss,
accuracy and macro F1 over the complete task vocabulary.

Training writes a Hugging Face model and tokenizer together with configuration
and validation metrics. The best validation checkpoint is retained. Checkpoints
used for emotion and strategy rewards require the named label maps to match
the dataset, preventing silent label-order mismatches.
Training and validation must contain disjoint conversations with matching split
tags. The supplied split is a deterministic 346/39 dialogue partition.

The commands provide CPU/GPU selection and bounded training through `--max-steps`.
For classifiers and MLE, a step is one optimizer update. For RL, a step is one
rollout batch followed by `--ppo-epochs` optimizer updates.
The full paper configurations require substantial model training; small-model
tests verify executable behavior and numerical objectives.
