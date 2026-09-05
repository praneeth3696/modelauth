# ModelAuth

Detecting silent model substitution in black-box LLM APIs, without a trusted
reference model.

You pay for a large model; the provider quietly serves a smaller one. The API
returns only text, so weights and logits are unavailable and API keys
authenticate the server, not the model. ModelAuth sends cheap probes alongside
normal traffic and watches the answer distribution for the moment it stops
matching what the endpoint used to look like.

**What is actually new here is the adversary.** Self-baselined sequential
monitoring already exists ([Leshin et al. 2026](https://arxiv.org/abs/2603.19022)),
and so does single-token fingerprinting ([Bruckner 2026](https://arxiv.org/abs/2607.10252)).
This repository reproduces those as baselines and measures what happens when the
provider fights back: partial routing, probe-aware caching, distribution
matching, and the cost saving each one buys.

See **[PAPER.md](PAPER.md)** for the argument and **[RESULTS.md](RESULTS.md)**
for every table with the command that produces it.

## Headline findings

- **Quantization substitution is close to invisible on the numeric probe
  channel.** q4_K_M vs q8_0 gives KL = 0.152 nats/probe and a single-probe Bayes
  error of 0.407. Even an oracle handed both distributions reaches 21.4% power.
- **Below 25% routing, no detector beats its own noise floor.** A provider
  cheating on a fifth of traffic is safe here.
- **Published probes are trivially filterable.** A one-line keyword match catches
  78.3% of the probe bank while misrouting 2.5% of organic traffic. The three
  templates this project started with are the most detectable probes in it.
- **The anytime-valid guarantee holds in practice.** Swept from α=1e-1 to 1e-6
  across three tiers, the e-process fires on 0 of 15 clean streams every time.
- **Identification never picks the wrong model family** — 42/42 streams — though
  it cannot separate quantizations of the same weights.

## Install

```bash
cd substitution-sim
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run

Everything below works on the logged data already in this repository. **No
Ollama required.**

```bash
cd substitution-sim
python audit_data.py               # parse completeness + per-tier separability
python evaluate.py                 # main 3-tier benchmark
python calibration.py              # nominal vs achieved false-alarm rate
python matched_operating_point.py  # every detector at a matched FA budget
python run_adversarial.py --tier easy --trials 30
python detector_lineup.py          # which model is being served?
python camouflage.py               # can a provider spot your probes?
python probe_bank.py               # candidate probe inventory
```

### Needs Ollama

```bash
ollama serve
python run_experiments.py --check          # verify models before generating
python run_experiments.py medium hard      # regenerate streams
python run_probe_survey.py --check         # ~25k generations, resumable
python run_probe_survey.py
python probe_selection.py                  # needs the survey above
python run_adversarial.py --online         # live routing, not resampling
```

## Layout

```
substitution-sim/
  config.py                    tiers, probe templates, generation budget
  probe_client.py              Ollama REST client
  simulator.py                 stream generator
  run_experiments.py           resumable generation, preflights models
  data_loader.py               parsing with an explicit reject taxonomy
  stats_categorical.py         PMF, KL, TV, Bayes error, Lorden reference

  detector_v1.py               sliding-window KS
  detector_cusum.py            adaptive CUSUM on the mean
  detector_variance_cusum.py   chi-square style variance CUSUM
  detector_fixed_reference.py  held-out reference comparison
  detector_compression.py      prequential MDL e-process (+ mixture alternative)
  detector_baselines_2026.py   energy distance [Leshin], JS fingerprint [Bruckner]
  detector_oracle.py           LR-CUSUM given both PMFs -- upper bound
  detector_lineup.py           posterior over candidate models

  adversary.py                 provider strategies
  run_adversarial.py           the detectability frontier
  probe_bank.py                83 candidate probes across 5 families
  run_probe_survey.py          sample the bank (needs Ollama)
  probe_selection.py           information-per-token ranking + portfolio
  camouflage.py                probe-vs-organic detectability

  evaluate.py                  main benchmark
  calibration.py               anytime-validity check
  matched_operating_point.py   matched false-alarm comparison
  audit_data.py                data completeness + separability

final-analysis/                figures, dashboards, CSV outputs
docs/archive/                  superseded reports, kept for provenance
```

## Two constructions that look right and are not

Both are documented in `detector_compression.py` because both cost real time.

**Excess surprisal.** `-log p(x) - H(p)` is mean-zero under the null but
`E[exp(·)] = K·exp(-H) >> 1`, so Ville's inequality does not apply.

**Frozen null vs adaptive alternative.** `E[q/p_0] = 1` requires `p_0` to be the
true null law, not a warmup estimate of it. The adaptive code compresses better
purely from having seen more data: **+92.9 nats of drift on a clean stream and a
100% false-alarm rate at every α**. Making both codes prequential fixes it.

## Caveats

- The adversarial frontier is measured offline by resampling logged responses,
  which destroys within-session run structure. Read the shape, not the intercept,
  until `--online` has been run.
- 15 streams per tier puts the smallest resolvable false-alarm rate at 1/15.
- The 2026 comparators are reconstructions from published descriptions, adapted
  to a single-token integer channel — not the authors' code.
- `camouflage.py` ships a 40-line stand-in organic corpus. Point `--organic` at a
  real prompt log before quoting its numbers.

## License

MIT.
