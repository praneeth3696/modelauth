# Results

Every table here is produced by a script in `substitution-sim/`. Nothing is
hand-transcribed. 14 test repetitions per tier; rep14 is held out for reference
fits, oracle fits, threshold tuning, and the lineup, so no number is scored on
data used to build it.

Reproduce all of it (no Ollama needed):

```bash
cd substitution-sim
python audit_data.py            # parse completeness + per-tier separability
python evaluate.py              # main benchmark
python calibration.py           # does the anytime-valid guarantee hold?
python matched_operating_point.py
python run_adversarial.py --tier easy --trials 30
python detector_lineup.py
python camouflage.py
```

---

## 1. Data audit

30 easy / 30 medium / 30 hard streams plus 75 cold-start streams. 86 of 12,000
easy-tier probes rejected (0.72%), the bulk of them `max_tokens` truncations that
the old `\d+` parser accepted as real values: `854` (×42), `842`, `814`, `8549`,
`8425`. Those 76 out-of-range values inflated the pooled `llama3.2:3b` standard
deviation from **18.25 to 349.09**.

## 2. The channel

`python audit_data.py`

| tier | substitution | entropy A/B (bits) | KL(B‖A) | TV | Bayes error | Lorden ref |
|---|---|---|---:|---:|---:|---:|
| easy | llama3.2:3b → qwen2.5:3b | 4.16 / 4.33 | 2.149 | 0.725 | 0.138 | 4.3 |
| medium | llama3.2:1b → llama3.2:3b | 5.31 / 4.55 | 1.450 | 0.635 | 0.183 | 6.4 |
| hard | q4_K_M → q8_0 | 4.15 / 4.59 | **0.152** | **0.186** | **0.407** | 60.6 |

The hard tier is 14× weaker than easy. A single probe distinguishes q4 from q8
barely better than a coin flip.

## 3. Main benchmark

`python evaluate.py` — `alpha = 1e-4`, defaults for everything else.

**easy**

| detector | delay | ×oracle | power | FA rate | streams w/ FA |
|---|---:|---:|---:|---:|---:|
| oracle LR-CUSUM *(upper bound)* | 1.29 | 1.0× | 100% | 0.000% | 0/14 |
| adaptive CUSUM | 8.50 | 6.6× | 100% | 0.220% | 7/14 |
| KS sliding window | 13.75 | 10.7× | 85.7% | 0.120% | 2/14 |
| fixed reference | 20.00 | 15.6× | 100% | 0.376% | 1/14 |
| **e-process (Ville)** | 30.29 | 23.6× | 100% | **0.000%** | **0/14** |
| variance CUSUM | 43.00 | 33.4× | 57.1% | 0.060% | 2/14 |
| e-process conditional | 66.21 | 51.5× | 100% | 0.000% | 0/14 |
| energy distance [Leshin] | 68.57 | 53.3× | 100% | 1.020% | 1/14 |
| MDL-CUSUM | 95.31 | 74.1× | 92.9% | 0.000% | 0/14 |
| JS fingerprint [Bruckner] | — | — | 0.0% | 0.000% | 0/14 |

**medium**

| detector | delay | ×oracle | power | FA rate | streams w/ FA |
|---|---:|---:|---:|---:|---:|
| oracle LR-CUSUM | 4.86 | 1.0× | 100% | 0.000% | 0/14 |
| fixed reference | 22.86 | 4.7× | 100% | 1.880% | 4/14 |
| adaptive CUSUM | 28.67 | 5.9× | 64.3% | 0.220% | 9/14 |
| **e-process (Ville)** | 38.36 | 7.9× | 100% | **0.000%** | **0/14** |
| e-process conditional | 66.14 | 13.6× | 100% | 0.000% | 0/14 |
| energy distance [Leshin] | 77.14 | 15.9× | 100% | 3.061% | 3/14 |
| MDL-CUSUM | 118.42 | 24.4× | 85.7% | 0.000% | 0/14 |
| KS sliding window | 52.33 | — | 21.4% | 0.179% | 2/14 |
| variance CUSUM | 153.00 | — | 21.4% | 0.000% | 0/14 |

**hard** — nothing works. The oracle itself reaches 21.4% power at 106 probes;
every deployable detector is at or below its own false-alarm rate. This is the
negative result, and it is the same fact as `KL = 0.152` above.

`×oracle` is suppressed below 50% power: a mean delay over three detected
streams is not comparable to one over fourteen. It is a ratio to the oracle's
*measured* delay, never to the Lorden bound — Lorden's criterion is worst-case
over change points and `h/KL` is asymptotic in `h`, so an average-case delay can
legitimately fall below it, as the oracle does (1.29 vs 4.29).

## 4. Does the guarantee hold?

`python calibration.py` — 15 null streams per tier, fraction that ever fire.

| detector | tier | α=1e-1 | 1e-2 | 1e-3 | 1e-4 | 1e-5 | 1e-6 |
|---|---|---:|---:|---:|---:|---:|---:|
| e-process (Ville) | easy | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| e-process (Ville) | medium | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| e-process (Ville) | hard | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| MDL-CUSUM (reset) | medium | **0.467** | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

The e-process holds everywhere. The CUSUM variant violates at α=0.1 on medium
(7/15) — its reset breaks the martingale, exactly as the theory says it should.
With 15 streams the smallest resolvable rate is 1/15 = 0.067, so anything below
that reads only as "no stream fired".

## 5. The fair comparison

`python matched_operating_point.py` — thresholds tuned on held-out rep14 to the
tightest setting that stays silent, then scored on reps 0–13.

**easy**

| detector | tuned to | delay | power | streams w/ FA |
|---|---|---:|---:|---:|
| adaptive CUSUM | h=5 | 8.50 | 100% | **7/14** |
| KS sliding window | α=1e-2 | 13.75 | 85.7% | 2/14 |
| JS fingerprint | thr=0.08 | 13.93 | 100% | 0/14 |
| **e-process** | α=1e-2 | 17.29 | 100% | **0/14** |
| variance CUSUM | h=5 | 43.00 | 57.1% | 2/14 |
| MDL-CUSUM | α=1e-2 | 48.86 | 100% | 0/14 |

**medium**

| detector | tuned to | delay | power | streams w/ FA |
|---|---|---:|---:|---:|
| JS fingerprint | thr=0.08 | 11.29 | 100% | **12/14** |
| **e-process** | α=1e-2 | 29.64 | 100% | **0/14** |
| adaptive CUSUM | h=6 | 40.14 | 50.0% | 4/14 |
| MDL-CUSUM | α=1e-2 | 70.00 | 100% | 0/14 |

Two things to read here. **A threshold tuned on one clean stream does not
generalise** — CUSUM at h=5 was silent on rep14 and still fired on 7 of 14 test
streams; the derived threshold held. And on medium the e-process wins outright:
100% power at zero false alarms against CUSUM's 50% at four.

Bruckner's published 0.30 JS cut never fires here (0% power) because KT smoothing
over 100 symbols compresses JS; the *separation* is fine (0.040 pre-switch vs
0.200 post). A fixed cut from another corpus does not transfer. Recalibrated to
0.08 it is competitive on easy and unusable on medium.

## 6. The detectability frontier

`python run_adversarial.py --tier easy --trials 30` — offline resampling,
1200 probes, switch at 200. `keeps` is the fraction of the full inference bill
the provider retains. `+N` is power **above** that detector's rate on honest
traffic; `!` marks an excess under 10 points.

| provider strategy | keeps | e-process | mixture-alt | adaptive CUSUM |
|---|---:|---|---|---|
| honest *(control)* | 0% | 0% | 0% | **20%** |
| full swap | 83% | 29 @ 100% +100 | 126 @ 100% +100 | 12 @ 93% +73 |
| route 75% | 62% | 109 @ 97% +97 | 281 @ 100% +100 | 79 @ 80% +60 |
| route 50% | 42% | — @ 0% ! | **665 @ 30% +30** | 226 @ 60% +40 |
| route 25% | 21% | — @ 0% ! | 551 @ 3% ! | 449 @ 30% +10 ! |
| route 10% | 8% | — @ 0% ! | — @ 0% ! | 411 @ 20% +0 ! |
| route 5% | 4% | — @ 0% ! | — @ 0% ! | 352 @ 30% +10 ! |
| probe-aware cache | **82%** | — @ 0% ! | — @ 0% ! | 442 @ 30% +10 ! |
| probe-cache 50% miss | 82% | — @ 0% ! | 665 @ 30% +30 | 329 @ 53% +33 |
| distribution-matched | 71% | — @ 0% ! | — @ 0% ! | 406 @ 20% +0 ! |

The honest row is the control and it does the most work in this table. Adaptive
CUSUM fires on honest traffic 20% of the time, so its 20–30% against weak routing
is not detection. The e-process reports nothing it cannot support. The mixture
alternative buys real detection at 50% routing, where a clean-swap alternative is
testing the wrong hypothesis.

**Below 25% routing, no detector beats its own noise floor.** A provider willing
to cheat on a fifth of traffic is safe on this channel, and `probe-aware cache`
keeps 82% of the saving against every detector here.

## 7. Camouflage

`python camouflage.py` — 83 probes vs a 40-line stand-in organic corpus.
Replace it with a real prompt log before quoting these numbers.

| filter | probes caught | organic misrouted |
|---|---:|---:|
| one-line keyword match | 78.3% | 2.5% |
| char n-gram classifier | 83.8% accuracy | — |

Overt probes are caught 81.2% of the time, the covert set 68.4% — the covert
probes are not covert enough, because several still carry "One word." or
"Number only." formatting instructions. Dropping those is the obvious next fix.

**The three templates this project shipped with are the most detectable probes in
the bank** (p ≈ 0.959–0.966). Least detectable: `covert_sep` (0.436),
`fmt_list` (0.473), `covert_indent` (0.583).

## 8. Identification

`python detector_lineup.py` — 5 candidate models, trailing 60-probe window.

| tier | exact model | model family |
|---|---:|---:|
| easy | 14/14 = **100%** | 14/14 = 100% |
| medium | 6/14 = 42.9% | 14/14 = 100% |
| hard | 3/14 = 21.4% | 14/14 = 100% |
| **all** | 23/42 = 54.8% | **42/42 = 100%** |

The lineup never picks the wrong model *family*. Every error is `llama3.2:3b`
confused with its own q4_K_M or q8_0 quantization — the same weights at different
precision, which this channel cannot separate. That is the hard-tier finding
again, not an independent failure.

## 9. What has not been run

- **The probe survey** (`run_probe_survey.py`) needs Ollama. Until it runs,
  `probe_selection.py` has no data and the leaderboard is empty. Roughly
  83 probes × 5 models × 60 samples ≈ 25k generations; fully resumable.
- **The online frontier** (`run_adversarial.py --online`) needs Ollama. Offline
  resampling destroys within-session run structure, so the delays in §6 are
  pessimistic and only their shape should be quoted.
- **A real organic corpus** for §7.
