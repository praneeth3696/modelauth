# The Price of Cheating

**Detectability frontiers for partial model substitution in black-box LLM APIs**

Working draft. Numbers are reproducible from this repository; see `RESULTS.md`
for the full tables and the commands that produce them.

---

## 1. The problem

You pay for a large model. The provider quietly serves a smaller one. Because
the API returns only text, you cannot inspect weights, logits, or layer
activations, and API keys authenticate *who owns the server*, not *which model
answered*. Detection has to work from the output distribution alone.

The version of this problem we take is the hard one: **self-baselining**. No
trusted reference model, no clean sample of the expensive model obtained some
other way. The monitor learns what the endpoint looks like from the endpoint
itself, then watches for that to stop being true.

## 2. What is already known

This area moved fast in 2025-2026, and the detector-shaped part of it is done.

| Work | Method | Reference-free | Sequential | Adversary |
|---|---|:---:|:---:|:---:|
| Gao, Liang & Guestrin, ICLR 2025 ([2410.20247](https://arxiv.org/abs/2410.20247)) | MMD, string kernel, batch two-sample | no | no | no |
| Leshin, Shah, Timmis & Kang, Mar 2026 ([2603.19022](https://arxiv.org/abs/2603.19022)) | embeddings + energy distance, e-values | **yes** | **yes** | no |
| Bruckner, Jul 2026 ([2607.10252](https://arxiv.org/abs/2607.10252)) | single-token PMF, Jensen-Shannon | no | no | no |
| Rank-based uniformity audit, Jun 2025 ([2506.06975](https://arxiv.org/abs/2506.06975)) | rank test | no | no | no |
| `fpverify` (tool) | single-token probes + betting e-process | no | **yes** | no |

Leshin et al. already do self-baselined anytime-valid sequential monitoring.
Bruckner already fingerprints models from single-token "pick a random number"
distributions across 165 models. `fpverify` already ships both combined. **We
reproduce these rather than claim them**, and the contribution moves to the
column none of them fills.

## 3. Contributions

1. **A detectability frontier against adaptive providers.** Partial routing,
   probe-aware caching, and distribution matching, each measured against each
   detector, with the provider's cost saving on the same axes. Leshin et al.
   name adversarial providers as an explicit limitation; this fills it.
2. **Probe camouflage as a first-class constraint.** `fpverify` published its
   probe, so any provider can filter audit traffic with one string match. We
   measure probe detectability and show what it costs the monitor.
3. **Sequential identification.** The lineup reports a posterior over candidate
   models at alarm time, not merely that something changed.
4. **A negative result with a stated boundary.** Quantization substitution
   (q4_K_M vs q8_0) is close to invisible on the numeric probe channel, and we
   quantify how close.

## 4. Method

The probe channel is **categorical, not continuous**. `llama3.2:3b` answers
53 about a third of the time; `qwen2.5:3b` answers 42 about a third of the time;
the ordering between them carries nothing. CUSUM on the mean and KS on the ECDF
both assume 53 and 54 are near-neighbours and discard most of the evidence.

Our detector treats the endpoint as a **codebook** and watches compression fail.
Both codes are prequential:

    null        q_all -- one KT codebook fed every observation from t=0
    alternative q_v   -- a KT codebook restarted at candidate change point v

    logLR(v,t) = sum_{i=v..t} [ log q_v(x_i | x_v..x_{i-1})
                                - log q_all(x_i | x_0..x_{i-1}) ]

Mixing over the unknown change point gives a Shiryaev-Roberts statistic with
`E[R_t] = t` under the null, so thresholding at `t/alpha` is anytime-valid by
Ville's inequality. **The threshold is derived from alpha; there is nothing to
tune.**

### Two constructions that look right and are not

Recorded because both cost us time and both would have produced publishable-looking
nonsense.

**Excess surprisal.** `-log p(x) - H(p)` is mean-zero under the null, but
`E[exp(·)] = K·exp(-H) >> 1`, so Ville does not apply and `log(1/alpha)` buys
nothing.

**Frozen null vs adaptive alternative.** Freeze `p_0` on a warmup window, run an
adaptive codebook against it, accumulate `log q/p_0`. `E[q/p_0] = 1` requires
`p_0` to be the *true* null law, not a 40-sample estimate of it. The adaptive
code compresses better purely by having seen more data. Measured: **+92.9 nats
of drift over one clean stream, and a 100% false-alarm rate at every alpha from
1e-1 to 1e-6.** Making both codes prequential removes the bias entirely.

## 5. Results in brief

Full tables in `RESULTS.md`.

**The channel.** Held-out estimates per tier:

| tier | substitution | KL(B‖A) | TV | Bayes error | oracle delay | oracle power |
|---|---|---:|---:|---:|---:|---:|
| easy | llama3.2:3b → qwen2.5:3b | 2.149 | 0.725 | 0.138 | 1.29 | 100% |
| medium | llama3.2:1b → llama3.2:3b | 1.450 | 0.635 | 0.183 | 4.86 | 100% |
| hard | q4_K_M → q8_0 | **0.152** | **0.186** | **0.407** | 106 | **21.4%** |

**Calibration.** Swept over alpha from 1e-1 to 1e-6 on clean null streams, the
e-process fires on 0 of 15 streams at every setting and every tier. The CUSUM
variant, whose reset breaks the martingale, violates at alpha=0.1 on the medium
tier (7/15). The theory's prediction and the measurement agree.

**Matched operating point.** At a zero-false-alarm budget the tuned CUSUM is
*faster* than the e-process on the easy tier (11.93 vs 17.29 probes). What the
e-process buys is not speed: its threshold is derived rather than swept, and a
threshold tuned on one held-out clean stream did not generalise — CUSUM at h=5
was silent on rep14 and still fired on 7 of 14 test streams. In deployment there
is no labelled substitution stream to sweep against.

**The frontier** (easy tier, 30 trials, offline resampling):

| provider strategy | keeps | e-process | mixture-alt | CUSUM |
|---|---:|---:|---:|---:|
| honest *(control)* | 0% | 0% | 0% | **20%** |
| full swap | 83% | 100% | 100% | 93% |
| route 75% | 62% | 97% | 100% | 80% |
| route 50% | 42% | 0% | **30%** | 60% |
| route 25% | 21% | 0% | 3% | 30% |
| route 10% | 8% | 0% | 0% | 20% |
| probe-aware cache | **82%** | **0%** | **0%** | 30% |
| distribution-matched | 71% | 0% | 0% | 20% |

Read against the honest control. CUSUM's 20-30% on undetectable strategies *is*
its false-alarm rate — it is not detecting, it is firing. The e-process claims
nothing it cannot support. The mixture alternative extends real detection to 50%
routing, where the clean-swap alternative is testing the wrong hypothesis.

**Below 25% routing, nothing detects.** A provider willing to cheat on a fifth
of traffic is, on this channel, safe.

**Camouflage.** A one-line keyword filter catches 78.3% of the probe bank while
misrouting 2.5% of organic traffic; a character n-gram classifier reaches 83.8%.
The three templates this project shipped with are the **most** detectable probes
in the bank (p ≈ 0.96). Probe-aware caching then costs the monitor everything
while the provider keeps 82% of the saving.

**Identification.** Top-1 exact-model accuracy is 100% on easy, 42.9% on medium,
21.4% on hard. Model-*family* accuracy is **100% on all 42 streams**: the lineup
never picks the wrong model, it only confuses quantizations of identical weights.

## 6. Limitations

- **The frontier is measured offline**, resampling from logged response pools.
  This destroys within-session run structure, so absolute delays are pessimistic.
  Shapes are trustworthy; intercepts need the live rerun (`run_adversarial.py --online`).
- **One probe family carries every result.** The probe bank is built and scored,
  but the survey that would rank it needs generation time that has not been spent.
- **`distribution_matched` is unbeatable on a single channel by construction**,
  and we do not claim otherwise. The interesting question is what it costs a
  provider to run across many channels at once.
- **15 streams per tier** puts the smallest resolvable false-alarm rate at 1/15,
  so alphas below 0.067 can only be confirmed as "no stream fired".
- Comparators are our reconstructions from published descriptions, adapted to a
  single-token integer channel, not the authors' code.

## 7. What would move this forward

Run the probe survey and rank the bank; rerun the frontier online; and take the
camouflage evaluation to a real prompt log rather than the 40-line stand-in.
The interesting open question is whether a *portfolio* of covert probes across
several channels raises the cost of distribution matching enough to make it
uneconomic — which is where an impossibility result turns back into a design
problem.
