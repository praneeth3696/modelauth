# Claims ledger

Every empirical claim the paper may make, the macro that carries it, the script
that produces it, and whether it is final. Check this before submission: any row
marked PROVISIONAL or BLOCKED that appears as a hard number in the manuscript is
a defect.

Regenerate the macro values with `python make_assets.py`; verify pending status
with `python make_assets.py --check` (exits nonzero while anything is pending).

---

## Final — safe to state as fact

| # | Claim | Value | Macro | Produced by |
|---|---|---|---|---|
| C1 | Truncation inflated the pooled std ~19× | 18.25 → 349.09 | *(prose)* | `audit_data.py` |
| C2 | Rejected probes, easy tier | 86 / 12000 = 0.72% | *(prose)* | `audit_data.py` |
| C3 | Easy-tier divergence | KL 2.149, TV 0.725 | `\channeleasykl` `\channeleasytv` | `evaluate.py` |
| C4 | Medium-tier divergence | KL 1.450, TV 0.635 | `\channelmediumkl` | `evaluate.py` |
| C5 | **Hard tier is near-blind** | KL 0.152, Bayes err 0.407 | `\channelhardkl` `\channelhardbayes` | `evaluate.py` |
| C6 | Oracle cannot do the hard tier | 21.4% power, 106 probes | `\benchhardoracleLRCUSUMpower` | `evaluate.py` |
| C7 | Oracle easy-tier delay | 1.29 probes, 100% power | `\bencheasyoracleLRCUSUMdelay` | `evaluate.py` |
| C8 | e-process holds calibration everywhere | 0/15 at every α, all tiers | `\calibeprocessmaxachieved` | `calibration.py` |
| C9 | CUSUM variant violates where predicted | α=0.1, medium, 7/15 | `\calibnviolations` | `calibration.py` |
| C10 | Tuned CUSUM is faster on easy | 8.50 vs 17.29 probes | `\matchedeasyadaptiveCUSUMdelay` | `matched_operating_point.py` |
| C11 | **Tuned threshold does not generalise** | silent on rep14, fires on 7/14 | `\matchedeasyadaptiveCUSUMfastreams` | `matched_operating_point.py` |
| C12 | e-process wins outright on medium | 100% @ 0 FA vs 50% @ 4 FA | `\matchedmediumeprocessVillepower` | `matched_operating_point.py` |
| C13 | Keyword filter catches most probes | 78.3% at 2.5% misroute | `\camkeywordcaught` | `camouflage.py` |
| C14 | **Current templates are the most detectable** | p ≈ 0.959–0.966 | `numbers.json` → `cam.most_detectable` | `camouflage.py` |
| C15 | Covert set is not covert enough | 68.4% still caught | `\camcovertcaught` | `camouflage.py` |
| C16 | Identification never errs on family | 42/42 = 100% | *(prose)* | `detector_lineup.py` |
| C17 | Exact-model identification | 54.8% overall, 100% easy | *(prose)* | `detector_lineup.py` |
| C18 | Adaptive CUSUM false-alarms on honest traffic | 20% | `\adveasyhonestadaptiveCUSUMpower` | `run_adversarial.py` |
| C19 | Probe-aware cache retains most of the saving | 82% | `\adveasyprobeawarecachekeeps` | `run_adversarial.py` |

## Provisional — power/shape final, absolute delays change with `--online`

| # | Claim | Value | Status |
|---|---|---|---|
| C20 | **Below 25% routing nothing beats its noise floor** | excess ≤ 10 pts | Power final. Safe to state. |
| C21 | Mixture alternative extends reach to 50% routing | +30 pts excess | Power final. Safe to state. |
| C22 | e-process detects full swap and 75% routing | 100% / 97% | Power final. Safe to state. |
| C23 | Delay at full swap | 29 probes | **Delay provisional** — offline resampling |
| C24 | Delay at 75% routing | 109 probes | **Delay provisional** |

C20–C22 are power claims and survive the online rerun; only C23–C24 move.
Prefer power language over delay language throughout §6.1 for this reason.

## Blocked — needs Ollama

| # | Claim | Unblocked by |
|---|---|---|
| C25 | Probe information-per-token leaderboard | `run_probe_survey.py` → `probe_selection.py` |
| C26 | Selected portfolio and its summed KL | same |
| C27 | Sweeps-to-detection for the chosen portfolio | same |
| C28 | Whether covert probes are competitive on information | same |

C28 is the one that could change the paper's message: if covert probes turn out
to be much weaker than overt ones, §7's argument becomes a genuine tension
rather than a design recommendation. Do not pre-commit the discussion either way.

## Claims to avoid

- **Do not** describe the e-process as faster or as dominating. C10 says
  otherwise on the easy tier. The defensible claim is calibration-free operation
  plus a win on medium.
- **Do not** divide by the Lorden bound. Worst-case over change points and
  asymptotic in *h*; the oracle legitimately beats it (1.29 vs 4.29).
- **Do not** quote raw power in the frontier. Use excess over the honest control
  or the CUSUM numbers read as capability when they are noise.
- **Do not** present the comparators as reproductions of the published results.
  They are reimplementations adapted to a single-token integer channel.
- **Do not** state camouflage numbers without the stand-in-corpus caveat.

## Pre-submission checklist

- [ ] `python make_assets.py --check` reviewed; every pending item either
      resolved or reflected as provisional in the text
- [ ] no hand-typed numerals in the manuscript (grep for digits outside macros)
- [ ] every citation in `references.bib` marked `VERIFIED` opened and checked
- [ ] Leshin et al. limitations section cited explicitly in §2
- [ ] offline-resampling caveat appears in §5, §6.1 and §8
- [ ] both failed constructions retained in §4.2
- [ ] identification reported with both columns, never 54.8% alone
