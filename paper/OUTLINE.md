# Writing guide

Section-by-section, with the status of every piece of evidence it rests on.
Read `CLAIMS.md` alongside this: it maps every claim to the script that
produces it, and it is the thing to check before submission.

**Rule for the whole manuscript: no number is typed by hand.** Everything comes
from `\input{numbers.tex}` (418 macros) or from `tables/*.tex`. If a number you
want has no macro, add it to `make_assets.py` rather than typing it — otherwise
the Ollama runs will silently invalidate it.

Status legend: **READY** — write it now, evidence is final. **PROVISIONAL** —
write it now, numbers change when Ollama runs. **BLOCKED** — leave a stub.

---

## Overall shape

Roughly 85% of this paper is writable today. The Ollama dependency is confined
to §6.3 (the probe leaderboard) and to the absolute delay values in §6.1.
Nothing in the framing, method, calibration, camouflage, identification or
limitations depends on it.

Target venue shape: 8–9 pages. The natural home is a security or ML-systems
venue rather than a statistics one — the statistical machinery is standard and
correctly attributed; the contribution is the threat model and the measurement.

---

## Abstract — **READY**

Four sentences that must survive: black-box APIs let providers substitute models
silently; sequential self-baselined detection of a *full* swap is solved; the
realistic cheat is *partial* routing against an adversary who knows it is being
audited, and there detection collapses; we measure exactly where.

Land the two hard numbers: **below 25% routing no detector beats its own noise
floor**, and **probe-aware caching keeps 82% of the saving against every
detector tested**. Do not put the e-process delay in the abstract — it is not
the contribution and it invites the wrong comparison.

## 1. Introduction — **READY**

Structure: the commercial incentive → why software authentication does not apply
(keys authenticate the server, not the model) → what the 2026 literature already
solved → the gap, stated as a question the existing work cannot answer: *how
much can a provider cheat before detection becomes impossible, and what does it
cost them to stay under that line?*

Contributions list, in this order (strongest first):
1. detectability frontier against five adaptive provider strategies
2. probe camouflage as a measured constraint, with the observation that
   published probes are now filterable
3. sequential identification, not just detection
4. a bounded negative result on quantization

State plainly in the intro that the detector is a reproduction, not a claim.
Reviewers punish overclaiming far harder than they punish a narrow contribution.

## 2. Related work — **READY**

Use the table from `PAPER.md` §2. The columns that matter are reference-free /
sequential / adversary; the last is empty for every row but ours, and that is
the paper's justification in one glance.

Be scrupulous with Leshin et al. — they got to self-baselined sequential
e-values first, and the honest framing is "we adopt their setting and extend the
threat model." Cite their limitations section explicitly; it names partial
routing and adversarial providers as open, which is the cleanest possible
motivation.

## 3. Threat model — **READY**, and write this early

This section does the most work and no experiment is needed for it. Define the
provider's action space (`adversary.py` is the formal spec), the monitor's
budget, and what the provider knows. The key axis nobody has drawn: **the
provider's knowledge of the audit**, from oblivious → knows probes exist → knows
the exact probe set (which is today's reality, since `fpverify` is public).

State the economic model here: routing fraction *r* maps to retained bill
`(1−r)`, and `cost_multiplier` in `adversary.py` is the implementation.

## 4. Method — **READY**

Two subsections.

**4.1 The channel is categorical.** Justify treating responses as symbols, not
reals. The 53/42 concentration is the whole argument and it is one sentence plus
`\channeleasykl` etc.

**4.2 Prequential MDL e-process.** Give the construction, then the Ville
argument. **Include the two failed constructions** — a short subsection or a
boxed remark. It costs a third of a column and it is the most useful thing in
the paper for anyone reimplementing: the frozen-null version looks correct,
passes casual inspection, and drifts +92.9 nats on clean data. Reviewers reward
this kind of honesty and it pre-empts "why not just do X".

## 5. Experimental setup — **READY**

Tiers, models, 15 reps, rep14 held out for every fit (reference, oracle,
threshold tuning, lineup). Say explicitly that no number is scored on data used
to build it. Table: `tables/channel.tex`.

Declare the offline-resampling caveat here, once, prominently, rather than
burying it in §6.1.

## 6. Results

### 6.1 The frontier — **PROVISIONAL** (shape final, delays change)

Figure: `figures/fig1_frontier.pdf`. Lead with the honest control. The single
most important sentence in the results: *adaptive CUSUM fires on honest traffic
20% of the time, so its apparent 20–30% power against weak routing is its own
false-alarm rate.* Plot excess power, never raw power.

What changes after Ollama: absolute delays. What does not: every power number,
every excess figure, the shape, and both headline claims. **Write this section
now**; the online rerun edits numbers, not prose.

### 6.2 Calibration — **READY**

Figure: `figures/fig2_calibration.pdf`. `\calibnviolations` = 1 violation across
`\calibnsettings` settings, and it is the CUSUM variant at α=0.1 exactly where
the theory says the reset should break. Note the 1/15 resolution floor honestly.

### 6.3 Probe economics — **BLOCKED**

Needs `run_probe_survey.py`. Leave a stub with the method described (worst-pair
KL per token, family-capped greedy selection) and a placeholder table. The
*method* is writable now; only the leaderboard is missing.

### 6.4 Camouflage — **READY**

Figure: `figures/fig4_camouflage.pdf`. Numbers: `\camkeywordcaught` = 78.3%
caught at 2.5% organic misroute. The finding that lands: **the three templates
this line of work has been using are the most detectable probes in the bank.**

Flag the 40-line stand-in corpus as a limitation in the caption, not just in §8.

### 6.5 Identification — **READY**

Report both columns. 54.8% exact / 100% family, and explain that every error is
a quantization confusion. Reporting only the first number would be misleading;
say why in the text.

### 6.6 The negative result — **READY**

Figure: `figures/fig3_channel.pdf`. Give it its own subsection rather than
burying it. `\channelhardkl` = 0.152, `\channelhardbayes` = 0.407, and the
oracle at 21.4% power is the sentence that closes it: *even a detector handed
both distributions cannot do this.*

## 7. Discussion — **READY**

The exchange-rate framing: what a provider buys per point of cheating. Then the
uncomfortable implication — a monitor that publishes its probes destroys its own
power, so this class of defence may need probe secrecy or probe diversity as a
design requirement, which is a different engineering problem from detector
design.

## 8. Limitations — **READY**, and do not compress it

Offline resampling; single probe family carrying every result; 15 streams and
the 1/15 floor; comparators are reconstructions; `distribution_matched` is
unbeatable on one channel by construction; the stand-in organic corpus. All six.

## 9. Conclusion — **READY**

Resist restating results. The forward-looking claim worth making: whether a
*portfolio* of covert probes across several channels raises the cost of
distribution matching enough to make it uneconomic. That turns an impossibility
result back into a design problem, and it is the natural next paper.

---

## Figures

| file | section | status |
|---|---|---|
| `fig1_frontier.pdf` | 6.1 | provisional (delays) |
| `fig2_calibration.pdf` | 6.2 | final |
| `fig3_channel.pdf` | 6.6 | final |
| `fig4_camouflage.pdf` | 6.4 | final |
| *(probe leaderboard)* | 6.3 | blocked |

## Build

```bash
cd substitution-sim
python audit_data.py && python evaluate.py && python calibration.py
python matched_operating_point.py && python detector_lineup.py && python camouflage.py
python run_adversarial.py --tier easy --trials 30
cd ../paper && python make_assets.py
python make_assets.py --check     # exits nonzero while anything is pending
```

Put `make_assets.py --check` in CI. It fails while any asset is provisional,
which is exactly the signal you want before submitting.

## When Ollama becomes available

```bash
python run_probe_survey.py --check     # confirm models, ~25k generations
python run_probe_survey.py
python probe_selection.py              # unblocks 6.3
python run_adversarial.py --online     # finalises 6.1 delays
cd ../paper && python make_assets.py
```

Then reread §6.1 and §6.3 only. Everything else stands.
