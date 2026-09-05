"""Single source of truth for every number and figure in the paper.

Nothing in the manuscript should be a hand-typed number. This script reads the
CSVs the experiment scripts emit and writes:

  numbers.tex     LaTeX macros, one per reported quantity
  numbers.json    the same values, for prose drafting and for checking claims
  tables/*.tex    booktabs tables ready to \\input
  figures/*.pdf   publication figures (also .png for slides)

When the Ollama runs finish, rerun the experiment scripts and then this, and
every figure and macro in the paper updates. No number gets edited by hand, so
no number can go stale without the build noticing.

Usage:
    cd paper && python make_assets.py
    python make_assets.py --check      # report which assets are still pending
"""

import argparse
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(HERE, "..", "final-analysis", "figures")
OUT_FIG = os.path.join(HERE, "figures")
OUT_TAB = os.path.join(HERE, "tables")

TIER_LABEL = {"easy": "Easy", "medium": "Medium", "hard": "Hard"}
TIER_PAIR = {
    "easy": r"llama3.2:3b $\rightarrow$ qwen2.5:3b",
    "medium": r"llama3.2:1b $\rightarrow$ llama3.2:3b",
    "hard": r"q4\_K\_M $\rightarrow$ q8\_0",
}


def read(name):
    path = os.path.join(FIGDIR, name)
    if not os.path.exists(path):
        return None
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def f(x, default=None):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def macro_name(*parts):
    """LaTeX macro names admit letters only -- no digits, spaces or punctuation.

    Digits are spelled out and everything else is dropped, so
    'matched.medium.JS fingerprint [Bruckner].delay' becomes
    '\\matchedmediumJSfingerprintBrucknerdelay'.
    """
    trans = str.maketrans({"0": "zero", "1": "one", "2": "two", "3": "three",
                           "4": "four", "5": "five", "6": "six", "7": "seven",
                           "8": "eight", "9": "nine"})
    s = "".join(parts).translate(trans)
    return "".join(c for c in s if c.isalpha())


# --------------------------------------------------------------- collection

def collect():
    N, pending = {}, []

    ch = read("channel_report.csv")
    if ch:
        for r in ch:
            t = r["difficulty"]
            N[f"channel.{t}.kl"] = round(f(r["kl_b_given_a"]), 3)
            N[f"channel.{t}.tv"] = round(f(r["total_variation"]), 3)
            N[f"channel.{t}.bayes"] = round(f(r["bayes_error"]), 3)
            N[f"channel.{t}.entropy_a"] = round(f(r["entropy_a_bits"]), 2)
            N[f"channel.{t}.entropy_b"] = round(f(r["entropy_b_bits"]), 2)
            N[f"channel.{t}.lorden"] = round(f(r["lorden_bound_probes"]), 1)
    else:
        pending.append("channel_report.csv  -> run: python audit_data.py && python evaluate.py")

    sm = read("summary_table_all_tiers.csv")
    if sm:
        for r in sm:
            t, m = r["difficulty"], r["method"]
            key = f"bench.{t}.{m}"
            N[key + ".delay"] = round(f(r["mean_delay"]), 2) if f(r["mean_delay"]) else None
            N[key + ".power"] = round(100 * f(r["detection_rate"], 0), 1)
            N[key + ".fa"] = round(100 * f(r["mean_false_alarm_rate"], 0), 3)
            N[key + ".fastreams"] = int(f(r["streams_with_false_alarm"], 0))
            N[key + ".nnull"] = int(f(r["n_null_streams"], 0))
    else:
        pending.append("summary_table_all_tiers.csv -> run: python evaluate.py")

    cal = read("calibration.csv")
    if cal:
        viol = [r for r in cal if r["verdict"] == "VIOLATED"]
        N["calib.n_settings"] = len(cal)
        N["calib.n_violations"] = len(viol)
        N["calib.eprocess_max_achieved"] = max(
            (f(r["achieved"], 0) for r in cal if "e-process" in r["detector"]), default=0.0)
        N["calib.n_streams"] = int(f(cal[0]["n_streams"], 0))
        N["calib.resolution_floor"] = round(1.0 / max(1, int(f(cal[0]["n_streams"], 1))), 3)
        for r in viol:
            N[f"calib.violation.{r['detector']}.{r['difficulty']}.alpha"] = f(r["nominal_alpha"])
            N[f"calib.violation.{r['detector']}.{r['difficulty']}.achieved"] = f(r["achieved"])
    else:
        pending.append("calibration.csv -> run: python calibration.py")

    mo = read("matched_operating_point.csv")
    if mo:
        for r in mo:
            key = f"matched.{r['difficulty']}.{r['detector']}"
            N[key + ".tuned"] = r["tuned_value"]
            N[key + ".delay"] = round(f(r["mean_delay"]), 2) if f(r["mean_delay"]) else None
            N[key + ".power"] = round(100 * f(r["power"], 0), 1)
            N[key + ".fastreams"] = int(f(r["fa_streams"], 0))
    else:
        pending.append("matched_operating_point.csv -> run: python matched_operating_point.py")

    ad = read("adversarial_frontier.csv")
    if ad:
        for r in ad:
            key = f"adv.{r['tier']}.{r['strategy']}.{r['detector']}"
            N[key + ".power"] = round(100 * f(r["power"], 0), 1)
            N[key + ".excess"] = round(100 * f(r.get("excess_power"), 0), 1)
            N[key + ".delay"] = round(f(r["mean_delay"]), 1) if f(r["mean_delay"]) else None
            N[f"adv.{r['tier']}.{r['strategy']}.keeps"] = round(100 * f(r["provider_saving"], 0), 1)
        N["adv.trials"] = int(f(ad[0]["trials"], 0))
        N["adv.mode"] = "offline resampling"
        pending.append("adversarial_frontier.csv is OFFLINE -> rerun with --online for final delays")
    else:
        pending.append("adversarial_frontier.csv -> run: python run_adversarial.py")

    cam = read("camouflage.csv")
    if cam:
        n = len(cam)
        kw = sum(1 for r in cam if r["keyword_flagged"] == "True")
        cov = [r for r in cam if r["declared_covert"] == "True"]
        ovt = [r for r in cam if r["declared_covert"] != "True"]
        N["cam.n_probes"] = n
        N["cam.keyword_caught"] = round(100 * kw / n, 1)
        N["cam.covert_caught"] = round(
            100 * sum(1 for r in cov if r["keyword_flagged"] == "True") / max(1, len(cov)), 1)
        N["cam.overt_caught"] = round(
            100 * sum(1 for r in ovt if r["keyword_flagged"] == "True") / max(1, len(ovt)), 1)
        N["cam.n_covert"] = len(cov)
        srt = sorted(cam, key=lambda r: f(r["classifier_prob_is_probe"], 0))
        N["cam.least_detectable"] = [(r["probe_id"], f(r["classifier_prob_is_probe"]))
                                     for r in srt[:3]]
        N["cam.most_detectable"] = [(r["probe_id"], f(r["classifier_prob_is_probe"]))
                                    for r in srt[-3:]]
    else:
        pending.append("camouflage.csv -> run: python camouflage.py")

    if not os.path.exists(os.path.join(FIGDIR, "probe_leaderboard.csv")):
        pending.append("probe_leaderboard.csv  ** NEEDS OLLAMA ** -> "
                       "python run_probe_survey.py && python probe_selection.py")

    return N, pending


# ------------------------------------------------------------------ writers

def write_macros(N):
    lines = ["% Auto-generated by paper/make_assets.py -- DO NOT EDIT.",
             "% Every number in the manuscript comes from here.", ""]
    for k, v in sorted(N.items()):
        if v is None or isinstance(v, (list, tuple)):
            continue
        lines.append(f"\\newcommand{{\\{macro_name(k)}}}{{{v}}}")
    path = os.path.join(HERE, "numbers.tex")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return path, sum(1 for l in lines if l.startswith("\\newcommand"))


def write_json(N, pending):
    path = os.path.join(HERE, "numbers.json")
    with open(path, "w") as fh:
        json.dump({"numbers": N, "pending": pending}, fh, indent=2, sort_keys=True)
    return path


def table_channel(N):
    rows = []
    for t in ("easy", "medium", "hard"):
        if f"channel.{t}.kl" not in N:
            continue
        rows.append(
            f"{TIER_LABEL[t]} & {TIER_PAIR[t]} & "
            f"{N[f'channel.{t}.entropy_a']:.2f} / {N[f'channel.{t}.entropy_b']:.2f} & "
            f"{N[f'channel.{t}.kl']:.3f} & {N[f'channel.{t}.tv']:.3f} & "
            f"{N[f'channel.{t}.bayes']:.3f} \\\\")
    if not rows:
        return None
    return "\n".join([
        r"\begin{tabular}{llccccc}", r"\toprule",
        r"Tier & Substitution & Entropy A/B (bits) & $KL(B\|A)$ & TV & Bayes err. \\",
        r"\midrule", *rows, r"\bottomrule", r"\end{tabular}"])


def table_matched(N):
    order = [("easy", "adaptive CUSUM"), ("easy", "KS sliding window"),
             ("easy", "e-process (Ville)"), ("medium", "e-process (Ville)"),
             ("medium", "adaptive CUSUM")]
    rows = []
    for t, d in order:
        k = f"matched.{t}.{d}"
        if k + ".delay" not in N:
            continue
        delay = N[k + ".delay"]
        rows.append(f"{TIER_LABEL[t]} & {d} & {N[k+'.tuned']} & "
                    f"{delay if delay else '--'} & {N[k+'.power']:.1f}\\% & "
                    f"{N[k+'.fastreams']}/14 \\\\")
    if not rows:
        return None
    return "\n".join([
        r"\begin{tabular}{llcccc}", r"\toprule",
        r"Tier & Detector & Tuned & Delay & Power & Streams w/ FA \\",
        r"\midrule", *rows, r"\bottomrule", r"\end{tabular}"])


# ------------------------------------------------------------------ figures

def figures(N):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return ["matplotlib not installed; figures skipped"]

    os.makedirs(OUT_FIG, exist_ok=True)
    plt.rcParams.update({
        "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 150, "savefig.bbox": "tight", "axes.grid": True,
        "grid.alpha": 0.25, "grid.linewidth": 0.5, "legend.frameon": False,
    })
    C = {"eproc": "#1f5673", "mix": "#33654f", "cusum": "#a83a2c", "grey": "#69737f"}
    made = []

    def save(fig, name):
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(OUT_FIG, f"{name}.{ext}"))
        plt.close(fig)
        made.append(name)

    # Fig 1 -- the frontier, excess power over the honest control
    ad = read("adversarial_frontier.csv")
    if ad:
        # One tier per panel: the CSV holds every tier that has been run, and
        # plotting them together silently doubles the points at each x.
        FIG_TIER = "easy"
        ad = [r for r in ad if r["tier"] == FIG_TIER]
    if ad:
        strat_x = {"route 5%": 5, "route 10%": 10, "route 25%": 25,
                   "route 50%": 50, "route 75%": 75, "full swap": 100}
        fig, ax = plt.subplots(figsize=(5.2, 3.0))
        for det, colour, style, lab in [
            ("e-process", C["eproc"], "-", "e-process"),
            ("mixture-alternative", C["mix"], "-", "mixture alternative"),
            ("adaptive CUSUM", C["cusum"], "--", "adaptive CUSUM"),
        ]:
            pts = sorted((strat_x[r["strategy"]], 100 * f(r.get("excess_power"), 0))
                         for r in ad if r["detector"] == det and r["strategy"] in strat_x)
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts], style,
                        color=colour, marker="o", ms=3.5, lw=1.6, label=lab)
        ax.set_xlabel("Traffic routed to the cheap model (\\%)" if False
                      else "Traffic routed to the cheap model (%)")
        ax.set_ylabel("Excess power over honest control")
        ax.set_xscale("log")
        ax.set_xticks([5, 10, 25, 50, 75, 100])
        ax.set_xticklabels(["5%", "10%", "25%", "50%", "75%", "100%"])
        ax.set_ylim(-4, 105)
        ax.axhline(0, color=C["grey"], lw=0.8, ls=":")
        ax.legend(loc="upper left", fontsize=8)
        save(fig, "fig1_frontier")

    # Fig 2 -- calibration, nominal vs achieved
    cal = read("calibration.csv")
    if cal:
        fig, ax = plt.subplots(figsize=(3.6, 3.2))
        lo, hi = 1e-7, 1.0
        ax.plot([lo, hi], [lo, hi], ls=":", color=C["grey"], lw=1,
                label="nominal = achieved")
        for det, colour, mk in [("e-process (Ville)", C["eproc"], "o"),
                                ("MDL-CUSUM (reset)", C["cusum"], "^")]:
            xs = [f(r["nominal_alpha"]) for r in cal if r["detector"] == det]
            ys = [max(f(r["achieved"], 0), lo) for r in cal if r["detector"] == det]
            if xs:
                ax.scatter(xs, ys, s=22, color=colour, marker=mk, label=det, alpha=.75,
                           zorder=3)
        floor = N.get("calib.resolution_floor", 1 / 15)
        ax.axhline(floor, color=C["grey"], lw=0.8, ls="--")
        ax.text(1.2e-7, floor * 1.25, f"resolution floor 1/{N.get('calib.n_streams', 15)}",
                fontsize=6.5, color=C["grey"])
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(hi, lo); ax.set_ylim(lo, 2)
        ax.set_xlabel(r"nominal $\alpha$")
        ax.set_ylabel("achieved false-alarm rate")
        ax.legend(loc="upper left", fontsize=7)
        save(fig, "fig2_calibration")

    # Fig 3 -- channel strength per tier, the negative result
    ch = read("channel_report.csv")
    if ch:
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(5.4, 2.5))
        tiers = [r["difficulty"] for r in ch]
        kl = [f(r["kl_b_given_a"]) for r in ch]
        be = [f(r["bayes_error"]) for r in ch]
        cols = [C["eproc"], C["mix"], C["cusum"]]
        a1.bar(tiers, kl, color=cols, width=.6)
        a1.set_ylabel("KL(B$\\|$A), nats/probe")
        for i, v in enumerate(kl):
            a1.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=7.5)
        a2.bar(tiers, be, color=cols, width=.6)
        a2.axhline(0.5, color=C["grey"], ls="--", lw=.9)
        a2.text(1.0, 0.505, "chance", fontsize=6.5, color=C["grey"], ha="center")
        a2.set_ylabel("single-probe Bayes error")
        a2.set_ylim(0, .58)
        for i, v in enumerate(be):
            a2.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=7.5)
        save(fig, "fig3_channel")

    # Fig 4 -- camouflage: detectability vs the current templates
    cam = read("camouflage.csv")
    if cam:
        fig, ax = plt.subplots(figsize=(5.2, 2.6))
        fams = {}
        for r in cam:
            fams.setdefault(r["family"], []).append(f(r["classifier_prob_is_probe"], 0))
        names = sorted(fams, key=lambda k: -sum(fams[k]) / len(fams[k]))
        ax.boxplot([fams[n] for n in names], tick_labels=names,
                   widths=.55, patch_artist=True,
                   boxprops=dict(facecolor="#e3edf3", lw=.8),
                   medianprops=dict(color=C["eproc"], lw=1.4),
                   flierprops=dict(ms=2.5, mfc=C["grey"], mec="none"))
        ax.axhline(0.5, color=C["grey"], ls=":", lw=.9)
        ax.set_ylabel("P(request is an audit probe)")
        ax.set_xticklabels(names, rotation=18, ha="right", fontsize=7.5)
        save(fig, "fig4_camouflage")

    return made


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)

    N, pending = collect()

    if args.check:
        print(f"{len(N)} values available.\n")
        if pending:
            print("PENDING:")
            for p in pending:
                print(f"  - {p}")
        else:
            print("Nothing pending.")
        return 0 if not pending else 1

    os.makedirs(OUT_TAB, exist_ok=True)
    mpath, nmac = write_macros(N)
    jpath = write_json(N, pending)

    for name, builder in [("channel", table_channel), ("matched", table_matched)]:
        tex = builder(N)
        if tex:
            with open(os.path.join(OUT_TAB, f"{name}.tex"), "w") as fh:
                fh.write(tex + "\n")

    figs = figures(N)

    print(f"wrote {os.path.relpath(mpath, HERE)}  ({nmac} macros)")
    print(f"wrote {os.path.relpath(jpath, HERE)}")
    print(f"wrote tables/  ({len(os.listdir(OUT_TAB))} files)")
    print(f"wrote figures/ ({', '.join(figs)})")
    if pending:
        print("\nPENDING -- these assets are not final:")
        for p in pending:
            print(f"  - {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
