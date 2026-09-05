"""Provider strategies: what a cost-cutting API provider might actually do.

Every result in this repo before now assumed the provider swaps 100% of traffic
at a clean instant and does not know it is being audited. That is the easiest
case and not the realistic one. This module makes the provider a first-class
object, so detectors are evaluated against an opponent rather than against a
step function.

Two execution modes, same strategy objects:

  offline  -- resample from response pools logged from real models. Needs no
              serving stack, so the whole frontier can be measured on the
              existing data. Destroys within-session run structure, so absolute
              delays are pessimistic; the shape of the curves is what to read.
  online   -- wrap a live probe client and route per request. Preserves run
              structure and is the number that belongs in a paper. Requires
              Ollama; see run_adversarial.py --online.

Strategies

  Honest                 never substitutes. The null.
  FullSwap               100% of traffic to the cheap model after t=switch.
  PartialRoute(r)        fraction r of requests to the cheap model. The realistic
                         cheat: the provider banks (1-r) of the quality and all
                         of the saving, and the change is in a mixture weight
                         rather than in the distribution itself.
  ProbeAwareCache        recognises audit-shaped requests and serves them from a
                         cache of genuine expensive-model answers, routing
                         everything else to the cheap model. Costs the provider
                         one lookup. Defeats any monitor whose probes are
                         identifiable -- which, since fpverify published the
                         "pick a random number" probe, now means every published
                         monitor.
  DistributionMatched    serves the cheap model but resamples its output to match
                         the expensive model's marginal on the probe channel.
                         The strongest attack and, on a single channel, an
                         information-theoretic win for the provider. Included to
                         mark the boundary honestly rather than to be beaten.
"""

import random


class Provider:
    name = "provider"

    def serve(self, index, prompt, rng):
        raise NotImplementedError

    def cost_multiplier(self, index):
        """Fraction of full-price inference actually paid. Lower = more saved."""
        return 1.0


class Honest(Provider):
    name = "honest"

    def __init__(self, pool_a):
        self.pool_a = pool_a

    def serve(self, index, prompt, rng):
        return rng.choice(self.pool_a), "A"


class FullSwap(Provider):
    name = "full_swap"

    def __init__(self, pool_a, pool_b, switch):
        self.pool_a, self.pool_b, self.switch = pool_a, pool_b, switch

    def serve(self, index, prompt, rng):
        if index >= self.switch:
            return rng.choice(self.pool_b), "B"
        return rng.choice(self.pool_a), "A"

    def cost_multiplier(self, index):
        return 0.0 if index >= self.switch else 1.0


class PartialRoute(Provider):
    """Route a fraction r of post-switch traffic to the cheap model."""

    def __init__(self, pool_a, pool_b, switch, route_fraction):
        self.pool_a, self.pool_b = pool_a, pool_b
        self.switch, self.r = switch, route_fraction
        self.name = f"partial_route_{route_fraction:g}"

    def serve(self, index, prompt, rng):
        if index >= self.switch and rng.random() < self.r:
            return rng.choice(self.pool_b), "B"
        return rng.choice(self.pool_a), "A"

    def cost_multiplier(self, index):
        return (1.0 - self.r) if index >= self.switch else 1.0


class ProbeAwareCache(Provider):
    """Serve audit-shaped requests honestly from cache; cheap-model everything else.

    is_probe decides what looks like an audit. The default recognises the
    published single-token numeric probe, which is the realistic starting point
    for a provider reading the monitoring literature.
    """

    def __init__(self, pool_a, pool_b, switch, is_probe=None, detect_rate=1.0):
        self.pool_a, self.pool_b, self.switch = pool_a, pool_b, switch
        self.detect_rate = detect_rate
        self.is_probe = is_probe or self._looks_like_probe
        self.name = (f"probe_aware_cache" if detect_rate >= 1.0
                     else f"probe_aware_cache_{detect_rate:g}")

    @staticmethod
    def _looks_like_probe(prompt):
        if not prompt:
            return False
        p = prompt.lower()
        return ("random" in p and ("number" in p or "integer" in p or "digit" in p))

    def serve(self, index, prompt, rng):
        if index >= self.switch:
            if self.is_probe(prompt) and rng.random() < self.detect_rate:
                return rng.choice(self.pool_a), "A"   # cached honest answer
            return rng.choice(self.pool_b), "B"
        return rng.choice(self.pool_a), "A"

    def cost_multiplier(self, index):
        # Probe traffic is a rounding error next to organic volume, so the
        # provider still banks essentially the whole saving.
        return 0.02 if index >= self.switch else 1.0


class DistributionMatched(Provider):
    """Cheap model, output resampled to match the expensive model's marginal.

    Implemented as: draw from pool_a's empirical distribution while the actual
    generation came from the cheap model. On the probe channel alone this is
    exactly indistinguishable from honest service -- the marginal is identical
    by construction -- so no test on this channel can have power above alpha.
    That is the point: it fixes where the boundary is.
    """

    def __init__(self, pool_a, pool_b, switch, wrapper_overhead=0.15):
        self.pool_a, self.pool_b, self.switch = pool_a, pool_b, switch
        self.overhead = wrapper_overhead
        self.name = "distribution_matched"

    def serve(self, index, prompt, rng):
        if index >= self.switch:
            return rng.choice(self.pool_a), "B"   # looks like A, generated by B
        return rng.choice(self.pool_a), "A"

    def cost_multiplier(self, index):
        # Cheap inference plus the matching wrapper's own cost.
        return self.overhead if index >= self.switch else 1.0


def simulate(provider, n, switch, seed=0, prompts=None):
    """Run a provider offline, returning (values, contexts, truth, cost)."""
    rng = random.Random(seed)
    values, ctxs, truth = [], [], []
    cost = 0.0
    for i in range(n):
        prompt = (rng.choice(prompts) if prompts else
                  "Pick a random number between 1 and 100. Reply with only the number.")
        v, who = provider.serve(i, prompt, rng)
        values.append(v)
        ctxs.append(prompt[:16])
        truth.append(who)
        cost += provider.cost_multiplier(i)
    return values, ctxs, truth, cost / n


def build_pools(difficulty, data_dir=None):
    """Response pools for each model in a tier, from the logged streams."""
    import os
    from config import MODEL_PAIRS
    from data_loader import DATA_DIR, load_numeric_stream

    data_dir = data_dir or DATA_DIR
    model_a, model_b = MODEL_PAIRS[difficulty]
    pools = {model_a: [], model_b: []}
    for cond in ("null", "substitution"):
        for rep in range(15):
            path = os.path.join(data_dir, f"{difficulty}_{cond}_rep{rep}.jsonl")
            if not os.path.exists(path):
                continue
            for r in load_numeric_stream(path):
                if r["true_model"] in pools:
                    pools[r["true_model"]].append(r["numeric_answer"])
    return pools[model_a], pools[model_b]
