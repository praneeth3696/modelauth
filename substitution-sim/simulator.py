import random
from probe_client import probe
from config import PROBE_TEMPLATES, MAX_TOKENS

def generate_probe_stream(model_a, model_b, total_requests, switch_point, temperature=1.0, max_tokens=MAX_TOKENS):
    results = []
    for i in range(total_requests):
        current_model = model_a
        if switch_point is not None and i >= switch_point:
            current_model = model_b

        prompt = random.choice(PROBE_TEMPLATES)
        answer = probe(current_model, prompt, temperature=temperature, max_tokens=max_tokens)

        results.append({
            "index": i,
            "prompt": prompt,
            "answer": answer,
            "true_model": current_model,
            "failed": answer is None,
        })

    return results

def generate_contaminated_stream(model_a, model_b, total_requests, warmup, contamination_fraction, temperature=1.0, max_tokens=MAX_TOKENS):
    """
    contamination_fraction: fraction of the warmup period already served by model_b
    (simulating substitution that happened before monitoring started).
    After warmup, the stream continues with model_a (the 'true' state the detector
    should settle into, if it can recover).
    """
    contaminated_count = int(warmup * contamination_fraction)
    results = []
    for i in range(total_requests):
        if i < contaminated_count:
            current_model = model_b   # contamination in early history
        else:
            current_model = model_a   # "true" state once contamination ends
        prompt = random.choice(PROBE_TEMPLATES)
        answer = probe(current_model, prompt, temperature=temperature, max_tokens=max_tokens)
        results.append({
            "index": i,
            "prompt": prompt,
            "answer": answer,
            "true_model": current_model,
            "failed": answer is None,
        })
    return results

