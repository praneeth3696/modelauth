MODEL_PAIRS = {
    "easy": ("llama3.2:3b", "qwen2.5:3b"),
}


PROBE_TEMPLATES = [
    "Pick a random number between 1 and 100. Reply with only the number.",
    "Name a random number from 1 to 100, just the digits.",
    "Give me a random integer in the range 1-100. Number only.",
]

TEMPERATURE = 1.0
MAX_TOKENS = 5
TOTAL_REQUESTS = 400
SWITCH_POINT = 200
N_REPETITIONS = 15
