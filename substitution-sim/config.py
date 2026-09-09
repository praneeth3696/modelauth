MODEL_PAIRS = {
    "medium": ("llama3.2:1b", "llama3.2:3b"),
    "hard": ("llama3.2:3b-instruct-q4_K_M", "llama3.2:3b-instruct-q8_0"),
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
