MODEL_PAIRS = {
    "easy": ("llama3.2:3b", "qwen2.5:3b"),
    "medium": ("llama3.2:1b", "llama3.2:3b"),
    "hard": ("llama3.2:3b-instruct-q4_K_M", "llama3.2:3b-instruct-q8_0"),
}

PROBE_TEMPLATES = [
    "Pick a random number between 1 and 100. Reply with only the number.",
    "Name a random number from 1 to 100, just the digits.",
    "Give me a random integer in the range 1-100. Number only.",
]

TEMPERATURE = 1.0

# Was 5, which truncated responses mid-digits: "854" and "8549" appeared in the
# easy-tier logs as fragments of longer answers, and the bare \d+ parser accepted
# them. 76 such values inflated the pooled llama3.2:3b std from 18.25 to 349.09.
MAX_TOKENS = 8

# Answers outside this range are recorded as out_of_range rather than parsed.
# The probe asks for 1-100; anything else is a truncation or a refusal preamble.
VALID_RANGE = (1, 100)

TOTAL_REQUESTS = 400
SWITCH_POINT = 200
N_REPETITIONS = 15
