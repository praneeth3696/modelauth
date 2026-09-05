import time
from openai import OpenAI
from config import MAX_TOKENS

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

def probe(model_name: str, prompt: str, temperature: float = 1.0, max_tokens: int = MAX_TOKENS, retries: int = 3):
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[warn] probe failed (attempt {attempt+1}/{retries}): {e}")
            time.sleep(2)
    return None
