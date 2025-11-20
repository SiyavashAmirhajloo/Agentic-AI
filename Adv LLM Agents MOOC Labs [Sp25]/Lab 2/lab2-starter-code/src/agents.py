# src/agents.py
from groq import Groq
import os

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

class LLM_Agent:
    def __init__(self, model="llama-3.3-70b-versatile"):
        self.model = model

    def get_response(self, messages, temperature=0.7, max_tokens=2048):
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=0.9
        )
        return response.choices[0].message.content

class Generation_Agent(LLM_Agent):
    pass  # We only need one strong model