import os
import time
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(
    
    api_key=os.getenv("GROQ_API_KEY")
)
MODEL = "llama-3.3-70b-versatile"

tests = [
    (
        "How many people attended Super Bowl 50?",
        "Super Bowl 50 was played Feb 7 2016 at Levis Stadium in Santa Clara CA."
    ),
    (
        "Which team won Super Bowl 50?",
        "The Denver Broncos defeated the Carolina Panthers 24-10."
    ),
    (
        "What year was Arthur Magazine founded?",
        "Arthur Magazine was an American literary periodical published in Philadelphia."
    ),
]

print("STRICT PROMPT:")
print("=" * 55)
for q, ctx in tests:
    try:
        prompt = "Answer ONLY from context. If not in context say so.\n"
        prompt += "Context: " + ctx + "\n"
        prompt += "Question: " + q + "\nAnswer:"
        r = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=60
        )
        print("Q:", q[:55])
        print("A:", r.choices[0].message.content.strip()[:100])
        print()
        time.sleep(20)
    except Exception as e:
        print("Error:", e)
        time.sleep(30)