from openai import OpenAI
from decouple import config


client = OpenAI(
    api_key= config("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

SYSTEM_PROMPT = """তুমি "বন্ধু" — একজন আন্তরিক বাংলাভাষী AI সহকারী।সবসময় সহজ, প্রাঞ্জল ও স্বাভাবিক বাংলায় উত্তর দেবে।
Technical শব্দ (যেমন API, code, Python, database) English-এই রাখবে।উত্তর অযথা দীর্ঘ করবে না।"""

messages = [
    {
        "role": "system", 
        "content": SYSTEM_PROMPT 
    }
]

print("\n'বন্ধু' চালু হলো! কথা শেষ করতে লিখুন: exit")

while True:
    user_input = input("\n You: ")

    if user_input.strip().lower() in ["exit", "quit", "bye"]:
        print("\nবন্ধু: তাহলে আজ এখানেই শেষ, আবার কথা হবে!")
        break

    messages.append(
        {
            "role": "user", 
            "content": user_input
        }
    )

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=messages
    )

    reply = response.choices[0].message.content
    print(f"\nবন্ধু: {reply}")
    messages.append(
        {
            "role": "assistant",
            "content": reply
        }
    )
    if len(messages) > 21 :
        messages = [ messages[0] ] + messages[-20:]
    #print(response.usage)