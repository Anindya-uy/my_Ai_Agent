from openai import OpenAI
#import os
#from dotenv import load_dotenv
from decouple import config

#load_dotenv()

client = OpenAI(
    #api_key= os.getenv("GROK_API_KEY"),
    api_key= config("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

response = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[
        {"role": "user", 
         "content": "Tell me, what is my name?"
         #"content": "বাংলায় আমাকে হ্যালো বলো আর এক লাইনে উৎসাহ দাও!"
        }
    ]
)

print(response.choices[0].message.content)
#print(response.usage)