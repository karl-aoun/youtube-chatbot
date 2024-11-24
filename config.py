from dotenv import load_dotenv
from groq import Groq
import os
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings


load_dotenv()


GROQ_API_KEY = os.environ.get('GROQ_API_KEY')
client = Groq(api_key=GROQ_API_KEY)


if GROQ_API_KEY:
    print("API Key loaded successfully")
else:
    print("ERROR: GROQ_API_KEY not found in .env file")

# embeddings
embeddings = FastEmbedEmbeddings()