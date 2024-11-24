import os
from groq import Groq
import requests
import yt_dlp
from google.colab import userdata

GROQ_API_KEY = userdata.get('GROQ_API_KEY')
client = Groq(api_key=GROQ_API_KEY)

def request_yt_audio(url):
    # Create the videos directory if it doesn't exist
    os.makedirs('./videos', exist_ok=True)

    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': './videos/%(id)s.%(ext)s',
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        audio_file = ydl.prepare_filename(info)
        # Change extension to mp3 as we've extracted audio
        audio_file = os.path.splitext(audio_file)[0] + '.mp3'
    return audio_file

def transcribe_youtube_video(url, output_dir: str = None):
    audio_file = request_yt_audio(url)

    # Read the audio content
    with open(audio_file, 'rb') as f:
        audio_content = f.read()

    # Prepare the files for the API request
    files = {
        'file': ('audio.mp3', audio_content, 'audio/mpeg')
    }

    # Prepare the data for the API request
    data = {
        'model': 'whisper-large-v3',
        'response_format': 'json',
        'language': 'en',
        'temperature': 0.0
    }

    print("STARTING")
    # Make the API request
    response = requests.post(
        'https://api.groq.com/openai/v1/audio/transcriptions',
        headers={'Authorization': f'Bearer {GROQ_API_KEY}'},
        files=files,
        data=data
    )

    if response.status_code == 200:
        return response.json()['text']
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None

import os
from groq import Groq
import requests
import yt_dlp
from google.colab import userdata
from uuid import uuid4
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings


GROQ_API_KEY = userdata.get('GROQ_API_KEY')
client = Groq(api_key=GROQ_API_KEY)


def request_yt_audio(url):
    # Create the videos directory if it doesn't exist
    os.makedirs('./videos', exist_ok=True)

    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': './videos/%(id)s.%(ext)s',
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        audio_file = ydl.prepare_filename(info)
        # Change extension to mp3 as we've extracted audio
        audio_file = os.path.splitext(audio_file)[0] + '.mp3'
    return audio_file

def transcribe_youtube_video(url, output_dir: str = None):
    audio_file = request_yt_audio(url)

    # Read the audio content
    with open(audio_file, 'rb') as f:
        audio_content = f.read()

    # Prepare the files for the API request
    files = {
        'file': ('audio.mp3', audio_content, 'audio/mpeg')
    }

    # Prepare the data for the API request
    data = {
        'model': 'whisper-large-v3',
        'response_format': 'json',
        'language': 'en',
        'temperature': 0.0
    }

    print("STARTING")
    # Make the API request
    response = requests.post(
        'https://api.groq.com/openai/v1/audio/transcriptions',
        headers={'Authorization': f'Bearer {GROQ_API_KEY}'},
        files=files,
        data=data
    )

    if response.status_code == 200:
        return response.json()['text']
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None


youtube_url = "https://youtu.be/FgnjdW-x7mQ?si=IUQSt8YKOz1ZxT6X"
transcription = transcribe_youtube_video(url=youtube_url, output_dir='data')
if transcription:
    print(transcription)





text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=500,
    chunk_overlap=50,
    length_function=len,
    is_separator_regex=False,
)

texts = text_splitter.create_documents([transcription])

texts[0]


# embeddings
embeddings = FastEmbedEmbeddings()




vector_store = Chroma(
    collection_name="karl",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
)

uuids = [str(uuid4()) for _ in range(len(texts))]

vector_store.add_documents(documents=texts, ids=uuids)

def get_context(query: str):
    context = vector_store.similarity_search(
        query,
        k=10
    )
    return context


def ask_ai(query: str):
    context = get_context(query)
    chat_completion = client.chat.completions.create(
        messages=[

            {
                "role": "system",
                "content": "you are a helpful assistant. Given a user query and a context, answer the human as clearly as possible."
            },
            # Set a user message for the assistant to respond to.
            {
                "role": "user",
                "content": f"Context: {context}, query: {query}"
            }
        ],

        # The language model which will generate the completion.
        model="llama-3.1-70b-versatile",
        temperature=0,
        max_tokens=1024,
        top_p=1,
        stop=None,
        stream=False,
    )

    # Print the completion returned by the LLM.
    return chat_completion.choices[0].message.content



ask_ai("what are black dwarfs?")

youtube_url = "https://youtu.be/FgnjdW-x7mQ?si=IUQSt8YKOz1ZxT6X"
transcription = transcribe_youtube_video(url=youtube_url, output_dir='data')
if transcription:
    print(transcription)

from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=500,
    chunk_overlap=50,
    length_function=len,
    is_separator_regex=False,
)

texts = text_splitter.create_documents([transcription])

texts[0]

# embeddings
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
embeddings = FastEmbedEmbeddings()

pip install -qU "langchain-chroma>=0.1.2"

from uuid import uuid4
from langchain_chroma import Chroma


vector_store = Chroma(
    collection_name="karl",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
)

uuids = [str(uuid4()) for _ in range(len(texts))]

vector_store.add_documents(documents=texts, ids=uuids)

def get_context(query: str):
    context = vector_store.similarity_search(
        query,
        k=10
    )
    return context

def ask_ai(query: str):
    context = get_context(query)
    chat_completion = client.chat.completions.create(
        messages=[

            {
                "role": "system",
                "content": "you are a helpful assistant. Given a user query and a context, answer the human as clearly as possible."
            },
            # Set a user message for the assistant to respond to.
            {
                "role": "user",
                "content": f"Context: {context}, query: {query}"
            }
        ],

        # The language model which will generate the completion.
        model="llama-3.1-70b-versatile",
        temperature=0,
        max_tokens=1024,
        top_p=1,
        stop=None,
        stream=False,
    )

    # Print the completion returned by the LLM.
    return chat_completion.choices[0].message.content

ask_ai("what are black dwarfs")