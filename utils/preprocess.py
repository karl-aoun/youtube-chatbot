import requests
import yt_dlp
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter


from config import GROQ_API_KEY


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
    



def split_text(transcription, chunk_size=500, chunk_overlap=50, length_function=len, is_separator_regex=False):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=length_function,
        is_separator_regex=is_separator_regex,
    )

    texts = text_splitter.create_documents([transcription])
    return texts