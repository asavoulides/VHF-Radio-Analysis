import os
import torch
import whisper
from pydub import AudioSegment
from concurrent.futures import ThreadPoolExecutor
import time
import threading
from openai import OpenAI

client = OpenAI()


# Function for handling GPT-4 request based on transcript
def gptReq(transcript):
    start_tokens = len(transcript) // 4  # Rough token estimate: 1 token ~ 4 characters
    try:
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are an insightful detective tasked with analyzing radio transcripts. Summarize events relevant to any criminal activities or noteworthy incidents without additional explanations.",
                },
                {
                    "role": "user",
                    "content": f"Using the following police radio transcript, provide a structured list of events focusing only on scene-relevant information:\n{transcript}",
                },
            ],
        )
        # Extract token usage and estimate cost
        token_used = completion.usage.total_tokens
        cost_estimate = (token_used / 1000) * 0.03  # $0.03 per 1k tokens

        # Display formatted output
        print("\n" + "=" * 60)
        print("[GPT-4 Response Summary]")
        print(f"Tokens Used: {token_used} | Estimated Cost: ~${cost_estimate:.2f}")
        print("\n[Event Analysis]:")
        print(completion.choices[0].message.content.strip())
        print("=" * 60)

        return completion.choices[0].message

    except Exception as e:
        print(f"[Error] GPT request failed: {e}")
        return None


# Function to transcribe an audio file to text using Whisper
def transcribe_audio(mp3_file_path, model_size="base"):
    try:
        print(f"[Live Update] Converting {mp3_file_path} to WAV format...")
        audio = AudioSegment.from_mp3(mp3_file_path)
        wav_file_path = mp3_file_path.replace(".mp3", ".wav")
        audio.export(wav_file_path, format="wav")
        print(f"[Live Update] Conversion completed for {mp3_file_path}.")

        print(f"[Live Update] Loading Whisper model ({model_size})...")
        model = whisper.load_model(model_size)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Live Update] Using device: {device}")
        model = model.to(device)

        print(f"[Live Update] Transcribing {wav_file_path}...")
        result = model.transcribe(wav_file_path)
        transcription = result["text"]
        print(
            f"[Live Update] Transcription completed. Transcript length: {len(transcription)} characters."
        )

        return transcription

    except Exception as e:
        print(f"[Error] Transcription failed for {mp3_file_path}: {e}")
        return None


# Function to list audio files in the designated directory
def list_audio_files(directory="Audio"):
    if not os.path.exists(directory):
        print(f"[Warning] Directory '{directory}' does not exist.")
        return []
    files = [
        f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))
    ]
    print(f"[Live Update] Found {len(files)} audio file(s) in '{directory}'.")
    return files


# Function to transcribe multiple files in parallel
def transcribe_files_in_parallel(files, model_size="base"):
    print(
        f"[Live Update] Starting transcription for {len(files)} file(s) in parallel..."
    )
    with ThreadPoolExecutor() as executor:
        transcripts = list(
            executor.map(lambda f: transcribe_audio(f"Audio/{f}", model_size), files)
        )
    print("[Live Update] Completed transcription for all files.")
    return transcripts


# Displaying a stopwatch in the background
def display_stopwatch():
    start_time = time.time()
    while True:
        elapsed_time = time.time() - start_time
        print(f"[Stopwatch] Elapsed Time: {elapsed_time:.2f} seconds", end="\r")
        time.sleep(0.1)


stopwatch_thread = threading.Thread(target=display_stopwatch, daemon=True)
stopwatch_thread.start()

# Main logic
if __name__ == "__main__":
    files = list_audio_files()
    if files:
        print("[Live Update] Starting the transcription process...")
        transcripts = transcribe_files_in_parallel(files)
        full_transcription = "\n".join(filter(None, transcripts))
        print("\n[Final Transcription Result]")
        gptReq(full_transcription)
    else:
        print("[Info] No files to transcribe.")
