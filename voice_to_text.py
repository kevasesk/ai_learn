import sys
import os
import torch
from transformers import pipeline
from pydub import AudioSegment
import math

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load the ASR pipeline
try:
    pipe = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3", device=device)
except Exception as e:
    print(f"Error loading pipeline: {e}")
    sys.exit(1)

audio_file = 'test.mp3'
output_dir = 'audio_chunks' # Directory to save temporary audio chunks
chunk_length_ms = 29 * 1000  # 29 seconds in milliseconds (slightly less than 30)

# Check if the file exists
if not os.path.exists(audio_file):
    print(f"Error: File '{audio_file}' does not exist.")
    sys.exit(1)

# Create output directory for chunks if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def split_audio(file_path, chunk_length_ms, output_folder):
    """
    Splits an audio file into chunks of a specified length.
    Returns a list of paths to the created chunk files.
    """
    audio = AudioSegment.from_mp3(file_path)
    total_length_ms = len(audio)
    
    num_chunks = math.ceil(total_length_ms / chunk_length_ms)
    
    chunk_files = []
    print(f"Splitting audio into {num_chunks} chunks...")

    for i in range(num_chunks):
        start_time = i * chunk_length_ms
        end_time = min((i + 1) * chunk_length_ms, total_length_ms)
        chunk = audio[start_time:end_time]
        
        chunk_filename = os.path.join(output_folder, f"{os.path.basename(file_path).split('.')[0]}_part_{i:03d}.mp3")
        chunk.export(chunk_filename, format="mp3")
        chunk_files.append(chunk_filename)
        print(f"Exported: {chunk_filename}")
    
    return chunk_files

def transcribe_chunks(chunk_files, pipeline_obj):
    """
    Transcribes a list of audio chunk files and combines their text.
    """
    full_transcription = []
    
    for i, chunk_file in enumerate(chunk_files):
        print(f"Processing chunk {i+1}/{len(chunk_files)}: {chunk_file}")
        try:
            # No need for return_timestamps=True here for individual chunks < 30s
            # but it doesn't hurt.
            transcription = pipeline_obj(chunk_file, generate_kwargs={"language": "en"})
            full_transcription.append(transcription["text"])
        except Exception as e:
            print(f"Error transcribing chunk {chunk_file}: {e}")
            full_transcription.append(f"[ERROR IN CHUNK {i+1}]") # Mark problematic chunks
    
    return " ".join(full_transcription)

# --- Main Processing ---
try:
    # 1. Split the audio file
    chunks = split_audio(audio_file, chunk_length_ms, output_dir)
    
    # 2. Process each chunk
    final_text = transcribe_chunks(chunks, pipe)
    
    print("\n--- Full Transcription ---")
    print(final_text)


except Exception as e:
    print(f"An error occurred during splitting or transcription: {e}")

finally:
    # 3. Clean up temporary chunk files
    print("\nCleaning up temporary audio chunks...")
    for f in os.listdir(output_dir):
        file_path = os.path.join(output_dir, f)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Error deleting file {file_path}: {e}")
    if os.path.exists(output_dir) and not os.listdir(output_dir): # Check if directory is empty
        os.rmdir(output_dir)
    print("Cleanup complete.")