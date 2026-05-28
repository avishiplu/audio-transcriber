import os
from io import BytesIO
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from pydub import AudioSegment

load_dotenv()

api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("OPENAI_API_KEY not found.")
    st.stop()

client = OpenAI(api_key=api_key)

MAX_SIZE_MB = 25

st.title("German Class Audio Transcriber")

uploaded_file = st.file_uploader(
    "Upload your audio file",
    type=["mp3", "mp4", "m4a", "wav", "mpeg", "mpg", "mpeg4"]
)

if uploaded_file is not None:
    original_name = Path(uploaded_file.name).stem

    st.success(f"Audio uploaded: {uploaded_file.name}")

    if st.button("Compress and Transcribe"):
        with st.spinner("Compressing audio..."):
            audio_bytes = uploaded_file.read()
            audio = AudioSegment.from_file(BytesIO(audio_bytes))

            audio = audio.set_channels(1)
            audio = audio.set_frame_rate(12000)

            compressed_audio = BytesIO()

            audio.export(
                compressed_audio,
                format="mp3",
                bitrate="24k"
            )

            compressed_audio.seek(0)
            compressed_audio.name = f"{original_name}_compressed.mp3"

        compressed_size_mb = len(compressed_audio.getvalue()) / (1024 * 1024)
        st.info(f"Compressed file size: {compressed_size_mb:.2f} MB")

        if compressed_size_mb > MAX_SIZE_MB:
            st.error("Compressed audio is still larger than 25 MB.")
            st.stop()

        with st.spinner("Transcribing compressed audio..."):
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=compressed_audio,
                language="de"
            )

        transcript_text = transcript.text
        transcript_file_name = f"{original_name}.txt"

        st.success("Transcription finished.")

        st.subheader("Transcript")
        st.write(transcript_text)

        st.download_button(
            label="Download Transcript",
            data=transcript_text,
            file_name=transcript_file_name,
            mime="text/plain"
        )