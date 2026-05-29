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

    button_clicked = st.button(
        "Split and Transcribe",
        disabled=st.session_state.get("processing", False)
    )

    if button_clicked:
        st.session_state["processing"] = True

        st.warning(
            "Processing started. Please wait. Do not click again or refresh the page."
        )
        with st.spinner("Reading audio..."):
            audio_bytes = uploaded_file.read()
            audio = AudioSegment.from_file(BytesIO(audio_bytes))

        chunk_length_ms = 10 * 60 * 1000  # 10 minutes per chunk
        chunks = []

        for start_ms in range(0, len(audio), chunk_length_ms):
            end_ms = start_ms + chunk_length_ms
            chunk = audio[start_ms:end_ms]

            chunk_file = BytesIO()
            chunk.export(
                chunk_file,
                format="mp3",
                bitrate="64k"
            )

            chunk_file.seek(0)
            chunk_file.name = f"{original_name}_part_{len(chunks) + 1}.mp3"
            chunks.append(chunk_file)

        st.info(f"Audio split into {len(chunks)} parts.")

        all_transcripts = []

        for index, chunk_file in enumerate(chunks, start=1):
            with st.spinner(f"Transcribing part {index} of {len(chunks)}..."):
                transcript = client.audio.transcriptions.create(
                    model="gpt-4o-transcribe-diarize",
                    file=chunk_file,
                    language="de",
                    response_format="diarized_json",
                    chunking_strategy="auto"
                )

            part_text = f"\n\n--- Part {index} ---\n\n"

            for segment in transcript.segments:
                speaker = segment.speaker
                text = segment.text
                part_text += f"{speaker}:\n{text}\n\n"

            all_transcripts.append(part_text)

        transcript_text = "".join(all_transcripts)

        with st.spinner("Organizing transcript with AI..."):
            cleanup_response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You clean and organize German class transcripts. "
                            "Do not add new information. Do not summarize. "
                            "Do not remove important content. "
                            "Only fix obvious transcription mistakes, improve spacing, "
                            "organize speaker labels, and make the transcript easier to read."
                        )
                    },
                    {
                        "role": "user",
                        "content": transcript_text
                    }
                ],
                temperature=0.2
            )

            transcript_text = cleanup_response.choices[0].message.content


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

        st.session_state["processing"] = False