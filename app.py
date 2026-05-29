import os
from io import BytesIO
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from pydub import AudioSegment
from faster_whisper import WhisperModel

load_dotenv()

api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("OPENAI_API_KEY not found.")
    st.stop()

client = OpenAI(api_key=api_key)

@st.cache_resource
def load_local_whisper_model():
    return WhisperModel(
        "large-v3-turbo",
        device="cpu",
        compute_type="int8"
    )

SAFE_CHUNK_SIZE_MB = 24
SAFE_CHUNK_SIZE_BYTES = SAFE_CHUNK_SIZE_MB * 1024 * 1024

st.title("German Class Audio Transcriber")

access_code = st.text_input(
    "Enter access code",
    type="password"
)

correct_code = st.secrets.get("APP_ACCESS_CODE", os.getenv("APP_ACCESS_CODE"))

if access_code != correct_code:
    st.warning("Please enter the correct access code to use this app.")
    st.stop()

input_method = st.radio(
    "Choose audio input method",
    ["Upload audio file", "Record audio directly"]
)

if input_method == "Upload audio file":
    uploaded_file = st.file_uploader(
        "Upload your audio file",
        type=["mp3", "mp4", "m4a", "wav", "mpeg", "mpg", "mpeg4"]
    )

else:
    uploaded_file = st.audio_input("Record your audio")

if uploaded_file is not None:
    original_name = Path(uploaded_file.name).stem

    st.success(f"Audio uploaded: {uploaded_file.name}")

    openai_button_clicked = st.button(
        "Transcribe with OpenAI",
        disabled=st.session_state.get("processing", False)
    )

    whisper_button_clicked = st.button(
        "Transcribe with Local Whisper",
        disabled=st.session_state.get("processing", False)
    )

    button_clicked = openai_button_clicked or whisper_button_clicked

    if button_clicked:
        st.session_state["processing"] = True

        st.warning(
            "Processing started. Please wait. Do not click again or refresh the page."
        )
        with st.spinner("Reading audio..."):
            audio_bytes = uploaded_file.read()

        with st.spinner("Preparing audio for transcription..."):
            prepared_audio = AudioSegment.from_file(BytesIO(audio_bytes))
            prepared_audio = prepared_audio.set_channels(1)
            prepared_audio = prepared_audio.set_frame_rate(16000)

        original_file_size_mb = len(audio_bytes) / (1024 * 1024)
        chunks = []

        if len(audio_bytes) <= SAFE_CHUNK_SIZE_BYTES:
            original_audio_file = BytesIO()
            prepared_audio.export(
                original_audio_file,
                format="wav"
            )

            original_audio_file.seek(0)
            original_audio_file.name = f"{original_name}.wav"
            chunks.append(original_audio_file)

            st.info(
                f"Audio size is {original_file_size_mb:.2f} MB. "
                "No splitting needed."
            )

        else:
            with st.spinner("Audio is larger than 24 MB. Preparing smaller parts..."):
                audio = prepared_audio
                estimated_chunk_count = len(audio_bytes) // SAFE_CHUNK_SIZE_BYTES

                if len(audio_bytes) % SAFE_CHUNK_SIZE_BYTES != 0:
                    estimated_chunk_count += 1

                chunk_length_ms = len(audio) // estimated_chunk_count

                temporary_chunks = []

                for start_ms in range(0, len(audio), chunk_length_ms):
                    end_ms = start_ms + chunk_length_ms
                    temporary_chunks.append(audio[start_ms:end_ms])

                for temporary_chunk in temporary_chunks:
                    parts_to_check = [temporary_chunk]

                    while parts_to_check:
                        current_part = parts_to_check.pop(0)

                        chunk_file = BytesIO()
                        current_part.export(
                            chunk_file,
                            format="mp3",
                            bitrate="64k"
                        )

                        chunk_file.seek(0)

                        if len(chunk_file.getvalue()) <= SAFE_CHUNK_SIZE_BYTES:
                            chunk_file.name = f"{original_name}_part_{len(chunks) + 1}.mp3"
                            chunks.append(chunk_file)

                        else:
                            middle_ms = len(current_part) // 2
                            first_half = current_part[:middle_ms]
                            second_half = current_part[middle_ms:]

                            parts_to_check.append(first_half)
                            parts_to_check.append(second_half)

            st.info(
                f"Audio size is {original_file_size_mb:.2f} MB. "
                f"Audio split into {len(chunks)} parts."
            )

        all_transcripts = []

        for index, chunk_file in enumerate(chunks, start=1):

            if openai_button_clicked:
                with st.spinner(f"Transcribing part {index} of {len(chunks)} with OpenAI..."):
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

            if whisper_button_clicked:
                with st.spinner(f"Transcribing part {index} of {len(chunks)} with Local Whisper..."):
                    local_whisper_model = load_local_whisper_model()
                    segments, info = local_whisper_model.transcribe(
                        chunk_file,
                        language="de"
                    )

                part_text = f"\n\n--- Part {index} ---\n\n"

                previous_line = ""

                for segment in segments:
                    current_line = segment.text.strip()

                    if current_line and current_line != previous_line:
                        part_text += f"{current_line}\n"
                        previous_line = current_line

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