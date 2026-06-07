import os
import subprocess
import tempfile
from io import BytesIO
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("OPENAI_API_KEY not found.")
    st.stop()

client = OpenAI(api_key=api_key)

groq_api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")

groq_client = None

if groq_api_key:
    groq_client = OpenAI(
        api_key=groq_api_key,
        base_url="https://api.groq.com/openai/v1"
    )

@st.cache_resource
def load_local_whisper_model():
    from faster_whisper import WhisperModel

    return WhisperModel(
        "small",
        device="cpu",
        compute_type="int8"
    )

WHISPER_LANGUAGES = {
    "Auto Detect": None,
    "Afrikaans": "af",
    "Albanian": "sq",
    "Amharic": "am",
    "Arabic": "ar",
    "Armenian": "hy",
    "Assamese": "as",
    "Azerbaijani": "az",
    "Bashkir": "ba",
    "Basque": "eu",
    "Belarusian": "be",
    "Bengali": "bn",
    "Bosnian": "bs",
    "Breton": "br",
    "Bulgarian": "bg",
    "Cantonese": "yue",
    "Catalan": "ca",
    "Chinese": "zh",
    "Croatian": "hr",
    "Czech": "cs",
    "Danish": "da",
    "Dutch": "nl",
    "English": "en",
    "Estonian": "et",
    "Faroese": "fo",
    "Finnish": "fi",
    "French": "fr",
    "Galician": "gl",
    "Georgian": "ka",
    "German": "de",
    "Greek": "el",
    "Gujarati": "gu",
    "Haitian Creole": "ht",
    "Hausa": "ha",
    "Hawaiian": "haw",
    "Hebrew": "he",
    "Hindi": "hi",
    "Hungarian": "hu",
    "Icelandic": "is",
    "Indonesian": "id",
    "Italian": "it",
    "Japanese": "ja",
    "Javanese": "jw",
    "Kannada": "kn",
    "Kazakh": "kk",
    "Khmer": "km",
    "Korean": "ko",
    "Lao": "lo",
    "Latin": "la",
    "Latvian": "lv",
    "Lingala": "ln",
    "Lithuanian": "lt",
    "Luxembourgish": "lb",
    "Macedonian": "mk",
    "Malagasy": "mg",
    "Malay": "ms",
    "Malayalam": "ml",
    "Maltese": "mt",
    "Maori": "mi",
    "Marathi": "mr",
    "Mongolian": "mn",
    "Myanmar": "my",
    "Nepali": "ne",
    "Norwegian": "no",
    "Nynorsk": "nn",
    "Occitan": "oc",
    "Pashto": "ps",
    "Persian": "fa",
    "Polish": "pl",
    "Portuguese": "pt",
    "Punjabi": "pa",
    "Romanian": "ro",
    "Russian": "ru",
    "Sanskrit": "sa",
    "Serbian": "sr",
    "Shona": "sn",
    "Sindhi": "sd",
    "Sinhala": "si",
    "Slovak": "sk",
    "Slovenian": "sl",
    "Somali": "so",
    "Spanish": "es",
    "Sundanese": "su",
    "Swahili": "sw",
    "Swedish": "sv",
    "Tagalog": "tl",
    "Tajik": "tg",
    "Tamil": "ta",
    "Tatar": "tt",
    "Telugu": "te",
    "Thai": "th",
    "Tibetan": "bo",
    "Turkish": "tr",
    "Turkmen": "tk",
    "Ukrainian": "uk",
    "Urdu": "ur",
    "Uzbek": "uz",
    "Vietnamese": "vi",
    "Welsh": "cy",
    "Yiddish": "yi",
    "Yoruba": "yo",
}

SAFE_CHUNK_SIZE_MB = 24
SAFE_CHUNK_SIZE_BYTES = SAFE_CHUNK_SIZE_MB * 1024 * 1024

def get_audio_duration_seconds(input_file_path):
    command = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(input_file_path)
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=True
    )

    return float(result.stdout.strip())


def create_safe_audio_chunks_with_ffmpeg(input_file_path, output_dir, original_name):
    segment_seconds = 30 * 60

    while segment_seconds >= 60:
        for old_chunk in Path(output_dir).glob("*.mp3"):
            old_chunk.unlink()

        output_pattern = str(Path(output_dir) / f"{original_name}_part_%03d.mp3")

        command = [
            "ffmpeg",
            "-y",
            "-i", str(input_file_path),
            "-ac", "1",
            "-ar", "16000",
            "-b:a", "64k",
            "-f", "segment",
            "-segment_time", str(segment_seconds),
            "-reset_timestamps", "1",
            output_pattern
        ]

        subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True
        )

        chunk_paths = sorted(Path(output_dir).glob("*.mp3"))

        if chunk_paths and all(chunk.stat().st_size <= SAFE_CHUNK_SIZE_BYTES for chunk in chunk_paths):
            return chunk_paths, segment_seconds

        segment_seconds = segment_seconds // 2

    raise ValueError("Could not create safe audio chunks below the size limit.")

st.title("Audio to Text Transcriber")
st.caption("Upload or record audio, convert it into clean text, and download your transcript.")

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

selected_language_name = st.selectbox(
    "Choose audio language",
    list(WHISPER_LANGUAGES.keys())
)

selected_language_code = WHISPER_LANGUAGES[selected_language_name]


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

    groq_button_clicked = st.button(
        "Transcribe with Groq Fast Engine",
        disabled=st.session_state.get("processing", False)
    )


    whisper_button_clicked = st.button(
        "Transcribe with Local Whisper",
        disabled=st.session_state.get("processing", False)
    )

    button_clicked = openai_button_clicked or groq_button_clicked or whisper_button_clicked

    if button_clicked:
        st.session_state["processing"] = True

        st.warning(
            "Processing started. Please wait. Do not click again or refresh the page."
        )
        with st.spinner("Saving uploaded audio safely..."):
            audio_bytes = uploaded_file.read()
            original_file_size_mb = len(audio_bytes) / (1024 * 1024)

            temp_dir = tempfile.mkdtemp()
            input_file_path = Path(temp_dir) / uploaded_file.name

            with open(input_file_path, "wb") as input_file:
                input_file.write(audio_bytes)

        chunks = []

        with st.spinner("Splitting audio into safe parts with FFmpeg..."):
            audio_duration_seconds = get_audio_duration_seconds(input_file_path)

            chunk_paths, used_segment_seconds = create_safe_audio_chunks_with_ffmpeg(
                input_file_path=input_file_path,
                output_dir=temp_dir,
                original_name=original_name
            )

            for chunk_path in chunk_paths:
                chunk_bytes = chunk_path.read_bytes()
                chunk_file = BytesIO(chunk_bytes)
                chunk_file.name = chunk_path.name
                chunks.append(chunk_file)

        st.info(
            f"Audio size is {original_file_size_mb:.2f} MB. "
            f"Audio length is {audio_duration_seconds / 60:.1f} minutes. "
            f"Audio prepared as {len(chunks)} part(s). "
            f"Each part is about {used_segment_seconds / 60:.1f} minutes or smaller."
        )

        all_transcripts = []

        for index, chunk_file in enumerate(chunks, start=1):

            if openai_button_clicked:
                with st.spinner(f"Transcribing part {index} of {len(chunks)} with OpenAI..."):
                    openai_transcription_kwargs = {
                        "model": "gpt-4o-transcribe-diarize",
                        "file": chunk_file,
                        "response_format": "diarized_json",
                        "chunking_strategy": "auto"
                    }

                    if selected_language_code is not None:
                        openai_transcription_kwargs["language"] = selected_language_code

                    transcript = client.audio.transcriptions.create(
                        **openai_transcription_kwargs
                    )

                part_text = f"\n\n--- Part {index} ---\n\n"

                for segment in transcript.segments:
                    speaker = segment.speaker
                    text = segment.text
                    part_text += f"{speaker}:\n{text}\n\n"

                all_transcripts.append(part_text)

            if groq_button_clicked:
                if groq_client is None:
                    st.error("GROQ_API_KEY not found. Please add it to Streamlit Secrets.")
                    st.stop()

                with st.spinner(f"Transcribing part {index} of {len(chunks)} with Groq Fast Engine..."):
                    groq_transcription_kwargs = {
                        "model": "whisper-large-v3-turbo",
                        "file": chunk_file,
                        "response_format": "text"
                    }

                    if selected_language_code is not None:
                        groq_transcription_kwargs["language"] = selected_language_code

                    transcript = groq_client.audio.transcriptions.create(
                        **groq_transcription_kwargs
                    )

                part_text = f"\n\n--- Part {index} ---\n\n"
                part_text += str(transcript)

                all_transcripts.append(part_text)

            if whisper_button_clicked:
                with st.spinner(f"Transcribing part {index} of {len(chunks)} with Local Whisper..."):
                    local_whisper_model = load_local_whisper_model()
                    whisper_transcription_kwargs = {}

                    if selected_language_code is not None:
                        whisper_transcription_kwargs["language"] = selected_language_code

                    segments, info = local_whisper_model.transcribe(
                        chunk_file,
                        **whisper_transcription_kwargs
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

        if openai_button_clicked:
            with st.spinner("Organizing transcript with AI..."):
                cleanup_response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You clean and organize audio transcripts in the same language as the transcript. "
                                "Do not translate the transcript. "
                                "Do not add new information. "
                                "Do not summarize. "
                                "Do not remove important content. "
                                "Only fix obvious transcription mistakes, improve spacing, punctuation, "
                                "speaker labels, and make the transcript easier to read."
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