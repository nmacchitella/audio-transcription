import streamlit as st
from groq import Groq, RateLimitError
from pydub import AudioSegment
from pathlib import Path
import tempfile
import subprocess
import time
import re

# Page config
st.set_page_config(
    page_title="Audio Transcription",
    page_icon="🎙️",
    layout="centered"
)

# Password protection
def check_password():
    """Returns True if the user entered the correct password."""

    def password_entered():
        if st.session_state["password"] == st.secrets["app"]["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if st.session_state.get("password_correct", False):
        return True

    st.text_input(
        "Password", type="password", on_change=password_entered, key="password"
    )
    if "password_correct" in st.session_state and not st.session_state["password_correct"]:
        st.error("Incorrect password")
    return False


# --- Core transcription functions (from notebook) ---

def preprocess_audio(input_path: Path) -> Path:
    """Preprocess audio file to 16kHz mono FLAC using ffmpeg."""
    with tempfile.NamedTemporaryFile(suffix='.flac', delete=False) as temp_file:
        output_path = Path(temp_file.name)

    subprocess.run([
        'ffmpeg',
        '-hide_banner',
        '-loglevel', 'error',
        '-i', str(input_path),
        '-ar', '16000',
        '-ac', '1',
        '-c:a', 'flac',
        '-y',
        str(output_path)
    ], check=True)
    return output_path


def transcribe_single_chunk(client: Groq, chunk: AudioSegment, chunk_num: int, total_chunks: int) -> tuple[dict, float]:
    """Transcribe a single audio chunk with Groq API."""
    total_api_time = 0

    while True:
        with tempfile.NamedTemporaryFile(suffix='.flac') as temp_file:
            chunk.export(temp_file.name, format='flac')

            start_time = time.time()
            try:
                result = client.audio.transcriptions.create(
                    file=("chunk.flac", temp_file, "audio/flac"),
                    model="whisper-large-v3",
                    language=st.session_state.get("language", "en"),
                    response_format="verbose_json"
                )
                api_time = time.time() - start_time
                total_api_time += api_time
                return result, total_api_time

            except RateLimitError:
                st.warning(f"Rate limit hit for chunk {chunk_num} - retrying in 60 seconds...")
                time.sleep(60)
                continue


def find_longest_common_sequence(sequences: list[str], match_by_words: bool = True) -> str:
    """Find the optimal alignment between sequences."""
    if not sequences:
        return ""

    if match_by_words:
        sequences = [
            [word for word in re.split(r'(\s+\w+)', seq) if word]
            for seq in sequences
        ]
    else:
        sequences = [list(seq) for seq in sequences]

    left_sequence = sequences[0]
    left_length = len(left_sequence)
    total_sequence = []

    for right_sequence in sequences[1:]:
        max_matching = 0.0
        right_length = len(right_sequence)
        max_indices = (left_length, left_length, 0, 0)

        for i in range(1, left_length + right_length + 1):
            eps = float(i) / 10000.0

            left_start = max(0, left_length - i)
            left_stop = min(left_length, left_length + right_length - i)
            left = left_sequence[left_start:left_stop]

            right_start = max(0, i - left_length)
            right_stop = min(right_length, i)
            right = right_sequence[right_start:right_stop]

            if len(left) != len(right):
                raise RuntimeError("Mismatched subsequences detected during transcript merging.")

            matches = sum(a == b for a, b in zip(left, right))
            matching = matches / float(i) + eps

            if matches > 1 and matching > max_matching:
                max_matching = matching
                max_indices = (left_start, left_stop, right_start, right_stop)

        left_start, left_stop, right_start, right_stop = max_indices
        left_mid = (left_stop + left_start) // 2
        right_mid = (right_stop + right_start) // 2

        total_sequence.extend(left_sequence[:left_mid])
        left_sequence = right_sequence[right_mid:]
        left_length = len(left_sequence)

    total_sequence.extend(left_sequence)

    if match_by_words:
        return ''.join(total_sequence)
    return ''.join(total_sequence)


def merge_transcripts(results: list[tuple[dict, int]]) -> dict:
    """Merge transcription chunks and handle overlaps."""
    final_segments = []
    processed_chunks = []

    for i, (chunk, _) in enumerate(results):
        data = chunk.model_dump() if hasattr(chunk, 'model_dump') else chunk
        segments = data['segments']

        if i < len(results) - 1:
            next_start = results[i + 1][1]
            current_segments = []
            overlap_segments = []

            for segment in segments:
                if segment['end'] * 1000 > next_start:
                    overlap_segments.append(segment)
                else:
                    current_segments.append(segment)

            if overlap_segments:
                merged_overlap = overlap_segments[0].copy()
                merged_overlap.update({
                    'text': ' '.join(s['text'] for s in overlap_segments),
                    'end': overlap_segments[-1]['end']
                })
                current_segments.append(merged_overlap)

            processed_chunks.append(current_segments)
        else:
            processed_chunks.append(segments)

    for i in range(len(processed_chunks) - 1):
        final_segments.extend(processed_chunks[i][:-1])

        last_segment = processed_chunks[i][-1]
        first_segment = processed_chunks[i + 1][0]

        merged_text = find_longest_common_sequence([last_segment['text'], first_segment['text']])
        merged_segment = last_segment.copy()
        merged_segment.update({
            'text': merged_text,
            'end': first_segment['end']
        })
        final_segments.append(merged_segment)

    if processed_chunks:
        final_segments.extend(processed_chunks[-1])

    final_text = ' '.join(segment['text'] for segment in final_segments)

    return {
        "text": final_text,
        "segments": final_segments
    }


def transcribe_audio(audio_path: Path, chunk_length: int = 600, overlap: int = 10, progress_callback=None) -> dict:
    """Main transcription function with progress tracking."""
    client = Groq(api_key=st.secrets["groq"]["api_key"], max_retries=1)

    processed_path = None
    try:
        processed_path = preprocess_audio(audio_path)
        audio = AudioSegment.from_file(processed_path, format="flac")

        duration = len(audio)
        chunk_ms = chunk_length * 1000
        overlap_ms = overlap * 1000
        total_chunks = (duration // (chunk_ms - overlap_ms)) + 1

        results = []

        for i in range(total_chunks):
            start = i * (chunk_ms - overlap_ms)
            end = min(start + chunk_ms, duration)

            if progress_callback:
                progress_callback(i + 1, total_chunks, start/1000, end/1000)

            chunk = audio[start:end]
            result, _ = transcribe_single_chunk(client, chunk, i + 1, total_chunks)
            results.append((result, start))

        return merge_transcripts(results)

    finally:
        if processed_path:
            Path(processed_path).unlink(missing_ok=True)


# --- Streamlit UI ---

def main():
    st.title("Audio Transcription")
    st.markdown("Upload an audio file and get the transcription.")

    # Language selection
    language_options = {
        "English": "en",
        "Italian": "it",
        "Spanish": "es",
        "French": "fr",
        "German": "de",
        "Portuguese": "pt",
        "Dutch": "nl",
        "Russian": "ru",
        "Chinese": "zh",
        "Japanese": "ja",
        "Korean": "ko",
    }

    selected_language = st.selectbox(
        "Audio Language",
        options=list(language_options.keys()),
        index=0
    )
    st.session_state["language"] = language_options[selected_language]

    # File upload
    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=["mp3", "wav", "m4a", "flac", "ogg", "mp4", "mpeg", "mpga", "webm"],
        help="Supported formats: MP3, WAV, M4A, FLAC, OGG, MP4, MPEG, MPGA, WEBM"
    )

    if uploaded_file is not None:
        st.audio(uploaded_file, format=f"audio/{uploaded_file.name.split('.')[-1]}")

        if st.button("Transcribe", type="primary"):
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = Path(tmp.name)

            try:
                progress_bar = st.progress(0)
                status_text = st.empty()

                def update_progress(current, total, start_time, end_time):
                    progress = current / total
                    progress_bar.progress(progress)
                    status_text.text(f"Processing chunk {current}/{total} ({start_time:.1f}s - {end_time:.1f}s)")

                with st.spinner("Transcribing..."):
                    result = transcribe_audio(tmp_path, progress_callback=update_progress)

                progress_bar.progress(1.0)
                status_text.text("Complete!")

                st.success("Transcription complete!")

                # Display result
                st.subheader("Transcription")
                st.text_area("Result", result["text"], height=300)

                # Download button
                st.download_button(
                    label="Download as TXT",
                    data=result["text"],
                    file_name=f"{uploaded_file.name.rsplit('.', 1)[0]}_transcription.txt",
                    mime="text/plain"
                )

            finally:
                tmp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    if check_password():
        main()
