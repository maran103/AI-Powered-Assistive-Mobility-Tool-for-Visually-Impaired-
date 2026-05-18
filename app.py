import os
import cv2
import tempfile
import mimetypes
import numpy as np
import streamlit as st
from ultralytics import YOLO
from gtts import gTTS
from groq import Groq

from pydub import AudioSegment

# ─────────────────────────────────────────
#  Page config
# ─────────────────────────────────────────
st.set_page_config(
    page_title="AI Assistive Mobility Tool",
    page_icon="👁️",
    layout="wide"
)

st.markdown("""
<style>
    .main-title {
        font-size: 42px;
        font-weight: bold;
        background: linear-gradient(90deg, #00c6ff, #0072ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
    }
    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #888;
        margin-bottom: 30px;
    }
    .card {
        background: rgba(0,0,0,0.05);
        border: 1px solid rgba(0,0,0,0.1);
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">👁️ AI Assistive Mobility Tool</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Real-time navigation guidance for the visually impaired — powered by YOLOv8 + LLM</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────
#  Load models (cached so they load once)
# ─────────────────────────────────────────
@st.cache_resource
def load_models():
    model_coco  = YOLO("yolov8n.pt")                  # auto-downloaded
    model_zebra = YOLO("models/best.pt")               # your custom zebra model
    model_light = YOLO("models/traffic_light.pt")      # your custom traffic light model
    return model_coco, model_zebra, model_light

@st.cache_resource
def load_groq_client():
    return Groq(api_key=os.environ["GROQ_API_KEY"])

with st.spinner("Loading models..."):
    model_coco, model_zebra, model_light = load_models()
    groq_client = load_groq_client()

# ─────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────
def generate_guidance(scene_description: str) -> str:
    """Call Groq LLaMA instead of Gemma — 10x faster, same quality."""
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{
                "role": "user",
                "content": (
                    "You are a human walking guide helping a blind pedestrian. "
                    f"Environment: {scene_description}. "
                    "Give ONLY one short, natural instruction in plain English. "
                    "Speak as if standing next to them. "
                    "Examples: 'Wait, red light ahead', 'Cross now at the zebra crossing', 'Walk forward carefully'. "
                    "Do NOT mention AI, detection, models, or technology. "
                    "Do NOT describe the task — just give the instruction."
                )
            }],
            max_tokens=40,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return ""

def format_counts(class_counts: dict) -> str:
    parts = []
    for lbl, cnt in class_counts.items():
        parts.append(f"one {lbl}" if cnt == 1 else f"{cnt} {lbl}s")
    return ", ".join(parts)

def process_frame(frame, current_time, last_narration_time, cooldown, audio_segments):
    """Run detection on one frame, return annotated frame + updated state."""
    frame = cv2.resize(frame, (1280, 720))
    annotated = frame.copy()

    results_coco  = model_coco(frame,  verbose=False)
    results_zebra = model_zebra(frame, verbose=False)
    results_light = model_light(frame, verbose=False)

    class_counts, zebra_count, light_color = {}, 0, None

    # Traffic lights
    for box in results_light[0].boxes:
        cls   = int(box.cls[0])
        label = model_light.names[cls]
        conf  = float(box.conf[0])
        if conf > 0.5:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            color = (0, 255, 0) if label.lower() == "green" else (0, 0, 255)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, f"Light: {label}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            light_color = label
            break

    # Zebra crossings
    for box in results_zebra[0].boxes:
        if float(box.conf[0]) > 0.3:
            zebra_count += 1
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(annotated, "Zebra crossing", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # Vehicles + people
    for box in results_coco[0].boxes:
        cls   = int(box.cls[0])
        label = model_coco.names[cls]
        conf  = float(box.conf[0])
        if conf > 0.6 and label in ["car", "bus", "truck", "motorcycle", "person"]:
            class_counts[label] = class_counts.get(label, 0) + 1
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 165, 255), 2)
            cv2.putText(annotated, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)

    # Build scene description
    desc = []
    if light_color:
        desc.append(f"traffic light is {light_color.lower()}")
    if zebra_count > 0:
        desc.append(f"{zebra_count} zebra crossing{'s' if zebra_count > 1 else ''} ahead")
    if class_counts:
        desc.append(f"{format_counts(class_counts)} ahead")

    # Generate narration if scene changed and cooldown passed
    if desc and (current_time - last_narration_time >= cooldown):
        guidance = generate_guidance(", ".join(desc))

        # Fallback if LLM returns empty
        if not guidance or len(guidance.split()) < 2:
            if light_color and light_color.lower() == "red":
                guidance = "Stop, red light ahead"
            elif light_color and light_color.lower() == "green":
                guidance = "Cross now, green light"
            elif zebra_count > 0:
                guidance = "Cross at the zebra crossing"
            else:
                guidance = "Walk carefully ahead"

        # Overlay narration text on frame
        cv2.putText(annotated, f'"{guidance}"', (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
                    cv2.LINE_AA)

        # Generate TTS
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            gTTS(text=guidance, lang="en").save(tmp.name)
            seg = AudioSegment.from_file(tmp.name, format="mp3")
            audio_segments.append(seg + AudioSegment.silent(duration=400))

        last_narration_time = current_time

    return annotated, last_narration_time

def merge_and_attach_audio(video_path, audio_segments, output_path):
    """Merge all TTS audio segments and attach to video."""
    from moviepy.editor import VideoFileClip, AudioFileClip

    if not audio_segments:
        return video_path

    combined = AudioSegment.empty()
    for seg in audio_segments:
        combined += seg

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        combined.export(tmp.name, format="wav")
        audio_tmp = tmp.name

    video_clip = VideoFileClip(video_path)
    audio_clip = AudioFileClip(audio_tmp)

    # Trim audio to video length if needed
    if audio_clip.duration > video_clip.duration:
        audio_clip = audio_clip.subclip(0, video_clip.duration)

    final = video_clip.set_audio(audio_clip)
    final.write_videofile(output_path, codec="libx264", audio_codec="aac", logger=None)
    video_clip.close()
    audio_clip.close()
    return output_path

# ─────────────────────────────────────────
#  Sidebar controls
# ─────────────────────────────────────────
st.sidebar.title("⚙️ Settings")
mode     = st.sidebar.radio("Input Mode", ["🖼️ Image", "🎥 Video", "📸 Webcam"])
cooldown = st.sidebar.slider("Narration cooldown (seconds)", 1, 10, 3)

st.sidebar.markdown("---")
st.sidebar.markdown("**About**")
st.sidebar.markdown("Detects traffic lights, zebra crossings, vehicles, and people. Generates voice guidance using LLaMA 3.1 via Groq.")

# ─────────────────────────────────────────
#  Main UI
# ─────────────────────────────────────────

# ── IMAGE MODE ──
if "Image" in mode:
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded:
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if frame is None:
            st.error("Could not read image.")
            st.stop()
        #file_bytes = np.frombuffer(uploaded.read(), np.uint8)
        #frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        try:
            with st.spinner("Running detection..."):
                audio_segments = []
                annotated, _ = process_frame(frame, 0, -999, cooldown, audio_segments)
        except Exception as e:
            st.error(f"Error occurred: {e}")
            st.stop()

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original")
            st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_column_width=True)
        with col2:
            st.subheader("Detected")
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_column_width=True)

        if audio_segments:
            combined = AudioSegment.empty()
            for seg in audio_segments:
                combined += seg
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                combined.export(tmp.name, format="mp3")
                st.subheader("🔊 Voice Guidance")
                st.audio(tmp.name)

# ── VIDEO MODE ──
elif "Video" in mode:
    uploaded = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    if uploaded:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(uploaded.read())
            input_path = tmp.name

        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 20
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        with tempfile.NamedTemporaryFile(suffix=".avi", delete=False) as tmp:
            temp_video_path = tmp.name

        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out = None
        audio_segments = []
        last_narration_time = -999
        frame_idx = 0

        progress = st.progress(0, text="Processing video...")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            current_time = frame_idx / fps
            annotated, last_narration_time = process_frame(
                frame, current_time, last_narration_time, cooldown, audio_segments
            )

            if out is None:
                h, w, _ = annotated.shape
                out = cv2.VideoWriter(temp_video_path, fourcc, fps, (w, h))
            out.write(annotated)

            frame_idx += 1
            if total_frames > 0:
                progress.progress(min(frame_idx / total_frames, 1.0),
                                   text=f"Processing frame {frame_idx}/{total_frames}...")

        cap.release()
        if out:
            out.release()
        progress.empty()

        with st.spinner("Attaching audio narration to video..."):
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                final_path = tmp.name
            merge_and_attach_audio(temp_video_path, audio_segments, final_path)

        st.success("✅ Done! Narrated video ready.")
        st.subheader("📺 Output Video with Voice Guidance")
        st.video(final_path)

        with open(final_path, "rb") as f:
            st.download_button(
                label="⬇️ Download Narrated Video",
                data=f,
                file_name="narrated_output.mp4",
                mime="video/mp4",
                use_container_width=True
            )

# ── WEBCAM MODE ──
elif "Webcam" in mode:
    st.info("📸 Capture a photo from your webcam — detection runs on the captured image.")
    captured = st.camera_input("Take a photo")

    if captured:
        file_bytes = np.frombuffer(captured.read(), np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        with st.spinner("Running detection..."):
            audio_segments = []
            annotated, _ = process_frame(frame, 0, -999, cooldown, audio_segments)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Captured")
            st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_column_width=True)
        with col2:
            st.subheader("Detected")
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_column_width=True)

        if audio_segments:
            combined = AudioSegment.empty()
            for seg in audio_segments:
                combined += seg
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                combined.export(tmp.name, format="mp3")
                st.subheader("🔊 Voice Guidance")
                st.audio(tmp.name)

# ─────────────────────────────────────────
#  Footer
# ─────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#888; font-size:14px;">
    Built by <b>Maharajmaran G</b> · 
    <a href="https://github.com/maran103" style="color:#0072ff;">GitHub</a> · 
    <a href="https://www.linkedin.com/in/maharajmaran-g-18684b257" style="color:#0072ff;">LinkedIn</a>
</div>
""", unsafe_allow_html=True)
