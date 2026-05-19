Set-Content README.md "# AI Assistive Mobility Tool for Visually Impaired

Real-time road navigation assistant that helps visually impaired pedestrians cross roads safely using computer vision and LLM-powered voice guidance.

## Live Demo
[Try it on Hugging Face Spaces](https://huggingface.co/spaces/Maranx/visual-assistive-tool)

## Model Performance
| Metric | Score |
|--------|-------|
| mAP50 | 98.9% |
| Precision | 95.6% |
| Recall | 98.0% |

## How It Works
- 3 YOLOv8 models detect traffic lights, zebra crossings, vehicles and people
- LLaMA 3.1 via Groq API converts detections into natural voice instructions
- gTTS converts instructions to speech overlaid on video output

## Tech Stack
Python, YOLOv8 (PyTorch), Groq API, LLaMA 3.1, OpenCV, gTTS, MoviePy, Streamlit, Hugging Face

## Features
- Image / Video / Webcam input modes
- Real-time bounding box annotations
- Voice narration synced to video output
- Offline-first model caching

## Run Locally
pip install -r requirements.txt
streamlit run app.py
"
