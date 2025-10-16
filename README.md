# 🦾 AI-Assisted Tool for Visually Impaired

This project is an **AI-powered assistive application** developed during the **Infosys Springboard Internship**.  
It helps **visually impaired users** navigate their environment safely using **real-time object detection, voice assistance, and navigation guidance**.

---

## 🚀 Features

- 🎯 **Real-Time Object Detection:** Detects nearby objects using **YOLOv8 (PyTorch)**.  
- 🗣️ **Voice Assistance:** Converts speech to text for user commands and provides voice feedback using **pyttsx3**.  
- 🧭 **Navigation Support:** Guides users to avoid obstacles and move safely.  
- ⚡ **Fast AI Processing:** Integrated **Groq API** for efficient inference and low latency.  

---

## 🧩 Tech Stack

| Category | Tools & Frameworks |
|-----------|--------------------|
| **Programming Language** | Python |
| **Deep Learning Framework** | PyTorch |
| **Model** | YOLOv8 |
| **APIs** | Groq API |
| **Libraries** | OpenCV, SpeechRecognition, pyttsx3 |
| **IDE** | VS Code / Jupyter Notebook |

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository
```bash
git clone https://github.com/<your-username>/AI-Assistive-Tool-For-Visually-Impaired.git
cd AI-Assistive-Tool-For-Visually-Impaired

python -m venv venv
source venv/bin/activate  # for Linux/Mac
venv\Scripts\activate     # for Windows

pip install -r requirements.txt

python app

# Output samples

