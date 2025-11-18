# 🧏‍♂️ Sign Language Recognition System (ISL | CNN + LSTM)

A real-time **Indian Sign Language (ISL)** recognition system that converts hand gestures into **text and speech**.  
This project uses a hybrid deep-learning approach combining:

- 🧠 **CNN** — for static alphabet recognition  
- 🔁 **LSTM** — for dynamic word/phrase recognition  
- 🤖 **MediaPipe** — for extracting hand landmarks  
- 🖥️ **Tkinter GUI** — for a simple desktop interface  

The goal of this system is to make communication easier between hearing-impaired individuals and the hearing community through an AI-powered translator.

---

## 🚀 Features

### 🔤 **Alphabet Recognition (CNN)**
- Detects static A–Z signs  
- Builds words from characters  
- Suggests correct words using a dictionary  

### 🗣️ **Phrase Recognition (LSTM)**
Recognizes dynamic ISL gestures like:
- Hello  
- Thank You  
- Yes / No  
- I Am Fine  
- Good Morning  
- Nice To Meet You  
- Good  

### 🔄 **Hybrid Mode**
Runs CNN + LSTM together:
- If a dynamic gesture is detected → show word  
- If a static gesture is detected → show letter  

### 🔊 **Speech Output**
The final text or word can be spoken aloud using **pyttsx3**.

### 🖼️ **User-Friendly GUI**
- Live webcam feed  
- Hand landmark drawing  
- Current letter & current word  
- Sentence builder  
- Suggestions panel  
- Clear & Speak buttons  
- Mode switch buttons: **Letter Mode / Phrase Mode / Both**

---

## 🧠 Tech Stack

- **Python 3**
- **TensorFlow & Keras** (CNN + LSTM models)
- **OpenCV**
- **MediaPipe**
- **cvzone**
- **pyttsx3** (text-to-speech)
- **Tkinter** (desktop GUI)

---

## 📂 Project Structure

