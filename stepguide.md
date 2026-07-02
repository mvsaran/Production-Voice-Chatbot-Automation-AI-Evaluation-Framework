# 🚀 Step-by-Step Layman Guide: Building & Testing Voice Chatbot Automation From Scratch

> 💡 **In Plain English:** Have you ever wondered how to build an AI robot that listens to your voice, understands your questions, and speaks back to you — and then how to **automatically test** if that robot is actually smart, fast, and polite without having a human sit there testing it all day? 
> 
> This guide is written in 100% plain English (layman's terms) to show you exactly how to build this project from scratch, what software you need, how all the AI pieces talk to each other, and the exact commands to run and verify your voice chatbot!

---

## 📑 Table of Contents
1. [👋 Welcome: What Does This Project Do?](#1--welcome-what-does-this-project-do)
2. [💻 Part 1: What Software Do You Need to Install? (Zero to Ready)](#2--part-1-what-software-do-you-need-to-install-zero-to-ready)
3. [🏗️ Part 2: Building & Setting Up the Project From Scratch](#3--part-2-building--setting-up-the-project-from-scratch)
4. [🔌 Part 3: How the AI Integrations Work Together (The Team)](#4--part-3-how-the-ai-integrations-work-together-the-team)
5. [🧪 Part 4: Commands to Execute Tests & Verify Voice Responses](#5--part-4-commands-to-execute-tests--verify-voice-responses)
6. [📊 Part 5: Viewing Your Visual Report Cards](#6--part-5-viewing-your-visual-report-cards)
7. [❓ Part 6: Layman Troubleshooting (What if something goes wrong?)](#7--part-6-layman-troubleshooting-what-if-something-goes-wrong)

---

## 1. 👋 Welcome: What Does This Project Do?

Imagine you run a bank or a hospital, and you want to put an **AI Voice Assistant** on your phone line. When a customer calls and asks: *"What is my checking account balance?"*, the AI needs to:
1. **Hear** what the customer said correctly (even if there is background noise or an accent).
2. **Think** of the correct answer without making up fake numbers.
3. **Speak** the answer back in a clear, friendly voice within a couple of seconds.

Before you let real customers talk to this AI, you need to test it. But manually calling a phone line 1,000 times to check every possible banking or healthcare question would take months!

**This project solves that problem.** It builds the voice chatbot AND provides an automated "robot QA inspector" that tests dozens of conversations in seconds, grading the chatbot on whether it heard correctly, answered accurately, and spoke fast enough!

---

## 2. 💻 Part 1: What Software Do You Need to Install? (Zero to Ready)

Before you touch any code, your computer needs three basic tools installed:

### 1️⃣ Python (The Engine)
* **What is it?** Python is the core programming language and engine that runs our chatbot and test scripts.
* **How to get it:** Download **Python 3.12 or 3.13+** from [python.org](https://www.python.org/downloads/).
* **How to check if you have it:** Open your Terminal (Mac/Linux) or Command Prompt / PowerShell (Windows) and type:
  ```powershell
  python --version
  ```
  *(If it says something like `Python 3.13.2`, you are ready to go!)*

### 2️⃣ An OpenAI API Key (The AI Brain & Voice Provider)
* **What is it?** We use OpenAI's supercomputers to do the actual hearing, thinking, and speaking. An API Key is like a secret digital passport and credit card that lets our code talk to OpenAI's servers.
* **How to get it:** Go to [platform.openai.com](https://platform.openai.com/), sign up, add a few dollars of credit, and create a new **Secret Key** (it starts with `sk-...`). Keep this key secret!

### 3️⃣ Audio Hardware & Drivers (For Live Microphone Testing)
* **What is it?** If you want to speak your own real voice into the computer and hear the bot talk back through your speakers, your computer needs audio drivers.
* **On Windows:** Audio drivers are usually built-in!
* **On Mac / Linux:** You may need to install a free audio tool called PortAudio by running `brew install portaudio` (Mac) or `sudo apt-get install portaudio19-dev` (Linux).

---

## 3. 🏗️ Part 2: Building & Setting Up the Project From Scratch

Now, let's build the project step-by-step on your machine. Follow these exact instructions in your command prompt:

### Step 1: Open your folder and create a "Virtual Environment"
A virtual environment is like a clean, private toolbox just for this project, so its software doesn't mess up anything else on your computer.

```powershell
# Navigate into your project folder
cd VOICECHATBOTAUTOMATION

# Create the clean toolbox (virtual environment named 'venv')
python -m venv venv

# Activate the toolbox (ON WINDOWS POWERSHELL):
.\venv\Scripts\activate

# OR Activate the toolbox (ON MAC / LINUX):
source venv/bin/activate
```
*(You will know it worked when you see `(venv)` appear at the very left of your command prompt!)*

### Step 2: Install All Project Libraries
Now, let's install the project building blocks. We have a file called `requirements.txt` that lists every tool we need. Run this single command:

```powershell
pip install -r requirements.txt
```

**What is being installed behind the scenes? (In plain English):**
* **`openai`**: The tool that connects us to Whisper (hearing), GPT-4o (thinking), and TTS (speaking).
* **`fastapi` & `uvicorn`**: The web server that acts like a receptionist, taking incoming requests and passing them to the AI.
* **`deepeval`**: The strict "AI Teacher" that grades our chatbot's answers for accuracy and honesty.
* **`pytest` & `pytest-html`**: The automated exam runner that executes our tests and creates visual report cards.
* **`sounddevice` & `scipy`**: The sound recorders that let Python use your microphone and speakers.

### Step 3: Configure Your Secret Key (`.env` file)
Our code looks for a settings sheet called `.env` to find your OpenAI key.
1. Make a copy of the template file `.env.example` and name it `.env`:
   ```powershell
   copy .env.example .env
   ```
2. Open the `.env` file in Notepad or VS Code.
3. Find the line that says `OPENAI_API_KEY=""` and paste your actual key inside the quotation marks:
   ```env
   OPENAI_API_KEY="sk-proj-your-actual-secret-key-goes-here"
   ```
4. Save and close the file. You are now 100% built and ready to run!

---

## 4. 🔌 Part 3: How the AI Integrations Work Together (The Team)

When a voice conversation happens, five different "specialists" work together like a relay race team:

```
+-------------------------------------------------------------------------------------------------------+
|                                  THE VOICE AUTOMATION RELAY TEAM                                      |
+-------------------------------------------------------------------------------------------------------+
|                                                                                                       |
|  [1. You / Test Script] ---> [2. Whisper API] ---> [3. GPT-4o LLM] ---> [4. OpenAI TTS]               |
|      Speaks Question             The Ear              The Brain            The Voice                  |
|                                     │                     │                     │                     |
|                                     └──────────┬──────────┴──────────┬──────────┘                     |
|                                                ▼                     ▼                                |
|                                      [5. DeepEval QA Judge & Pytest Reporter]                         |
|                                        Grades Hearing, Thinking, and Timing                           |
|                                                                                                       |
+-------------------------------------------------------------------------------------------------------+
```

| Team Member | Official Name | What is their job in layman's terms? |
| :--- | :--- | :--- |
| 👂 **The Ear** | **OpenAI Whisper (`whisper-1`)** | Listens to the raw spoken audio recording and types out the words in text. It filters out background noise and understands accents. |
| 🧠 **The Brain** | **OpenAI GPT-4o** | Reads the typed words, understands what the customer wants, remembers what was said earlier in the call, and writes out an intelligent, polite answer. |
| 🔊 **The Voice** | **OpenAI Text-to-Speech (`tts-1`)** | Takes the written answer from the Brain and reads it out loud in a smooth, natural human voice (saved as a playable `.wav` sound file). |
| 🕵️ **The Inspector** | **DeepEval (LLM-as-a-Judge)** | A separate, impartial AI that reads the customer question and the bot's answer, checking if the bot told the truth or if it "hallucinated" (made up fake facts). |
| 📋 **The Examiner** | **Pytest & Pytest-HTML** | The test automation manager that runs dozens of fake phone calls automatically, times with a stopwatch how fast they took, and builds a clickable visual report card. |

---

## 5. 🧪 Part 4: Commands to Execute Tests & Verify Voice Responses

We provide four easy ways to test and verify your voice chatbot. Choose the one that fits what you want to do:

### 🥉 Method A: The 1-Minute Connection Check (`test_voice_pipeline.py`)
**When to use:** You just installed the project and want a quick check to make sure your OpenAI API key works and the Ear, Brain, and Voice can talk to each other.

```powershell
python test_voice_pipeline.py
```
* **What happens:** The script pretends a customer asked: *"Hello! Can you tell me what services this voice automation chatbot offers?"*. It turns that into speech, sends it to Whisper, gets an answer from GPT-4o, and creates a spoken voice file.
* **How to verify:** Look at your command prompt! You will see a timing table showing exactly how many seconds the Ear, Brain, and Voice took. Then, open the folder `audio/output/` and double-click `assistant_response.wav` to hear the AI speak!

---

### 🥈 Method B: Have a Real Voice Conversation With Your Mic (`talk_to_bot.py`) 🎙️
**When to use:** You want to talk to the AI using your own physical microphone and hear it reply through your speakers in real-time!

```powershell
python talk_to_bot.py
```
* **Step-by-Step Instructions:**
  1. When the command prompt says `Press [ENTER] to start recording...`, hit your **Enter** key.
  2. Speak clearly into your microphone! (For example: *"Can you explain what generative AI is and how it helps in banking?"*).
  3. Press **Enter** again when you are done speaking to stop the recording.
  4. Watch the screen! You will see Whisper type out what you said, and GPT-4o type out its reply.
  5. **Listen!** Your computer speakers will automatically play the chatbot speaking its answer back to you!
  6. You can keep replying for a back-and-forth conversation, or type `q` and hit Enter to quit.

---

### 🥇 Method C: Run the Automated Exam for 30+ Enterprise Scenarios (`pytest -v`) 🧪
**When to use:** You want to test how the bot performs across **33 different real-world business scenarios** (Banking balance checks, Healthcare doctor appointments, Insurance claims, E-commerce refunds, Flight bookings, etc.) without you having to speak a single word!

```powershell
# Run ALL automated tests (takes roughly 10-15 seconds)
pytest -v

# OR run ONLY the speech recognition accuracy and latency tests
pytest -m voice -v

# OR run ONLY the security tests (checking if the bot blocks hackers and bad words)
pytest -m safety -v
```
* **What happens:** The automated examiner opens our pre-written test scripts in `test_data/conversations/`. It fires them through the pipeline, checks Word Error Rates (WER), runs DeepEval to ensure zero hallucinations, and prints a big green **`PASSED`** or red **`FAILED`** score for every single test!

---

### 🏆 Method D: Generate the Beautiful Visual Dashboard (`generate_dashboard.py`) 📊
**When to use:** You want to generate a stunning, professional graphical dashboard with charts and readability scores that you can show to your manager, team, or clients.

```powershell
python generate_dashboard.py
```
* **What happens:** The script takes all 33 test scenarios (and your live microphone turns!) and compiles two beautiful HTML visual report cards:
  * 📁 `reports/html/report.html` (Primary enterprise QA report).
  * 📁 `reports/html/voice_qa_dashboard.html` (Dedicated graphical UI dashboard).

---

## 6. 📊 Part 5: Viewing Your Visual Report Cards

After running `python generate_dashboard.py` or `pytest -v`, your computer has created interactive webpages right on your hard drive! You do NOT need internet access or a web server to view them.

### How to open and view your dashboards:
1. Open your File Explorer (Windows) or Finder (Mac).
2. Go inside your project folder $\rightarrow$ double-click the **`reports`** folder $\rightarrow$ double-click the **`html`** folder.
3. Double-click **`report.html`** or **`voice_qa_dashboard.html`**. They will open right inside Google Chrome, Microsoft Edge, or Mozilla Firefox!

### What will you see on the screen?
* 🎨 **Sleek Dark Mode Design**: A modern, glass-like dark theme with vibrant neon colors that is easy on the eyes.
* 📈 **Radar Graph (The SLA Compliance Chart)**: A web-like radar chart comparing our chatbot's current build against target enterprise SLAs for Correctness, Relevancy, Faithfulness, Non-Hallucination, and Speech Accuracy.
* ⏱️ **Bar Graph (The Latency Breakdown)**: A color-coded stacked bar chart showing exactly how many seconds Whisper STT (blue), GPT-4o LLM (cyan), and OpenAI TTS (purple) took for each turn.
* 🎛️ **Interactive Filter Buttons**: Click buttons like `🏦 Banking`, `🏥 Healthcare`, `✈️ Travel`, or `🛒 E-Commerce` to instantly filter and view specific business conversations.
* 📑 **Conversation Cards**: Every test call is shown in a neat box displaying:
  * *"What the User Spoke"* vs *"What the Chatbot Answered"*.
  * **Word Error Rate (WER) Badge**: Shows how accurately the Ear heard the words.
  * **DeepEval Score Pills**: Shows exact grades for Correctness and Hallucination.

---

## 7. ❓ Part 6: Layman Troubleshooting (What if something goes wrong?)

Don't panic! Here are simple solutions to the most common questions:

### 🔴 Problem: I ran `python talk_to_bot.py` and it says `PortAudio library not found` or `Error querying device`.
* **Layman Cause:** Python is having trouble finding your microphone drivers.
* **Layman Solution:** Make sure your microphone is plugged in and not muted in Windows/Mac system settings. On Windows, try reinstalling sounddevice with `pip install sounddevice --upgrade`. On Mac, run `brew install portaudio`.

### 🔴 Problem: It fails with `AuthenticationError: Incorrect API key provided` or `You exceeded your current quota`.
* **Layman Cause:** Either the secret key in your `.env` file was copied incorrectly, or your OpenAI account has run out of billing credit ($0 balance).
* **Layman Solution:** Open `.env`, make sure there are no extra spaces around your key (`OPENAI_API_KEY="sk-..."`), and log into [platform.openai.com/account/billing](https://platform.openai.com/account/billing) to make sure you have at least $5 of credit.

### 🔴 Problem: What does `WER: 0.05` mean on my report card? Is 0.05 good or bad?
* **Layman Explanation:** WER stands for **Word Error Rate**. It is simply the percentage of words the AI misheard.
* A WER of `0.05` means the AI only made mistakes on **5% of the words** (95% accuracy!). That is an excellent grade! Our system requires WER to be `0.10` or lower (90%+ accuracy) to pass.

### 🔴 Problem: Can I run automated tests if I am offline or don't want to spend OpenAI credit?
* **Layman Explanation:** **Yes!** Our project has a built-in "stunt double" system called `ServiceResolver`. If you disconnect from the internet or set mock keys, the system automatically uses **Mock Adapters** (fake Ear, fake Brain, fake Voice) that simulate realistic timing and return test answers for zero cost!

---

## ✍️ Author
**Saran Kumar**
