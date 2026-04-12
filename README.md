Hybrid-Fusion Rag UI 

# Project Setup and Run Instructions

This guide explains how to install dependencies, start Redis, and run the application.

---

## 1. Prerequisites

Make sure the following are installed on your system:

- Python 3.8 or higher
- pip (Python package manager)
- Redis server
- Git

---

## 2. Clone the Repository

```bash
git clone <YOUR_GITHUB_REPO_URL>
cd <YOUR_PROJECT_DIRECTORY>

3. Install and Start Redis
Install Redis

macOS (Homebrew):

brew install redis

Ubuntu / Debian:

sudo apt update
sudo apt install redis-server

Windows:
Use WSL (recommended) or run Redis via Docker.

Start Redis Server

Open a separate terminal and run:

redis-server

Keep this terminal open while running the app.

To verify Redis is working:

redis-cli ping

Expected output:

PONG
4. Install Python Dependencies

Open a second terminal and navigate to the project directory:

cd <YOUR_PROJECT_DIRECTORY>

(Optional but recommended) Create and activate a virtual environment:

python3 -m venv venv
source venv/bin/activate   # macOS / Linux
# venv\Scripts\activate    # Windows

Install dependencies:

pip install -r requirements.txt
5. Run the Application

With Redis running and dependencies installed, start the app:

python3 alpha_app2.py
6. Notes
Redis must be running before starting the application.
Default Redis port is 6379.
If you get connection errors, ensure redis-server is active.
If pip installation fails, upgrade pip:
pip install --upgrade pip
7. Quick Start Summary

Terminal 1:

redis-server

Terminal 2:

cd <YOUR_PROJECT_DIRECTORY>
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 alpha_app2.py


<img width="1022" height="965" alt="Screenshot 2026-01-15 at 21 48 03" src="https://github.com/user-attachments/assets/f7eb643e-c229-44d8-993a-95568c758606" />



<img width="1383" height="772" alt="Screenshot 2025-12-31 at 6 37 46" src="https://github.com/user-attachments/assets/274e0653-9ff0-4f8e-8d65-87db4363c4ed" />
STT-TTS real time dashboard
<img width="1299" height="967" alt="Screenshot 2026-01-13 at 16 45 24" src="https://github.com/user-attachments/assets/1b005d9e-4751-474a-98eb-56db46518fc3" />

Hybrid-Fusion Fine Tuning
<img width="1104" height="964" alt="Screenshot 2026-01-13 at 21 18 01" src="https://github.com/user-attachments/assets/84d11b96-7b56-49d3-86f8-cb7a5bde6f78" />



One-shot, Internal Multi-Hop via fine tuning for an accurate context window 



<img width="1128" height="964" alt="Screenshot 2026-01-13 at 18 52 20" src="https://github.com/user-attachments/assets/4030db22-cf03-482e-a540-570be01ccb30" />





