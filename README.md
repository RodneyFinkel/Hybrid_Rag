Hybrid-Fusion Rag UI 

# Project Setup and Run Instructions

## 1. Clone the repository

```bash
git clone <GITHUB_REPO_URL>
cd <YOUR_PROJECT_DIRECTORY>
2. Create and activate environment
python3 -m venv venv

macOS / Linux:

source venv/bin/activate

Windows (PowerShell):

venv\Scripts\activate
3. Install Redis
macOS (Homebrew)
brew install redis
Ubuntu / Debian
sudo apt update
sudo apt install redis-server
Windows

Option A: WSL (recommended)

wsl --install

Inside WSL:

sudo apt update
sudo apt install redis-server
redis-server

Option B: Docker

docker run -p 6379:6379 redis
4. Install Python dependencies
pip install -r requirements.txt
5. Start Redis
redis-server

(Keep this terminal running)

6. Run the app
python3 alpha_app2.py
Notes
Redis must be running before starting the app.
Default Redis port: 6379.











