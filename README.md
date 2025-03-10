# complex_networks_project
Scripts for my Complex Networks exam

## Setup
1. **Clone the repository**:
```bash
   git clone https://github.com/StefanoSpadano/complex_networks_project.git
   cd complex_networks_project
```
2. **Install dependencies used in this project**
```bash
pip install -r requirements.txt
```
The first script is used to collect data online from Reddit using the Reddit API wrapper library; a config.py file is needed to store your Reddit credentials to start the data scraping process

3. **Create a config.py file in the project folder and add your credentials to this file**
# config.py
REDDIT_CLIENT_ID = "your-client-id"

REDDIT_CLIENT_SECRET = "your-client-secret"

REDDIT_USER_AGENT = "your-user-agent"
