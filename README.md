# Complex networks project

## Description
This project aims at studying the network dynamics of a subreddit online. The subreddit chosen is the OnePiece subreddit and in order to study user engagement, user sentiments and network dynamics some python libraries have been used such as PRAW, Pandas, VADER Sentiment Analysis and networkx. 


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
3. Once you have created a Reddit account you can go to (https://www.reddit.com/prefs/apps) and click on:

- "Create an app" or "Create another app";

- select script as the app type since we are using [PRAW](https://praw.readthedocs.io/en/stable/index.html) and not a web app;

- fill in:

  - app name (for example OnePiece scraper);
  
  - set Redirect URI to "http://localhost:8080";
  
  - leave other fields empty.

After submitting this spaces you will get access to:

- Client ID (a short alphanumeric string);

- Client secret (a long alphanumeric string).


4. A config.py file is needed to store your Reddit credentials to start the data scraping process using the PRAW library. Create a python file naming it "config.py" and save it in the project folder then proceed to add your credentials just retrieved in the following way:
## config.py
REDDIT_CLIENT_ID = "your-client-id"

REDDIT_CLIENT_SECRET = "your-client-secret"

REDDIT_USER_AGENT = "your-user-agent"
