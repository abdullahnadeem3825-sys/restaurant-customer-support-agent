# Restaurant Customer Support Agent

An AI-powered customer support system for restaurants built using CrewAI. The system leverages a local DeepSeek model (DeepSeek-V3.1-Terminus via Hugging Face) to handle customer queries, reservations, and support tasks through intelligent multi-agent collaboration.

## Features

* Automated customer query handling
* Reservation management
* Menu-related assistance
* Multi-agent collaboration using CrewAI
* Local LLM support using DeepSeek (via Hugging Face)

## Tech Stack

* Python
* CrewAI
* Hugging Face

## How It Works

The system uses multiple AI agents, each responsible for specific tasks such as handling queries, managing reservations, and providing recommendations. These agents collaborate to deliver accurate and efficient customer support. A local DeepSeek model ensures privacy and offline capability.

## Project Structure

```id="j38p3p"
customer_support.py                       # Core logic and agents
requirements.txt              # Dependencies
```

## Setup and Usage

1. Clone the repository:

```bash id="7njw5n"
git clone https://github.com/your-username/restaurant-customer-support-agent.git
cd restaurant-customer-support-agent
```

2. Install dependencies:

```bash id="fapn3y"
pip install -r requirements.txt
```

3. Pull the DeepSeek model using Hugging face:


4. Run the application:

```bash id="l6qvzb"
python customer_support.py
```


## Future Improvements

* Integration with real-time restaurant databases
* Voice-based customer support
* Deployment on cloud platforms
* Enhanced UI/UX

## Author

Abdullah Nadeem
