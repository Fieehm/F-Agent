# !pip install smolagents[litellm]
from smolagents import CodeAgent, OpenAIServerModel
from dotenv import load_dotenv
import os

load_dotenv()
model = OpenAIServerModel(
    model_id="deepseek-chat",
    api_base="https://api.deepseek.com/v1/", # Leave this blank to query OpenAI servers.
    api_key=os.getenv("DEEPSEEK_API_KEY"), # Switch to the API key for the server you're targeting.
)
agent = CodeAgent(tools=[], 
                  model=model, 
                  add_base_tools=True)

agent.run(
    "Could you give me the 118th number in the Fibonacci sequence?",
)

print(agent.logs)