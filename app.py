from smolagents import CodeAgent,DuckDuckGoSearchTool, LiteLLMModel ,load_tool,tool, OpenAIServerModel
import datetime
import requests
import pytz
import yaml
from tools.final_answer import FinalAnswerTool
import os
from dotenv import load_dotenv
from tools.web_search import DuckDuckGoSearchTool
from Gradio_UI import GradioUI
from tools.visit_webpage import VisitWebpageTool
import re

load_dotenv()
# Below is an example of a tool that does nothing. Amaze us with your creativity !
@tool
def my_custom_tool(arg1:str, arg2:int)-> str: #it's import to specify the return type
    #Keep this format for the description / args / args description but feel free to modify the tool
    """A tool that does nothing yet 
    Args:
        arg1: the first argument
        arg2: the second argument
    """
    return "What magic will you build ?"

@tool
def get_weather(city: str, country_code: str = None) -> str:
    """A tool that fetches the current weather for a specified city using OpenWeatherMap API
    Args:
        city: The name of the city to get weather for
        country_code: Optional two-letter country code (e.g., 'US', 'UK')
    """
    # You'll need to set this as an environment variable or configure it securely
    API_KEY = os.getenv("GET_WEATHER_API")
    base_url = f"http://api.openweathermap.org/data/2.5/weather"

    try:
        # Construct the location query
        location = f"{city}"
        if country_code:
            location = f"{city},{country_code}"

        # Parameters for the API request
        params = {
            'q': location,
            'appid': API_KEY,
            'units': 'metric'  # Use metric units (Celsius)
        }

        # Make the API request
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        
        data = response.json()

        # Format the weather information
        weather_info = (
            f"Weather in {data['name']}:\n"
            f"Temperature: {data['main']['temp']}°C\n"
            f"Feels like: {data['main']['feels_like']}°C\n"
            f"Condition: {data['weather'][0]['main']}\n"
            f"Description: {data['weather'][0]['description']}\n"
            f"Humidity: {data['main']['humidity']}%\n"
            f"Wind Speed: {data['wind']['speed']} m/s"
        )

        return weather_info

    except requests.exceptions.RequestException as e:
        return f"Failed to fetch weather data: {str(e)}"


@tool
def get_current_time_in_timezone(timezone: str) -> str:
    """A tool that fetches the current local time in a specified timezone.
    Args:
        timezone: A string representing a valid timezone (e.g., 'America/New_York').
    """
    try:
        # Create timezone object
        tz = pytz.timezone(timezone)
        # Get current time in that timezone
        local_time = datetime.datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
        return f"The current local time in {timezone} is: {local_time}"
    except Exception as e:
        return f"Error fetching time for timezone '{timezone}': {str(e)}"


final_answer = FinalAnswerTool()
web_search = DuckDuckGoSearchTool()
visit_webpage = VisitWebpageTool()

# If the agent does not answer, the model is overloaded, please use another model or the following Hugging Face Endpoint that also contains qwen2.5 coder:
# model_id='https://pflgm2locj2t89co.us-east-1.aws.endpoints.huggingface.cloud' 

model = OpenAIServerModel(
max_tokens=2096,
temperature=0.5,
#model_id='Qwen/Qwen2.5-Coder-32B-Instruct',# it is possible that this model may be overloaded,
#model_id='https://pflgm2locj2t89co.us-east-1.aws.endpoints.huggingface.cloud',
model_id = "deepseek-coder",
api_base = "https://api.deepseek.com/v1",
api_key = os.getenv("DEEPSEEK_API_KEY"),
custom_role_conversions={
    "user" : "user",
    "assistant" : "assistant"}
)


# Import tool from Hub
image_generation_tool = load_tool("agents-course/text-to-image", trust_remote_code=True)

with open("prompts.yaml", 'r') as stream:
    prompt_templates = yaml.safe_load(stream)
    
agent = CodeAgent(
    model=model,
    tools=[final_answer, get_current_time_in_timezone, get_weather, web_search, visit_webpage], ## add your tools here (don't remove final answer)
    max_steps=6,
    verbosity_level=1,
    grammar=None,
    planning_interval=None,
    name=None,
    description=None,
    prompt_templates=prompt_templates,
    
)


GradioUI(agent).launch()