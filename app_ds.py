from smolagents import CodeAgent, DuckDuckGoSearchTool, load_tool, tool
import datetime
import requests
import pytz
import yaml
from tools.final_answer import FinalAnswerTool
from Gradio_UI import GradioUI
from openai import OpenAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Custom tools remain the same as before
@tool
def my_custom_tool(arg1:str, arg2:int)-> str:
    """A tool that does nothing yet 
    Args:
        arg1: the first argument
        arg2: the second argument
    """
    return "What magic will you build ?"

@tool
def get_current_time_in_timezone(timezone: str) -> str:
    """A tool that fetches the current local time in a specified timezone.
    Args:
        timezone: A string representing a valid timezone (e.g., 'America/New_York').
    """
    try:
        tz = pytz.timezone(timezone)
        local_time = datetime.datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
        return f"The current local time in {timezone} is: {local_time}"
    except Exception as e:
        return f"Error fetching time for timezone '{timezone}': {str(e)}"

class DeepSeekModel:
    def __init__(self, temperature=0.7, max_tokens=2096):
        self.client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com/v1"
        )
        self.temperature = temperature
        self.max_tokens = max_tokens
        
    def __call__(self, prompt, stop_sequences=None, **kwargs):
        """Make the class callable with proper parameter handling
        
        Args:
            prompt: The input text prompt
            stop_sequences: Sequences where the model should stop generating
            **kwargs: Additional arguments that might be passed
        """
        try:
            response = self.client.chat.completions.create(
                model="deepseek-coder",
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stop=stop_sequences if stop_sequences else None
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: {str(e)}"

# Initialize tools
final_answer = FinalAnswerTool()
image_generation_tool = load_tool("agents-course/text-to-image", trust_remote_code=True)

# Load prompts
with open("prompts.yaml", 'r') as stream:
    prompt_templates = yaml.safe_load(stream)

# Initialize model
model = DeepSeekModel(
    temperature=0.5,
    max_tokens=2096
)

# Initialize agent
agent = CodeAgent(
    model=model,
    tools=[final_answer, get_current_time_in_timezone],
    max_steps=6,
    verbosity_level=1,
    grammar=None,
    planning_interval=None,
    name=None,
    description=None,
    prompt_templates=prompt_templates
)

# Launch Gradio UI
if __name__ == "__main__":
    GradioUI(agent).launch()