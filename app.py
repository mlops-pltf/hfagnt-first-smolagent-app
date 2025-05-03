from smolagents import CodeAgent,DuckDuckGoSearchTool, HfApiModel,load_tool,tool
import datetime
import requests
import pytz
import yaml
from tools.final_answer import FinalAnswerTool
from tools.visit_webpage import VisitWebpageTool
from tools.web_search import DuckDuckGoSearchTool

from Gradio_UI import GradioUI

# Below is an example of a tool that does nothing. Amaze us with your creativity !
@tool
def eligibility_duration_finder(current_year:int, birth_year:int)-> str: #it's import to specify the return type
    """A tool that accepts current year as input and then calculates the eligible number of years for the person
    Args:
        current_year: The current Year value
        birth_year: The birth year of the person
    Returns: Eliglible number of years for the person
    """
    return (current_year - birth_year)/2

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



# If the agent does not answer, the model is overloaded, please use another model or the following Hugging Face Endpoint that also contains qwen2.5 coder:
# model_id='https://pflgm2locj2t89co.us-east-1.aws.endpoints.huggingface.cloud' 

model = HfApiModel(
    max_tokens=2096,
    temperature=0.5,
    model_id='Qwen/Qwen2.5-Coder-32B-Instruct',# it is possible that this model may be overloaded
    # model_id='meta-llama/Llama-2-7b-chat-hf',
    custom_role_conversions=None,
    provider="together"
)


with open("prompts.yaml", 'r') as stream:
    prompt_templates = yaml.safe_load(stream)

image_generator = load_tool("agents-course/text-to-image", trust_remote_code=True) # Import tool from Hub
agent = CodeAgent(
    model=model,
    tools=[
        FinalAnswerTool()
        , VisitWebpageTool()
        , DuckDuckGoSearchTool()
        , image_generator
        , eligibility_duration_finder
    ], ## add your tools here (don't remove final answer)
    max_steps=6,
    verbosity_level=1,
    grammar=None,
    planning_interval=None,
    name=None,
    description=None,
    prompt_templates=prompt_templates
)


GradioUI(agent).launch()