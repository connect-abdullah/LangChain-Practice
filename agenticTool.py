import os 
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.tools import tool
import requests
from ddgs import DDGS
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub

load_dotenv()

# Making a custom web search tool using ddgs
@tool
def web_search(query: str) -> str:
    """Search the web for information using DuckDuckGo"""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
            if results:
                return "\n".join([f"- {result['title']}: {result['body']}" for result in results])
            else:
                return "No search results found."
    except Exception as e:
        return f"Search error: {str(e)}"

# Making a get_weather tool by using @tool
@tool
def get_weather(city: str) -> str:
    """This function fetches the current weather data for a given city"""  # this description must be added for tools, because due to this llm knows which tool to execute and when.
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid=b49d0ddb07546697f27b3953d7b74930"
    response = requests.get(url)
    return response.json()

# Initialize the LLM
llm = ChatOpenAI(model_name="openai/gpt-3.5-turbo",
               base_url="https://openrouter.ai/api/v1",
               openai_api_key=os.getenv("OPENROUTER_API_KEY"),
               temperature=0.7,
               max_tokens=500,
               )

# Pull the ReAct prompt from LangChain Hub
prompt = hub.pull("hwchase17/react") # pulls the standard ReAct agent prompt

# Making Agent
agent = create_react_agent(
    llm=llm,
    prompt=prompt,
    tools=[web_search,get_weather]
)

# Making Agentic Executor and passing agent into it so agent thinks and then give executor the action to execute
agent_executor = AgentExecutor(
    agent=agent,
    tools=[web_search,get_weather],
    verbose=True
)

# Invoke
response = agent_executor.invoke({"input": "Find the capital of Pakistan, then find it's current weather condition"})
print(response['output'])