import os 
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents import AgentExecutor, create_tool_calling_agent, tool
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()


# --- Tools ---
@tool
def multiply_numbers(a: int, b: int) -> int:
    """Multiply two numbers and return the product."""
    return a * b

wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

tools = [multiply_numbers, wikipedia]

# Initialize the LLM
llm = ChatOpenAI(model_name="openai/gpt-3.5-turbo",
               base_url="https://openrouter.ai/api/v1",
               openai_api_key=os.getenv("OPENROUTER_API_KEY"),
               temperature=0.7,
               max_tokens=500,
               )

# --- Prompt ---
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a research assistant. Use these {tools} when needed."),
    ("user", "{input}"),
    ("placeholder", "{agent_scratchpad}"),  # stores agent reasoning/tool use
])

# --- Agent ---
agent = create_tool_calling_agent(llm, tools, prompt)

# --- Executor ---
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# --- Example queries ---
response = agent_executor.invoke({
                    "input": "Who is Albert Einstein, and multiply the number of his age by 2.", 
                    "tools" : tools}
                    )
print(response['output'])
