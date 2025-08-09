from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize the LLM
llm = ChatOpenAI(model_name="openai/gpt-3.5-turbo",
               base_url="https://openrouter.ai/api/v1",
               openai_api_key=os.getenv("OPENROUTER_API_KEY"),
               temperature=0.7,
               max_tokens=500,
               )

# Will Create A Prompt Template for a single message
# prompt = ChatPromptTemplate.from_template("Tell me a joke about {subject}")

# If you want to create prompt from multiple messages, and tune the ai according to it, you can use this below
prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "You are an expert linguist and thesaurus. "
     "Your task is to generate a list of 10 high-quality, contextually appropriate synonyms for the given word. "
     "Consider nuances in meaning, register, and usage. "
     "Return the synonyms as a comma-separated list, and avoid repeating the original word. "
     "If the word has multiple senses, choose the most common one unless context is provided."
    ),
    ("human", "{input}")
])

# We will then create a chain
chain = prompt | llm 
    # output  -> input 
# In a chain, the output of the first component is passed as input to the next component using the | operator, allowing data to flow sequentially through each step.

response = chain.invoke({"input" : "happy"})

print(response.content)