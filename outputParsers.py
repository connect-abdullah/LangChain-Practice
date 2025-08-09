from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv
# Importing Output Parser
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


# Initialize the LLM
llm = ChatOpenAI(model_name="openai/gpt-3.5-turbo",
               base_url="https://openrouter.ai/api/v1",
               openai_api_key=os.getenv("OPENROUTER_API_KEY"),
               temperature=0.7,
               max_tokens=500,
               )

def call_prompt():
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Tell me a joke about following word"),
        ("human", "{input}")
    ])
    # setting up parser
    parser = StrOutputParser()
    
    # then passing the response to parser to get parsed and show the output formatted
    chain = prompt | llm | parser


    return chain.invoke({"input" : "happy"})
    

print(call_prompt())