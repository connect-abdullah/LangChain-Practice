from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv
# Importing Output Parser
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

load_dotenv()


# Initialize the LLM
llm = ChatOpenAI(model_name="openai/gpt-3.5-turbo",
               base_url="https://openrouter.ai/api/v1",
               openai_api_key=os.getenv("OPENROUTER_API_KEY"),
               temperature=0.7,
               max_tokens=500,
               )

def extract_info():
    # Prompting to AI
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Extract the information from the following phrase. \
        Only return field in the provided schema. Ignore extra details. \
        Follow these instructions exactly: {follow_instructions}"),
        ("human", "{phrase}")
    ])
    
    # Telling the ai to use this and extract format according to it
    class Person(BaseModel):
        name : str = Field(description="Name of the person")
        age : int = Field(description="Age of the person")
        university : int = Field(description="University Name")
        
    
    # Setting the json parser with pydantic object of person to use the saqme schema while parsing
    parser = JsonOutputParser(pydantic_object=Person)

    # Chain to execute
    chain = prompt | llm | parser 
    
    return chain.invoke({
        "phrase" : "Aima is 20 years old and she studies in COMSATS University Islamabad",
        "follow_instructions" : parser.get_format_instructions()
    })

print(extract_info())