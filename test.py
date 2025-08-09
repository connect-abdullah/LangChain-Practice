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


extract = ChatPromptTemplate.from_template("extract a list of countries from text: {input}")
sort = ChatPromptTemplate.from_template("Sort the following list of countries alphabetically : {input}")
uppercase = ChatPromptTemplate.from_template("Give the output in UPPERCASE : {input}")

# Prompt → LLM → Prompt → LLM → ...
extract_chain = extract | llm
sort_chain = sort | llm
uppercase_chain = uppercase | llm

chain = extract_chain  | sort_chain  | uppercase_chain 

response = chain.invoke({"input": "While planning our international marketing campaign, we decided to focus on emerging markets in Southeast Asia, including Vietnam, Thailand, and Indonesia. We're also considering expanding into South America, particularly Brazil and Argentina. Although we've ruled out China for now, we may revisit that in the next fiscal year. Additionally, Nigeria and Kenya present interesting opportunities on the African continent."})

print(response.content)

