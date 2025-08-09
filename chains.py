from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
parser = StrOutputParser()

# Initialize the LLM
llm = ChatOpenAI(model_name="openai/gpt-3.5-turbo",
                 base_url="https://openrouter.ai/api/v1",
                 openai_api_key=os.getenv("OPENROUTER_API_KEY"),
                 temperature=0.7,
                 max_tokens=500,
                 )

# Summarizing the input into 2 sentences
summarize = ChatPromptTemplate.from_messages([
    ("system", "Summarize the following text into maximum 3 words"),
    ("human", "{input}")
])

# Translating the input into french
translate = ChatPromptTemplate.from_template(
    "Translate the following English text into French: {input}")

# Rephrase into more formal way of french
rephrase = ChatPromptTemplate.from_template(
    "Rephrase the following French text to sound more formal and professional wording: {input}")

# Return the output in double quotes
output = ChatPromptTemplate.from_template(
    "Return ONLY the following text inside double quotes: {input}")

# calling each by their own chain and getting their responses from llm
summarize_chain = summarize | llm | parser
translate_chain = translate | llm | parser
rephrase_chain = rephrase | llm | parser
output_chain = output | llm | parser

final_chain = summarize_chain | translate_chain | rephrase_chain | output_chain

text = ("Artificial intelligence is transforming industries across the globe, automating repetitive tasks, improving decision-making, and enabling new business models. However, it also raises ethical concerns about job displacement, bias, and privacy that must be addressed responsibly.")

response = final_chain.invoke({"input": text})

print(response)
