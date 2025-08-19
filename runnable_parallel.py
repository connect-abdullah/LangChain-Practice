from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableSequence
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatOpenAI(model_name="openai/gpt-3.5-turbo",
               base_url="https://openrouter.ai/api/v1",
               openai_api_key=os.getenv("OPENROUTER_API_KEY"),
               temperature=0.7,
               max_tokens=500,
               )

prompt1 = ChatPromptTemplate.from_template("Generate a concise tweet about the following topic: {topic}")

prompt2 = ChatPromptTemplate.from_template("Generate a concise linkedin post about the following topic: {topic}")

parser = StrOutputParser()

# Using RunnableParallel so we can get responses from the both llms and then displace the text
runnable_chain = RunnableParallel({
    # You can write chains in both ways, Using | or Using RunnableSequence. Both will work in the exact same way
    "tweet" : prompt1 | llm | parser,
    "linkedin" : RunnableSequence(prompt2, llm, parser)
})

result = runnable_chain.invoke({
    "topic" : "AI is revolving"
})

print(result)