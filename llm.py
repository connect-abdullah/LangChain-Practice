from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()


llm=ChatOpenAI(model_name="openai/gpt-3.5-turbo",
               base_url="https://openrouter.ai/api/v1",
               openai_api_key=os.getenv("OPENROUTER_API_KEY"),
               temperature=0.7,
               max_tokens=500,
               )

response=llm.invoke("Hello, how are you?")
print(response.content)


# For Streaming Response
# for chunk in llm.stream("Hello, how are you?"):
#     print(chunk.content, end="", flush=True)



# for Multiple Prompt Requests

# response = llm.batch([
#     "Hello, how are you?",
#     "What is the capital of France?",
#     "What is the capital of Germany?",
#     "What is the capital of Italy?",
#     "What is the capital of Spain?",
#     "What is the capital of Portugal?",
#     "What is the capital of Greece?",
# ])

# for r in response:
#     print(r.content)