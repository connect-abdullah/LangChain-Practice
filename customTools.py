from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

class MultiplyInput(BaseModel):
    a : int = Field(required=True,description="The First Number")
    b : int = Field(required=True,description="The Second Number")
    
def multiply_func(a,b):
    return a * b

multiply_tool = StructuredTool.from_function(
    func=multiply_func,
    name="multiply",
    description="Multiple two numbers",
    args_schema=MultiplyInput
)

result = multiply_tool.invoke({
    "a" : 3,
    "b" : 3
})

print(result)