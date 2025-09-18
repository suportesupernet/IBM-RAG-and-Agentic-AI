from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

template = """Your job is to come up with a classic dish from the area that the users suggests.
{location}
  YOUR RESPONSE:
"""

prompt = PromptTemplate.from_template(template)

location_chain_lcel = prompt | llama_llm | StrOutputParser()

result = location_chain_lcel.invoke({"location": "China"})

print(result)
