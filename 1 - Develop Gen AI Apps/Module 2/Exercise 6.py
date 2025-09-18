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
-----------------------------
# Sequential Chain
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

location_template = """Your job is to come up with a classic dish from the area that the users suggests.
{location}
YOUR RESPONSE:
"""

dish_template = """Given a meal {meal}, give a short and simple recipe on how to make that dish at home.

YOURSE RESPONSE:
"""

time_template = """Given the recipe {recipe}, estimate how much time I need to cook it.

YOUR RESPONSE:
"""

location_chain_lcel = (
  PromptTemplate.from_template(location_template)
  | llama_llm
  | StrOutputParser()
)

dish_chain_lcel = (
  PromptTemplate.from_template(dish_template)
  | llama_llm
  | StrOutputParser()
)

time_chain_lcel = (
  PromptTemplate.from_template(time_template)
  | llama_llm
  | StrOutputParser()
)

overall_chain_lcel = (
  RunnablePassthrough.assign(meal=lambda x: location_chain_lcel.invoke({"location": x["location"]}))
  | RunnablePassthrough.assign(recipe=lambda x: dish_chain_lcel.invoke({"meal": x["meal"]}))
  | RunnablePassthrough.assign(time=lambda x: time_chain_lcel.invoke({"recipe": x["recipe"]}))
)

result = overall_chain_lcel.invoke({"location": "China"})
pprint(result)
