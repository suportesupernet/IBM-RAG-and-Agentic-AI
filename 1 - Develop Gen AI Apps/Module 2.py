def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')
import os
os.environ['ANONYMIZED_TELEMETRY'] = 'False'

from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes
from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM

model_id = 'meta-llama/llama-3-3-70b-instruct'

parameters = {
    GenParams.MAX_NEW_TOKENS: 256,
    GenParams.TEMPERATURE: 0.2,
}

credentials = {
    "url": "https://us-south.ml.cloud.ibm.com"
    # "api_key": "your api key"
}

project_id = "skills-network"

model = ModelInference(
    model_id=model_id,
    params=parameters,
    credentials=credentials,
    project_id=project_id
)

msg = model.generate("In today's sales meeting, we ")
print(msg['results'][0]['generated_text'])

# Chat Model
llama_llm - WatsonxLLM(model = model)
print(llama_llm.invoke("Whos is woman's best friend?"))

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

msg = llama_llm.invoke([
    SystemMessage(content="You are a helpful assistant that translates English to French."),
    HumanMessage(content="Translate the following sentence to French: 'I love programming.'")
])

msg = llama_llm.invoke(
    [
        SystemMessage(content="You are a supportive AI bot that suggests fitness activities to a user in one short sentence"),
        HumanMessage(content="I like high-intensity workouts, what should I do?"),
        AIMessage(content="You should try a CrossFit class"),
        HumanMessage(content="How often should I attend?")
    ]
)

# Prompt Templates
from langchain_core.prompts import PromptTemplate
prompt = PromptTemplate.from_template("Tell me one {adjective} joke about {topic}")
input_ = {"adjective": "funny", "topic": "cats"}

prompt.invoke(input_)

# Chat prompt templates
from langchain_core.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant"),
    ("user", "Tell me a joke about {topic}")
])

input_ = {"topic": "cats"}
prompt.invoke(input_)

# Messages Placeholder
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant"),
    MessagesPlaceholder("msgs")
])

input_ = {"msgs": [HumanMessage(content="What is the day after Tuesday?")]}
prompt.invoke(input_)

chain = prompt | llama_llm
response = chain.invoke(input = input_)
print(response)

# Output parses
from langchain_core.output_parses import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

class Joke(BaseModel):
    setup: str = Field(description="question to set up a joke")
    punchline: str = Field(description="answer to resolve the joke")

joke_query = "Tell me a joke"
output_parser = JsonOutputParser(pydantic_object=Joke)
format_instructions = output_parser.get_format_instruction()

prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": format_instructions},
)

chain = prompt | llama_llm | output_parser
chain.invoke({"query": joke_query})

# Comma-separated list parser
from langchain.output_parsers import CommaSeparatedListOutputParser
output_parser = CommaSeparatedListOutputParser()
format_instructions = output_parser.get_format_instructions()

prompt = PromptTemplate(
    template="Answer the user query. {format_instruction}\nList five {subject}.",
    input_variables=["subject"],
    partial_variables={"format_instruction": format_instructions},
)

chain = prompt | llama_llm | output_parser
chain.invoke({"subject": "ice cream flavors"})

# Exercise 2: Creating and using a JSON output parser
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

json_parser = JsonOutputParser()
format_instructions = """RESPONSE FORMAT: Return ONLY a single JSON object—no markdown, no examples, no extra keys.  It must look exactly like:
{
  "title": "movie title",
  "director": "director name",
  "year": 2000,
  "genre": "movie genre"
}

IMPORTANT: Your response must be *only* that JSON.  Do NOT include any illustrative or example JSON."""

prompt_template = PromptTemplate(
    template="""You are a JSON-only assistant.

Task: Generate info about the movie "{movie_name}" in JSON format.

{format_instructions}
""",
    input_variables=["movie_name"],
    partial_variables={"format_instructions": format_instructions},
)
movie_chain = prompt_template | llama_llm | json_parser
movie_name = "Abigail"
result = movie_chain.invoke({"movie_name": movie_name})

print("Parsed result:")
print(f"Title: {result['title']}")
print(f"Director: {result['director']}")
print(f"Year: {result['year']}")
print(f"Genre: {result['genre']}")