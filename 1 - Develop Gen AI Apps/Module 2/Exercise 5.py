from langchain.memory import ConversationBufferMemory, ChatMessageHistory
from langchain.chains import ConversationChain
from langchain_core.messages import HumanMessage, AIMessage
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM

model_id = 'meta-llama/llama-4-maverick-17b-128e-instruct-fp8'
parameters = {
  GenParams.MAX_NEW_TOKENS: 256,
  GenParams.TEMPERATURE: 0.2,
}
credentials = {"url": "https://url.com"}
project_id = "project_id"

model = ModelInference(
  model_id=model_id,
  params=parameters,
  credentials=credentials,
  project_id=project_id
)
llm = WatsonxLLM(model=model)

history = ChatMessageHistory()

history.add_user_message("Hello, my name is Alice.")
history.add_ai_message("Hello Alice! It's nice to meet you. How can I help you today?")

print("Initial Chat History:")
for message in history.messages:
  sender = "Human" if isInstance(message, HumanMessage) else "AI"
  print(f"{sender}: {message.content}")

memory = ConversationBufferMemory(chat_memory=history)
conversation = ConversationChain(
  llm=llm,
  memory=memory,
  verbose=True
)

def chat_simulation(conversation, inputs):
  print("\n=== Beginning Chat Simulation ===")

for i, user_input in enumerate(inputs):
  print(f"\n--- Turn {i+1} ---")
  print(f"Human: {user_input}")

  response = conversation.invoke(input=user_input)
  print(f"AI: {response['response']}")

print("\n=== End of Chat Simulation ===")

test_inputs = [
  "My favorite color is blue.",
    "I enjoy hiking in the mountains.",
    "What activities would you recommend for me?",
    "What was my favorite color again?",
    "Can you remember both my name and my favorite color?"
]

chat_simulation(conversation, test_inputs)

print("\nFinal Memory Contents:")
print(conversation.memory.buffer)

# Create a summarizing memory that will compress the conversation
summary_memory = ConversationSummaryMemory(llm=llm)

summary_memory.save_context(
  {"input": "Hello, my name is Alice."},
  {"output": "Hello Alice! It's nice to meet you. How can I help you today?"}
)

summary_conversation = ConversationChain(
  llm=llm,
  memory=summary_memory,
  verbose=True
)

print("\\\\n\\n=== Testing Conversation Summary Memory ===")
chat_simulation(summary_conversation, test_inputs)

print("\\nFinal Summary Memory Contents:")
print(summary_memory.buffer)

print("\n=== Memory Comparison ===")
print(f"Buffer Memory Size: {len(conversation.memory.buffer)} characters")
print(f"Summary Memory Size: {len(summary_memory.buffer)} characters")
print("\nThe conversation summary memory typically creates a more compact repreesntation of the chat history.")
