# Chat message history

from langchain.memory import ChatmessageHistory
chat = llama_llm
history = ChatMessageHistory()
history.add_ai_message("hi!")
history.add_user_message("what is the capital of France?")
print(history.messages)

ai_response = chat.invoke(history.messages)
print(ai_response)

history.add_ai_message(ai_response)

# Conversational buffer
from langchain.memory import ConversationBuffermemory
from langchain.chains import ConversationChain

conversation = ConversationChain(
  llm=llama_llm,
  verbose=True,
  memory=ConversationBufferMemory()
)
conversation.invoke(input="Hello, I am a little cat. Who are you?")
conversation.invoke(input="What can you do?")
conversation.invoke(input="Who am I?")
