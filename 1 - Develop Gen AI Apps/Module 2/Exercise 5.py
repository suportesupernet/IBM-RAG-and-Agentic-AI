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
