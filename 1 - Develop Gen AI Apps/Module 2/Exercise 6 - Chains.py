from langchain.chains import LLMChain, SequentialChain
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

positive_review = """I absolutely love this coffee maker! It brews quickly and the coffee tastes amazing.
The built-in grinder saves me so much time in the morning, and the programmable timer means
I wake up to fresh coffee every day. Worth every penny and highly recommended to any coffee enthusiast."""

negative_review = """Disappointed with this laptop. It's constantly overheating after just 30 minutes of use,
and the battery life is nowhere near the 8 hours advertised - I barely get 3 hours.
The keyboard has already started sticking on several keys after just two weeks. Would not recommend to anyone."""

# Step 1: Define the prompt templates for each processing step
sentiment_template = """Analyze the sentiment of the following product review as positive, negative, or neutral.
Provide your analysis in the format: "SENTIMENT: [positive/negative/neutral]"

Review: {review}

Your analysis:
"""

summary_template = """Summarize the following product review into 3-5 key bullet points. Each bullet point should be concise and capture an important aspect mentioned in the review.

Review: {review}
Sentiment: {sentiment}

Key points:
"""

response_template = """Write a helpful response to a customer based on their product review. If the sentiment is positive, thank them for their feedback. If negative, express understanding and suggest a solution or next steps. Personalize based on the specific points they mentioned.

Review: {review}
Sentiment: {sentiment}
Key points: {summary}

Response to customer:
"""

sentiment_prompt = PromptTemplate.from_template(sentiment_template)
summary_prompt = PromptTemplate.from_template(summary_template)
response_prompt = PromptTemplate.from_template(response_template)

sentiment_chain_llmchain = LLMChain(llm=llama_llm, prompt=sentiment_prompt, output_key='sentiment')
summary_chain_llmchain = LLMChain(llm=llama_llm, prompt=summary_prompt, output_key='summary')
response_chain_llmchain = LLMChain(llm=llama_llm, prompt=response_prompt, output_key='response')

overall_chain_llmchain = SequentialChain(
    chains=[sentiment_chain_llmchain, summary_chain_llmchain, response_chain_llmchain],
    input_variables=['review'],
    output_variables=['sentiment', 'summary', 'response'],
    verbose=True
    )

sentiment_chain = sentiment_prompt | llama_llm | StrOutputParser()
summary_chain = summary_prompt | llama_llm |StrOutputParser()
response_chain = response_prompt | llama_llm | StrOutputParser()

overall_chain_lcel = (
    RunnablePassthrough.assign(sentiment=lambda x: sentiment_chain.invoke({"review": x["review"]}))
    | RunnablePassthrough.assign(summary=lambda x: summary_chain(invoke({"review": x["review"], "sentiment": x["sentiment"]})))
    | RunnablePassthrough.assign(response=lambda x: response_chain.invoke({"review": x["review"], "sentiment": x["sentiment"], "summary": x["summary"]}))
    )

test_chains(review):
    print("Test both chain implementations with the given review")
    print("\n" + "="*50)
    print(f"TESTING WITH REVIEW:\n{review[:100]}...\n")

    print("TRADITIONAL CHAIN RESULTS:")
    result_llmchain=overall_chain_llmchain.invoke(input={'review': review})
    print(result_llmchain)

    print("\nLCEL CHAIN RESULTS:")
    result = overall_chain_lcel.invoke("review": review)
    print(result)
    print("="*50)

test_chains(positive_review)
test_chains(negative_review)
