# Embeding Models

from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames

embed_params = {
	EmbedTextParamsMetaNames.TRUNCATE_INPUT_TOKENS: 3,
	EmbedTextParamsMetaNames.RETURN_OPTIONS: {"input_text": True},
}

from langchain_ibm import WatsonxEmbeddings

watson_embedding = WatsonEmbeddings(
	model_id="model",
	url="url",
	project_id="project nam-",
	params=embed_params,
)

texts = [text.page_content for text in chunks]
embedding_result = watsonx_embedding.embed_documents(texts)
embedding_result[0][:5]

# Vector stores
from langchain.vectorstores import Chroma

docsearch = Chroma.from_documents(chunks, watson_embedding)
docs = docsearch.similarity_search("Langchain")
print(docs[0].page_content)

# Retrievers
retriever = docsearch.as_retriever()
docs = retriever.invoke("Langchain")
print(docs[0])

# Parent document retrievers
from langchain.retrievers import ParentDocumentRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.storage import InMemoryStore

parent_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=20, separator='\n')
child_splitter = CharacterTextSplitter(chunk_size=400, chunk_overlap=20, separator='\n')
vectorstore = Chroma(
	collection_name="split_parents", embedding_function=watsonx_embedding
)
store = InMemoryStore()
retriever = ParentDocumentRetriever(
	vectorstore=vectorstore,
	docstore=store,
	child_splitter=child_splitter,
	parent_splitter=parent_splitter,
)
retriever.add_documents(document)
len(list(store.yield_keys()))
sub_docs = vectorstore.similarity_search("Langchain")
print(sub_docs[0].page_content)
retrieved_docs = retriever.invoke("Langchain")
print(retrieved_docs[0].page_content)

# RetrievalQA
from langchain.chains import RetrievalQA
qa = RetrievalQA.from_chain_type(
	llm=llama_llm,
	chain_type="stuff",
	retriever=docsearch.as_retriever(),
	return_source_documents=False
	)
qa.invoke("what is this paper discussing?")
