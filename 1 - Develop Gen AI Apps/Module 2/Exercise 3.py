from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, WebBaseloader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter

paper_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/96-FDF8f7coh0ooim7NyEQ/langchain-paper.pdf"
pdf_loader = PyPDFLoader(paper_url)
pdf_document = pdf_loader.load()

web_url = "https://python.langchain.com/v0.2/docs/introduction/"
web_loader = WebBaseLoader(web_url)
web_document = web_loader.load()

splitter1 = CharacterTextSplitter(chunk_size=300, chunk_overlap=30, separator="\n")
splitter2 = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

chunks_1 = splitter1.split_documents(pdf_document)
chunks_2 = splitter2.split_documents(web_document)

def display_document_stats(docs, name):
    total_chunks = len(docs)
    total_chars = sum(len(doc.page_content) for doc in docs)
    avg_chunk_size = total_chars / total_chunks if total_chunks > 0 else 0

    all_metadata_keys = set()
    for doc in docs:
        all_metadata_keys.update(doc.metada.keys())

    print(f"\n=== {name} Statistics ===")
    print(f"Total number of chunks: {total_chunks}")
    print(f"Average chunk size: {avg_chunk_size: 2f} characters")
    print(f"Metadata keys preserved: {', '.join(all_metadata_keys)}")

    if docs:
        print("\nExample chunk:")
        example_doc = docs[min(5, total_chunks-1)]
        print(f"Content (first 150 chars): {example_doc.page_content[:150]}...")
        print(f"Metadata: {example_doc.metadata}")

        lengths = [len(doc.page_content) for doc in docs]
        min_len = min(lengths)
        max_len = max(lengths)
        print(f"Min chunk size: {min_len} characters")
        print(f"Max chunk size: {max_len} characters")

display_document_stats(chunks_1, "Splitter 1")
display_document_stats(chunks_2, "Splitter 2")

