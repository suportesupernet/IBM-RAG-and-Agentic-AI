from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, WebBaseloader

paper_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/96-FDF8f7coh0ooim7NyEQ/langchain-paper.pdf"
pdf_loader = PyPDFLoader(paper_url)
pdf_document = pdf_loader.load()
