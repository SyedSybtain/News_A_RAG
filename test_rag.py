from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import UnstructuredFileLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

load_dotenv()

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vectordb = Chroma(
    persist_directory="chroma_db_news",
    embedding_function=embedding_model
)

vectordb.persist()

llm = ChatGroq(temperature=0, model_name="openai/gpt-oss-120b")

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectordb.as_retriever(),
    return_source_documents=True
)

query = "How much targets US has Hit"
result = qa_chain.invoke({"query": query})

print("\nAnswer:", result["result"])
print("\nSources:")
for doc in result["source_documents"]:
    print(doc.metadata)
