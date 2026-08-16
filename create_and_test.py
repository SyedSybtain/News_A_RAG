from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import UnstructuredFileLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

load_dotenv()

# 1. Load your document
loader = UnstructuredFileLoader("files/news_dataset.csv")  # could be .txt, .docx, etc.
documents = loader.load()

# 2. Split into smaller chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
docs = splitter.split_documents(documents)

# 3. Embedding model
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 4. Chroma vector store (with persistence)
persist_directory = "chroma_db_news"
vectordb = Chroma.from_documents(
    documents=docs,
    embedding=embedding_model,
    persist_directory=persist_directory
)
# If you've already created the Chroma DB once, you can reload it directly like this:
vectordb = Chroma(
    persist_directory="chroma_db_news",
    embedding_function=embedding_model
)
#########################
vectordb.persist()

# 5. Groq LLM
llm = ChatGroq(temperature=0, model_name="openai/gpt-oss-120b")

# 6. Create RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectordb.as_retriever(),
    return_source_documents=True
)

# 7. Ask a question
query = "How Much Target US Has Hit"
result = qa_chain.invoke({"query": query})

print("\nAnswer:", result["result"])
print("\nSources:")
for doc in result["source_documents"]:
    print(doc.metadata)

# from langchain_community.document_loaders import UnstructuredFileLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.vectorstores import Chroma
# from langchain.chains import RetrievalQA
# from langchain_groq import ChatGroq
# from langchain_core.prompts import ChatPromptTemplate
