
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv

from agents import create_agents

load_dotenv()

embedding_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

vectordb = Chroma(
    persist_directory="chroma_db_news",
    embedding_function=embedding_model
)

retriever = vectordb.as_retriever(
    search_kwargs={
        "k": 5
    }
)

llm = ChatGroq(
    temperature=0,
    model_name="openai/gpt-oss-120b"
)

retriever_agent, analyst_agent, answer_agent = create_agents(llm)

def ask_question(question):

    print("\n" + "=" * 60)
    print("AGENT 1 - INFORMATION RETRIEVAL")
    print("=" * 60)

    documents = retriever.invoke(question)

    context = "\n\n".join(
        [
            f"Document {i + 1}:\n{doc.page_content}"
            for i, doc in enumerate(documents)
        ]
    )

    retrieved_information = retriever_agent.invoke({
        "question": question,
        "context": context
    })

    print("\nRetrieved Information:")
    print(retrieved_information)

    print("\n" + "=" * 60)
    print("AGENT 2 - RESEARCH ANALYSIS")
    print("=" * 60)

    analysis = analyst_agent.invoke({
        "question": question,
        "retrieved_information": retrieved_information
    })

    print("\nAnalysis:")
    print(analysis)

    print("\n" + "=" * 60)
    print("AGENT 3 - FINAL ANSWER")
    print("=" * 60)

    final_answer = answer_agent.invoke({
        "question": question,
        "retrieved_information": retrieved_information,
        "analysis": analysis
    })

    print("\nFinal Answer:")
    print(final_answer)

    return {
        "answer": final_answer,
        "analysis": analysis,
        "retrieved_information": retrieved_information,
        "source_documents": documents
    }

if __name__ == "__main__":

    query = "How much targets US has Hit"

    result = ask_question(query)

    print("\n" + "=" * 60)
    print("SOURCE DOCUMENTS")
    print("=" * 60)

    for i, doc in enumerate(result["source_documents"]):

        print(f"\nSource {i + 1}")

        print("Metadata:")
        print(doc.metadata)

        print("Content:")
        print(doc.page_content[:500])
