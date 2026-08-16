from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ============================================================
# AGENT 1: RETRIEVER AGENT
# ============================================================

def create_retriever_agent(llm):

    prompt = ChatPromptTemplate.from_template("""
You are Agent 1: Information Retrieval Agent.

Your job is to extract the most relevant information from
the retrieved documents to answer the user's question.

USER QUESTION:
{question}

RETRIEVED DOCUMENTS:
{context}

Instructions:
1. Read all retrieved documents carefully.
2. Identify information directly related to the question.
3. Extract important facts, numbers, dates, names and events.
4. Ignore irrelevant information.
5. Do not invent or assume information.
6. Only use information available in the documents.
7. If the documents do not contain enough information,
   clearly state that the evidence is insufficient.

Return the relevant evidence in a concise format.
""")

    chain = prompt | llm | StrOutputParser()

    return chain


# ============================================================
# AGENT 2: ANALYST AGENT
# ============================================================

def create_analyst_agent(llm):

    prompt = ChatPromptTemplate.from_template("""
You are Agent 2: Research Analyst Agent.

Your job is to analyze the evidence extracted by Agent 1
and determine what it means for the user's question.

USER QUESTION:
{question}

INFORMATION FROM AGENT 1:
{retrieved_information}

Instructions:
1. Carefully analyze the information.
2. Identify the facts that directly answer the question.
3. Compare information when multiple pieces of evidence exist.
4. Check numbers, dates and important details.
5. Do not introduce facts that are not present in the evidence.
6. Do not hallucinate.
7. If the information is insufficient, clearly state this.
8. Provide a concise evidence-based analysis.

Your output will be passed to Agent 3.
""")

    chain = prompt | llm | StrOutputParser()

    return chain


# ============================================================
# AGENT 3: FINAL ANSWER AGENT
# ============================================================

def create_answer_agent(llm):

    prompt = ChatPromptTemplate.from_template("""
You are Agent 3: Final Answer Agent.

Your job is to provide the final answer to the user based
on the retrieved evidence and analysis.

USER QUESTION:
{question}

RETRIEVED INFORMATION:
{retrieved_information}

ANALYSIS:
{analysis}

Instructions:
1. Answer the user's question directly.
2. Use only information supported by the evidence.
3. Do not invent facts.
4. Do not hallucinate.
5. Keep the answer clear and easy to understand.
6. Include important numbers, dates and names when relevant.
7. If the evidence is insufficient, say that the available
   documents do not provide enough information.
8. Do not mention Agent 1, Agent 2 or Agent 3 in the final answer.
9. Do not describe the internal RAG pipeline.
10. Return only the final answer.

FINAL ANSWER:
""")

    chain = prompt | llm | StrOutputParser()

    return chain


# ============================================================
# CREATE ALL THREE AGENTS
# ============================================================

def create_agents(llm):

    retriever_agent = create_retriever_agent(llm)

    analyst_agent = create_analyst_agent(llm)

    answer_agent = create_answer_agent(llm)

    return (
        retriever_agent,
        analyst_agent,
        answer_agent
    )
