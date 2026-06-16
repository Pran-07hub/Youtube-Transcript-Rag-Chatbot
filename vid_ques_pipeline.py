from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings
# pyrefly: ignore [missing-import]
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
# pyrefly: ignore [missing-import]
from langchain_community.vectorstores import FAISS
# pyrefly: ignore [missing-import]
from langchain_classic.retrievers import MultiQueryRetriever
from transcript_extract import transcript_fetch
# pyrefly: ignore [missing-import]
# from langchain_google_genai import ChatGoogleGenerativeAI
# pyrefly: ignore [missing-import]
from dotenv import load_dotenv
load_dotenv()


splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)

def preprocess(video_id, logger=None):

    if logger:
        logger("Extracting Transcript...")
    print("Extracting Transcript...")
    transcript = transcript_fetch(video_id)

    if logger:
        logger("Chunking Transcript...")
    print("Chunking Transcript...")
    chunks = splitter.create_documents([transcript])
    if not chunks or not any(c.page_content.strip() for c in chunks):
        raise ValueError("The transcript is empty or could not be chunked.")

    if logger:
        logger("Creating Embeddings...")
    print("Creating Embeddings...")
    embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

    if logger:
        logger("Creating Vector Store...")
    print("Creating Vector Store...")
    vector_store = FAISS.from_documents(chunks, embedding)

    if logger:
        logger("Creating Retriever...")
    print("Creating Retriever...")
    retriever = MultiQueryRetriever.from_llm(
        retriever=vector_store.as_retriever(search_type = "mmr", search_kwargs={"k": 5, "lambda_mult": 0.5}),
        llm=model
    )

    print("Asking LLM...")
    return retriever

llm = HuggingFaceEndpoint(repo_id="meta-llama/Meta-Llama-3-8B-Instruct", task="text-generation")

model = ChatHuggingFace(llm=llm)
# model1 = ChatGoogleGenerativeAI(model = "gemini-2.5-flash")

chat_store = {}

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are a helpful assistant.

Answer ONLY from the transcript context.

If the answer is not fully present in the transcript:
- answer the transcript-supported part first
- then explicitly write:

Additional reasoning:

and provide your own reasoning.

Never pretend transcript information exists when it does not.
"""
    ),

    MessagesPlaceholder("chat_history"),

    (
        "human",
        """
Context:
{context}

Question:
{question}
"""
    )
])

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

rewrite_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are a query rewriting assistant.

Your task is to convert the user's latest question into a standalone question that can be understood without the chat history.

Rules:
- Preserve the original meaning.
- Use the chat history to resolve references such as:
  - it
  - they
  - this
  - that
  - he
  - she
  - these
  - those

- Do NOT answer the question.
- Do NOT add new information.
- Do NOT explain your reasoning.
- If the question seems complete, do NOT change anything
- Return ONLY the rewritten question.
"""
    ),

    MessagesPlaceholder("chat_history"),

    ("human", "{question}")
])

def context(retrieved_docs):
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
    return context_text

retriever_store = {}
def final_chain(video_id, query, logger=None):

    if logger:
        logger("Starting Chain...")
    print("Starting Chain...")
    parser = StrOutputParser()

    if video_id not in retriever_store:
        retriever_store[video_id] = preprocess(video_id, logger=logger)

    rewrite_chain = RunnableParallel({
        'question': RunnablePassthrough(),
        'chat_history': RunnableLambda(lambda x: chat_store[video_id] if video_id in chat_store else [])
    }) | rewrite_prompt | model | parser

    parallel_chain = RunnableParallel({
        'context': retriever_store[video_id] | RunnableLambda(context),
        'question': RunnablePassthrough(),
        'chat_history': RunnableLambda(lambda x: chat_store[video_id] if video_id in chat_store else [])
    })

    main_chain = rewrite_chain | parallel_chain | prompt | model | parser

    answer = main_chain.invoke(query)

    if video_id not in chat_store:
        chat_store[video_id] = []
    chat_store[video_id].append(HumanMessage(content=query))
    chat_store[video_id].append(AIMessage(content=answer))

    return answer

# if __name__ == '__main__':
#     print(final_chain('jNJH6uD5LQE', 'What is being discussed here'))
#     print(final_chain('jNJH6uD5LQE', 'what are ai agents as per the video'))
#     print(final_chain('jNJH6uD5LQE', 'what is the use of such agents'))