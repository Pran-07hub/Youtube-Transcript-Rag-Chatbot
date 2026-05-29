from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings
# pyrefly: ignore [missing-import]
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
# pyrefly: ignore [missing-import]
from langchain_community.vectorstores import FAISS
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

def preprocess(video_id):
    transcript = transcript_fetch(video_id)

    chunks = splitter.create_documents([transcript])

    embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

    vector_store = FAISS.from_documents(chunks, embedding)

    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    return retriever

llm = HuggingFaceEndpoint(repo_id="meta-llama/Meta-Llama-3-8B-Instruct", task="text-generation")

model = ChatHuggingFace(llm=llm)
# model1 = ChatGoogleGenerativeAI(model = "gemini-2.5-flash")

prompt = PromptTemplate(
     template="""
      You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, answer accordingly.

      {context}
      Question: {question}
    """,
    input_variables = ['context', 'question']
)

def context(retrieved_docs):
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
    return context_text

def final_chain(video_id, query):

    parser = StrOutputParser()

    parallel_chain = RunnableParallel({
        'context': preprocess(video_id) | RunnableLambda(context),
        'question': RunnablePassthrough()
    })

    final_chain = parallel_chain | prompt | model | parser

    answer = final_chain.invoke(query)
    return answer

if __name__ == '__main__':
    print(final_chain('B7TQYJLfYZQ', 'What is being discussed here'))