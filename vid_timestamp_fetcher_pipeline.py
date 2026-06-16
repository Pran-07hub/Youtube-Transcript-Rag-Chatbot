from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import PydanticOutputParser
# pyrefly: ignore [missing-import]
from langchain_classic.retrievers import MultiQueryRetriever
from pydantic import BaseModel, Field
from typing import List, Optional
# pyrefly: ignore [missing-import]
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from transcript_extract import transcript_fetch_with_time
from youtube_vid_search import search_youtube
from datetime import timedelta
import time,random
# pyrefly: ignore [missing-import]
from langchain_google_genai import ChatGoogleGenerativeAI
# pyrefly: ignore [missing-import]
from dotenv import load_dotenv
load_dotenv()

def video_fetch(query,max_results = 5,logger=None):
    if logger:
        logger("Fetching Videos...")
    print("Fetching Videos...")
    videos = search_youtube(query=query, max_results=max_results)
    return videos

def transcript_fetcher(videos,logger=None):
    if logger:
        logger("Fetching Transcripts...")
    print("Fetching Transcripts...")
    transcripts =[]
    
    for video in videos:
        try:
            transcripts.append(transcript_fetch_with_time(video['video_id']))
        except Exception as e:
            print(f"Failed to fetch transcript for video {video['video_id']}: {e}")
        time.sleep(random.uniform(1.0, 4.0))
    return transcripts

def chunk_former(transcripts,chunk_duration = 120,chunk_overlap = 0,logger=None):
    if logger:
        logger("Chunking Transcripts...")
    print("Chunking Transcripts...")
    chunks = []
    for i,transcript in enumerate(transcripts):
        if not transcript:
            continue
        duration = 0
        script = ""
        timestamp_index = 0
        for j,snippet in enumerate(transcript):
            script += snippet[0]
            duration += snippet[2]
            if duration > chunk_duration:
                chunks.append(Document(page_content=script, metadata={'timestart':transcript[timestamp_index][1],'duration':duration,'video_id':i}))
                script = ""
                timestamp_index = j+1
                duration = 0
        chunks.append(Document(page_content=script, metadata={'timestart':transcript[timestamp_index][1],'duration':duration,'video_id':i}))
    return chunks

embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

def form_vector_store(chunks,embedding, logger=None):
    if logger:
        logger("Forming Vector Store...")
    print("Forming Vector Store...")
    vector_store = FAISS.from_documents(chunks, embedding)
    return vector_store

llm = HuggingFaceEndpoint(repo_id="meta-llama/Meta-Llama-3-8B-Instruct", task="text-generation")
model = ChatHuggingFace(llm=llm)
model1 = ChatGoogleGenerativeAI(model = "gemini-2.5-flash")


class SingleResponse(BaseModel):
    summary: str = Field(description="Summaries of the relevant content in order")
    video_id: int = Field(description="Video IDs of the relevant content in order")
    timestamp: Optional[float] = Field(default = None, description="Timestamps of the relevant content in order")

class Answer(BaseModel):
    responses : List[SingleResponse]    

parser = PydanticOutputParser(pydantic_object=Answer)

prompt = PromptTemplate(
     template="""
You are an expert video-analysis assistant.

Your task is to answer the user's question ONLY using the provided transcript context.

The transcript context comes from multiple YouTube videos.
Each transcript chunk contains:
- transcript text
- video_id
- timestamps
- additional metadata

INSTRUCTIONS:

1. Use ONLY the provided transcript context.
Do not use outside knowledge.

2. If the answer is not present in the context, return blank json object

3. Find the most relevant information from the transcript chunks.

4. If only 1 relevant answer exists, return only 1 response.
Do NOT generate filler, placeholder, or "not related" responses.
Return max 4 answers

5. Each response MUST come from a DIFFERENT video_id.
If multiple chunks from the same video are relevant:
- combine them into ONE summarized response
- OR choose the most relevant one

6. For each response:
- summarize the relevant content clearly
- mention the main insight
- include the video_id
- include the most relevant timestamp

7. Do NOT repeat similar answers.

8. If transcript chunks partially answer the question:
- dont mention it
- do not invent missing details

9. Rank responses by relevance and informativeness.

CONTEXT:{context_metadata}
QUESTION: {question}

OUTPUT : 10.Format Instructions to be stricty followed:  {format_instruction}

11.Return ONLY raw JSON as mentioned in the schema. Start answer with dictionary "responses":[
Do NOT explain the output.
Do NOT add notes before or after JSON.
    """,
    input_variables = ['context_metadata', 'question'],
    partial_variables = {'format_instruction': parser.get_format_instructions()}
)

def data_prep(query, max_results=5, chunk_duration = 120, chunk_overlap = 0, logger=None):
    if logger:
        logger("Preparing Data...")
    print("Preparing Data...")
    videos = video_fetch(query=query,max_results = max_results, logger=logger)
    if not videos:
        raise ValueError("No videos found on YouTube for the search query.")
        
    transcripts = transcript_fetcher(videos, logger=logger)
    chunks = chunk_former(transcripts,chunk_duration=chunk_duration,chunk_overlap=chunk_overlap, logger=logger)
    if not chunks:
        raise ValueError("Could not retrieve transcripts for any of the found videos (captions may be disabled, or the YouTube IP request limit has been reached).")
        
    vector_store = form_vector_store(chunks=chunks, embedding=embedding, logger=logger)
    retriever = MultiQueryRetriever.from_llm(
        retriever=vector_store.as_retriever(search_type = "mmr", search_kwargs={"k": 10, "lambda_mult": 0.5}),
        llm=model
    )
    return retriever, videos

def context_metadata(retrieved_docs,logger=None):
    if logger:
        logger("Formatting Context...")
    print("Formatting Context...")
    context_metadata = "\n\n".join(f"{doc.page_content}, {doc.metadata}" for doc in retrieved_docs)
    if logger:
        logger("Asking LLM...")
    print("Asking LLM....")
    return context_metadata

# Execution

def final_chain(query,max_results=5, chunk_duration = 120, chunk_overlap = 0, logger=None):
    retriever, videos = data_prep(query,max_results=max_results, chunk_duration = chunk_duration, chunk_overlap = chunk_overlap, logger=logger)
    parallel_chain = RunnableParallel({
        'context_metadata' : retriever | RunnableLambda(lambda docs: context_metadata(docs, logger=logger)),
        'question' : RunnablePassthrough()
    })

    final_chain = parallel_chain | prompt | model1 | parser

    answer = final_chain.invoke(query)

    return answer, videos

if __name__ == '__main__':
    try:
        answer, videos = final_chain('how to replace a bicycle paddle')
        print(answer)
        print(videos)
        for item in answer.responses:
            video_id = item.video_id
            timestamp = item.timestamp
            summary = item.summary
            video = videos[video_id]
            video_url = video["url"]
            print('Video Title: ', video['title'])
            print('Video URL: ', video_url)
            print('Timestamp to watch from: ', str(timedelta(seconds=timestamp)))
            print('Summary: ', summary)
            print('\n')
    except Exception as e:
        print(f"Error occurred: {e}")
