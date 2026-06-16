# pyrefly: ignore [missing-import]
import streamlit as st
from datetime import timedelta
from vid_timestamp_fetcher_pipeline import final_chain as multi_video_chain
from vid_ques_pipeline import final_chain as single_video_chain

st.set_page_config(
page_title="YouTube Transcript Chatbot",
page_icon="🎥",
layout="wide"
)

st.title("🎥 YouTube Transcript Chatbot")

st.markdown(
"""
Two modes available:

1. **Single Video QA**

   * Give one YouTube URL
   * Ask questions from that specific video

2. **Multi Video Search**

   * Give a general query
   * System fetches multiple videos
   * Returns timestamp-grounded answers
     """
     )

# ======================================================
# SIDEBAR MODE SELECTION
# ======================================================

st.sidebar.title("Mode Selection")

mode = st.sidebar.radio(
"Choose Mode",
[
"Single Video QA",
"Multi Video Search"
]
)

# ======================================================
# SINGLE VIDEO MODE
# ======================================================

if mode == "Single Video QA":

    st.header("📺 Ask Questions From One Video")

    video_id = st.text_input(
        "Enter YouTube Video ID",
        placeholder="rGvh2F6Y5Lg"
    )

    question = st.text_input(
        "Ask Question",
        placeholder="What does the speaker say about transformers?"
    )

    ask_button = st.button(
        "Analyze Video"
    )

    if (
        video_id
        and question
        and ask_button
    ):

        with st.spinner("Analyzing video transcript..."):

            try:
                
                result = single_video_chain(video_id,question)

                st.success("Answer Generated")

                st.video(f"https://www.youtube.com/watch?v={video_id}")

                st.markdown("### Answer")

                st.write(result)

            except Exception as e:

                st.error(f"Error: {str(e)}")
              
# ======================================================
# MULTI VIDEO MODE
# ======================================================

elif mode == "Multi Video Search":
    st.header("🔍 Multi-Video Semantic Search")

    query = st.text_input(
        "Enter Search Query",
        placeholder="Explain how llms function"
    )

    search_button = st.button(
        "Search Videos"
    )

    if search_button and query:

        with st.spinner("Searching videos and generating answers..."):

            try:

                answer, fetched_videos = multi_video_chain(query)

                if len(answer.responses) == 0:

                    st.warning("No relevant context found.")

                else:

                    st.success("Results Generated")

                    for item in answer.responses:

                        video_id = item.video_id

                        timestamp = item.timestamp

                        summary = item.summary

                        video = fetched_videos[video_id]

                        video_url = video["url"]

                        timestamp_int = (
                            int(timestamp)
                            if timestamp is not None
                            else 0
                        )

                        youtube_timestamp_url = (
                            f"{video_url}&t={timestamp_int}s"
                        )

                        st.divider()

                        st.subheader(f"📺 {video['title']}")

                        st.markdown(f"""**Video ID:** {video_id}
                            **Timestamp:** {str(timedelta(seconds=timestamp_int))}""")

                        st.markdown(f"### Summary {summary}")

                        st.link_button("Watch Relevant Segment",youtube_timestamp_url)

                        st.video(video_url)

            except Exception as e:

                st.error(f"Error: {str(e)}")
