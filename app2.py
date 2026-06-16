# pyrefly: ignore [missing-import]

# pyrefly: ignore [missing-import]
import streamlit as st
from datetime import timedelta
from urllib.parse import urlparse, parse_qs

from vid_timestamp_fetcher_pipeline import (
    final_chain as multi_video_chain
)

from vid_ques_pipeline import (
    final_chain as single_video_chain
)

# ======================================================
# PAGE CONFIG
# ======================================================

st.set_page_config(
    page_title="YouTube Transcript Chatbot",
    page_icon="🎥",
    layout="wide"
)

# ======================================================
# HELPERS
# ======================================================

def extract_video_id(url: str):

    try:
        parsed = urlparse(url)
        return parse_qs(parsed.query).get("v", [None])[0]

    except Exception:
        return None


# ======================================================
# SESSION STATE
# ======================================================

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

if "current_video_id" not in st.session_state:
    st.session_state.current_video_id = None

# ======================================================
# TITLE
# ======================================================

st.title("🎥 YouTube Transcript Chatbot")

st.markdown(
"""
Two modes available:

### 📺 Single Video QA

- Provide a YouTube video URL
- Chat with the video transcript
- Supports conversational follow-up questions

### 🔍 Multi Video Search

- Ask a broad question
- System discovers relevant videos
- Returns timestamp-grounded answers
"""
)

# ======================================================
# SIDEBAR
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
# SINGLE VIDEO CHAT MODE
# ======================================================

if mode == "Single Video QA":

    st.header("📺 Chat With A Video")

    video_url = st.text_input(
        "Enter YouTube Video URL",
        placeholder="https://www.youtube.com/watch?v=..."
    )

    video_id = extract_video_id(video_url)

    if video_id:

        # Clear chat if user changes video
        if st.session_state.current_video_id != video_id:

            st.session_state.current_video_id = video_id
            st.session_state.chat_messages = []

        st.video(video_url)

        # Display chat history
        for msg in st.session_state.chat_messages:

            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        user_query = st.chat_input(
            "Ask something about this video..."
        )

        if user_query:

            st.session_state.chat_messages.append(
                {
                    "role": "user",
                    "content": user_query
                }
            )

            with st.chat_message("user"):
                st.markdown(user_query)

            with st.chat_message("assistant"):

                try:

                    with st.status(
                        "Analyzing Video...",
                        expanded=True
                    ) as status:

                        def logger(message):
                            status.write(message)

                        answer = single_video_chain(
                            video_id,
                            user_query,
                            logger=logger
                        )

                        status.update(
                            label="Completed",
                            state="complete"
                        )

                    st.markdown(answer)

                    st.session_state.chat_messages.append(
                        {
                            "role": "assistant",
                            "content": answer
                        }
                    )

                except Exception as e:

                    status.update(
                        label="Failed",
                        state="error"
                    )

                    st.error(str(e))

    elif video_url:

        st.error("Invalid YouTube URL")

# ======================================================
# MULTI VIDEO MODE
# ======================================================

elif mode == "Multi Video Search":

    st.header("🔍 Multi-Video Semantic Search")

    query = st.text_input(
        "Enter Search Query",
        placeholder="Explain how transformers work"
    )

    search_button = st.button(
        "Search Videos"
    )

    if search_button and query:

        try:

            with st.status(
                "Running Multi-Video Retrieval...",
                expanded=True
            ) as status:

                def logger(message):
                    status.write(message)

                answer, fetched_videos = (
                    multi_video_chain(
                        query,
                        logger=logger
                    )
                )

                status.update(
                    label="Completed",
                    state="complete"
                )

                if len(answer.responses) == 0:

                    st.warning(
                        "No relevant context found."
                    )

                else:

                    st.success(
                        "Results Generated"
                    )

                    for item in answer.responses:

                        video_id = item.video_id
                        timestamp = item.timestamp
                        summary = item.summary

                        video = fetched_videos[
                            video_id
                        ]

                        video_url = video["url"]

                        timestamp_int = (
                            int(timestamp)
                            if timestamp is not None
                            else 0
                        )

                        youtube_timestamp_url = (
                            f"{video_url}&t={timestamp_int}s"
                        )

                        with st.expander(
                            f"📺 {video['title']}",
                            expanded=False
                        ):

                            st.markdown(
                                f"**Video ID:** {video_id}"
                            )

                            st.markdown(
                                f"**Timestamp:** {str(timedelta(seconds=timestamp_int))}"
                            )

                            st.markdown("### Summary")

                            st.write(summary)

                            st.link_button(
                                "Watch Relevant Segment",
                                youtube_timestamp_url
                            )

                            st.video(video_url)

        except Exception as e:

            try:
                status.update(
                    label="Failed",
                    state="error"
                )
            except:
                pass

            st.error(f"Error: {str(e)}")

# ======================================================
# FOOTER
# ======================================================

st.divider()

st.caption(
    "Built using LangChain, FAISS, HuggingFace, Streamlit, yt-dlp, and YouTube Transcript API"
)