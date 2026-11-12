import os
import streamlit as st
from vca_core import run_pipeline, fmt_time

st.set_page_config(page_title="Video Chapter Agent", page_icon="🎬", layout="wide")
st.title("🎬 Video Chapter Agent")
st.caption("Chapters, key takeaways and a quiz from a YouTube transcript (free Gemini API)")

def server_key():
    """Key stored in Streamlit Secrets / env. Never sent to the browser."""
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    return os.environ.get("GEMINI_API_KEY", "") or os.environ.get("GOOGLE_API_KEY", "")


with st.sidebar:
    user_key = st.text_input("Gemini API key", type="password",
                             help="Free key: https://aistudio.google.com/apikey")
    if server_key() and not user_key:
        st.caption("Using the app's built-in key. Paste your own to use your own quota.")
    api_key = user_key or server_key()
    n_q = st.slider("Quiz questions", 3, 10, 5)
    use_cache = st.checkbox("Use cache", value=True)

url = st.text_input("YouTube link")
with st.expander("No captions? Paste the transcript here (optional)"):
    pasted = st.text_area("Transcript (lines like '12:35 some text', or plain text)", height=150)

if st.button("Generate", type="primary"):
    if not (url or pasted):
        st.error("Enter a YouTube link or paste a transcript.")
    elif not api_key:
        st.error("Enter your Gemini API key in the sidebar.")
    else:
        try:
            with st.spinner("Working... (free tier is slow on purpose, about 6 s per call)"):
                st.session_state.result = run_pipeline(
                    url_or_id=url or None, pasted_text=pasted or None, api_key=api_key,
                    use_cache=use_cache, n_questions=n_q, progress=lambda m: None)
                st.session_state.start = 0
                st.session_state.graded = False
        except Exception as e:
            msg = str(e)
            if "Could not retrieve a transcript" in msg or "blocking requests" in msg or "IpBlocked" in type(e).__name__ or "RequestBlocked" in type(e).__name__:
                st.error("YouTube blocked the transcript request from this server (common on cloud hosting). "
                         "Open the video on YouTube, click **...more → Show transcript**, copy the text, "
                         "and paste it into the box above.")
            else:
                st.error(f"Something went wrong: {msg}")

res = st.session_state.get("result")
if res:
    is_youtube = not res["video_id"].startswith("pasted-")
    left, right = st.columns([3, 2])

    with left:
        if is_youtube:
            st.video(f"https://www.youtube.com/watch?v={res['video_id']}",
                     start_time=int(st.session_state.get("start", 0)))
        st.subheader("Chapters")
        for i, ch in enumerate(res["chapters"]):
            if st.button(f"{fmt_time(ch['start'])}  —  {ch['title']}", key=f"ch{i}", use_container_width=True):
                st.session_state.start = ch["start"]
                st.rerun()

    with right:
        st.subheader("Key takeaways")
        for p in res["takeaways"]:
            st.markdown(f"- {p}")

    st.subheader("Quiz")
    answers = {}
    for i, q in enumerate(res["quiz"]):
        answers[i] = st.radio(f"**{i + 1}. {q['question']}**", q["options"], index=None, key=f"q{i}")
    if st.button("Check answers"):
        st.session_state.graded = True
    if st.session_state.get("graded"):
        score = 0
        for i, q in enumerate(res["quiz"]):
            right_answer = q["options"][q["answer_index"]]
            if answers[i] == right_answer:
                score += 1
                st.success(f"Q{i + 1}: correct. {q['explanation']}")
            else:
                st.error(f"Q{i + 1}: answer is **{right_answer}**. {q['explanation']}")
        st.metric("Score", f"{score} / {len(res['quiz'])}")
