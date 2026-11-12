# 🎬 Video Chapter Agent

Turn a YouTube video into **auto-generated chapters, key takeaways and a quiz**, using the free Gemini API and a Streamlit web app.

Paste a YouTube link (or a transcript) and the app produces:

- **Chapters** with timestamps. Clicking a chapter jumps the embedded video to that moment.
- **Key takeaways**, 5 to 8 short points summarising the video.
- **An interactive quiz** (3 to 10 multiple-choice questions) with explanations and a score.

## How it works

1. **Transcript**: fetched with `youtube-transcript-api`, or pasted by the user if the video has no captions.
2. **Chaptering**: the transcript is split into 15-minute windows. Gemini proposes chapter titles and start times for each window using structured (Pydantic) JSON output. Start times are snapped to real transcript timestamps, and near-duplicate boundaries are merged.
3. **Takeaways and quiz**: generated from the transcript with schema-validated JSON. Quiz options are shuffled and validated (exactly 4 options, valid answer index).
4. **Reliability**: automatic retries, model fallback, rate-limit handling (about 6 s between calls on the free tier), and a local SQLite cache so repeated videos don't re-use API quota.

The notebook also includes an **evaluation** section that compares the Gemini chapters against the creator's own chapters, using three baselines (equal splits, TextTiling, sentence embeddings) and these metrics: Pk, WindowDiff and tolerance-based precision/recall/F1.

## Project structure

```
├── app.py              # Streamlit user interface
├── vca_core.py         # Core pipeline: transcript, Gemini calls, caching, evaluation metrics
├── requirements.txt    # Python dependencies
├── Video_Chapter_Agent_Clean.ipynb   # Development and evaluation notebook
└── .streamlit/secrets.toml.example   # Example of how to store the API key
```

## Run locally

```bash
git clone <your-repo-url>
cd <your-repo-folder>
pip install -r requirements.txt

# Option A: paste your key in the app's sidebar
# Option B: set it once
export GEMINI_API_KEY="your-key"      # Windows PowerShell: $env:GEMINI_API_KEY="your-key"

streamlit run app.py
```

Get a free Gemini API key at <https://aistudio.google.com/apikey>.

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub. **Do not commit your API key.**
2. Go to <https://share.streamlit.io>, click **Create app**, and select the repo and branch.
3. Set **Main file path** to `app.py`.
4. Open **Advanced settings → Secrets** and add:
   ```toml
   GEMINI_API_KEY = "your-gemini-api-key"
   ```
   Skip this step if you want every visitor to paste their own key in the sidebar.
5. Click **Deploy**.

## Known limitations

- **YouTube transcripts on cloud hosting**: YouTube often blocks transcript requests from cloud server IPs, so link-based fetching may fail on the deployed app even though it works locally. In that case, use the **"Paste the transcript"** box in the app.
- **Free-tier limits**: the Gemini free tier has per-minute and per-day quotas, so long videos can be slow.
- **Cache**: the SQLite cache (`vca_cache.db`) is temporary on Streamlit Cloud and resets when the app restarts.
- Chapter quality depends on the transcript quality; videos without captions need a pasted transcript.

## Tech stack

Python, Streamlit, Google Gemini API (`google-genai`), Pydantic, `youtube-transcript-api`, NumPy, SQLite.
