import os, re, json, time, sqlite3, hashlib, random
from typing import List

import numpy as np
from pydantic import BaseModel, Field, model_validator, ValidationError
from google import genai
from google.genai import types

MODELS = ["gemini-3.6-flash", "gemini-3.5-flash", "gemini-3.7-flash", "gemini-3.8-flash"]
CALL_GAP_SECONDS = 6
DB_PATH = "vca_cache.db"

def fmt_time(sec):
    sec = int(sec)
    h, m, s = sec // 3600, (sec % 3600) // 60, sec % 60
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def extract_video_id(url_or_id):
    text = url_or_id.strip()
    m = re.search(r"(?:v=|youtu\.be/|embed/|shorts/|live/)([A-Za-z0-9_-]{11})", text)
    if m:
        return m.group(1)
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", text):
        return text
    raise ValueError("Could not find a YouTube video id in: " + url_or_id)


def fetch_transcript(video_id, languages=("en", "en-US", "en-GB", "en-IN")):
    from youtube_transcript_api import YouTubeTranscriptApi

    if hasattr(YouTubeTranscriptApi, "fetch"):
        api = YouTubeTranscriptApi()
        try:
            data = api.fetch(video_id, languages=list(languages))
        except Exception:
            first = next(iter(api.list(video_id)))
            data = first.fetch()
        return [{"start": float(s.start), "text": s.text.replace("\n", " ")} for s in data]
    else:
        data = YouTubeTranscriptApi.get_transcript(video_id, languages=list(languages))
        return [{"start": float(d["start"]), "text": d["text"].replace("\n", " ")} for d in data]


def parse_pasted_transcript(text, words_per_minute=150):
    """Accepts '12:35 some text' lines, or YouTube's copied format where the
    timestamp is on its own line and the text is on the next line(s)."""
    ts_pat = re.compile(r"^\s*\[?((?:\d+:)?\d{1,2}:\d{2})\]?\s*[-–]?\s*(.*)$")
    spoken_pat = re.compile(r"^\s*\d+\s+(?:hours?|minutes?|seconds?)(?:,?\s*\d+\s+(?:hours?|minutes?|seconds?))*\s*$", re.I)
    segs, cur = [], None
    for line in text.splitlines():
        line = line.strip()
        if not line or spoken_pat.match(line):
            continue
        m = ts_pat.match(line)
        if m:
            parts = [int(x) for x in m.group(1).split(":")]
            secs = parts[0] * 60 + parts[1] if len(parts) == 2 else parts[0] * 3600 + parts[1] * 60 + parts[2]
            cur = {"start": float(secs), "text": m.group(2).strip()}
            segs.append(cur)
        elif cur is not None:
            cur["text"] = (cur["text"] + " " + line).strip()
    segs = [s for s in segs if s["text"]]
    if len(segs) >= 3:
        return segs
    sentences = re.split(r"(?<=[.!?])\s+", " ".join(text.split()))
    segs, words = [], 0
    for s in sentences:
        if s:
            segs.append({"start": words / words_per_minute * 60, "text": s})
            words += len(s.split())
    return segs


def make_lines(segments, step=15):
    lines, cur = [], None
    for s in segments:
        if cur is None:
            cur = {"start": s["start"], "text": s["text"]}
        elif s["start"] - cur["start"] < step:
            cur["text"] += " " + s["text"]
        else:
            lines.append(cur)
            cur = {"start": s["start"], "text": s["text"]}
    if cur:
        lines.append(cur)
    return lines


def make_blocks(lines, block_sec=30):
    return make_lines(lines, step=block_sec)


def split_windows(lines, window_sec=900):
    windows = {}
    for l in lines:
        windows.setdefault(int(l["start"] // window_sec), []).append(l)
    return [windows[k] for k in sorted(windows)]


def lines_to_text(lines):
    return "\n".join(f"[{int(l['start'])}s] {l['text']}" for l in lines)


_last_call = [0.0]


def detect_models(client):
    global MODELS
    try:
        names = [m.name.replace("models/", "") for m in client.models.list()
                 if "generateContent" in (getattr(m, "supported_actions", None) or [])]
        bad = ("image", "tts", "live", "audio", "embedding", "robotics", "computer",
               "thinking", "exp", "learnlm", "veo", "imagen")
        flash = [n for n in names if "flash" in n and not any(b in n for b in bad)]
        flash.sort(key=lambda n: ("preview" in n, "lite" in n, [-int(x) for x in re.findall(r"\d+", n)]))
        if flash:
            MODELS = flash[:4]
    except Exception as e:
        print("Could not list models:", e)
    print("Using models:", MODELS)


def get_client(api_key=None):
    key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise ValueError("No Gemini API key. Get a free one at https://aistudio.google.com/apikey")
    return genai.Client(api_key=key, http_options=types.HttpOptions(timeout=90_000))


def ask_gemini(client, prompt, schema, temperature=0.2, retries=2):
    global MODELS
    last_error = None
    for round_no in range(2):
        not_found = 0
        tried = list(MODELS)
        for model in tried:
            for attempt in range(retries):
                wait = CALL_GAP_SECONDS - (time.time() - _last_call[0])
                if wait > 0:
                    time.sleep(wait)
                _last_call[0] = time.time()
                t0 = time.time()
                try:
                    resp = client.models.generate_content(
                        model=model, contents=prompt,
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            response_schema=schema, temperature=temperature))
                    out = resp.parsed if resp.parsed is not None else schema.model_validate_json(resp.text)
                    print(f"  [{model}] ok in {time.time() - t0:.0f}s")
                    return out
                except ValidationError as e:
                    last_error = e
                    print(f"  [{model}] bad JSON, retrying ({attempt + 1}/{retries})")
                except Exception as e:
                    last_error = e
                    msg = str(e)
                    short = msg[:120].replace("\n", " ")
                    if any(x in msg for x in ("504", "DEADLINE", "timed out", "Timeout", "timeout")):
                        print(f"  [{model}] too slow ({time.time() - t0:.0f}s), moving it to the end")
                        MODELS = [m for m in MODELS if m != model] + [model]
                        break
                    if "404" in msg or "NOT_FOUND" in msg:
                        print(f"  [{model}] not available, next model")
                        not_found += 1
                        break
                    if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                        if "limit: 0" in msg or "PerDay" in msg or "per day" in msg.lower():
                            print(f"  [{model}] no free quota left, next model ({short})")
                            break
                        m = re.search(r"retry in ([\d.]+)s", msg) or re.search(r"retryDelay.*?(\d+)s", msg)
                        delay = min(float(m.group(1)) + 2, 65) if m else 20
                        print(f"  [{model}] rate limit, waiting {delay:.0f}s")
                        time.sleep(delay)
                    elif any(x in msg for x in ("503", "UNAVAILABLE", "500", "overloaded")):
                        print(f"  [{model}] busy, next model")
                        break
                    else:
                        raise
        if round_no == 0 and not_found == len(tried):
            detect_models(client)
            continue
        break
    raise RuntimeError(f"Gemini failed on all models. Last error: {last_error}")


class Chapter(BaseModel):
    start_seconds: int = Field(description="Start time in seconds, copied from a [..s] marker")
    title: str = Field(description="Short chapter title, 3 to 7 words")


class ChapterList(BaseModel):
    chapters: List[Chapter]


class Takeaways(BaseModel):
    points: List[str] = Field(description="5 to 8 short key takeaways")


class MCQ(BaseModel):
    question: str
    options: List[str] = Field(description="Exactly 4 answer options")
    answer_index: int = Field(description="Index 0-3 of the correct option")
    explanation: str = Field(description="One or two sentences on why the answer is right")

    @model_validator(mode="after")
    def check(self):
        if len(self.options) != 4:
            raise ValueError("need exactly 4 options")
        if not 0 <= self.answer_index <= 3:
            raise ValueError("answer_index must be 0..3")
        return self


class Quiz(BaseModel):
    questions: List[MCQ]


CHAPTER_PROMPT = """You are dividing a video into chapters, like YouTube chapters.
Below is part {i} of {n} of the transcript. Each line starts with [N s] = seconds from video start.

Rules:
- Split this part into topical chapters, roughly one chapter every 3 to 6 minutes.
- start_seconds MUST be copied from one of the [N s] markers shown (just the number N).
- Give each chapter a short, clear title (3 to 7 words). No time in the title.
- List chapters in time order.

TRANSCRIPT PART:
{text}
"""


def _snap(value, valid_starts):
    return int(min(valid_starts, key=lambda s: abs(s - value)))


def merge_boundaries(found, duration, min_gap=45):
    found = sorted(found, key=lambda c: c["start"])
    kept = []
    for c in found:
        if kept and c["start"] - kept[-1]["start"] < min_gap:
            continue
        if kept and c["title"].lower() == kept[-1]["title"].lower():
            continue
        kept.append(dict(c))
    if not kept:
        kept = [{"start": 0, "title": "Full video"}]
    kept[0]["start"] = 0
    for i, c in enumerate(kept):
        c["end"] = kept[i + 1]["start"] if i + 1 < len(kept) else int(duration)
    return kept


def generate_chapters(client, lines, duration, window_min=15, min_gap=45, progress=print):
    windows = split_windows(lines, window_min * 60)
    found = []
    for i, w in enumerate(windows, 1):
        progress(f"  chaptering part {i}/{len(windows)} ...")
        out = ask_gemini(client, CHAPTER_PROMPT.format(i=i, n=len(windows), text=lines_to_text(w)), ChapterList)
        valid = [l["start"] for l in w]
        for ch in out.chapters:
            found.append({"start": _snap(ch.start_seconds, valid), "title": ch.title.strip()})
    return merge_boundaries(found, duration, min_gap)


def build_context(lines, max_chars=120_000):
    text = " ".join(l["text"] for l in lines)
    if len(text) <= max_chars:
        return text
    n = 40
    size, step = max_chars // n, len(text) // n
    return " ... ".join(text[i * step: i * step + size] for i in range(n))


TAKEAWAY_PROMPT = """Read this video transcript and write 5 to 8 key takeaways.
Each takeaway is ONE clear sentence in simple words. Only use facts that are in the transcript.

TRANSCRIPT:
{text}
"""

QUIZ_PROMPT = """Read this video transcript and write {n} multiple-choice questions to test understanding.
Rules:
- Exactly 4 options per question, only one correct.
- Wrong options should be believable, not silly.
- Questions must be answerable from the transcript only.
- Give a short explanation for the correct answer.

TRANSCRIPT:
{text}
"""


def generate_takeaways(client, lines):
    out = ask_gemini(client, TAKEAWAY_PROMPT.format(text=build_context(lines)), Takeaways)
    return [p.strip() for p in out.points]


def generate_quiz(client, lines, n_questions=5):
    out = ask_gemini(client, QUIZ_PROMPT.format(n=n_questions, text=build_context(lines)), Quiz, temperature=0.4)
    quiz = []
    for q in out.questions:
        correct = q.options[q.answer_index]
        opts = q.options[:]
        random.shuffle(opts)
        quiz.append({"question": q.question, "options": opts,
                     "answer_index": opts.index(correct), "explanation": q.explanation})
    return quiz


def _db():
    con = sqlite3.connect(DB_PATH)
    con.execute("CREATE TABLE IF NOT EXISTS cache (video_id TEXT PRIMARY KEY, data TEXT)")
    return con


def load_cache(video_id):
    con = _db()
    row = con.execute("SELECT data FROM cache WHERE video_id=?", (video_id,)).fetchone()
    con.close()
    return json.loads(row[0]) if row else {}


def save_cache(video_id, data):
    con = _db()
    con.execute("INSERT OR REPLACE INTO cache VALUES (?, ?)", (video_id, json.dumps(data)))
    con.commit()
    con.close()


def cached_step(video_id, key, fn, use_cache=True):
    data = load_cache(video_id) if use_cache else {}
    if key in data:
        return data[key]
    value = fn()
    data[key] = value
    save_cache(video_id, data)
    return value


def run_pipeline(url_or_id=None, pasted_text=None, api_key=None, use_cache=True, n_questions=5, progress=print):
    if pasted_text:
        video_id = "pasted-" + hashlib.md5(pasted_text.encode()).hexdigest()[:8]
        segments = parse_pasted_transcript(pasted_text)
    else:
        video_id = extract_video_id(url_or_id)
        progress("Fetching transcript ...")
        segments = fetch_transcript(video_id)
    lines = make_lines(segments)
    duration = int(lines[-1]["start"] + 10)
    client = get_client(api_key)

    progress("Generating chapters ...")
    chapters = cached_step(video_id, "chapters",
                           lambda: generate_chapters(client, lines, duration, progress=progress), use_cache)
    progress("Generating key takeaways ...")
    takeaways = cached_step(video_id, "takeaways", lambda: generate_takeaways(client, lines), use_cache)
    progress("Generating quiz ...")
    quiz = cached_step(video_id, "quiz", lambda: generate_quiz(client, lines, n_questions), use_cache)
    return dict(video_id=video_id, duration=duration, lines=lines,
                chapters=chapters, takeaways=takeaways, quiz=quiz)


def equidistant_boundaries(duration, k):
    return [int(i * duration / k) for i in range(k)]


def texttiling_boundaries(lines, block_sec=45):
    import nltk
    from nltk.tokenize import TextTilingTokenizer
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords", quiet=True)
    try:
        from nltk.corpus import stopwords
        stop = stopwords.words("english")
    except Exception:
        stop = ("a an the and or but if of to in on at for with is are was were be been it this that these those "
                "i you he she we they them his her our your as by from so not no do does did have has had will "
                "would can could just also very there here what which who how when where than then about into").split()
    blocks = make_blocks(lines, block_sec)
    text = "\n\n".join(re.sub(r"\s+", " ", b["text"]) for b in blocks)
    try:
        segments = TextTilingTokenizer(stopwords=stop).tokenize(text)
    except Exception as e:
        print("TextTiling could not run:", e)
        return [0]
    starts, idx = [], 0
    for seg in segments:
        starts.append(int(blocks[min(idx, len(blocks) - 1)]["start"]))
        idx += len([p for p in seg.split("\n\n") if p.strip()])
    return starts


def embedding_boundaries(lines, k, block_sec=30, model_name="all-MiniLM-L6-v2"):
    from sentence_transformers import SentenceTransformer
    blocks = make_blocks(lines, block_sec)
    if len(blocks) < 4 or k <= 1:
        return [0]
    emb = SentenceTransformer(model_name).encode([b["text"] for b in blocks], normalize_embeddings=True)
    w, sims = 3, []
    for i in range(1, len(blocks)):
        left, right = emb[max(0, i - w):i].mean(0), emb[i:i + w].mean(0)
        sims.append((float(left @ right / (np.linalg.norm(left) * np.linalg.norm(right) + 1e-9)), i))
    sims.sort()
    chosen = []
    for _, i in sims:
        if all(abs(i - j) >= 2 for j in chosen):
            chosen.append(i)
        if len(chosen) == k - 1:
            break
    return [0] + sorted(int(blocks[i]["start"]) for i in chosen)


def parse_chapter_timestamps(text):
    out = []
    for m in re.finditer(r"(?m)^\s*(?:[-•*]\s*)?\(?((?:\d+:)?\d{1,2}:\d{2})\)?", text):
        parts = [int(x) for x in m.group(1).split(":")]
        out.append(parts[0] * 60 + parts[1] if len(parts) == 2 else parts[0] * 3600 + parts[1] * 60 + parts[2])
    return sorted(set(out))


def to_labels(boundaries, duration):
    b = sorted(set(int(x) for x in boundaries if x > 0))
    labels, seg, bi = [], 0, 0
    for t in range(int(duration)):
        while bi < len(b) and t >= b[bi]:
            seg += 1
            bi += 1
        labels.append(seg)
    return np.array(labels)


def _window_size(ref):
    n_segments = int(ref[-1]) + 1
    return max(2, int(round(len(ref) / n_segments / 2)))


def pk_score(ref_b, hyp_b, duration):
    ref, hyp = to_labels(ref_b, duration), to_labels(hyp_b, duration)
    k = _window_size(ref)
    return float(np.mean((ref[:-k] == ref[k:]) != (hyp[:-k] == hyp[k:])))


def windowdiff_score(ref_b, hyp_b, duration):
    ref, hyp = to_labels(ref_b, duration), to_labels(hyp_b, duration)
    k = _window_size(ref)
    return float(np.mean((ref[k:] - ref[:-k]) != (hyp[k:] - hyp[:-k])))


def tolerance_f1(ref_b, hyp_b, tol=30):
    ref = [b for b in ref_b if b > 0]
    hyp = [b for b in hyp_b if b > 0]
    if not ref or not hyp:
        return 0.0, 0.0, 0.0
    used, tp = set(), 0
    for h in hyp:
        best = min((r for r in ref if r not in used and abs(r - h) <= tol),
                   key=lambda r: abs(r - h), default=None)
        if best is not None:
            used.add(best)
            tp += 1
    p, r = tp / len(hyp), tp / len(ref)
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f1
