# ─────────────────────────────────────────────────────────────
# app.py — Semantic Search Engine
# Streamlit web interface with side-by-side comparison
# ─────────────────────────────────────────────────────────────

# ── Crash prevention — must be before all other imports ───────
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import warnings
warnings.filterwarnings("ignore")

# ── Standard libraries ─────────────────────────────────────────
import time
import numpy as np
import pandas as pd
import faiss
import pickle

# ── ML libraries ───────────────────────────────────────────────
from sentence_transformers import SentenceTransformer

# ── Streamlit ──────────────────────────────────────────────────
# Streamlit turns Python scripts into web apps.
# Every time the user interacts with the app (types a query,
# clicks a button), Streamlit re-runs the entire script from top.
# st.cache_resource() prevents expensive operations (like loading
# the model) from re-running on every interaction.
import streamlit as st

# ─────────────────────────────────────────────────────────────
# PAGE CONFIGURATION
# Must be the very first Streamlit call in the script.
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "Semantic Search Engine",
    page_icon  = "🔍",
    layout     = "wide",       # use full browser width
)

# ─────────────────────────────────────────────────────────────
# FILE PATHS
# All files live in the same folder as app.py.
# Using relative paths keeps the app portable.
# ─────────────────────────────────────────────────────────────
BASE_PATH   = r"C:\Users\ishaa\OneDrive\Documents\Projects\Semantic Search Engine"
DATA_PATH   = os.path.join(BASE_PATH, "ag_news_1000.csv")
EMB_PATH    = os.path.join(BASE_PATH, "embeddings.npy")
MEAN_PATH   = os.path.join(BASE_PATH, "mean_vector.npy")
FAISS_PATH  = os.path.join(BASE_PATH, "faiss_index.bin")
BM25_PATH   = os.path.join(BASE_PATH, "bm25_model.pkl")

# ─────────────────────────────────────────────────────────────
# CACHED RESOURCE LOADING
# @st.cache_resource means: run this function ONCE when the app
# first loads, then reuse the result forever until the app restarts.
# Without this, the model would reload on every single keystroke.
# ─────────────────────────────────────────────────────────────

@st.cache_resource
def load_all_resources():
    """
    Load all files and models exactly once.
    Returns everything the search functions need.
    """
    # Load dataset
    df          = pd.read_csv(DATA_PATH)
    texts       = df["content"].tolist()

    # Load embeddings and mean vector
    embeddings  = np.load(EMB_PATH)
    mean_vector = np.load(MEAN_PATH)

    # Load FAISS index
    index       = faiss.read_index(FAISS_PATH)

    # Load BM25 model
    with open(BM25_PATH, "rb") as f:
        bm25    = pickle.load(f)

    # Load sentence transformer model
    # This is the slowest step — ~3 seconds on first load
    model       = SentenceTransformer("all-MiniLM-L12-v2")

    return df, texts, embeddings, mean_vector, index, bm25, model


# ─────────────────────────────────────────────────────────────
# SEARCH FUNCTIONS
# ─────────────────────────────────────────────────────────────

def semantic_search(query, top_k, df, embeddings,
                    mean_vector, index, model):
    """
    FAISS semantic search.
    Returns results list and time taken in milliseconds.
    """
    t0    = time.time()

    # Embed query, apply mean centering, normalise
    q_vec = model.encode(query, convert_to_numpy=True)
    q_vec = (q_vec - mean_vector).astype(np.float32).reshape(1, -1)
    faiss.normalize_L2(q_vec)

    # Search FAISS index
    scores, indices = index.search(q_vec, top_k)

    elapsed = (time.time() - t0) * 1000   # milliseconds

    results = []
    for r in range(top_k):
        idx = indices[0][r]
        results.append({
            "rank"    : r + 1,
            "score"   : round(float(scores[0][r]), 4),
            "category": df.loc[idx, "category"],
            "text"    : df.loc[idx, "content"],
            "index"   : int(idx),
        })
    return results, round(elapsed, 1)


def keyword_search(query, top_k, df, bm25):
    """
    BM25 keyword search.
    Returns results list and time taken in milliseconds.
    """
    t0      = time.time()

    tokens  = query.lower().split()
    scores  = bm25.get_scores(tokens)
    top_idx = np.argsort(scores)[-top_k:][::-1]

    elapsed = (time.time() - t0) * 1000

    results = []
    for r, idx in enumerate(top_idx):
        results.append({
            "rank"    : r + 1,
            "score"   : round(float(scores[idx]), 4),
            "category": df.loc[idx, "category"],
            "text"    : df.loc[idx, "content"],
            "index"   : int(idx),
        })
    return results, round(elapsed, 1)


# ─────────────────────────────────────────────────────────────
# HELPER — CATEGORY COLOUR
# Maps each category to a colour for visual badges.
# ─────────────────────────────────────────────────────────────

CATEGORY_COLORS = {
    "Business"   : "#1f77b4",
    "Technology" : "#2ca02c",
    "Sports"     : "#ff7f0e",
    "World"      : "#9467bd",
}

def category_badge(category):
    """Returns an HTML coloured badge for a category label."""
    color = CATEGORY_COLORS.get(category, "#888888")
    return (f'<span style="background-color:{color};color:white;'
            f'padding:2px 8px;border-radius:4px;font-size:12px;'
            f'font-weight:bold">{category}</span>')


# ─────────────────────────────────────────────────────────────
# HELPER — RENDER RESULT CARD
# Displays one search result as a clean card.
# ─────────────────────────────────────────────────────────────

def render_result_card(result, system):
    """
    Renders one result as a styled card using st.markdown.
    system = "semantic" or "keyword" — controls score label.
    """
    badge      = category_badge(result["category"])
    score_label= "Similarity" if system == "semantic" else "BM25 score"
    # Show first 300 characters of article text
    preview    = result["text"][:300].replace("\n", " ")
    if len(result["text"]) > 300:
        preview += "..."

    st.markdown(f"""
    <div style="border:1px solid #e0e0e0; border-radius:8px;
                padding:12px; margin-bottom:10px;
                background-color:#fafafa;">
        <div style="display:flex; justify-content:space-between;
                    align-items:center; margin-bottom:6px;">
            <span style="font-weight:bold; font-size:15px;">
                #{result['rank']}
            </span>
            <div>
                {badge}
                <span style="color:#666; font-size:12px; margin-left:8px;">
                    {score_label}: {result['score']}
                </span>
            </div>
        </div>
        <p style="margin:0; font-size:14px; color:#333;
                  line-height:1.5">{preview}</p>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# MAIN APP
# Everything below builds the actual UI.
# Streamlit runs this top-to-bottom on every interaction.
# ─────────────────────────────────────────────────────────────

def main():

    # ── Load all resources (cached after first run) ───────────
    with st.spinner("Loading models and indexes..."):
        (df, texts, embeddings,
         mean_vector, index, bm25, model) = load_all_resources()

    # ── Header ─────────────────────────────────────────────────
    st.title("🔍 Semantic Search Engine")
    st.markdown(
        "Search 1,000 AG News articles using **semantic search** "
        "(meaning) vs **keyword search** (exact words). "
        "Built with `sentence-transformers`, `FAISS`, and `BM25`."
    )
    st.divider()

    # ── Sidebar — settings ────────────────────────────────────
    # st.sidebar puts controls in a collapsible left panel
    with st.sidebar:
        st.header("⚙️ Settings")

        # Number of results slider
        # st.slider(label, min, max, default)
        top_k = st.slider(
            "Number of results", 
            min_value = 1,
            max_value = 10,
            value     = 5,
        )

        # Category filter
        # Lets user restrict search to one category
        all_categories = ["All"] + sorted(df["category"].unique().tolist())
        selected_cat   = st.selectbox(
            "Filter by category",
            options = all_categories,
        )

        st.divider()

        # About section
        st.markdown("### About")
        st.markdown(f"""
        - **Dataset:** AG News (1,000 articles)
        - **Model:** all-MiniLM-L12-v2
        - **Embedding dim:** 384
        - **Categories:** Business, Technology, Sports, World
        """)

        # Category legend
        st.markdown("### Category colours")
        for cat, color in CATEGORY_COLORS.items():
            st.markdown(
                f'<span style="background:{color};color:white;'
                f'padding:2px 8px;border-radius:4px;'
                f'font-size:12px">{cat}</span>',
                unsafe_allow_html=True
            )
            st.write("")   # small spacer

    # ── Search bar ─────────────────────────────────────────────
    query = st.text_input(
        label       = "Enter your search query",
        placeholder = "e.g. stock market crash, football world cup...",
    )

    # ── Example queries ────────────────────────────────────────
    st.markdown("**Try an example:**")

    # st.columns splits the row into equal-width columns
    ex_cols = st.columns(5)
    examples = [
        "NASA space exploration",
        "stock market crash",
        "mobile phone features",
        "Olympic Games Athens",
        "political election results",
    ]

    # Session state stores values that persist across reruns.
    # When user clicks an example button, we store that query
    # in session state so the search bar picks it up.
    if "example_query" not in st.session_state:
        st.session_state.example_query = ""

    for col, example in zip(ex_cols, examples):
        with col:
            if st.button(example, use_container_width=True):
                st.session_state.example_query = example

    # If an example was clicked, use it as the query
    if st.session_state.example_query:
        query = st.session_state.example_query

    st.divider()

    # ── Run search when query is not empty ────────────────────
    if query.strip():

        # Run both search systems
        sem_results, sem_time = semantic_search(
            query, top_k, df, embeddings, mean_vector, index, model
        )
        kw_results,  kw_time  = keyword_search(
            query, top_k, df, bm25
        )

        # Apply category filter if selected
        if selected_cat != "All":
            sem_results = [r for r in sem_results
                           if r["category"] == selected_cat]
            kw_results  = [r for r in kw_results
                           if r["category"] == selected_cat]

        # ── Results header with speed ─────────────────────────
        st.markdown(f"### Results for: *\"{query}\"*")

        speed_col1, speed_col2, speed_col3 = st.columns(3)
        with speed_col1:
            st.metric("Semantic search time", f"{sem_time} ms")
        with speed_col2:
            st.metric("Keyword search time",  f"{kw_time} ms")
        with speed_col3:
            # Speed ratio — how much faster is keyword than semantic
            ratio = round(sem_time / kw_time, 1) if kw_time > 0 else "∞"
            st.metric("Keyword speedup", f"{ratio}×")

        st.divider()

        # ── Side-by-side results ──────────────────────────────
        # Two equal columns — semantic on left, keyword on right
        col_sem, col_kw = st.columns(2)

        with col_sem:
            st.markdown("### 🔵 Semantic Search (FAISS)")
            st.caption(
                "Finds articles by *meaning*. "
                "Works even if your words don't appear in the article."
            )
            if sem_results:
                for r in sem_results:
                    render_result_card(r, "semantic")
            else:
                st.info("No results for this category filter.")

        with col_kw:
            st.markdown("### 🟡 Keyword Search (BM25)")
            st.caption(
                "Finds articles by *exact words*. "
                "Fast and reliable when specific terms are present."
            )
            if kw_results:
                for r in kw_results:
                    render_result_card(r, "keyword")
            else:
                st.info("No results for this category filter.")

        # ── Analytics section ─────────────────────────────────
        st.divider()
        st.markdown("### 📊 Search Analytics")

        a1, a2, a3 = st.columns(3)

        # ── Metric 1: category distribution ──────────────────
        with a1:
            st.markdown("**Category breakdown — Semantic**")
            sem_cats = pd.Series(
                [r["category"] for r in sem_results]
            ).value_counts()
            # st.bar_chart takes a pandas Series directly
            st.bar_chart(sem_cats)

        with a2:
            st.markdown("**Category breakdown — Keyword**")
            kw_cats = pd.Series(
                [r["category"] for r in kw_results]
            ).value_counts()
            st.bar_chart(kw_cats)

        # ── Metric 2: result overlap ───────────────────────────
        with a3:
            st.markdown("**Result overlap**")
            sem_idx = set(r["index"] for r in sem_results)
            kw_idx  = set(r["index"] for r in kw_results)
            overlap = sem_idx & kw_idx   # intersection

            overlap_pct = (len(overlap) / top_k * 100
                           if top_k > 0 else 0)

            st.metric(
                "Shared articles",
                f"{len(overlap)} / {top_k}",
                help="Articles appearing in both result lists"
            )
            st.metric(
                "Agreement",
                f"{overlap_pct:.0f}%",
                help="How much both systems agree"
            )
            st.metric(
                "Semantic-only",
                len(sem_idx - kw_idx),
                help="Articles only semantic found"
            )
            st.metric(
                "Keyword-only",
                len(kw_idx - sem_idx),
                help="Articles only keyword found"
            )

        # ── Score table ────────────────────────────────────────
        st.divider()
        st.markdown("### 📋 Full Results Table")

        # Build a comparison dataframe
        # Shows both systems side by side in a table
        table_rows = []
        max_r = max(len(sem_results), len(kw_results))

        for i in range(min(top_k, max_r)):
            row = {"Rank": i + 1}

            if i < len(sem_results):
                s = sem_results[i]
                row["Sem Category"] = s["category"]
                row["Sem Score"]    = s["score"]
                row["Sem Preview"]  = s["text"][:80] + "..."
            else:
                row["Sem Category"] = "-"
                row["Sem Score"]    = "-"
                row["Sem Preview"]  = "-"

            if i < len(kw_results):
                k = kw_results[i]
                row["KW Category"]  = k["category"]
                row["KW Score"]     = k["score"]
                row["KW Preview"]   = k["text"][:80] + "..."
            else:
                row["KW Category"]  = "-"
                row["KW Score"]     = "-"
                row["KW Preview"]   = "-"

            table_rows.append(row)

        st.dataframe(
            pd.DataFrame(table_rows),
            use_container_width = True,
            hide_index          = True,
        )

    else:
        # ── Empty state — shown before any query is entered ───
        st.markdown("### 👆 Enter a query above to begin searching")
        st.markdown("""
        **What this app demonstrates:**
        - 🔵 **Semantic search** finds results by *meaning* —
          even when your words don't appear in the article
        - 🟡 **Keyword search** finds results by *exact word match* —
          fast and precise for specific terms
        - 📊 **Analytics** show how much both systems agree
          and where they differ

        **Interesting queries to try:**
        - `automobile industry` — semantic finds car articles
          even though "automobile" isn't in them
        - `NASA space exploration` — both systems agree strongly
        - `deadly armed conflict` — semantic understands meaning,
          keyword matches words
        - `Apple` — see how each system handles ambiguity
        """)


# ── Entry point ────────────────────────────────────────────────
# This is standard Python — only run main() if this file is
# executed directly (not imported as a module)
if __name__ == "__main__":
    main()