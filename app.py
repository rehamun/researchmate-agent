import os
import streamlit as st
import pandas as pd

from utils.pdf_utils import extract_text_from_pdf, chunk_pages
from utils.rag_utils import build_chunk_index
from utils.agent_utils import (
    analyze_single_paper,
    make_comparison_dataframe,
    generate_literature_review,
    generate_research_gaps,
    answer_question_with_sources
)

st.set_page_config(page_title="ResearchMate Agent", layout="wide")

# Load secrets from Streamlit Cloud if available
try:
    if "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    if "OPENAI_MODEL" in st.secrets:
        os.environ["OPENAI_MODEL"] = st.secrets["OPENAI_MODEL"]
    if "EMBEDDING_MODEL" in st.secrets:
        os.environ["EMBEDDING_MODEL"] = st.secrets["EMBEDDING_MODEL"]
except Exception:
    pass

if "analyses" not in st.session_state:
    st.session_state.analyses = []

if "comparison_df" not in st.session_state:
    st.session_state.comparison_df = pd.DataFrame()

if "indexed_chunks" not in st.session_state:
    st.session_state.indexed_chunks = []

if "processed" not in st.session_state:
    st.session_state.processed = False

if "literature_review_text" not in st.session_state:
    st.session_state.literature_review_text = ""

if "research_gaps_text" not in st.session_state:
    st.session_state.research_gaps_text = ""

st.sidebar.title("Research Setup")

research_topic = st.sidebar.text_area(
    "Research Topic",
    placeholder="Example: Artificial intelligence in higher education"
)

keywords = st.sidebar.text_input(
    "Keywords",
    placeholder="Example: AI, higher education, adaptive learning"
)

review_style = st.sidebar.selectbox(
    "Literature Review Style",
    ["thematic", "chronological", "methodological"]
)

top_k_chunks = st.sidebar.slider(
    "Top retrieved chunks for Q&A",
    min_value=3,
    max_value=10,
    value=6
)

st.title("ResearchMate Agent")
st.caption("AI Agent for Literature Review, Study Comparison, and Research Gap Detection")

uploaded_files = st.file_uploader(
    "Upload research papers (PDF)",
    type=["pdf"],
    accept_multiple_files=True
)

process_btn = st.button("Process Papers")

if process_btn:
    if not research_topic.strip():
        st.error("Please enter a research topic.")
    elif not uploaded_files:
        st.error("Please upload at least one PDF file.")
    else:
        analyses = []
        all_indexed_chunks = []

        progress = st.progress(0)
        status = st.empty()
        total_files = len(uploaded_files)

        for idx, uploaded_file in enumerate(uploaded_files, start=1):
            status.write(f"Processing: {uploaded_file.name}")

            full_text, pages = extract_text_from_pdf(uploaded_file)

            if not full_text.strip():
                st.warning(f"No readable text found in {uploaded_file.name}. Skipping.")
                continue

            analysis = analyze_single_paper(
                full_text=full_text,
                paper_name=uploaded_file.name,
                research_topic=research_topic,
                keywords=keywords
            )
            analyses.append(analysis)

            chunks = chunk_pages(pages, chunk_size=1200, overlap=200)
            indexed_chunks = build_chunk_index(chunks)

            for chunk in indexed_chunks:
                chunk["paper_name"] = uploaded_file.name

            all_indexed_chunks.extend(indexed_chunks)
            progress.progress(idx / total_files)

        if analyses:
            comparison_df = make_comparison_dataframe(analyses)

            st.session_state.analyses = analyses
            st.session_state.comparison_df = comparison_df
            st.session_state.indexed_chunks = all_indexed_chunks
            st.session_state.processed = True

            st.success("Papers processed successfully.")
        else:
            st.error("No papers were processed successfully.")

        status.empty()

if st.session_state.processed:
    st.markdown("---")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Study Analysis",
        "Comparison Table",
        "Literature Review",
        "Research Gaps",
        "Q&A"
    ])

    with tab1:
        st.subheader("Analyzed Papers")
        for idx, analysis in enumerate(st.session_state.analyses, start=1):
            with st.expander(f"Paper {idx}: {analysis.get('paper_name', 'Unknown')}"):
                st.json(analysis)

    with tab2:
        st.subheader("Comparison Table")
        st.dataframe(st.session_state.comparison_df, use_container_width=True)

        csv_data = st.session_state.comparison_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Comparison Table as CSV",
            data=csv_data,
            file_name="comparison_table.csv",
            mime="text/csv"
        )

    with tab3:
        st.subheader("Literature Review Draft")

        if st.button("Generate Literature Review Draft"):
            with st.spinner("Generating literature review..."):
                review_text = generate_literature_review(
                    analyses=st.session_state.analyses,
                    research_topic=research_topic,
                    review_style=review_style
                )
                st.session_state.literature_review_text = review_text

        if st.session_state.literature_review_text:
            st.text_area(
                "Generated Literature Review",
                value=st.session_state.literature_review_text,
                height=400
            )

    with tab4:
        st.subheader("Research Gaps and Future Directions")

        if st.button("Detect Research Gaps"):
            with st.spinner("Analyzing research gaps..."):
                gaps_text = generate_research_gaps(
                    analyses=st.session_state.analyses,
                    research_topic=research_topic
                )
                st.session_state.research_gaps_text = gaps_text

        if st.session_state.research_gaps_text:
            st.text_area(
                "Research Gap Analysis",
                value=st.session_state.research_gaps_text,
                height=350
            )

    with tab5:
        st.subheader("Ask Questions About the Uploaded Papers")
        user_question = st.text_input(
            "Enter your question",
            placeholder="Example: What methodologies were most commonly used across the uploaded studies?"
        )

        if st.button("Answer Question"):
            if not user_question.strip():
                st.warning("Please enter a question.")
            else:
                with st.spinner("Searching papers and generating answer..."):
                    answer, sources = answer_question_with_sources(
                        question=user_question,
                        indexed_chunks=st.session_state.indexed_chunks,
                        analyses=st.session_state.analyses,
                        top_k=top_k_chunks
                    )

                st.markdown("### Answer")
                st.write(answer)

                st.markdown("### Retrieved Source Chunks")
                for source in sources:
                    with st.expander(
                        f"{source.get('paper_name', 'Unknown')} | Page {source.get('page_number', 'N/A')} | Score {source.get('score', 0):.4f}"
                    ):
                        st.write(source["text"])
else:
    st.info("Upload PDFs, fill in the research setup, and click 'Process Papers'.")