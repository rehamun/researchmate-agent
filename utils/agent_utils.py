import json
import pandas as pd
from .llm_utils import call_llm_json, call_llm_text
from .pdf_utils import build_paper_context
from .rag_utils import retrieve_top_chunks


def analyze_single_paper(full_text, paper_name, research_topic, keywords):
    context = build_paper_context(full_text)

    system_prompt = """
You are an expert academic research assistant.

Your job is to analyze a research paper and extract structured information.
Only use information that is supported by the provided paper text.
If something is missing, return "Not clearly stated".

Return valid JSON only.
"""

    user_prompt = f"""
Research topic:
{research_topic}

Keywords:
{keywords}

Paper file name:
{paper_name}

Paper content:
{context}

Return a JSON object with exactly these keys:
{{
  "paper_name": "",
  "title": "",
  "authors": "",
  "year": "",
  "objective": "",
  "research_problem": "",
  "methodology": "",
  "sample": "",
  "instrument": "",
  "context_or_location": "",
  "key_findings": "",
  "recommendations": "",
  "limitations": "",
  "keywords_from_paper": "",
  "relevance_to_topic": "",
  "short_summary": ""
}}
"""

    return call_llm_json(system_prompt, user_prompt)


def make_comparison_dataframe(analyses):
    rows = []
    for item in analyses:
        rows.append({
            "Paper": item.get("paper_name", ""),
            "Title": item.get("title", ""),
            "Year": item.get("year", ""),
            "Objective": item.get("objective", ""),
            "Methodology": item.get("methodology", ""),
            "Sample": item.get("sample", ""),
            "Key Findings": item.get("key_findings", ""),
            "Relevance": item.get("relevance_to_topic", "")
        })

    return pd.DataFrame(rows)


def generate_literature_review(analyses, research_topic, review_style="thematic"):
    system_prompt = """
You are an academic writing assistant.

Write a literature review draft in formal academic English.
Use only the supplied study summaries.
Do not fabricate citations or facts.
If author/year is unclear, refer to the study by its title or file name.
Organize the review clearly and coherently.
"""

    studies_json = json.dumps(analyses, ensure_ascii=False, indent=2)

    user_prompt = f"""
Research topic:
{research_topic}

Preferred review style:
{review_style}

Analyzed studies:
{studies_json}

Write:
1. An introductory paragraph for the previous studies section.
2. A comparative literature review draft in academic English.
3. A short concluding paragraph summarizing overall patterns.

The output should be plain text, not JSON.
"""

    return call_llm_text(system_prompt, user_prompt, temperature=0.3)


def generate_research_gaps(analyses, research_topic):
    system_prompt = """
You are an expert research gap analyst.

Based only on the supplied studies, identify realistic research gaps.
Do not invent unsupported claims.
Write in clear academic English.
"""

    studies_json = json.dumps(analyses, ensure_ascii=False, indent=2)

    user_prompt = f"""
Research topic:
{research_topic}

Analyzed studies:
{studies_json}

Please provide:
1. Main patterns across the studies
2. Main differences across the studies
3. Possible research gaps
4. Suggested future research directions

Use clear headings and bullet points.
"""

    return call_llm_text(system_prompt, user_prompt, temperature=0.2)


def answer_question_with_sources(question, indexed_chunks, analyses, top_k=6):
    top_chunks = retrieve_top_chunks(question, indexed_chunks, top_k=top_k)

    sources_text = []
    for chunk in top_chunks:
        source = f"[Source: {chunk.get('paper_name', 'Unknown')} | Page: {chunk.get('page_number', 'N/A')}]"
        sources_text.append(source + "\n" + chunk["text"])

    joined_sources = "\n\n".join(sources_text)
    analyses_json = json.dumps(analyses, ensure_ascii=False, indent=2)

    system_prompt = """
You are a careful academic research assistant.

Answer the user's question using only the provided sources and analyzed study summaries.
If the answer is not supported, say so clearly.
At the end, include a short section called "Sources Used".
"""

    user_prompt = f"""
User question:
{question}

Analyzed studies:
{analyses_json}

Retrieved source chunks:
{joined_sources}

Write the answer in formal English.
"""

    answer = call_llm_text(system_prompt, user_prompt, temperature=0.2)
    return answer, top_chunks