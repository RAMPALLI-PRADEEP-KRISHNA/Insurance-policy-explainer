"""Streamlit web application for Insurance Policy Explainer."""

import hashlib
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple
import streamlit as st

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import config
from src.generation.answer_generator import answer_generator, AnswerGenerator
from src.ingestion.chunker import chunk_document
from src.ingestion.index_store import index_store, IndexStore
from src.ingestion.pdf_extractor import process_pdf
from src.models.data_models import (
    Chunk,
    Document,
    DocumentQuality,
    DocumentStatus,
    EvidenceStatus,
    GroundedAnswer,
    RedFlag,
    RedFlagReport,
    RiskCategory,
    RiskSeverity,
    ComparisonCategory,
    ComparisonCellStatus,
    ComparisonFact,
    ComparisonRow,
    ComparisonReport,
)
from src.comparison.comparison_engine import (
    comparison_engine,
    ComparisonEngine,
    compare_policies,
)
from src.risk_scanner.red_flag_scanner import (
    red_flag_scanner,
    RedFlagScanner,
    scan_policy,
)
from src.retrieval.evidence_validator import evidence_validator, EvidenceValidator
from src.retrieval.retriever import retriever, Retriever
from src.utils.exceptions import (
    InsuranceExplainerError,
    MissingAPIKeyError,
    ModelUnavailableError,
    GenerationTimeoutError,
    GeminiServiceUnavailableError,
    MistralRateLimitError,
    IndexNotFoundError,
)
from src.utils.timing import timed
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

class ProcessResult(tuple):
    def __new__(cls, document, chunks, cache_hit, elapsed_seconds):
        return super().__new__(cls, (document, chunks))

    def __init__(self, document, chunks, cache_hit, elapsed_seconds):
        self.cache_hit = cache_hit
        self.elapsed_seconds = elapsed_seconds


@st.cache_resource(show_spinner="Loading language model (first run only)")
def preload_embedder():
    from src.ingestion.embedder import embedder
    embedder.load()
    return embedder


# =====================================================================
# SERVICE FUNCTIONS (Decoupled for clean testability and UI reuse)
# =====================================================================

def process_uploaded_policy(
    file_bytes: bytes,
    filename: str,
    indexes_store: Optional[IndexStore] = None,
    progress_callback=None,
) -> ProcessResult:
    import time
    t0 = time.time()
    store = indexes_store or index_store
    
    file_hash = hashlib.sha256(file_bytes).hexdigest()
    
    # 1. Fast path cache check
    if store.is_fresh(file_hash, file_hash):
        try:
            _, chunks, meta = store.load_index(file_hash)
            page_count = max((c.page_number for c in chunks), default=1) if chunks else 1
            from datetime import datetime, timezone
            doc = Document(
                doc_id=file_hash,
                filename=meta.filename,
                file_hash=meta.file_hash,
                page_count=page_count,
                chunk_count=meta.chunk_count,
                status=DocumentStatus.READY,
                quality=DocumentQuality.GOOD
            )
            elapsed = time.time() - t0
            return ProcessResult(doc, chunks, True, elapsed)
        except Exception as e:
            logger.warning("Cache valid but failed to load, falling back to full rebuild: %s", e)

    # 2. Full pipeline
    temp_dir = Path(tempfile.gettempdir()) / "insurance_explainer_uploads"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / filename
    temp_path.write_bytes(file_bytes)
    
    try:
        with timed("PDF extraction"):
            document = process_pdf(temp_path, save_meta=True)
            # Override doc_id to match hash for deterministic index caching
            document.doc_id = file_hash
            document.file_hash = file_hash
        if document.status == DocumentStatus.FAILED or document.quality == DocumentQuality.FAILED:
            return ProcessResult(document, [], False, time.time() - t0)
            
        with timed("chunking"):
            chunks = chunk_document(document)
            
        with timed("FAISS build + persist"):
            store.build_index(document, chunks=chunks, force_rebuild=True, progress_callback=progress_callback)
            
        elapsed = time.time() - t0
        return ProcessResult(document, chunks, False, elapsed)
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass


def answer_policy_question(
    doc_id: str,
    query: str,
    language: str = "en",
    retriever_instance: Optional[Retriever] = None,
    validator_instance: Optional[EvidenceValidator] = None,
    generator_instance: Optional[AnswerGenerator] = None,
) -> GroundedAnswer:
    """
    Execute complete grounded question answering pipeline:
    Retriever -> EvidenceValidator -> AnswerGenerator -> GroundedAnswer

    Args:
        doc_id: Active policy document ID.
        query: User question in English or Tamil.
        language: 'en' for English, 'ta' for Tamil.
        retriever_instance: Optional custom Retriever.
        validator_instance: Optional custom EvidenceValidator.
        generator_instance: Optional custom AnswerGenerator.

    Returns:
        GroundedAnswer containing answer text, citations, and evidence status.
    """
    r = retriever_instance or retriever
    v = validator_instance or evidence_validator
    g = generator_instance or answer_generator

    # 1. Hybrid dense retrieval with numeric/keyword boosting
    results = r.retrieve(doc_id=doc_id, query=query, top_k=config.RETRIEVAL_FINAL_K)

    # 2. Four-state evidence validation
    with timed("evidence validation"):
        evidence = v.validate(query=query, results=results, doc_id=doc_id)

    # 3. Grounded answer generation via Gemini (or controlled refusal on INSUFFICIENT)
    with timed("LLM request"):
        return g.generate_answer(query=query, evidence=evidence, language=language)


def scan_policy_red_flags(
    document: Document,
    chunks: Optional[List[Chunk]] = None,
    scanner_instance: Optional[RedFlagScanner] = None,
) -> RedFlagReport:
    """
    Execute evidence-grounded red-flag scan on an insurance policy.
    Identifies high-impact restrictions (waiting periods, exclusions, co-pay, room rent limits).

    Args:
        document: Ingested policy Document model.
        chunks: Optional list of Chunks (resolved automatically if None).
        scanner_instance: Optional RedFlagScanner instance.

    Returns:
        RedFlagReport with detected clauses, review priorities, and page citations.
    """
    scanner = scanner_instance or red_flag_scanner
    return scanner.scan(document=document, chunks=chunks)


def compare_uploaded_policies(
    doc_a: Document,
    doc_b: Document,
    chunks_a: Optional[List[Chunk]] = None,
    chunks_b: Optional[List[Chunk]] = None,
    engine_instance: Optional[ComparisonEngine] = None,
) -> ComparisonReport:
    """
    Execute side-by-side policy comparison between Policy A and Policy B.

    Args:
        doc_a: Policy A Document.
        doc_b: Policy B Document.
        chunks_a: Optional pre-loaded chunks for Policy A.
        chunks_b: Optional pre-loaded chunks for Policy B.
        engine_instance: Optional ComparisonEngine instance.

    Returns:
        ComparisonReport with side-by-side rows, facts, and difference descriptions.
    """
    engine = engine_instance or comparison_engine
    return engine.compare(doc_a=doc_a, doc_b=doc_b, chunks_a=chunks_a, chunks_b=chunks_b)


# =====================================================================
# STREAMLIT UI COMPONENTS
# =====================================================================

def _format_status_badge(status: ComparisonCellStatus) -> str:
    """Format visual status badge preserving uncertainty."""
    if status == ComparisonCellStatus.FOUND:
        return "🟢 **FOUND**"
    elif status == ComparisonCellStatus.AMBIGUOUS:
        return "🟡 **AMBIGUOUS**"
    elif status == ComparisonCellStatus.NOT_MENTIONED:
        return "⚪ **NOT MENTIONED**"
    elif status == ComparisonCellStatus.EXPLICITLY_ABSENT:
        return "🔵 **EXPLICITLY ABSENT**"
    return str(status)


def render_comparison_report(report: ComparisonReport) -> None:
    """Render structured two-policy comparison matrix with isolated citations."""
    st.markdown("### ⚖️ Side-by-Side Policy Comparison Matrix")
    st.caption("Objective comparison of documented policy terms. Preserves uncertainty and avoids subjective ranking.")

    # Headers for Policy A and Policy B
    col_a, col_b = st.columns(2)
    with col_a:
        st.info(f"**Policy A:** `{report.filename_a}`\n\nDoc ID: `{report.doc_id_a[:8]}...`")
    with col_b:
        st.info(f"**Policy B:** `{report.filename_b}`\n\nDoc ID: `{report.doc_id_b[:8]}...`")

    st.markdown("---")

    for idx, row in enumerate(report.rows, start=1):
        st.markdown(f"#### {idx}. {row.label}")

        col_fa, col_fb, col_diff = st.columns([1, 1, 1.2])

        # Policy A Fact Cell
        with col_fa:
            st.caption(f"**Policy A ({report.filename_a}):**")
            badge_a = _format_status_badge(row.policy_a_fact.status)
            st.markdown(f"{badge_a}")
            if row.policy_a_fact.original_value:
                st.markdown(f"**Value:** `{row.policy_a_fact.original_value}`")
            if row.policy_a_fact.conditions:
                st.caption(f"Conditions: *{row.policy_a_fact.conditions}*")
            st.write(row.policy_a_fact.description)

            if row.policy_a_fact.citations:
                with st.expander(f"📌 Policy A Source (Page {row.policy_a_fact.citations[0].page_number})"):
                    st.caption(f"Section: {row.policy_a_fact.citations[0].section or 'N/A'}")
                    st.info(f'"{row.policy_a_fact.citations[0].source_snippet}"')

        # Policy B Fact Cell
        with col_fb:
            st.caption(f"**Policy B ({report.filename_b}):**")
            badge_b = _format_status_badge(row.policy_b_fact.status)
            st.markdown(f"{badge_b}")
            if row.policy_b_fact.original_value:
                st.markdown(f"**Value:** `{row.policy_b_fact.original_value}`")
            if row.policy_b_fact.conditions:
                st.caption(f"Conditions: *{row.policy_b_fact.conditions}*")
            st.write(row.policy_b_fact.description)

            if row.policy_b_fact.citations:
                with st.expander(f"📌 Policy B Source (Page {row.policy_b_fact.citations[0].page_number})"):
                    st.caption(f"Section: {row.policy_b_fact.citations[0].section or 'N/A'}")
                    st.info(f'"{row.policy_b_fact.citations[0].source_snippet}"')

        # Difference Cell
        with col_diff:
            st.caption("**Comparison / Difference:**")
            st.markdown(f"💡 *{row.difference}*")

        st.markdown("---")

    st.caption(f"⚠️ **Disclaimer:** {report.disclaimer}")

def render_red_flags(report: RedFlagReport) -> None:
    """Render categorized red flags with review priority badges and citations."""
    if report.total_flags == 0:
        st.info("No significant restrictive clauses or red flags were identified in this policy.")
        return

    # Metrics summary row
    high_count = sum(1 for f in report.findings if f.severity == RiskSeverity.HIGH)
    med_count = sum(1 for f in report.findings if f.severity == RiskSeverity.MEDIUM)
    low_count = sum(1 for f in report.findings if f.severity == RiskSeverity.LOW)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Flags Found", report.total_flags)
    c2.metric("High Review Priority", high_count)
    c3.metric("Medium Review Priority", med_count)
    c4.metric("Low Review Priority", low_count)

    st.markdown("---")

    # Filter controls
    col_filter, col_info = st.columns([1, 2])
    with col_filter:
        sev_filter = st.selectbox(
            "Filter by review priority:",
            ["All Severities", "HIGH", "MEDIUM", "LOW"],
            index=0,
        )

    filtered_findings = report.findings
    if sev_filter != "All Severities":
        filtered_findings = [f for f in filtered_findings if f.severity.value == sev_filter]

    if not filtered_findings:
        st.write(f"No flags found with severity '{sev_filter}'.")
    else:
        for idx, flag in enumerate(filtered_findings, start=1):
            if flag.severity == RiskSeverity.HIGH:
                sev_badge = "🔴 **HIGH REVIEW PRIORITY**"
            elif flag.severity == RiskSeverity.MEDIUM:
                sev_badge = "🟡 **MEDIUM REVIEW PRIORITY**"
            else:
                sev_badge = "🔵 **LOW REVIEW PRIORITY**"

            cat_label = flag.category.value.replace("_", " ").title()

            with st.container():
                st.markdown(f"#### {idx}. {flag.title}")
                st.markdown(f"{sev_badge} &nbsp;|&nbsp; **Category:** `{cat_label}` &nbsp;|&nbsp; **Page:** {flag.page_number}")
                st.write(flag.description)

                with st.expander(f"📌 View Source Excerpt & Citation (Page {flag.page_number})"):
                    st.caption(f"**Section:** {flag.section or 'N/A'} | **Clause:** {flag.heading or 'N/A'}")
                    st.info(f'"{flag.source_snippet}"')
                st.markdown("---")

    st.caption(f"⚠️ **Disclaimer:** {report.disclaimer}")


def render_citations(citations: list) -> None:
    """Render authoritative page-level citations in clean expandable UI."""
    if not citations:
        return

    st.markdown("##### 📌 Authoritative Sources & Citations")
    for idx, c in enumerate(citations, start=1):
        pages_str = ", ".join(str(p) for p in c.page_numbers)
        header_title = f"Page {pages_str} — {c.heading or c.section or 'Policy Clause'}"

        with st.expander(header_title, expanded=(idx == 1)):
            col1, col2 = st.columns([1, 2])
            with col1:
                st.caption(f"**Document:** {c.filename}")
                st.caption(f"**Page(s):** {pages_str}")
                if c.section:
                    st.caption(f"**Section:** {c.section}")
                if c.heading and c.heading != c.section:
                    st.caption(f"**Clause:** {c.heading}")
            with col2:
                st.markdown("**Exact Policy Excerpt:**")
                st.info(f'"{c.source_snippet}"')


def render_evidence_status_badge(status: EvidenceStatus, confidence: float) -> None:
    """Render visually distinct evidence state indicator."""
    if status == EvidenceStatus.SUFFICIENT:
        st.success(f"**Evidence Status: SUFFICIENT** (Retrieval Confidence: {confidence:.0%})")
    elif status == EvidenceStatus.MARGINAL:
        st.warning(
            f"**Evidence Status: MARGINAL** (Retrieval Confidence: {confidence:.0%})\n\n"
            "⚠️ *The policy evidence is related but incomplete for your specific query. The answer below is cautious and limited to verified facts.*"
        )
    elif status == EvidenceStatus.INSUFFICIENT:
        st.info(
            "**Evidence Status: INSUFFICIENT**\n\n"
            "ℹ️ *The uploaded policy document does not contain sufficient verified evidence to answer this question. No coverage assumptions were made.*"
        )
    elif status == EvidenceStatus.CONFLICTING:
        st.error(
            "**Evidence Status: CONFLICTING**\n\n"
            "🚨 *Contradictory policy terms were detected across retrieved clauses. Please verify with the insurer before relying on these figures.*"
        )


def main() -> None:
    """Main Streamlit application entry point."""
    st.set_page_config(
        page_title="Insurance Policy Explainer",
        page_icon="🛡️",
        layout="wide",
    )

    # Initialize Session State
    if "active_doc" not in st.session_state:
        st.session_state.active_doc = None
    if "active_chunks" not in st.session_state:
        st.session_state.active_chunks = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "uploaded_file_hash" not in st.session_state:
        st.session_state.uploaded_file_hash = None
    if "red_flag_report" not in st.session_state:
        st.session_state.red_flag_report = None
    if "comparison_report" not in st.session_state:
        st.session_state.comparison_report = None
    if "policy_b_doc" not in st.session_state:
        st.session_state.policy_b_doc = None
    if "policy_b_chunks" not in st.session_state:
        st.session_state.policy_b_chunks = []

    # Sidebar: Policy Upload & Configuration
    with st.sidebar:
        st.title("🛡️ Policy Setup")
        st.markdown("Upload your health or motor insurance policy PDF to start asking grounded questions.")

        uploaded_file = st.file_uploader(
            "Upload Policy PDF",
            type=["pdf"],
            help="Upload an insurance policy document (text or scanned PDF).",
        )

        # Handle File Upload & Processing
        if uploaded_file is not None and hashlib.sha256(uploaded_file.getvalue()).hexdigest() != st.session_state.uploaded_file_hash:
            import time
            t0 = time.time()
            progress_bar = st.progress(0, text="Starting policy ingestion...")
            
            def on_progress(processed, total):
                progress_bar.progress(processed / total, text=f"Embedding chunks: {processed}/{total} (Elapsed: {time.time()-t0:.1f}s)")

            with st.spinner("Processing policy document and building FAISS index..."):
                try:
                    file_bytes = uploaded_file.read()
                    res = process_uploaded_policy(file_bytes, uploaded_file.name, progress_callback=on_progress)
                    doc, chunks = res

                    # Update active document in session state (enforces strict document isolation)
                    st.session_state.active_doc = doc
                    st.session_state.active_chunks = chunks
                    st.session_state.uploaded_file_hash = hashlib.sha256(uploaded_file.getvalue()).hexdigest()
                    st.session_state.chat_history = []  # Reset chat history for new policy
                    
                    progress_bar.empty()
                    if res.cache_hit:
                        st.success(f"Loaded existing index in {res.elapsed_seconds:.2f} s")
                    st.session_state.red_flag_report = None  # Reset report for new policy
                    st.session_state.comparison_report = None
                    st.session_state.policy_b_doc = None
                    st.session_state.policy_b_chunks = []

                    st.toast("Policy processed and indexed successfully!", icon="✅")
                except Exception as e:
                    logger.error("Error processing uploaded PDF: %s", e)
                    st.error(f"Failed to process policy: {str(e)}")

        # Language Selection
        st.divider()
        st.subheader("🌐 Answer Language")
        lang_choice = st.radio(
            "Select language:",
            ["English", "தமிழ் (Tamil)"],
            index=0,
            horizontal=True,
        )
        selected_language = "ta" if "Tamil" in lang_choice else "en"

        # Active Policy Metadata Card
        st.divider()
        st.subheader("📄 Active Policy Information")
        active_doc: Optional[Document] = st.session_state.active_doc

        if active_doc is not None:
            st.write(f"**File:** `{active_doc.filename}`")
            st.write(f"**Pages:** {active_doc.page_count}")
            st.write(f"**Chunks:** {len(st.session_state.active_chunks)}")

            # Quality Display
            if active_doc.quality == DocumentQuality.GOOD:
                st.success("Quality: GOOD (Native text layer verified)")
            elif active_doc.quality == DocumentQuality.WARNING:
                st.warning("Quality: WARNING (Some pages have low text yield)")
            elif active_doc.quality == DocumentQuality.OCR_REQUIRED:
                st.info("Quality: OCR REQUIRED (Scanned pages detected)")
            else:
                st.error(f"Quality: {active_doc.quality.value} (Extraction unreliable)")

            if active_doc.ocr_page_ids:
                st.caption(f"OCR applied on pages: {active_doc.ocr_page_ids}")

            if st.button("🗑️ Clear Active Policy"):
                st.session_state.active_doc = None
                st.session_state.active_chunks = []
                st.session_state.chat_history = []
                st.session_state.uploaded_file_hash = None
                st.session_state.red_flag_report = None
                st.session_state.comparison_report = None
                st.session_state.policy_b_doc = None
                st.session_state.policy_b_chunks = []
                st.rerun()
        else:
            st.info("No policy active. Please upload a PDF above.")

    # Main Area: Chat & Question Interface
    st.header("📋 Insurance Policy Explainer")
    st.caption("Ask questions in English or Tamil, or scan for important policy restrictions and exclusions.")

    if active_doc is None:
        st.info("👈 Please upload an insurance policy PDF in the sidebar to get started.")
        st.markdown(
            """
            ### What this application does:
            - **Strict Grounding:** Answers are generated exclusively from your uploaded policy clauses.
            - **Zero Hallucination:** Never guesses or assumes unwritten terms.
            - **Page Citations:** Every claim links directly to the source page and clause.
            - **Bilingual:** Ask and receive answers in either English or Tamil.
            - **Red-Flag Scanner:** Identifies waiting periods, exclusions, co-payments, room-rent limits, and sub-limits.
            """
        )
        return

    # Check if document quality allows querying
    if active_doc.quality in {DocumentQuality.FAILED, DocumentQuality.POOR}:
        st.error(
            "⚠️ The uploaded document text could not be extracted with sufficient reliability. "
            "Analysis is disabled to prevent inaccurate or misleading insurance information."
        )
        return

    # Tabs for Question Answering, Red-Flag Scanner, and Policy Comparison
    tab_chat, tab_scanner, tab_compare = st.tabs([
        "💬 Ask Questions",
        "🚩 Policy Red-Flag Scanner",
        "⚖️ Compare Policies",
    ])

    with tab_chat:
        # Render Chat History
        for chat_item in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(chat_item["query"])

            with st.chat_message("assistant"):
                grounded_ans: GroundedAnswer = chat_item["answer"]
                if getattr(grounded_ans, "llm_unavailable", False):
                    reason = getattr(grounded_ans, "llm_unavailable_reason", "Timeout or quota exceeded")
                    st.warning(f"Language model unavailable ({reason})")
                    ans_text = grounded_ans.answer.strip()
                    if not ans_text:
                        ans_text = "\n\n".join(f"- {c.source_snippet}" for c in grounded_ans.citations)
                    st.markdown(f"**Retrieved Sources:**\n\n{ans_text}")
                    render_citations(grounded_ans.citations)
                else:
                    render_evidence_status_badge(grounded_ans.evidence_status, grounded_ans.confidence)
                    st.markdown(grounded_ans.answer)
                    model = getattr(grounded_ans, "generation_model", "")
                    if model and "gemini" in model.lower() and config.LLM_PROVIDER == "mistral":
                        st.caption("Answered by Gemini (Mistral was unavailable)")
                    render_citations(grounded_ans.citations)

        # Chat Input Box
        query = st.chat_input("Ask a question about this policy (e.g., 'What is the waiting period for pre-existing diseases?')...")

        if query:
            # Display user message immediately
            with st.chat_message("user"):
                st.write(query)

            # Process question through RAG pipeline
            with st.chat_message("assistant"):
                with st.spinner("Searching policy clauses, validating evidence, and generating grounded answer..."):
                    try:
                        with timed("Total time per question"):
                            grounded_ans = answer_policy_question(
                            doc_id=active_doc.doc_id,
                            query=query,
                            language=selected_language,
                        )

                        if getattr(grounded_ans, "llm_unavailable", False):
                            reason = getattr(grounded_ans, "llm_unavailable_reason", "Timeout or quota exceeded")
                            st.warning(f"Language model unavailable ({reason})")
                            ans_text = grounded_ans.answer.strip()
                            if not ans_text:
                                ans_text = "\n\n".join(f"- {c.source_snippet}" for c in grounded_ans.citations)
                            st.markdown(f"**Retrieved Sources:**\n\n{ans_text}")
                            render_citations(grounded_ans.citations)
                        else:
                            render_evidence_status_badge(grounded_ans.evidence_status, grounded_ans.confidence)
                            st.markdown(grounded_ans.answer)
                            model = getattr(grounded_ans, "generation_model", "")
                            if model and "gemini" in model.lower() and config.LLM_PROVIDER == "mistral":
                                st.caption("Answered by Gemini (Mistral was unavailable)")
                            render_citations(grounded_ans.citations)

                        # Append to session state chat history
                        st.session_state.chat_history.append({
                            "query": query,
                            "answer": grounded_ans,
                            "language": selected_language,
                        })

                    except MissingAPIKeyError:
                        st.warning(
                            "⚠️ **Gemini API key is not configured.**\n\n"
                            "To generate AI answers, please configure `MISTRAL_API_KEY` (and optionally `GEMINI_API_KEY`) in your `.env` file. "
                            "Retrieval and evidence validation are fully operational."
                        )
                    except ModelUnavailableError as e:
                        st.error(f"Configured Gemini model is unavailable: {str(e)}")
                    except GenerationTimeoutError:
                        st.error("The answer generation request timed out. Please try asking again.")
                    except GeminiServiceUnavailableError:
                        st.warning(
                            "⏳ **Gemini is temporarily unavailable.**\n\n"
                            "The AI model is currently experiencing high demand. "
                            "Your policy was processed and the relevant clauses were retrieved successfully, "
                            "but an AI-generated answer could not be produced right now.\n\n"
                            "**Please try again in a moment.**"
                        )
                    except MistralRateLimitError as e:
                        st.warning(f"? **Mistral is temporarily unavailable.**\n\n{str(e)}")
                    except IndexNotFoundError:
                        st.error("The policy index could not be located. Please re-upload the document.")
                    except InsuranceExplainerError as e:
                        st.error(f"Application error: {str(e)}")
                    except Exception as e:
                        logger.exception("Unexpected error in chat answering: %s", e)
                        st.error("An unexpected error occurred while answering your question. Please try again.")

    with tab_scanner:
        st.subheader("🚩 Evidence-Grounded Red-Flag Scanner")
        st.caption("Identifies policy exclusions, waiting periods, room-rent limits, and co-payment obligations.")

        col_btn, col_help = st.columns([1, 2])
        with col_btn:
            scan_clicked = st.button("🔍 Scan Policy for Red Flags", type="primary")

        if scan_clicked or st.session_state.red_flag_report is not None:
            if scan_clicked or st.session_state.red_flag_report is None:
                with st.spinner("Scanning policy clauses and analyzing restrictive terms..."):
                    report = scan_policy_red_flags(
                        document=active_doc,
                        chunks=st.session_state.active_chunks,
                    )
                    st.session_state.red_flag_report = report

            render_red_flags(st.session_state.red_flag_report)
        else:
            st.info("Click '🔍 Scan Policy for Red Flags' to analyze this policy for important clauses and restrictions.")

    with tab_compare:
        st.subheader("⚖️ Two-Policy Comparison Engine")
        st.caption("Side-by-side grounded comparison across 12 standardized insurance dimensions.")

        st.info(f"**Policy A (Active Policy):** `{active_doc.filename}` ({active_doc.page_count} pages)")

        uploaded_file_b = st.file_uploader(
            "Upload Policy B PDF to Compare:",
            type=["pdf"],
            key="compare_policy_b_uploader",
            help="Upload a second insurance policy to compare against Policy A.",
        )

        col_cmp_btn, col_cmp_info = st.columns([1, 2])
        with col_cmp_btn:
            compare_clicked = st.button("⚖️ Compare Policies", type="primary")

        if compare_clicked and uploaded_file_b is not None:
            with st.spinner("Processing Policy B and running grounded side-by-side comparison..."):
                try:
                    file_b_bytes = uploaded_file_b.read()
                    res_b = process_uploaded_policy(file_b_bytes, uploaded_file_b.name)
                    doc_b, chunks_b = res_b
                    st.session_state.policy_b_doc = doc_b
                    st.session_state.policy_b_chunks = chunks_b

                    report = compare_uploaded_policies(
                        doc_a=active_doc,
                        doc_b=doc_b,
                        chunks_a=st.session_state.active_chunks,
                        chunks_b=chunks_b,
                    )
                    st.session_state.comparison_report = report
                    st.toast("Comparison generated successfully!", icon="✅")
                except Exception as e:
                    logger.exception("Error running policy comparison: %s", e)
                    st.error(f"Comparison failed: {str(e)}")

        elif compare_clicked and uploaded_file_b is None:
            st.warning("Please upload Policy B above before comparing.")

        if st.session_state.comparison_report is not None:
            render_comparison_report(st.session_state.comparison_report)
        else:
            st.info("Upload Policy B above and click '⚖️ Compare Policies' to view the side-by-side comparison matrix.")


if __name__ == "__main__":
    main()











