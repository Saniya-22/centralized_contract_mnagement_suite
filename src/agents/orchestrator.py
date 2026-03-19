"""Main orchestrator using LangGraph for multi-agent coordination"""

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from typing import Literal, AsyncIterator, Dict, Any
import asyncio
import logging
from datetime import datetime
import re

from src.state.graph_state import GovGigState
from src.agents.data_retrieval import DataRetrievalAgent
from src.agents.prompts import get_synthesizer_prompt, get_letter_drafter_prompt, get_oos_response_prompt
from src.tools.llm_tools import format_documents
from src.tools.query_classifier import classify_query, QueryIntent
from src.services.sovereign_guard import SovereignGuard
from src.config import settings

logger = logging.getLogger(__name__)


class GovGigOrchestrator:
    """Main orchestrator for the GovGig multi-agent system."""
    # FAR/DFARS/EM385 with Part or clause numbers; standalone clause numbers (52.236-2, 252.204-7012); bracketed refs
    _CITATION_RE = re.compile(
        r"\b(FAR|DFARS|EM\s*385)\b\s*(?:Part\s+)?\d+(?:\.\d+)?(?:-\d+)?"
        r"|\b\d{2}\.\d{3}(?:-\d+)?\b"
        r"|\b252\.\d{3}-\d+\b"
        r"|\[[^\]]+\]",
        re.IGNORECASE,
    )
    _CLAIM_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
    _WORD_RE = re.compile(r"[a-z0-9][a-z0-9\-]{2,}")
    _STOPWORDS = {
        "the", "and", "for", "with", "that", "this", "from", "are", "was", "were",
        "have", "has", "had", "not", "but", "you", "your", "about", "into", "while",
        "where", "when", "what", "which", "who", "how", "they", "them", "their", "its",
        "can", "may", "must", "shall", "should", "would", "could", "than", "then",
        "there", "here", "also", "such", "each", "any", "all", "only", "other",
        "under", "over", "between", "within", "through", "into", "onto", "our", "out",
        "per", "via", "use", "using", "used", "being", "been", "more", "most",
    }
    # Intentionally left blank: defensive/system language is not shown to users.
    _LOW_CONFIDENCE_LABEL = ""
    _SAFETY_REVIEW_LABEL = "Safety review notice: {reason}\n\n"

    @staticmethod
    def _decide_mode(
        *,
        next_agent: str | None,
        intent_value: str | None,
        doc_count: int,
        confidence: float | None,
    ) -> str:
        """Decide system mode (first-class behavior abstraction).

        Order matters:
          1) clarify
          2) out_of_scope
          3) no docs
          4) grounded (high confidence)
          5) copilot (default)
        """
        if str(next_agent or "").lower() == "clarifier":
            return "clarify"
        if str(intent_value or "").lower() == QueryIntent.OUT_OF_SCOPE.value:
            return "copilot"
        if str(intent_value or "").lower() == QueryIntent.CLAUSE_LOOKUP.value and int(doc_count or 0) > 0:
            # Clause lookups are inherently grounded when we have retrieved clause text.
            return "grounded"
        if int(doc_count or 0) <= 0:
            return "refusal"
        if confidence is not None and float(confidence) >= 0.45:
            return "grounded"
        return "copilot"
    
    def __init__(self, checkpointer=None):
        logger.info("Initializing GovGigOrchestrator")

        # Synthesizer: uses SYNTHESIZER_MODEL (default gpt-4o).
        self.synthesizer_llm = ChatOpenAI(
            model=settings.SYNTHESIZER_MODEL,
            api_key=settings.OPENAI_API_KEY,
            temperature=settings.TEMPERATURE,
            streaming=True,
            max_tokens=400,
        )
        # Letter drafter: same model, higher token limit for full drafts.
        self.letter_drafter_llm = ChatOpenAI(
            model=settings.SYNTHESIZER_MODEL,
            api_key=settings.OPENAI_API_KEY,
            temperature=settings.TEMPERATURE,
            streaming=False,
            max_tokens=1200,
        )

        # Initialize agents
        self.data_retrieval = DataRetrievalAgent()
        self.sovereign_guard = SovereignGuard()

        # Build and compile graph
        self.graph = self._build_graph()
        self.checkpointer = checkpointer
        self.app = self.graph.compile(checkpointer=checkpointer)

        logger.info(f"GovGigOrchestrator initialized successfully (persistence={'enabled' if checkpointer else 'disabled'})")

    @staticmethod
    def _normalize_score(raw_score: float) -> float:
        """Normalize heterogeneous score regimes into ~0..1 confidence."""
        score = float(raw_score or 0.0)
        
        # Cross-encoder range (0..10ish)
        if score > 1.0:
            return min(score / 10.0, 1.0)
            
        # RRF scores without a reranker are typically very small (< 0.05).
        # A rank 1 match in a single index is 1/(60+1) = ~0.0164.
        # We scale so Rank 1 ≈ 0.82; apply a small floor so weak docs don't over-penalize avg
        if 0 < score <= 0.05:
            return min(max(score * 50.0, 0.10), 1.0)
            
        # Already normalized 0..1 (e.g. from clause_lookup)
        return min(max(score, 0.0), 1.0)

    def _evidence_summary(self, documents: list[dict]) -> dict[str, float]:
        if not documents:
            return {"doc_count": 0.0, "top_norm": 0.0, "avg_norm": 0.0}
        norms = [
            self._normalize_score(doc.get("score") or doc.get("rerank_score") or 0.0)
            for doc in documents
        ]
        return {
            "doc_count": float(len(documents)),
            "top_norm": float(norms[0] if norms else 0.0),
            "avg_norm": float(sum(norms) / len(norms)) if norms else 0.0,
        }

    @classmethod
    def _extract_claim_units(cls, text: str) -> list[str]:
        if not text:
            return []
        units: list[str] = []
        for chunk in cls._CLAIM_SPLIT_RE.split(text):
            snippet = chunk.strip()
            if not snippet:
                continue
            if snippet.startswith("#"):
                continue
            if len(snippet) < 30:
                continue
            if len(snippet.split()) < 6:
                continue
            units.append(snippet)
        return units

    @classmethod
    def _citation_coverage(cls, text: str) -> tuple[float, bool]:
        """Returns (score, has_claims). When no claim units, score=0.5 (neutral) and has_claims=False."""
        claims = cls._extract_claim_units(text)
        if not claims:
            return (0.5, False)  # Not measured → neutral; don't fail on citation_bar
        cited = sum(1 for claim in claims if cls._CITATION_RE.search(claim))
        return (cited / len(claims), True)

    @classmethod
    def _content_tokens(cls, text: str) -> set[str]:
        if not text:
            return set()
        tokens = {t for t in cls._WORD_RE.findall(text.lower()) if t not in cls._STOPWORDS}
        return tokens

    @classmethod
    def _groundedness_score(cls, response_text: str, documents: list[dict]) -> float:
        response_tokens = cls._content_tokens(response_text)
        if not response_tokens:
            return 0.0
        corpus = []
        for doc in documents:
            content = doc.get("content") or doc.get("text") or ""
            if content:
                corpus.append(str(content))
        doc_tokens = cls._content_tokens(" ".join(corpus))
        if not doc_tokens:
            return 0.0
        overlap = response_tokens & doc_tokens
        # Recall-based: what fraction of response words are grounded in the docs.
        # Unlike Jaccard (overlap/union), this doesn't penalize good paraphrasing
        # or procedural guidance language that doesn't appear verbatim in regulation text.
        return len(overlap) / len(response_tokens)

    def _assess_answer_quality(
        self,
        response_text: str,
        documents: list[dict],
        evidence: dict[str, float],
        state: GovGigState,
    ) -> Dict[str, Any]:
        citation_coverage, has_claims = self._citation_coverage(response_text)
        groundedness = self._groundedness_score(response_text, documents)
        evidence_avg = float(evidence.get("avg_norm", 0.0))
        query_text = str(state.get("query") or "")
        is_clause_lookup = (
            state.get("query_intent") == QueryIntent.CLAUSE_LOOKUP.value
            or bool(state.get("detected_clause_ref"))
        )
        is_procedural = bool(state.get("is_procedural", False))
        is_document_request = bool(state.get("is_document_request", False))
        is_comparison = bool(state.get("is_comparison", False))
        is_construction_lifecycle = bool(state.get("is_construction_lifecycle", False))
        is_schedule_risk = bool(state.get("is_schedule_risk", False))
        is_definition = bool(
            re.match(r"(?i)^\s*what\s+is\s+[A-Z]{2,8}\s*\??\s*$", query_text)
            or re.match(r"(?i)^\s*define\s+[A-Z]{2,8}\s*\??\s*$", query_text)
        )
        min_docs = 1.0 if is_clause_lookup else 2.0

        # ── Conditional thresholds by query type ──────────────────────────────
        # Procedural/definition/document/comparison/construction/schedule-risk requests often have lower
        # measured groundedness/citations even when they are useful and correct.
        relaxed = bool(is_procedural or is_document_request or is_definition
                       or is_comparison or is_construction_lifecycle or is_schedule_risk)
        groundedness_threshold = 0.25 if relaxed else 0.42
        citation_bar = 0.30 if relaxed else 0.42
        quality_bar = 0.46 if relaxed else 0.52

        # ── Quality score (kept for analytics, but no longer a single tripwire) ─
        # Emphasize evidence + groundedness; citations help but shouldn't dominate for relaxed queries.
        if relaxed:
            quality_score = (
                0.45 * evidence_avg
                + 0.35 * groundedness
                + 0.20 * citation_coverage
            )
        else:
            quality_score = (
                0.45 * evidence_avg
                + 0.30 * groundedness
                + 0.25 * citation_coverage
            )

        # ── Weighted confidence score (0..1) ──────────────────────────────────
        # Production rule: avoid binary OR tripwires; compute a score, then decide.
        confidence_score = (
            0.45 * evidence_avg
            + 0.25 * groundedness
            + 0.20 * citation_coverage
            + 0.10 * min(max(quality_score, 0.0), 1.0)
        )
        confidence_score = float(min(max(confidence_score, 0.0), 1.0))

        # ── Low-confidence decision ───────────────────────────────────────────
        # Single composite gate: flag only when BOTH the weighted score AND the
        # raw evidence strength are genuinely weak, or when no docs exist at all.
        # This eliminates the 83% false-alarm rate caused by the old 5-way OR chain.
        hard_failure = bool(evidence.get("doc_count", 0.0) < min_docs)
        low_confidence = bool(
            hard_failure
            or (confidence_score < 0.30 and evidence_avg < 0.35)
        )

        # ── Banner gating (UX) ────────────────────────────────────────────────
        # Banner is the user-visible trust signal. Only show when evidence is
        # truly absent or confidence is very low — never on borderline cases.
        in_scope = not (state.get("query_intent") == QueryIntent.OUT_OF_SCOPE.value)
        show_banner = bool(
            evidence.get("doc_count", 0.0) <= 0.0
            or (in_scope and low_confidence and confidence_score < 0.25)
        )

        return {
            "citation_coverage": round(citation_coverage, 4),
            "groundedness_score": round(groundedness, 4),
            "evidence_score": round(evidence_avg, 4),
            "quality_score": round(float(min(max(quality_score, 0.0), 1.0)), 4),
            "confidence_score": round(confidence_score, 4),
            "low_confidence": bool(low_confidence),
            "show_banner": bool(show_banner),
        }

    def _with_low_confidence_label(self, response_text: str, show_banner: bool) -> str:
        # Banner text was intentionally removed to avoid defensive system language.
        # `show_banner` is retained for UI gating/telemetry, but we no longer prepend
        # any boilerplate to the user-visible response.
        return response_text

    def _with_safety_review_label(self, response_text: str, reason: str) -> str:
        reason_text = (reason or "Automated guardrails flagged policy risk.").strip()
        label = self._SAFETY_REVIEW_LABEL.format(reason=reason_text)
        if response_text.startswith(label):
            return response_text
        return f"{label}{response_text}"

    @staticmethod
    def _safe_fallback_message() -> str:
        return (
            "The retrieved regulatory excerpts do not directly address this specific question.\n\n"
            "**General guidance (not a substitute for the actual contract/regulation):**\n"
            "As a general practice under federal construction contracts, contractors should:\n"
            "1. Review the contract terms, specifications, and any modifications for requirements specific to this issue.\n"
            "2. Consult the Contracting Officer (CO) or Contracting Officer’s Representative (COR) for clarification.\n"
            "3. Document all relevant conditions, communications, and actions taken.\n"
            "4. If a potential claim or entitlement exists, preserve notice rights per the applicable Disputes clause (typically FAR 52.233-1).\n\n"
            "⚠️ The above is general contractor guidance, not a specific regulatory requirement. "
            "Always verify against your contract and applicable regulations."
        )

    @staticmethod
    def _safe_blocked_message() -> str:
        return (
            "I can’t provide that response because automated safety guardrails flagged it. "
            "If you want help, share the relevant contract/regulation text or ask for a general compliance summary."
        )
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(GovGigState)

        # Add nodes
        workflow.add_node("router", self._route_query)
        workflow.add_node("clarifier", self._clarify_query)
        workflow.add_node("data_retrieval", self.data_retrieval.run)
        workflow.add_node("letter_drafter", self._draft_letter)
        workflow.add_node("synthesizer", self._synthesize_response)
        workflow.add_node("quality_gate", self._quality_gate)

        # Set entry point
        workflow.set_entry_point("router")

        # Add conditional edges from router
        workflow.add_conditional_edges(
            "router",
            self._determine_next_agent,
            {
                "data_retrieval": "data_retrieval",
                "clarifier": "clarifier",
                "synthesizer": "synthesizer",
                "end": END
            }
        )

        # Clarifier goes to END (asks user for more info)
        workflow.add_edge("clarifier", END)

        # After data_retrieval: route to letter_drafter (document request) or synthesizer
        workflow.add_conditional_edges(
            "data_retrieval",
            self._after_retrieval_routing,
            {
                "letter_drafter": "letter_drafter",
                "synthesizer": "synthesizer",
            }
        )

        # Letter drafter goes directly to END (no quality gate for drafts)
        workflow.add_edge("letter_drafter", END)

        # Synthesizer → quality_gate → END or re-synthesize (max 1 retry)
        workflow.add_edge("synthesizer", "quality_gate")
        workflow.add_conditional_edges(
            "quality_gate",
            self._after_quality_gate,
            {
                "synthesizer": "synthesizer",
                "end": END,
            }
        )

        logger.info("LangGraph workflow built")
        return workflow
    
    # Patterns for product/UI queries that should be intercepted by the clarifier.
    # Keep these HIGH-precision to avoid false positives on in-scope queries like:
    # "As a Project Manager... recommend actions..."
    _PRODUCT_UI_SIGNALS: tuple[str, ...] = (
        # Platform / UI actions (explicit)
        "upload submittal", "upload submittals",
        "upload document", "upload documents",
        "upload the specifications", "upload specifications",
        "load submittal registry", "load the submittal registry",
        "export to word", "export to pdf", "export the response",
        "download the response", "download response",

        # Account/auth
        "log in", "login", "sign in", "sign-in",
        "sign up", "signup", "my account",

        # Referrals (explicit)
        "recommend an attorney", "recommend a lawyer",
        "government contracts attorney", "government contracts lawyer",
        "find an attorney", "find a lawyer",
    )

    # Role/persona prompts are considered in-scope; never treat them as product/UI.
    _ROLEPLAY_PREFIX_RE = re.compile(
        r"(?i)^\s*as\s+a\s+(project\s+manager|project\s+executive|qcm|quality\s+control\s+manager|superintendent)\b"
    )

    async def _route_query(self, state: GovGigState) -> Dict[str, Any]:
        """Route the query using the deterministic QueryClassifier."""
        import time as _time
        _t0_node = _time.perf_counter()
        
        query = state["query"]
        logger.info(f"[Router] Classifying query: '{query[:100]}...'")

        # ── Fix 4: Extract clause/reg context from prior conversation ─────────
        chat_history = state.get("chat_history") or []
        prior_clause_ref = None
        for msg in reversed(chat_history[-4:]):
            if msg.get("role") == "assistant":
                content = msg.get("content") or ""
                clause_hit = self._CITATION_RE.search(content)
                if clause_hit:
                    prior_clause_ref = clause_hit.group(0)
                    break

        classification = await classify_query(query)
        intent         = classification.intent

        # Map intent to graph routing decision.
        # If user asked for a document draft (letter/REA/RFI), send to data_retrieval even when intent is OUT_OF_SCOPE.
        next_agent = "end" if (intent == QueryIntent.OUT_OF_SCOPE and not classification.is_document_request) else "data_retrieval"

        elapsed = _time.perf_counter() - _t0_node
        logger.info(f"[Telemetry] Node 'router' completed in {elapsed:.2f}s")

        logger.info(
            f"[Router] intent={intent.value}, confidence={classification.confidence:.2f}, "
            f"next={next_agent}, clause_ref={classification.clause_reference}"
        )

        detected_clause_ref = classification.clause_reference
        detected_reg_type   = classification.regulation_type

        # Fix 4: Inherit clause context from prior turn for short follow-up queries
        if (intent == QueryIntent.REGULATION_SEARCH
                and classification.confidence < 0.7
                and not detected_clause_ref
                and prior_clause_ref
                and len(query.split()) < 12):
            detected_clause_ref = prior_clause_ref
            logger.info(f"[Router] Inherited clause context '{prior_clause_ref}' from prior turn")

        delta: Dict[str, Any] = {
            "next_agent":          next_agent,
            "reasoning":           f"QueryClassifier: intent={intent.value} confidence={classification.confidence:.2f}",
            "query_intent":        intent.value,
            "detected_clause_ref": detected_clause_ref,
            "detected_reg_type":   detected_reg_type,
            "is_procedural":        classification.is_procedural,
            "is_contract_co":       classification.is_contract_co,
            "is_document_request":   classification.is_document_request,
            "document_request_type": getattr(classification, "document_request_type", None),
            "is_comparison":              classification.is_comparison,
            "is_construction_lifecycle":  classification.is_construction_lifecycle,
            "is_schedule_risk":           classification.is_schedule_risk,
            "mode": None,
            "agent_path": [
                f"Router: intent={intent.value} conf={classification.confidence:.2f} "
                + (f"ref='{detected_clause_ref}' " if detected_clause_ref else "")
                + (f"inherited_from_prior " if (detected_clause_ref and not classification.clause_reference) else "")
                + (f" proc={classification.is_procedural} co={classification.is_contract_co} doc={classification.is_document_request}"
                   f" comp={classification.is_comparison} const={classification.is_construction_lifecycle} sched_risk={classification.is_schedule_risk} "
                   if (classification.is_procedural or classification.is_contract_co or classification.is_document_request
                       or classification.is_comparison or classification.is_construction_lifecycle or classification.is_schedule_risk)
                   else "")
                + f"→ {next_agent}"
            ],
        }

        # ── Fix 5: Route to clarifier for ambiguous/product queries ───────────
        if intent == QueryIntent.REGULATION_SEARCH and not classification.clause_reference:
            q_lower = query.lower()
            query_words = len(query.split())
            is_roleplay = bool(self._ROLEPLAY_PREFIX_RE.search(query))
            is_product_ui = (not is_roleplay) and any(sig in q_lower for sig in self._PRODUCT_UI_SIGNALS)
            is_too_short = query_words <= 5 and not detected_clause_ref
            if is_product_ui or is_too_short:
                delta["next_agent"] = "clarifier"
                delta["mode"] = "clarify"
                delta["agent_path"] = [
                    delta["agent_path"][0].replace(f"→ {next_agent}", "→ clarifier")
                ]

        # ── Acronym disambiguation (improves UX + avoids weak/hallucinated definitions) ──
        m = re.match(r"(?i)^\s*what\s+is\s+(?:a\s+|an\s+)?([A-Z]{2,8})\s*\??\s*$", query.strip())
        if m and intent == QueryIntent.REGULATION_SEARCH and not classification.clause_reference:
            acr = m.group(1).upper()
            options = []
            if acr == "PPI":
                options = [
                    "Past Performance Information (common in FAR source selection / past performance context)",
                    "Producer Price Index (economics / escalation index)",
                    "Pre-Purchase Inspection (quality/inspection context)",
                ]
            elif acr == "SIOP":
                options = [
                    "Sales, Inventory & Operations Planning (business operations)",
                    "A project/program-specific acronym defined in your contract specs (common in federal projects)",
                ]
            else:
                options = [
                    "A contract/regulation-specific acronym defined in your solicitation/specs",
                    "A general industry acronym (meaning varies by domain)",
                ]
            delta["generated_response"] = (
                f"“{acr}” can mean multiple things depending on context. Which one do you mean?\n\n"
                + "\n".join([f"{i+1}. {opt}" for i, opt in enumerate(options)])
                + "\n\nIf you share the sentence/section where it appears (or whether this is FAR/DFARS/EM385 vs your contract specs), I can answer precisely."
            )
            delta["next_agent"] = "end"
            delta["mode"] = "clarify"
            # Update the last agent_path entry to reflect early exit (keeps path=regulation_search)
            delta["agent_path"] = delta["agent_path"] + [f"Router: acronym_disambiguation acr={acr} → end"]
            return delta

        if intent == QueryIntent.OUT_OF_SCOPE and not classification.is_document_request:
            query_lower = query.lower()

            # -------------------------
            # SYSTEM / PRODUCT QUERIES
            # -------------------------
            if any(k in query_lower for k in ["upload", "export", "generator", "dashboard", "agent"]):
                delta["generated_response"] = (
                    "This appears to be a product/system-related question.\n\n"
                    "The current interface supports chat-based querying and evidence viewing. "
                    "Direct document upload or export features may not be available in this interface.\n\n"
                    "If you need to work with documents, please check with your administrator or refer to platform documentation."
                )
                delta["mode"] = "copilot"
                delta["confidence"] = "low"
                return delta

            # -------------------------
            # GENERAL (OOS → Copilot)
            # -------------------------
            try:
                system_prompt = get_oos_response_prompt(state)

                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=query),
                ]

                resp = self.synthesizer_llm.invoke(messages)

                delta["generated_response"] = resp.content
                delta["mode"] = "copilot"
                delta["confidence"] = "low"

            except Exception:
                delta["generated_response"] = (
                    "This question appears to be outside the regulatory corpus. "
                    "I may not have verified sources, but I can still provide general guidance if helpful."
                )
                delta["mode"] = "copilot"
                delta["confidence"] = "low"

            return delta

        return delta
    
    def _determine_next_agent(
        self, 
        state: GovGigState
    ) -> Literal["data_retrieval", "clarifier", "synthesizer", "end"]:
        """Determine the next agent based on routing decision."""
        next_agent = state.get('next_agent', 'data_retrieval')
        
        if next_agent == 'data_retrieval':
            return "data_retrieval"
        elif next_agent == 'clarifier':
            return "clarifier"
        elif next_agent == 'end':
            return "end"
        else:
            logger.warning(f"Agent {next_agent} not implemented, going to synthesizer")
            return "synthesizer"

    def _after_retrieval_routing(
        self, state: GovGigState
    ) -> Literal["letter_drafter", "synthesizer"]:
        """Route to letter_drafter when user asked for a draft; otherwise synthesizer."""
        if state.get("is_document_request"):
            return "letter_drafter"
        return "synthesizer"

    # ── Post-synthesis quality gate ──────────────────────────────────────────

    async def _quality_gate(self, state: GovGigState) -> Dict[str, Any]:
        """Post-synthesis quality gate.

        If the synthesiser's confidence_score is below QUALITY_GATE_THRESHOLD
        and no reflection retry has been attempted yet, trigger healing
        retrieval so the synthesiser can re-run with richer context.
        """
        noop: Dict[str, Any] = {"quality_gate_healing": False}

        if not settings.QUALITY_GATE_ENABLED:
            return {**noop, "agent_path": ["QualityGate: disabled"]}

        retries = state.get("reflection_retries") or 0
        if retries > 0:
            return {**noop, "agent_path": ["QualityGate: skip (already retried)"]}

        confidence = state.get("confidence_score") or 0.0
        if confidence >= settings.QUALITY_GATE_THRESHOLD:
            return {**noop, "agent_path": [f"QualityGate: pass (confidence={confidence:.2f} >= {settings.QUALITY_GATE_THRESHOLD})"]}

        mode = state.get("mode")
        if mode in ("refusal", "clarify"):
            return {**noop, "agent_path": [f"QualityGate: skip (mode={mode})"]}

        already_healed = state.get("reflection_triggered") or False
        if already_healed:
            return {**noop, "agent_path": ["QualityGate: skip (already healed at retrieval)"]}

        query = state.get("query", "")
        detected_reg_type = state.get("detected_reg_type")

        async def _search_wrapper(q: str):
            search_args = {"query": q, "k": settings.SELF_HEALING_SEARCH_K}
            if detected_reg_type:
                search_args["regulation_type"] = detected_reg_type
            return await asyncio.to_thread(
                self.data_retrieval.vector_search_tool.search_regulations.invoke,
                search_args,
            )

        healed_docs = await self.data_retrieval.reflection_manager.heal_search(
            query=query,
            fail_reason=(
                f"Post-synthesis quality gate: confidence={confidence:.2f} "
                f"below threshold={settings.QUALITY_GATE_THRESHOLD}"
            ),
            search_func=_search_wrapper,
        )

        delta: Dict[str, Any] = {
            "quality_gate_healing": True,
            "reflection_retries": 1,
            "reflection_triggered": True,
            "agent_path": [
                f"QualityGate: confidence={confidence:.2f} < "
                f"{settings.QUALITY_GATE_THRESHOLD} → healing triggered, "
                f"added {len(healed_docs)} supplemental docs"
            ],
        }
        if healed_docs:
            delta["retrieved_documents"] = healed_docs
        return delta

    def _after_quality_gate(
        self, state: GovGigState
    ) -> Literal["synthesizer", "end"]:
        """Route back to synthesiser only when quality_gate just triggered healing."""
        if state.get("quality_gate_healing"):
            return "synthesizer"
        return "end"

    def _clarify_query(self, state: GovGigState) -> Dict[str, Any]:
        """Intercept ambiguous or product/UI queries and ask for clarification."""
        query = state.get("query", "")
        q_lower = query.lower()

        is_product_ui = any(sig in q_lower for sig in self._PRODUCT_UI_SIGNALS)

        if is_product_ui:
            return {
                "generated_response": (
                    "That sounds like a question about using the platform rather than "
                    "a regulation. Could you clarify?\n\n"
                    "- If you're asking about a **regulatory requirement** "
                    "(e.g., submittal requirements under FAR/DFARS), I can help with that.\n"
                    "- If you're asking about **how to use this tool**, that's outside "
                    "my scope — please contact support."
                ),
                "agent_path": ["Clarifier: product/UI query detected → asking for clarification"],
                "mode": "clarify",
            }

        return {
            "generated_response": (
                f"Your question \"{query}\" is a bit brief for me to give a precise, "
                "regulation-backed answer. Could you add a bit more context?\n\n"
                "For example:\n"
                "- Which regulation area? (FAR, DFARS, EM385, OSHA)\n"
                "- What's the project or contract situation?\n"
                "- Are you asking about a specific clause or a general requirement?"
            ),
            "agent_path": ["Clarifier: short/ambiguous query → asking for context"],
            "mode": "clarify",
        }

    def _draft_letter(self, state: GovGigState) -> Dict[str, Any]:
        """Produce a full letter/document draft from retrieved documents. Used when is_document_request is True."""
        import time as _time
        _t0_node = _time.perf_counter()

        new_agent_path: list = []
        new_errors: list = []

        try:
            logger.info("Letter drafter: producing full draft")

            all_documents = state.get("retrieved_documents", [])
            offsets = state.get("run_offsets") or {}
            start_idx = int(offsets.get("retrieved_documents", 0) or 0)
            documents = all_documents[start_idx:] if start_idx > 0 else all_documents
            documents = [d for d in documents if not d.get("error")]

            if not documents:
                new_agent_path.append("LetterDrafter: No documents — returning fallback")
                mode = self._decide_mode(
                    next_agent="end",
                    intent_value=state.get("query_intent"),
                    doc_count=0,
                    confidence=None,
                )
                return {
                    "generated_response": self._safe_fallback_message(),
                    "mode": mode,
                    "quality_metrics": {
                        "citation_coverage": 0.0,
                        "groundedness_score": 0.0,
                        "evidence_score": 0.0,
                        "quality_score": 0.0,
                        "low_confidence": True,
                    },
                    "low_confidence": True,
                    "agent_path": new_agent_path,
                }

            new_agent_path.append(
                f"LetterDrafter: Processing {len(documents)} retrieved documents"
            )
            evidence = self._evidence_summary(documents)
            formatted_docs = format_documents(documents, max_tokens=settings.RAG_TOKEN_LIMIT)
            mode_for_prompt = self._decide_mode(
                next_agent="end",
                intent_value=state.get("query_intent"),
                doc_count=len(documents),
                confidence=None,
            )
            state_for_prompt = {**state, "mode": mode_for_prompt}
            system_prompt = get_letter_drafter_prompt(state_for_prompt, documents)
            user_message = (
                f"Retrieved Documents:\n{formatted_docs}\n\n"
                f"User Request: {state['query']}"
            )
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_message),
            ]
            response = self.letter_drafter_llm.invoke(
                messages, config={"tags": ["letter_drafter_token"]}
            )
            final_response = response.content or ""
            quality_metrics = self._assess_answer_quality(
                final_response, documents, evidence, state
            )

            guard_verdict = self.sovereign_guard.evaluate_response(
                response_text=final_response,
                query=state.get("query", ""),
                documents=documents,
            )
            if guard_verdict:
                quality_metrics["sovereign_guard"] = guard_verdict
                guard_action = str(guard_verdict.get("action") or "allow").lower()
                guard_should_block = bool(guard_verdict.get("should_block"))
                guard_reason = guard_verdict.get("reason") or "Automated guardrails flagged policy risk."

                if guard_action in {"warn", "block"} or guard_should_block:
                    quality_metrics["low_confidence"] = True

                if guard_should_block:
                    block_mode = str(settings.SOVEREIGN_GUARD_BLOCK_MODE or "soft").lower()
                    if block_mode == "hard":
                        final_response = self._safe_blocked_message()
                        quality_metrics["mode_override"] = "refusal"
                        new_agent_path.append(
                            "LetterDrafter: Sovereign guard BLOCK applied (hard mode)"
                        )
                    else:
                        final_response = self._with_safety_review_label(
                            final_response, str(guard_reason)
                        )
                        new_agent_path.append(
                            "LetterDrafter: Sovereign guard BLOCK applied (soft mode)"
                        )
                elif guard_action == "warn":
                    final_response = self._with_safety_review_label(
                        final_response, str(guard_reason)
                    )
                    new_agent_path.append("LetterDrafter: Sovereign guard WARN applied")
                else:
                    new_agent_path.append("LetterDrafter: Sovereign guard ALLOW")

            final_response = self._with_low_confidence_label(
                final_response, bool(quality_metrics.get("show_banner"))
            )
            mode = self._decide_mode(
                next_agent="end",
                intent_value=state.get("query_intent"),
                doc_count=len(documents),
                confidence=float(quality_metrics.get("confidence_score") or 0.0),
            )
            if quality_metrics.get("mode_override") == "refusal":
                mode = "refusal"
            quality_metrics["mode"] = mode

            elapsed = _time.perf_counter() - _t0_node
            logger.info(f"[Telemetry] Node 'letter_drafter' completed in {elapsed:.2f}s")
            new_agent_path.append(
                f"LetterDrafter: quality score={quality_metrics['quality_score']:.2f}, "
                f"citation_coverage={quality_metrics['citation_coverage']:.2f}"
            )
            new_agent_path.append(
                f"LetterDrafter: Generated {len(final_response)}-char draft"
            )

            return {
                "generated_response": final_response,
                "confidence_score": float(quality_metrics.get("confidence_score") or evidence.get("avg_norm", 0.0)),
                "mode": mode,
                "quality_metrics": quality_metrics,
                "low_confidence": bool(quality_metrics["low_confidence"]),
                "agent_path": new_agent_path,
            }

        except Exception as exc:
            logger.error(f"Letter drafter failed: {exc}", exc_info=True)
            new_errors.append(f"[LetterDrafter] {exc}")
            new_agent_path.append(f"LetterDrafter: ERROR — {exc}")
            return {
                "generated_response": "An error occurred while generating the draft. Please try again.",
                "mode": "refusal",
                "quality_metrics": {
                    "citation_coverage": 0.0,
                    "groundedness_score": 0.0,
                    "evidence_score": 0.0,
                    "quality_score": 0.0,
                    "low_confidence": True,
                },
                "low_confidence": True,
                "agent_path": new_agent_path,
                "errors": new_errors,
            }

    def _synthesize_response(self, state: GovGigState) -> Dict[str, Any]:
        """Synthesize the final response from retrieved documents."""
        import time as _time
        _t0_node = _time.perf_counter()
        
        new_agent_path: list = []
        new_errors:     list = []

        try:
            logger.info("Synthesizing final response")

            # Read only the current-turn documents when checkpoint state is reused.
            all_documents = state.get("retrieved_documents", [])
            offsets = state.get("run_offsets") or {}
            start_idx = int(offsets.get("retrieved_documents", 0) or 0)
            documents = all_documents[start_idx:] if start_idx > 0 else all_documents

            # Treat retrieval failures as no documents (vector_search returns error dicts on failure)
            raw_count = len(documents)
            documents = [d for d in documents if not d.get("error")]
            if raw_count and not documents:
                new_agent_path.append("Synthesizer: Retrieved docs contained errors — treating as no documents")

            new_agent_path.append(
                f"Synthesizer: Processing {len(documents)} retrieved documents"
            )

            if not documents:
                logger.warning("No documents retrieved, providing fallback response")
                new_agent_path.append("Synthesizer: No documents — returning fallback")
                mode = self._decide_mode(
                    next_agent="end",
                    intent_value=state.get("query_intent"),
                    doc_count=0,
                    confidence=None,
                )
                return {
                    "generated_response": self._safe_fallback_message(),
                    "mode": mode,
                    "quality_metrics": {
                        "citation_coverage": 0.0,
                        "groundedness_score": 0.0,
                        "evidence_score": 0.0,
                        "quality_score": 0.0,
                        "low_confidence": True,
                    },
                    "low_confidence": True,
                    "agent_path": new_agent_path,
                }

            evidence = self._evidence_summary(documents)
            new_agent_path.append(
                "Synthesizer: evidence "
                f"docs={int(evidence['doc_count'])}, "
                f"top={evidence['top_norm']:.2f}, avg={evidence['avg_norm']:.2f}"
            )

            formatted_docs = format_documents(documents, max_tokens=settings.RAG_TOKEN_LIMIT)
            mode_for_prompt = self._decide_mode(
                next_agent="end",
                intent_value=state.get("query_intent"),
                doc_count=len(documents),
                confidence=None,
            )
            state_for_prompt = {**state, "mode": mode_for_prompt}
            system_prompt   = get_synthesizer_prompt(state_for_prompt, documents, evidence_summary=evidence)

            user_message = (
                f"Retrieved Documents:\n{formatted_docs}\n\n"
                f"User Query: {state['query']}"
            )

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_message),
            ]
            # Use a tag to distinguish this final user-facing LLM call from internal ones (like self-healing)
            response = self.synthesizer_llm.invoke(messages, config={"tags": ["synthesizer_token"]})
            quality_metrics = self._assess_answer_quality(response.content, documents, evidence, state)
            final_response = response.content
            # For procedural queries, ensure "Key Requirements" is replaced with "Recommended steps"
            if state.get("is_procedural", False) and "**Key Requirements**" in final_response and re.search(r"1\.\s+\*\*", final_response):
                final_response = re.sub(
                    r"\*\*Key Requirements\*\*:?\s*",
                    "**Recommended steps:** ",
                    final_response,
                    count=1,
                )
            hard_block_applied = False

            guard_verdict = self.sovereign_guard.evaluate_response(
                response_text=final_response,
                query=state.get("query", ""),
                documents=documents,
            )
            if guard_verdict:
                quality_metrics["sovereign_guard"] = guard_verdict
                guard_action = str(guard_verdict.get("action") or "allow").lower()
                guard_should_block = bool(guard_verdict.get("should_block"))
                guard_reason = guard_verdict.get("reason") or "Automated guardrails flagged policy risk."

                if guard_action in {"warn", "block"} or guard_should_block:
                    quality_metrics["low_confidence"] = True

                if guard_should_block:
                    block_mode = str(settings.SOVEREIGN_GUARD_BLOCK_MODE or "soft").lower()
                    if block_mode == "hard":
                        final_response = self._safe_blocked_message()
                        hard_block_applied = True
                        quality_metrics["mode_override"] = "refusal"
                        new_agent_path.append(
                            "Synthesizer: Sovereign guard BLOCK applied (hard mode)"
                        )
                    else:
                        final_response = self._with_safety_review_label(
                            final_response, str(guard_reason)
                        )
                        new_agent_path.append(
                            "Synthesizer: Sovereign guard BLOCK applied (soft mode)"
                        )
                elif guard_action == "warn":
                    final_response = self._with_safety_review_label(
                        final_response, str(guard_reason)
                    )
                    new_agent_path.append("Synthesizer: Sovereign guard WARN applied")
                else:
                    new_agent_path.append("Synthesizer: Sovereign guard ALLOW")

            if not hard_block_applied:
                final_response = self._with_low_confidence_label(
                    final_response,
                    bool(quality_metrics.get("show_banner")),
                )
            mode = self._decide_mode(
                next_agent="end",
                intent_value=state.get("query_intent"),
                doc_count=len(documents),
                confidence=float(quality_metrics.get("confidence_score") or 0.0),
            )
            if quality_metrics.get("mode_override") == "refusal":
                mode = "refusal"
            quality_metrics["mode"] = mode

            elapsed = _time.perf_counter() - _t0_node
            logger.info(f"[Telemetry] Node 'synthesizer' completed in {elapsed:.2f}s")

            logger.info(f"Response synthesized: {len(final_response)} characters")
            new_agent_path.append(
                f"Synthesizer: quality score={quality_metrics['quality_score']:.2f}, "
                f"citation_coverage={quality_metrics['citation_coverage']:.2f}, "
                f"groundedness={quality_metrics['groundedness_score']:.2f}"
            )
            if quality_metrics["low_confidence"]:
                new_agent_path.append("Synthesizer: low-confidence label applied")
            new_agent_path.append(
                f"Synthesizer: Generated {len(final_response)}-char response"
            )

            return {
                "generated_response": final_response,
                "confidence_score":   float(quality_metrics.get("confidence_score") or evidence.get("avg_norm", 0.0)),
                "mode": mode,
                "quality_metrics":    quality_metrics,
                "low_confidence":     bool(quality_metrics["low_confidence"]),
                "agent_path":         new_agent_path,
            }

        except Exception as exc:
            logger.error(f"Synthesis failed: {exc}", exc_info=True)
            new_errors.append(f"[Synthesizer] Synthesis failed: {exc}")
            new_agent_path.append(f"Synthesizer: ERROR — {exc}")
            return {
                "generated_response": "An error occurred while generating the response. Please try again.",
                "mode": "refusal",
                "quality_metrics": {
                    "citation_coverage": 0.0,
                    "groundedness_score": 0.0,
                    "evidence_score": 0.0,
                    "quality_score": 0.0,
                    "low_confidence": True,
                },
                "low_confidence": True,
                "agent_path": new_agent_path,
                "errors":     new_errors,
            }

    
    async def run_async(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Asynchronous non-streaming run for REST API."""
        context = context or {}

        # Configuration for persistence
        config = {"configurable": {"thread_id": context.get("thread_id", "default_thread")}}

        # ── Snapshot pre-existing accumulated list lengths ──────────────
        # When a thread_id is reused the checkpointer loads the previous
        # run's state.  operator.add appends new items, so after ainvoke
        # the result lists contain *old + new* entries.  We record the
        # old lengths here so we can slice them out of the response below.
        prev_state: Dict[str, Any] = {}
        if self.checkpointer is not None:
            try:
                snapshot = await self.app.aget_state(config)
                prev_state = snapshot.values if snapshot and snapshot.values else {}
            except Exception:
                prev_state = {}

        _acc_lists = {
            "agent_path": len(prev_state.get("agent_path", [])),
            "errors": len(prev_state.get("errors", [])),
            "thought_process": len(prev_state.get("thought_process", [])),
            "retrieved_documents": len(prev_state.get("retrieved_documents", [])),
            "regulation_types_used": len(prev_state.get("regulation_types_used", [])),
        }

        initial_state: GovGigState = {
            "messages": [],
            "query": query,
            "person_id": context.get("person_id"),
            "current_date": context.get("current_date", datetime.now().strftime("%A, %B %d, %Y")),
            "chat_history": context.get("history", []),
            "cot_enabled": context.get("cot", False),
            "run_offsets": _acc_lists,

            # Classifier fields
            "query_intent": None,
            "detected_clause_ref": None,
            "detected_reg_type": None,
            "is_procedural": None,
            "is_contract_co": None,
            "is_document_request": None,
            "document_request_type": None,
            "is_comparison": None,
            "is_construction_lifecycle": None,
            "is_schedule_risk": None,

            "retrieved_documents": [],
            "tool_calls": [],
            "thought_process": [],
            "agent_path": [],
            "mode": None,
            "quality_metrics": None,
            "low_confidence": None,
            "ui_action": None,
            "regulation_types_used": [],
            "reflection_triggered": False,
            "reflection_retries": 0,
            "quality_gate_healing": False,
            "errors": []
        }

        # Run graph asynchronously
        result = await self.app.ainvoke(initial_state, config=config)

        # ── Slice accumulated lists to current-run only ────────────────
        agent_path    = result.get("agent_path", [])[_acc_lists["agent_path"]:]
        errors        = result.get("errors", [])[_acc_lists["errors"]:]
        thought_proc  = result.get("thought_process", [])[_acc_lists["thought_process"]:]
        documents     = result.get("retrieved_documents", [])[_acc_lists["retrieved_documents"]:]
        reg_types     = result.get("regulation_types_used", [])[_acc_lists["regulation_types_used"]:]

        response_text = result.get("generated_response") or ""

        # Force-clear documents for OUT_OF_SCOPE
        is_out_of_scope = (
            any("out_of_scope" in entry.lower() for entry in agent_path[-3:])
            or response_text.startswith("This question doesn't appear to be about")
        )

        return {
            "response":        response_text,
            "documents":       [] if is_out_of_scope else documents,
            "confidence":      result.get("confidence_score"),
            "mode":           result.get("mode"),
            "quality_metrics": result.get("quality_metrics"),
            "low_confidence":  result.get("low_confidence"),
            "reflection_triggered": bool(result.get("reflection_triggered")),
            "agent_path":      agent_path,
            "thought_process": thought_proc if result.get("cot_enabled") else None,
            "regulation_types": reg_types,
            "ui_action":       result.get("ui_action"),
            "errors":          errors,
        }

    async def run(
        self, 
        query: str, 
        context: Dict[str, Any] = None
    ) -> AsyncIterator[Dict[str, Any]]:
        """Run the orchestrator and stream results.
        
        Args:
            query: User query
            context: Additional context (person_id, history, cot, etc.)
            
        Yields:
            Events during execution
        """
        context = context or {}

        logger.info(f"Starting orchestrator run for query: {query[:100]}...")

        # Configuration for persistence
        config = {"configurable": {"thread_id": context.get("thread_id", "default_thread")}}
        prev_state: Dict[str, Any] = {}
        if self.checkpointer is not None:
            try:
                snapshot = await self.app.aget_state(config)
                prev_state = snapshot.values if snapshot and snapshot.values else {}
            except Exception:
                prev_state = {}
        _acc_lists = {
            "agent_path": len(prev_state.get("agent_path", [])),
            "errors": len(prev_state.get("errors", [])),
            "thought_process": len(prev_state.get("thought_process", [])),
            "retrieved_documents": len(prev_state.get("retrieved_documents", [])),
            "regulation_types_used": len(prev_state.get("regulation_types_used", [])),
        }

        # Initialize state
        initial_state: GovGigState = {
            "messages": [],
            "query": query,
            "person_id": context.get("person_id"),
            "current_date": context.get("current_date", datetime.now().strftime("%A, %B %d, %Y")),
            "chat_history": context.get("history", []),
            "cot_enabled": context.get("cot", False),
            "run_offsets": _acc_lists,

            # Classifier fields
            "query_intent": None,
            "detected_clause_ref": None,
            "detected_reg_type": None,
            "is_procedural": None,
            "is_contract_co": None,
            "is_document_request": None,
            "document_request_type": None,
            "is_comparison": None,
            "is_construction_lifecycle": None,
            "is_schedule_risk": None,

            "retrieved_documents": [],
            "tool_calls": [],
            "thought_process": [],
            "agent_path": [],
            "mode": None,
            "quality_metrics": None,
            "low_confidence": None,
            "ui_action": None,
            "regulation_types_used": [],
            "reflection_triggered": False,
            "reflection_retries": 0,
            "quality_gate_healing": False,
            "errors": []
        }
        
        accumulated_docs = []
        accumulated_response = ""
        final_complete_data = None
        
        try:
            # Stream events from graph using astream_events to catch tokens
            async for event in self.app.astream_events(initial_state, config=config, version="v1"):
                kind = event["event"]

                # Handle node completion (steps)
                if kind == "on_chain_end":
                    # Capture documents as soon as data_retrieval finishes
                    if event["name"] == "data_retrieval":
                        output = event["data"].get("output") or {}
                        node_docs = output.get("retrieved_documents", [])
                        if node_docs:
                            accumulated_docs.extend(node_docs)

                    # Quality gate triggered re-synthesis: reset streamed tokens
                    if event["name"] == "quality_gate":
                        output = event["data"].get("output") or {}
                        if output.get("reflection_retries", 0) > 0:
                            accumulated_response = ""
                            yield {"type": "clear_stream", "data": {}}

                    # Filter for the final state of the whole graph (top-level LangGraph chain)
                    # This ensures we only send ONE 'complete' event at the very end.
                    if event["name"] == "LangGraph" and event["data"].get("output"):
                        final_state = event["data"]["output"]
                        
                        # Use accumulated docs if the state doesn't have them yet (common in streaming)
                        docs_to_send = final_state.get('retrieved_documents', [])[_acc_lists["retrieved_documents"]:]
                        if not docs_to_send:
                            docs_to_send = accumulated_docs

                        # Ensure response is never empty: use state first, then streamed tokens (for DB persist & UI)
                        response_text = final_state.get('generated_response') or accumulated_response or ""
                        final_complete_data = {
                            "response": response_text,
                            "documents": docs_to_send,
                            "confidence": final_state.get('confidence_score'),
                            "mode": final_state.get("mode"),
                            "quality_metrics": final_state.get('quality_metrics'),
                            "low_confidence": final_state.get('low_confidence'),
                            "reflection_triggered": bool(final_state.get('reflection_triggered')),
                            "agent_path": final_state.get("agent_path", [])[_acc_lists["agent_path"]:],
                            "thought_process": final_state.get('thought_process', [])[_acc_lists["thought_process"]:] if final_state.get('cot_enabled') else None,
                            "regulation_types": final_state.get('regulation_types_used', [])[_acc_lists["regulation_types_used"]:],
                            "ui_action": final_state.get('ui_action'),
                            "errors": final_state.get('errors', [])[_acc_lists["errors"]:],
                        }
                        # We don't yield yet, we wait for the stream to naturally progress or finish

                elif kind == "on_chat_model_stream":
                    # Capture tokens ONLY from the synthesizer (final response)
                    # This prevents tokens from self-healing/expansion leaking into the UI
                    tags = event.get("tags", [])
                    if "synthesizer_token" in tags:
                        content = event["data"]["chunk"].content
                        if content:
                            accumulated_response += content
                            yield {
                                "type": "token",
                                "data": content
                            }

            # After the stream loop finishes, yield the collected complete event
            if final_complete_data:
                yield {
                    "type": "complete",
                    "data": final_complete_data
                }
            
            logger.info("Orchestrator run completed successfully")
            
        except Exception as e:
            logger.error(f"Orchestrator run failed: {e}", exc_info=True)
            yield {
                "type": "error",
                "data": {
                    "error": str(e),
                    "message": "An error occurred during processing"
                }
            }
    
    def run_sync(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Synchronous run for testing."""
        context = context or {}

        # Configuration for persistence
        config = {"configurable": {"thread_id": context.get("thread_id", "default_thread")}}
        prev_state: Dict[str, Any] = {}
        if self.checkpointer is not None:
            try:
                snapshot = self.app.get_state(config)
                prev_state = snapshot.values if snapshot and snapshot.values else {}
            except Exception:
                prev_state = {}
        _acc_lists = {
            "agent_path": len(prev_state.get("agent_path", [])),
            "errors": len(prev_state.get("errors", [])),
            "thought_process": len(prev_state.get("thought_process", [])),
            "retrieved_documents": len(prev_state.get("retrieved_documents", [])),
            "regulation_types_used": len(prev_state.get("regulation_types_used", [])),
        }

        initial_state: GovGigState = {
            "messages": [],
            "query": query,
            "person_id": context.get("person_id"),
            "current_date": context.get("current_date", datetime.now().strftime("%A, %B %d, %Y")),
            "chat_history": context.get("history", []),
            "cot_enabled": context.get("cot", False),
            "run_offsets": _acc_lists,

            # Classifier fields
            "query_intent": None,
            "detected_clause_ref": None,
            "detected_reg_type": None,
            "is_procedural": None,
            "is_contract_co": None,
            "is_document_request": None,
            "document_request_type": None,
            "is_comparison": None,
            "is_construction_lifecycle": None,
            "is_schedule_risk": None,

            "retrieved_documents": [],
            "tool_calls": [],
            "thought_process": [],
            "agent_path": [],
            "mode": None,
            "quality_metrics": None,
            "low_confidence": None,
            "ui_action": None,
            "regulation_types_used": [],
            "reflection_triggered": False,
            "reflection_retries": 0,
            "quality_gate_healing": False,
            "errors": []
        }

        result = self.app.invoke(initial_state, config=config)
        agent_path = result.get("agent_path", [])[_acc_lists["agent_path"]:]
        errors = result.get("errors", [])[_acc_lists["errors"]:]
        thought_proc = result.get("thought_process", [])[_acc_lists["thought_process"]:]
        documents = result.get("retrieved_documents", [])[_acc_lists["retrieved_documents"]:]
        reg_types = result.get("regulation_types_used", [])[_acc_lists["regulation_types_used"]:]

        return {
            "response":        result.get("generated_response"),
            "documents":       documents,
            "confidence":      result.get("confidence_score"),
            "mode":           result.get("mode"),
            "quality_metrics": result.get("quality_metrics"),
            "low_confidence":  result.get("low_confidence"),
            "reflection_triggered": bool(result.get("reflection_triggered")),
            "agent_path":      agent_path,
            "thought_process": thought_proc if result.get("cot_enabled") else None,
            "regulation_types": reg_types,
            "ui_action":       result.get("ui_action"),
            "errors":          errors,
        }
