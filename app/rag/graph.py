"""
LangGraph RAG workflow definition.

Build the compiled graph once at startup (via dependencies.py) and reuse
it for every /ask request.  Each node is a pure async function that accepts
RAGState and returns a partial dict — LangGraph handles merging.

Workflow:
  START
    → query_classifier
        → [smalltalk]  safe_direct_answer → END
        → [unsafe]     refusal            → END
        → [unclear]    clarification      → END
        → [retrieval]  retrieve
                         → permission_filter
                           → reranker
                             → context_budgeter
                               → answer_generator
                                 → answer_validator
                                     → [grounded|cautious] END
                                     → [unsupported]       refusal → END

TODO (Phase 2):
  - Add a "retrieve" node that dispatches to keyword/vector/hybrid based on
    state["retrieval_mode"] and stores results in state["raw_chunks"].
  - Implement safe_direct_answer for smalltalk (short, friendly, no retrieval).
  - Implement refusal node (return a canned safe-refusal message).
  - Implement clarification node (ask the user to rephrase).
  - Consider adding a query-rewriting node between classifier and retrieve:
    expand acronyms, add synonyms, strip noise (improves recall).
"""

from langgraph.graph import END, StateGraph

from app.models.search import SearchRequest
from app.rag.nodes.answer_generator import answer_generator_node
from app.rag.nodes.answer_validator import answer_validator_node, route_validation
from app.rag.nodes.context_budgeter import context_budgeter_node
from app.rag.nodes.permission_filter import permission_filter_node
from app.rag.nodes.query_classifier import query_classifier_node, route_query
from app.rag.nodes.reranker import reranker_node
from app.rag.state import RAGState
from app.search.base import SearchService


def _make_retrieve_node(
    keyword: SearchService,
    vector: SearchService,
    hybrid: SearchService,
):
    """
    Closure that captures the search services and returns a LangGraph-compatible
    async node function.

    TODO (Phase 2): implement actual retrieval dispatch.
    """

    async def retrieve_node(state: RAGState) -> dict:
        mode = state.get("retrieval_mode", "hybrid")
        service = {"keyword": keyword, "vector": vector, "hybrid": hybrid}[mode]

        request = SearchRequest(
            query=state["query"],
            user_role=state.get("user_role", "customer"),
        )
        response = await service.search(request)
        return {"raw_chunks": response.results}

    return retrieve_node


def _make_terminal_node(answer: str, confidence: str):
    """Factory for simple terminal nodes (refusal, clarification, smalltalk)."""

    async def node(state: RAGState) -> dict:
        return {"answer": answer, "confidence": confidence, "citations": []}

    return node


def build_rag_graph(
    keyword_search: SearchService,
    vector_search: SearchService,
    hybrid_search: SearchService,
    settings,
):
    """
    Compile and return the LangGraph CompiledGraph.

    Called once at application startup from dependencies.py.
    """
    graph = StateGraph(RAGState)

    # ---- Nodes ----
    graph.add_node("query_classifier", query_classifier_node)
    graph.add_node(
        "retrieve",
        _make_retrieve_node(keyword_search, vector_search, hybrid_search),
    )
    graph.add_node("permission_filter", permission_filter_node)
    graph.add_node("reranker", reranker_node)
    graph.add_node("context_budgeter", context_budgeter_node)
    graph.add_node("answer_generator", answer_generator_node)
    graph.add_node("answer_validator", answer_validator_node)

    # Terminal short-circuit nodes
    graph.add_node(
        "safe_direct_answer",
        _make_terminal_node(
            "I'm here to help with fintech product questions. What would you like to know?",
            confidence="grounded",
        ),
    )
    graph.add_node(
        "refusal",
        _make_terminal_node(
            "I cannot help with that request. Please ask a question related to the "
            "documents you are authorised to access.",
            confidence="refused",
        ),
    )
    graph.add_node(
        "clarification",
        _make_terminal_node(
            "Your question is a bit broad. Could you provide more detail so I can find "
            "the most relevant information?",
            confidence="cautious",
        ),
    )

    # ---- Edges ----
    graph.set_entry_point("query_classifier")

    graph.add_conditional_edges(
        "query_classifier",
        route_query,
        {
            "retrieval": "retrieve",
            "smalltalk": "safe_direct_answer",
            "unsafe": "refusal",
            "unclear": "clarification",
        },
    )

    graph.add_edge("retrieve", "permission_filter")
    graph.add_edge("permission_filter", "reranker")
    graph.add_edge("reranker", "context_budgeter")
    graph.add_edge("context_budgeter", "answer_generator")
    graph.add_edge("answer_generator", "answer_validator")

    graph.add_conditional_edges(
        "answer_validator",
        route_validation,
        {"end": END, "refusal": "refusal"},
    )

    graph.add_edge("safe_direct_answer", END)
    graph.add_edge("refusal", END)
    graph.add_edge("clarification", END)

    return graph.compile()
