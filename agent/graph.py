from langgraph.graph import StateGraph, END
from models.schemas import ResearchState
from .nodes import (
    understand_query,
    retrieve_arxiv,
    select_paper,
    ensure_indexed,
    generate_briefing,
    retrieve_for_qa,
    answer_qa,
)

def build_research_graph():
    graph = StateGraph(ResearchState)

    graph.add_node("understand_query", understand_query)
    graph.add_node("retrieve_arxiv", retrieve_arxiv)
    graph.add_node("select_paper", select_paper)
    graph.add_node("ensure_indexed", ensure_indexed)
    graph.add_node("generate_briefing", generate_briefing)

    graph.set_entry_point("understand_query")
    graph.add_edge("understand_query", "retrieve_arxiv")
    graph.add_edge("retrieve_arxiv", "select_paper")
    graph.add_edge("select_paper", "ensure_indexed")
    graph.add_edge("ensure_indexed", "generate_briefing")
    graph.add_edge("generate_briefing", END)

    return graph.compile()

def build_qa_graph():
    graph = StateGraph(ResearchState)

    graph.add_node("retrieve_for_qa", retrieve_for_qa)
    graph.add_node("answer_qa", answer_qa)

    graph.set_entry_point("retrieve_for_qa")
    graph.add_edge("retrieve_for_qa", "answer_qa")
    graph.add_edge("answer_qa", END)

    return graph.compile()
