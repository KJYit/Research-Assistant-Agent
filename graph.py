"""LangGraph construction for stateless research turns and checkpointed sessions."""

from functools import partial
from threading import Lock
from weakref import WeakValueDictionary

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from nodes import research_turn as stages
from state import SessionState, TurnState
from tools.research_providers import ResearchProviders, default_providers
from tools.retrieval import SemanticIndex


def build_turn_graph(providers: ResearchProviders | None = None):
    """Build the per-turn pipeline. Intermediate state has no checkpointer."""
    providers = providers or default_providers()
    index = SemanticIndex(providers.embed_documents, providers.embed_queries)
    builder = StateGraph(TurnState)
    builder.add_node("prepare_queries", partial(stages.prepare_queries, prepare=providers.prepare))
    builder.add_node("retrieve_local", partial(stages.retrieve_local, index=index))
    builder.add_node("retrieve_web", partial(stages.retrieve_web, web_search=providers.web_search))
    builder.add_node("fuse_candidates", stages.fuse_candidates)
    builder.add_node("rerank_candidates", partial(stages.rerank_candidates, rerank=providers.rerank))
    builder.add_node("check_coverage", stages.check_coverage)
    builder.add_node("respond", partial(stages.respond, answer=providers.answer))

    builder.add_edge(START, "prepare_queries")
    builder.add_conditional_edges("prepare_queries", stages.after_prepare,
                                  ["retrieve_local", "retrieve_web", "respond"])
    builder.add_edge(["retrieve_local", "retrieve_web"], "fuse_candidates")
    builder.add_edge("fuse_candidates", "rerank_candidates")
    builder.add_edge("rerank_candidates", "check_coverage")
    builder.add_conditional_edges("check_coverage", stages.after_coverage,
                                  ["prepare_queries", "respond"])
    builder.add_edge("respond", END)
    return builder.compile(checkpointer=False)


def build_graph(providers: ResearchProviders | None = None):
    """Checkpoint conversation and public results by thread ID."""
    turn = build_turn_graph(providers)

    def research_turn(state: SessionState) -> SessionState:
        topic = state["topic"]
        history = state.get("messages", [])
        result = turn.invoke({"topic": topic, "history": history[-4:]})
        answer = result["final_answer"]
        return {
            "messages": history + [HumanMessage(content=topic), AIMessage(content=answer)],
            "final_answer": answer,
            "sources": result["sources"],
        }

    builder = StateGraph(SessionState)
    builder.add_node("research_turn", research_turn)
    builder.add_edge(START, "research_turn")
    builder.add_edge("research_turn", END)
    return builder.compile(checkpointer=MemorySaver())


research_graph = build_graph()

_thread_locks: WeakValueDictionary[str, Lock] = WeakValueDictionary()
_registry_lock = Lock()


def invoke_research_turn(topic: str, thread_id: str, *, graph=None) -> SessionState:
    """Run a turn using a process-local lock for each active conversation."""
    with _registry_lock:
        lock = _thread_locks.get(thread_id)
        if lock is None:
            lock = _thread_locks[thread_id] = Lock()
    if graph is None:
        graph = research_graph
    with lock:
        return graph.invoke({"topic": topic}, config={"configurable": {"thread_id": thread_id}})
