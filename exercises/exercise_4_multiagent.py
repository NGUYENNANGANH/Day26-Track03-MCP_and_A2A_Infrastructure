"""Bài Tập 4: Thêm Privacy Agent vào Multi-Agent System

Sử dụng LangGraph StateGraph để phối hợp các agents.
"""

import asyncio
import os
import sys
from typing import Annotated, TypedDict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from common.llm import get_llm


def _last_wins(left: str | None, right: str | None) -> str:
    """Reducer: giá trị mới ghi đè giá trị cũ."""
    return right if right is not None else (left or "")


class State(TypedDict):
    question: str
    law_analysis: Annotated[str, _last_wins]
    tax_analysis: Annotated[str, _last_wins]
    compliance_analysis: Annotated[str, _last_wins]
    privacy_analysis: Annotated[str, _last_wins]
    final_response: str


def law_agent(state: State) -> dict:
    llm = get_llm()
    prompt = f"Bạn là luật sư cao cấp. Phân tích khía cạnh pháp lý tổng quát: {state['question']}"
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"law_analysis": response.content}


def check_routing(state: State) -> list[Send]:
    """Sử dụng Send API để chạy các agents song song."""
    question_lower = state["question"].lower()
    tasks = []
    
    if any(kw in question_lower for kw in ["tax", "irs", "thuế", "fee"]):
        tasks.append(Send("tax_agent", state))
    
    if any(kw in question_lower for kw in ["compliance", "sec", "regulation", "tuân thủ"]):
        tasks.append(Send("compliance_agent", state))
    
    if any(kw in question_lower for kw in ["data", "privacy", "gdpr", "dữ liệu", "rò rỉ", "leak"]):
        tasks.append(Send("privacy_agent", state))
    
    if not tasks:
        tasks.append(Send("aggregate_results", state))
    
    return tasks


def tax_agent(state: State) -> dict:
    llm = get_llm()
    prompt = f"Chuyên gia thuế phân tích: {state['question']}"
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"tax_analysis": response.content}


def compliance_agent(state: State) -> dict:
    llm = get_llm()
    prompt = f"Chuyên gia compliance phân tích: {state['question']}"
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"compliance_analysis": response.content}


def privacy_agent(state: State) -> dict:
    llm = get_llm()
    prompt = f"Chuyên gia bảo mật dữ liệu phân tích: {state['question']}"
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"privacy_analysis": response.content}


def aggregate_results(state: State) -> dict:
    llm = get_llm()
    combined = f"""
Pháp lý: {state.get('law_analysis')}
Thuế: {state.get('tax_analysis')}
Tuân thủ: {state.get('compliance_analysis')}
Bảo mật: {state.get('privacy_analysis')}
"""
    prompt = f"Tổng hợp báo cáo từ các phân tích sau:\n{combined}"
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"final_response": response.content}


def build_graph() -> StateGraph:
    graph = StateGraph(State)
    
    graph.add_node("law_agent", law_agent)
    graph.add_node("tax_agent", tax_agent)
    graph.add_node("compliance_agent", compliance_agent)
    graph.add_node("privacy_agent", privacy_agent)
    graph.add_node("aggregate_results", aggregate_results)
    
    graph.add_edge(START, "law_agent")
    
    # Ở LangGraph 1.0+, cách gọi conditional edges với Send là dùng list node đích
    graph.add_conditional_edges(
        "law_agent",
        check_routing,
        ["tax_agent", "compliance_agent", "privacy_agent", "aggregate_results"]
    )
    
    graph.add_edge("tax_agent", "aggregate_results")
    graph.add_edge("compliance_agent", "aggregate_results")
    graph.add_edge("privacy_agent", "aggregate_results")
    graph.add_edge("aggregate_results", END)
    
    return graph.compile()


async def main():
    load_dotenv()
    question = "Nếu công ty bị rò rỉ dữ liệu khách hàng, hậu quả pháp lý và thuế là gì?"
    print(f"Câu hỏi: {question}\n")
    
    graph = build_graph()
    result = await graph.ainvoke({"question": question})
    
    print("\n" + "=" * 70)
    print("KẾT QUẢ CUỐI CÙNG")
    print("=" * 70)
    print(result["final_response"])


if __name__ == "__main__":
    asyncio.run(main())
