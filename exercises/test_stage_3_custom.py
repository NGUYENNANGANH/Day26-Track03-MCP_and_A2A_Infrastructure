"""Thực hành Stage 3: Tùy chỉnh Agent ReAct.

Nhiệm vụ:
1. Thay đổi SYSTEM_PROMPT để thay đổi phong cách Agent.
2. Quan sát log suy nghĩ của Agent thông qua vòng lặp stream.
"""

import asyncio
import os
import sys

# Thêm đường dẫn để import common
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
from stages.stage_3_single_agent.main import TOOLS, QUESTION
from common.llm import get_llm
from langgraph.prebuilt import create_react_agent

# TODO: Thay đổi SYSTEM_PROMPT để Agent trả lời theo phong cách "Luật sư nghiêm khắc và ngắn gọn"
SYSTEM_PROMPT = (
    "Bạn là một Luật sư cấp cao cực kỳ nghiêm khắc và chuyên nghiệp. "
    "Hãy phân tích các vấn đề pháp lý một cách quyết đoán, sử dụng thuật ngữ chuyên môn. "
    "Đừng nói lời thừa thãi, hãy đi thẳng vào các rủi ro và hình phạt. "
    "Luôn sử dụng các công cụ tra cứu trước khi đưa ra kết luận. "
    "Trả lời bằng tiếng Việt, trình bày theo dạng danh sách gạch đầu dòng rõ ràng."
)

async def main():
    load_dotenv()
    llm = get_llm()
    
    # Khởi tạo agent
    # Lưu ý: create_react_agent không có tham số verbose trực tiếp như mẫu cũ của LangChain,
    # chúng ta quan sát qua stream updates.
    graph = create_react_agent(model=llm, tools=TOOLS, prompt=SYSTEM_PROMPT)

    inputs = {"messages": [{"role": "user", "content": QUESTION}]}

    print(f"--- ĐANG CHẠY AGENT VỚI PHONG CÁCH MỚI ---\n")
    print(f"Câu hỏi: {QUESTION}\n")

    step = 0
    async for chunk in graph.astream(inputs, stream_mode="updates"):
        for node_name, update in chunk.items():
            step += 1
            messages = update.get("messages", [])
            for msg in messages:
                # Log Think + Act
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    print(f"\n[BƯỚC {step}] 🤔 AI ĐANG SUY NGHĨ & GỌI TOOL:")
                    for tc in msg.tool_calls:
                        print(f"  - Công cụ: {tc['name']}")
                        print(f"  - Tham số: {tc['args']}")
                
                # Log Observe
                elif msg.type == "tool":
                    print(f"\n[BƯỚC {step}] 🔍 KẾT QUẢ TỪ CÔNG CỤ:")
                    print(f"  {msg.content[:200]}...")
                
                # Log Final Answer
                elif msg.type == "ai" and msg.content:
                    print(f"\n[BƯỚC {step}] 👨‍⚖️ CÂU TRẢ LỜI CUỐI CÙNG (Phong cách Luật sư):")
                    print("-" * 50)
                    print(msg.content)

if __name__ == "__main__":
    asyncio.run(main())
