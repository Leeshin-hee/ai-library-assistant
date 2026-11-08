
import json
from dataclasses import dataclass
from typing import Literal
from uuid import uuid4

from langchain.agents import create_agent
from langchain.agents.middleware import after_agent, before_agent, wrap_model_call
from langchain.messages import AIMessage
from langchain.tools import ToolRuntime, tool
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore


MODEL = "qwen3:4b-instruct-2507-q4_K_M"

BOOKS = [
    {"title": "달빛 도서관의 비밀", "author": "최현범"},
    {"title": "내일이 사라진 도시", "author": "오다영"},
    {"title": "고양이는 자정에 책을 읽는다", "author": "김하린"},
    {"title": "시간을 파는 서점", "author": "이도윤"},
    {"title": "여름 끝에서 너를 만났다", "author": "박서연"},
    {"title": "기억을 잃은 탐정의 편지", "author": "정민재"},
    {"title": "열세 번째 문이 열리는 밤", "author": "한유진"},
    {"title": "구름 위 우체국의 마지막 편지", "author": "윤가람"},
    {"title": "마법사가 사라진 왕국", "author": "오지호"},
    {"title": "우리의 비밀은 별빛보다 깊다", "author": "강다은"},
    {"title": "사과의 매력을 찾아서", "author": "서정욱"},
]

LOAN_INFO = {
    "policy": "실습용 대출 정책은 1인 최대 3권, 대출 기간 14일입니다.",
    "availability": "현재 실습 데이터에서는 개별 도서의 대출 가능 여부를 확인할 수 없습니다.",
    "personal": "개인별 대출 내역은 이 실습 에이전트에서 조회할 수 없습니다.",
}

FORMAT_NAMES = {"plain": "기본 형식", "numbered": "번호 목록"}
REFUSAL = "도서 검색, 대출 안내, 응답 형식 설정을 도와드릴 수 있습니다."
PRIVACY_REFUSAL = "다른 이용자의 대출 내역은 안내할 수 없습니다."
NO_EVIDENCE = "등록 목록이나 대출 안내에서 확인할 수 있는 질문으로 다시 말씀해 주세요."


@dataclass
class Context:
    librarian_name: str
    librarian_role: str
    user_id: str


def normalize(text: str) -> str:
    return "".join(text.split()).casefold()


def current_turn(messages):
    """가장 최근 사용자 입력 이후의 메시지만 반환합니다."""
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].type == "human":
            return messages[index:]
    return []


@tool
def search_books(query: str) -> dict:
    """등록 도서를 제목 일부 또는 저자명으로 검색합니다. 검색어만 전달하세요."""
    keyword = normalize(query)
    if not keyword:
        return {"status": "empty", "message": "도서 제목이나 저자 이름을 알려 주세요."}

    matches = [
        book for book in BOOKS
        if keyword in normalize(book["title"]) or keyword in normalize(book["author"])
    ]
    if not matches:
        return {"status": "not_found", "message": "등록 목록에서 해당 도서를 찾지 못했습니다."}
    return {"status": "found", "books": matches}


@tool
def get_loan_info(topic: Literal["policy", "availability", "personal"]) -> dict:
    """대출 권수·기간은 policy, 특정 책 대출 가능 여부는 availability,
    개인 대출 내역은 personal을 선택하세요.
    """
    return {"status": topic, "message": LOAN_INFO[topic]}


@tool
def save_preference(style: Literal["plain", "numbered"], runtime: ToolRuntime) -> str:
    """이용자가 앞으로도 사용할 검색 결과 형식을 명시적으로 요청하면 저장합니다."""
    runtime.store.put(("library", runtime.context.user_id), "format", {"style": style})
    return f"검색 결과 표시를 {FORMAT_NAMES[style]}으로 저장했습니다."


def grounded_reply(messages, store, context) -> str:
    """현재 요청에서 확인된 조회 결과로 이용자 회신문을 만듭니다."""
    item = store.get(("library", context.user_id), "format")
    numbered = item is not None and item.value.get("style") == "numbered"
    sections = []

    for message in current_turn(messages):
        if message.type != "tool" or message.status == "error":
            continue
        if message.name == "save_preference":
            sections.append(message.content)
        elif message.name in {"search_books", "get_loan_info"}:
            result = json.loads(message.content)
            if message.name == "search_books" and result["status"] == "found":
                rows = []
                for index, book in enumerate(result["books"], start=1):
                    prefix = f"{index}. " if numbered else "- "
                    rows.append(f"{prefix}『{book['title']}』 / {book['author']}")
                sections.append("등록 도서 검색 결과입니다.\n" + "\n".join(rows))
            else:
                sections.append(result["message"])

    return "\n\n".join(dict.fromkeys(sections))


@tool
def draft_reply(runtime: ToolRuntime) -> str:
    """검색·대출 조회 또는 선호 저장을 마친 뒤, 확인된 결과로 회신 초안을 만듭니다."""
    reply = grounded_reply(runtime.state["messages"], runtime.store, runtime.context)
    return reply or NO_EVIDENCE


@before_agent(can_jump_to=["end"])
def library_input_guardrail(state, runtime):
    turn = current_turn(state["messages"])
    if not turn:
        return None

    text = normalize(turn[0].content)
    third_party = ("다른학생", "다른이용자", "친구", "타인", "남의")
    borrowing = ("대출", "빌린책", "빌렸")
    off_topic = ("파이썬코드", "코드짜", "주식추천", "날씨알려", "게임공략")

    if any(word in text for word in third_party) and any(word in text for word in borrowing):
        print("[입력 차단] 타인 대출 정보")
        return {"messages": [AIMessage(content=PRIVACY_REFUSAL)], "jump_to": "end"}

    if any(word in text for word in off_topic):
        print("[입력 차단] 도서관 업무 범위 밖")
        return {"messages": [AIMessage(content=REFUSAL)], "jump_to": "end"}

    return None


@wrap_model_call
def enrich_context(request, handler):
    ctx = request.runtime.context
    item = request.runtime.store.get(("library", ctx.user_id), "format")
    style = item.value["style"] if item else "plain"

    prompt = (
        f"당신은 {ctx.librarian_name}이며 역할은 {ctx.librarian_role}입니다.\n"
        f"이용자의 검색 결과 표시 선호는 {FORMAT_NAMES[style]}입니다.\n"
        "제목 일부나 저자 이름을 말하며 찾아 달라면 search_books를 호출하세요.\n"
        "'달빛 도서관 찾아줘'의 검색어는 '달빛 도서관'입니다.\n"
        "후속 질문 '그 책 저자는?'에는 대화에서 책 제목을 찾아 다시 search_books로 확인하세요.\n"
        "대출 규정이나 대출 가능 여부는 get_loan_info를 호출하세요.\n"
        "앞으로 번호 목록 또는 기본 형식을 써 달라고 명시하면 save_preference를 호출하세요.\n"
        "조회나 저장을 마쳤으면 draft_reply를 한 번 호출하고, 그 결과를 그대로 최종 답변으로 반환하세요.\n"
        "도서관 업무 범위 밖이면 지원 범위를 안내하세요. 확인하지 않은 도서나 대출 정보는 말하지 마세요."
    )
    return handler(request.override(system_prompt=prompt))


@after_agent
def review_reply(state, runtime):
    last = state["messages"][-1]
    if not isinstance(last, AIMessage):
        return None

    if last.content in (PRIVACY_REFUSAL, REFUSAL):
        return None
    expected = grounded_reply(state["messages"], runtime.store, runtime.context)
    safe_reply = expected or NO_EVIDENCE
    if last.content == safe_reply:
        print("[최종 검수] 통과")
        return None

    print("[최종 검수] 확인된 정보로 교체" if expected else "[최종 검수] 조회 근거 없음")
    return {"messages": [last.model_copy(update={"content": safe_reply})]}


agent = create_agent(
    ChatOllama(model=MODEL, temperature=0, num_ctx=4096),
    tools=[search_books, get_loan_info, save_preference, draft_reply],
    checkpointer=InMemorySaver(),
    store=InMemoryStore(),
    context_schema=Context,
    middleware=[library_input_guardrail, enrich_context, review_reply],
)


def main():
    user_id = "demo-user"
    thread_id = uuid4().hex
    print("=== AI 사서 보조 에이전트 ===")
    print("명령  /new 새 대화  |  /user 다른 실습 이용자  |  /exit 종료")

    while True:
        question = input("\n이용자 > ").strip()
        if not question:
            continue
        if question == "/exit":
            break
        if question == "/new":
            thread_id = uuid4().hex
            print("[새 대화] 같은 이용자의 응답 형식 설정을 유지합니다.")
            continue
        if question == "/user":
            user_id = "demo-other" if user_id == "demo-user" else "demo-user"
            thread_id = uuid4().hex
            print(f"[이용자 전환] {user_id}")
            continue

        context = Context("도담", "도서 검색과 대출 안내를 지원하는 사서 보조", user_id)
        result = agent.invoke(
            {"messages": [{"role": "user", "content": question}]},
            config={"configurable": {"thread_id": thread_id}, "recursion_limit": 18},
            context=context,
        )
        for message in current_turn(result["messages"]):
            if message.type == "tool":
                print(f"[도구] {message.name}")
        print(f"사서 보조 > {result['messages'][-1].content}")


if __name__ == "__main__":
    main()
