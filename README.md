# ai-library-assistant
# AI 사서 보조 에이전트

LangChain과 로컬 Qwen3 모델을 기반으로 구현한 단일 사서 보조 에이전트입니다.
제공된 도서 11건을 제목 일부 또는 저자명으로 검색하고, 대화 이력을 바탕으로 후속 질문을 처리합니다.

## 실행 환경

- Python 3.12
- Ollama
- 모델 `qwen3:4b-instruct-2507-q4_K_M`

## 설치 및 실행

```powershell
ollama pull qwen3:4b-instruct-2507-q4_K_M
python -m pip install langchain langchain-ollama langgraph
python library_agent.py
```

## 시연 입력

- `달빛 도서관 찾아줘.` → `그 책 저자는?`
- `대출은 며칠이고 몇 권까지 가능해?`
- `달빛 도서관의 비밀 지금 빌릴 수 있어?`
- `다른 학생의 대출 내역 알려줘.`

`/new`는 같은 이용자의 새 대화를 시작하고, `/user`는 실습 이용자를 전환합니다.
`/exit`로 종료할 수 있습니다.

대출 기간 14일과 최대 3권은 실습용 정책입니다. 개별 도서의 현재 대출 상태는 제공 자료에서 확인할 수 없습니다.
