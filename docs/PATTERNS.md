# 에이전트 패턴·용어 정리

> 기준일 2026-11-01. basic_agent가 무엇을 어떤 이름으로 제공하는지와, 업계에서 쓰는 표준 명칭을
> 대응시키는 문서다. 업계 전체가 합의한 단일 분류표는 없다. 그래서 많이 인용되는 용어들을
> **5개 축**으로 나눠 정리했다.

## 0. 읽는 법

**출처 확인 표기** (각 항목의 `출처` 열)
- **원문**: 2026-11-01에 원문을 직접 읽음
- **초록**: 논문의 제목·날짜·초록을 확인함 (본문은 읽지 않음)
- **확인 안 함**: 학습 지식에 의존함

**9B 열**: 로컬 9B 모델(vLLM)에서 쓸 만한지에 대한 **판단**이다. 측정값이 아니다. 실측 결과가 생기면 바꾼다.

**basic_agent 열**
- `있음`: 등록 이름
- `계획 N.N`: `WORKPLAN.md`의 해당 항목
- `범위 밖`: 이번 작업에서 다루지 않음

---

## 1. 가장 기본이 되는 구분: workflow와 agent

| 명칭 | 정의 | 출처 |
|---|---|---|
| **Workflow** | LLM과 도구가 **미리 정해진 코드 경로**를 따라 실행됨 | [A1] 원문 |
| **Agent** | LLM이 자기 프로세스와 도구 사용을 **스스로 결정**함 | [A1] 원문 |
| **Agentic system** | 위 둘을 합친 상위 개념 | [A1] 원문 |
| Agent (최근 정의) | "루프 안에서 도구를 자율적으로 쓰는 LLM". Anthropic이 이 정의로 수렴했다고 밝힘 | [A5] 원문 |

**원칙** [A1]: 가장 단순한 해법에서 시작하고, 필요할 때만 복잡도를 올린다. 예측 가능한 작업은 workflow로, 유연성이 필요한 작업은 agent로 처리한다.

**최근 방향** [A3][A6][A9]: 모델이 강해질수록 하네스(모델 바깥의 루프·도구·상태 관리)를 얇게 하고, 컨텍스트를 필요할 때 펼쳐 쓰는 단일 에이전트 루프를 기본으로 둔다. 서브에이전트는 검증, 병렬 탐색, 권한 분리 용도로 좁혀 쓴다. 다만 이 방향의 전제는 **강한 모델**이다. 약한 메인 모델은 "언제 도움을 청할지"를 몰라서 실패했다는 보고가 있다 [A9].

---

## 2. 축별 분류

### 축 1. 흐름을 누가 정하나
§1 참조. basic_agent에서는 `plan_execute`가 workflow 쪽(그래프가 고정되어 있고, 계획만 LLM이 세움), `react`가 agent 쪽에 가깝다.

### 축 2. 에이전트가 계획을 어떻게 세우나

| 표준 명칭 | 정의 | 언제 쓰나 | 9B | basic_agent | 출처 |
|---|---|---|---|---|---|
| **ReAct** | 추론과 행동을 한 스텝씩 번갈아 진행. 관찰 결과가 다음 추론에 들어감 | 열린 문제, 다음 단계가 앞 결과에 달린 탐색 | 중간: 스텝이 길어지면 흔들림 | `react` 있음 | [P1] 초록 (Yao et al., 2022-10) |
| **Plan-and-Solve / Plan-and-Execute** | 먼저 계획을 세운 뒤 하위 과제로 나눠 실행 | 분해가 가능하고 구조가 있는 작업 | 높음: 구조가 약한 모델을 보완 | `plan_execute` 있음 | [P2] 초록 (Wang et al., 2023-05). "Plan-and-Execute"는 LangChain의 구현 명칭 (확인 안 함) |
| **ReWOO** | 추론을 관찰에서 분리. 계획을 한 번에 세우고 도구 결과는 나중에 모음 → 반복되는 프롬프트를 줄임 | 토큰 절약이 중요하고 경로가 예측 가능할 때 | 높음 | 구조상 유사 [판단]: planner/worker/solver ↔ planner/executor/joiner | [P3] 초록 (Xu et al., 2023-05) |
| **LLMCompiler** | 함수 호출을 DAG로 계획하고 병렬 실행 | 독립적인 호출이 많고 지연 시간이 중요할 때 | 높음 | `plan_execute` 있음 (실행 스케줄은 "레벨 동기화"로 단순화) | [P4] 초록 (Kim et al., 2023-12) |
| **Reflexion** | 실패를 말로 된 피드백으로 기억해 다음 시도에 반영 (가중치 학습 없음) | 같은 과제를 반복 시도할 수 있을 때 | 낮음~중간 | 범위 밖. 관련 패턴: Evaluator-optimizer (계획 6.4) | [P5] 제목·날짜 (Shinn et al., 2023-03) |

### 축 3. 에이전트가 여럿일 때 어떻게 연결하나

| 표준 명칭 | 별칭 | 정의 | 언제 쓰나 | 9B | basic_agent | 출처 |
|---|---|---|---|---|---|---|
| **Single-agent** | main agent loop | 에이전트 하나 + 도구 | **기본값**. 대부분의 작업 | — | `react` / `plan_execute` 단독 | [A3][A10] 원문 |
| **Orchestrator-subagent** | Subagents (LangChain), Supervisor (LangGraph), Manager / agents-as-tools (OpenAI), Orchestrator-workers (Anthropic workflow) | 메인 에이전트가 하위 에이전트를 **도구처럼 호출**. 대화와 종합은 메인이 맡음 | 검증, 넓은 병렬 탐색, 권한 분리, 컨텍스트 격리 | 중간 | **`AgentTool` 있음** (정적 등록). 실행 중 동적 생성은 계획 6.5 | [A3][A4][A8][L2] 원문 |
| **Handoff** | Decentralized (OpenAI) | 대화 주도권 자체를 다른 에이전트에 넘김 | 하위 에이전트가 사용자와 **직접** 대화해야 할 때만 [A3] | — | 범위 밖 | [A3][L2] 원문 |
| **Router** | Routing (Anthropic workflow) | 입력을 분류해 전문 경로 하나로 보냄 | 입력 유형이 뚜렷하게 나뉠 때 | 높음 (분류 작업) | 계획 6.2 | [A1][L2] 원문 |
| **Hierarchical** | — | 관리자 아래 관리자가 있는 다층 구조 | 매우 큰 작업 | 낮음 | `AgentTool` 중첩으로 구성 가능 (3단 중첩은 계획 4B.5에서 검증) | 확인 안 함 |
| **Network / Swarm** | — | 에이전트끼리 자유롭게 통신 | 근거 자료가 비권장 ("mostly a distraction") | — | 범위 밖 | [A9] 원문 |
| **Map-reduce-and-manage** | — | 관리자가 쪼개고 자식이 실행하고 관리자가 종합 | Cognition이 실제로 쓸 만하다고 본 형태 | — | 계획 6.5 | [A9] 원문 |
| **Smart Friend** | — | 약한 메인이 어려운 지점에서 강한 모델에게 물음 | 비용·속도와 품질의 절충 | 메인이 9B면 위험 ("언제 물을지"를 모름) | 자식마다 다른 `llm`을 쓰는 것으로 구성 가능 (계획 4B.6) | [A9] 원문 |

### 축 4. 행동을 어떤 형식으로 내나

| 표준 명칭 | 정의 | 9B | basic_agent | 출처 |
|---|---|---|---|---|
| **Tool calling / Function calling** | 구조화된 형식(JSON)으로 도구를 지정하고 인자를 넘김 | — | JSON 스키마 + vLLM guided decoding 방식 (모델의 네이티브 tool calling 아님, 계획 4C) | 공통 |
| **Parallel tool calls** | 한 턴에 여러 도구 호출 | — | `plan_execute`는 레벨 단위로 가능. `react`는 아직 불가 (계획 4A.2) | [A8] 원문 |
| **CodeAct** | 실행 가능한 Python 코드로 행동을 통일. 여러 도구를 코드로 조합 | 낮음 (샌드박스 필요) | 범위 밖 | [P6] 초록 (Wang et al., 2024-02) |
| **Programmatic tool calling** | 모델이 짠 프로그램이 도구를 호출. 흐름이 예측 가능한 구간에 사용하고, 판단이나 승인이 필요한 호출은 직접 호출로 둠 | 낮음 (샌드박스 필요) | 범위 밖 | [A8] 원문 |

### 축 5. 컨텍스트를 어떻게 관리하나 (context engineering)

| 표준 명칭 | 정의 | 9B | basic_agent | 출처 |
|---|---|---|---|---|
| **Context engineering** | 모델이 보는 컨텍스트 전체를 설계하는 일. 프롬프트 엔지니어링의 확장 | — | — | [A5] 원문 |
| **Compaction** | 컨텍스트가 한계에 가까워지면 요약하고 새로 시작 | 요약 품질이 관건 | 있음 (opt-in): `ContextPolicy(compact_at_chars=…)`, react | [A5] 원문 |
| **Tool-result clearing** | 오래된 도구 결과를 지움. 가장 가벼운 형태의 compaction | 높음 | 있음 (opt-in): `keep_recent=N`, react | [A5] 원문 |
| **Structured note-taking** (agentic memory) | 컨텍스트 밖에 메모를 남기고 나중에 다시 읽음 | 중간 | 있음 (opt-in): `notes=True`, react | [A5][A7] 원문 |
| **Just-in-time context** | 데이터 대신 가벼운 식별자(경로, 핸들)를 들고 있다가 필요할 때 로드 | 높음 | 있음: `offload=True` + `read_observation(step, offset | find)`. **react 기본값** (2026-11-01 실모델 평가 근거) | [A5] 원문 |
| **Progressive disclosure** | 스킬과 도구의 짧은 설명만 보여주고 본문은 필요할 때 로드 (Skills, tool search) | 중간 (스스로 찾아 써야 함) | 있음 (opt-in): `skills=…` + `load_skill`, `tool_search_threshold=…` + `tool_search`, react | [A3][A8] 원문, [L2] Skills 패턴 |
| **Context isolation** | 하위 에이전트에 별도 컨텍스트를 줌 | — | `AgentTool` 있음 (자식은 빈 history로 시작). 선택적 전달은 계획 4B.1 | [A4][A6] 원문 |

---

## 3. Anthropic workflow 5종 [A1] 원문

| 패턴 | 정의 | 언제 쓰나 (원문 요지) | basic_agent |
|---|---|---|---|
| **Prompt chaining** | 작업을 단계로 나눠 앞 단계의 출력을 다음 단계 입력으로. 단계 사이에 프로그램 검사(gate) 가능 | 고정된 하위 과제로 깔끔하게 나뉠 때. 지연을 감수하고 정확도를 얻음 | 계획 6.1 |
| **Routing** | 입력을 분류해 전문화된 후속 처리로 보냄 | 입력 유형이 뚜렷하게 나뉘고 분류를 정확히 할 수 있을 때 | 계획 6.2 |
| **Parallelization** | **Sectioning**: 독립 하위 과제를 병렬 실행. **Voting**: 같은 과제를 여러 번 실행해 다양한 출력을 얻음 | 속도를 위해 나누거나, 확신을 높이려 여러 관점·시도가 필요할 때 | 계획 6.3 (`plan_execute` 레벨은 도구 호출 단위의 sectioning) |
| **Orchestrator-workers** | 중앙 LLM이 작업을 **동적으로** 나눠 워커에게 맡기고 결과를 종합 | 필요한 하위 과제를 **미리 알 수 없을 때**. Parallelization과의 차이는 하위 과제가 입력에 따라 정해진다는 점 | 계획 6.5. `plan_execute`는 도구 호출 DAG를 미리 짜는 방식이라 구분됨 |
| **Evaluator-optimizer** | 한 LLM이 생성하고 다른 LLM이 평가·피드백하는 루프 | 평가 기준이 명확하고, 반복 개선의 가치가 측정 가능할 때 | 계획 6.4 |

---

## 4. 용어집

| 용어 | 뜻 | 출처 |
|---|---|---|
| **Harness** | 모델 바깥의 루프·도구·상태·컨텍스트 관리 계층. 구성요소 하나하나가 "모델이 혼자 못 하는 것"에 대한 가정이고, 모델이 좋아지면 낡는다 | [A6] 원문 |
| **ACI** (agent-computer interface) | 에이전트가 쓰는 도구 인터페이스. 사람용 UI처럼 문서화와 테스트에 공을 들여야 함 | [A1] 원문 |
| **Telephone game** | 에이전트 간 전달을 거칠 때마다 정보가 손실되는 현상 | [A4][A10] 원문 |
| **Context-centric decomposition** | 작업 유형(기획/구현/테스트)이 아니라 **필요한 컨텍스트** 기준으로 에이전트를 나눔 | [A10] 원문 |
| **Verification subagent** | 메인의 결과를 깨끗한 컨텍스트에서 검사하는 하위 에이전트. 컨텍스트 전달이 적어 잘 동작함 | [A10] 원문 |
| **Single-threaded writes** | 쓰기는 한 에이전트만 하고, 다른 에이전트는 판단만 보탬 | [A9] 원문 |
| **Effort scaling** | 질문 난이도에 맞춰 에이전트·도구 호출 수를 정하는 규칙 (예: 단순 사실 확인은 에이전트 1개, 도구 호출 3~10회) | [A4] 원문 |
| **HITL** (human-in-the-loop) | 실행 중 사람의 승인·입력을 받는 지점 | 공통 |
| **Deep agents** | 상세한 시스템 프롬프트 + 계획 도구(todo) + 서브에이전트 + 파일시스템을 묶은 구성 | [L1] 원문 (LangChain, 2025-07) |

---

## 5. basic_agent의 조합 규칙 (2026-11-01 측정)

근거: `examples/compose_matrix.py` (73/73 PASS), 상세 결과는 `docs/baseline.md`.

0. **등록부**: 모든 패턴은 `agent.patterns.registry`에 표준 명칭과 함께 등록된다 (`python -m agent.run --list-patterns`). 빌더 시그니처는 `build(llm=, registry=, **config)`로 통일되어 있다.
1. **중첩**: 공통 계약(`core/contract.py`)을 지키는 그래프는 `AgentTool`로 감싸 어떤 패턴의 도구로든 넣을 수 있다. 측정한 쌍: react⊃react, react⊃plan, plan⊃react, plan⊃plan.
   - 입력: `query`, `history`, `context`(선택), `iteration`
   - 출력: `final`, `result`, `status`(`ok` / `error` / `capped`), `error`
   - `AgentTool(...)` 호출 인자에 `context`를 넣으면 자식 프롬프트에 전달된다.
   - 자식이 `ok`가 아닌 상태로 끝나면 부모는 실패 표식(`__failure__`)이 붙은 결과를 받는다. 그래서 "자식이 실패했다"와 "자식이 빈 답을 냈다"가 구분된다.
2. **도구 실행**: 모든 패턴은 `core/execution.py`의 `execute_tool` 하나로 도구를 실행한다. 승인 게이트, 실패 표식(`__failure__`), interrupt 통과, 실행 기록이 패턴과 무관하게 같다.
3. **HITL 중첩**: 자식 안의 승인 게이트와 `ask_human`은 부모 그래프를 멈춘다. 재개하면 자식이 멈춘 지점부터 이어진다. 부모에만 checkpointer를 주면 된다. 승인·거절·그래프 재생성 후 재개·같은 자식 2회 호출을 측정했다.
4. **동시 interrupt**: `plan_execute`의 병렬 레벨에서 여러 자식이 동시에 멈추면 `aresume(graph, {interrupt_id: verdict})`로 한 번에 재개한다.
5. **현재 한계** (계획 항목)
   - 자식에게 컨텍스트를 넘기는 것은 부모 모델이 `context` 인자를 직접 채울 때만 된다. 부모의 history나 결과를 골라 자동으로 넘기는 기능은 없다 (4B.1)
   - 자식을 읽기 전용으로 제한할 수 없다 (4B.3)

---

## 6. 출처

| ID | 자료 | 확인 |
|---|---|---|
| A1 | Anthropic, *Building effective agents* (2024-12) — anthropic.com/engineering/building-effective-agents | 원문 |
| A3 | Anthropic, *Building agents: When to fold subagents into your main agent loop* (2026-10) — claude.com/resources/articles/building-agents-when-to-fold-subagents-into-your-main-agent-loop | 원문 |
| A4 | Anthropic, *How we built our multi-agent research system* — anthropic.com/engineering/multi-agent-research-system | 원문 |
| A5 | Anthropic, *Effective context engineering for AI agents* — anthropic.com/engineering/effective-context-engineering-for-ai-agents | 원문 |
| A6 | Anthropic, *Harness design for long-running apps* — anthropic.com/engineering/harness-design-long-running-apps | 원문 |
| A7 | Anthropic, *Effective harnesses for long-running agents* — anthropic.com/engineering/effective-harnesses-for-long-running-agents | 원문 |
| A8 | OpenAI API docs: *Multi-agent*, *Programmatic tool calling*, *Skills* — developers.openai.com/api/docs/guides/… | 원문 |
| A9 | Cognition, *Multi-Agents: What's Actually Working* (2026-04) — cognition.com/blog/multi-agents-working | 원문 |
| A10 | Anthropic, *When to use multi-agent systems (and when not to)* — claude.com/blog/building-multi-agent-systems-when-and-how-to-use-them | 원문 |
| L1 | LangChain, *Deep Agents* (Harrison Chase, 2025-07-30) — blog.langchain.com/deep-agents | 원문 |
| L2 | LangChain docs, *Multi-agent* (Subagents / Handoffs / Skills / Router) — docs.langchain.com/oss/python/langchain/multi-agent | 원문 |
| P1 | Yao et al., *ReAct: Synergizing Reasoning and Acting in Language Models* — arXiv:2210.03629 | 초록 |
| P2 | Wang et al., *Plan-and-Solve Prompting* — arXiv:2305.04091 | 초록 |
| P3 | Xu et al., *ReWOO: Decoupling Reasoning from Observations* — arXiv:2305.18323 | 초록 |
| P4 | Kim et al., *An LLM Compiler for Parallel Function Calling* — arXiv:2312.04511 | 초록 |
| P5 | Shinn et al., *Reflexion: Language Agents with Verbal Reinforcement Learning* — arXiv:2303.11366 | 제목·날짜 |
| P6 | Wang et al., *Executable Code Actions Elicit Better LLM Agents* (CodeAct) — arXiv:2402.01030 | 초록 |
