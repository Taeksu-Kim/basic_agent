# Agent — 재사용 가능한 에이전트 골격 (LangGraph)

LangGraph 기반 범용 에이전트 골격. TDD로 작성했습니다. **하나의 substrate를
공유하는 두 개의 에이전트 타입**으로 구성돼 있어, 서로 조합(compose)됩니다
(한쪽을 다른 쪽의 tool로 감쌀 수 있음):

| 타입 | 패키지 | 패러다임 | 적합한 경우 |
|---|---|---|---|
| **Plan-and-Execute** | `plan_execute/` | orchestrated: task DAG를 계획 → 실행 → 리플랜 | 구조적 / 분해 가능 / 병렬화 가능 |
| **ReAct** | `react/` | reactive: LLM이 스스로 한 스텝씩 진행 | 오픈엔드 / 예측 불가 |

공유 substrate는 `core/`에 있습니다: `core/llm.py`(LLM 추상화)와
`core/tools.py`(tool + registry). 두 타입 모두 동일한 `ainvoke` 계약을 가진
그래프로 컴파일되고 `AgentTool`로 감쌀 수 있습니다. 주식 도메인의
*뉴스 → 주가 반응 예측* 에이전트는 이 위에 실제 tool과 프롬프트를 끼워 넣어
만듭니다.

## `plan_execute` — 패턴

두 표준 패턴을 합친 것입니다:

- **Plan-and-Execute** (planner → 실행 → 리플랜 루프), 그리고
- **LLMCompiler** (planner가 task **DAG**를 생성; `$N` 인자로 한 task의 출력을
  다른 task에 전달; 플랜 전체 실행 후 **joiner**가 종료/리플랜 판단).

오케스트레이터 = **planner + joiner + 루프 제어** (별도의 query rewriter /
intent classifier 없음 — planner가 흡수). 라우팅은 **정적**: 각 task의 `tool`
필드가 바인딩입니다.

```
START ─▶ plan ─(dispatch)─▶ run_task* ─▶ advance ─(dispatch)─▶ … ─▶ join
                                                                     │
                            finish / cap ◀──(route_join)─────────────┤
                            replan ───────────────────────────────────┘  (→ plan)
```

### 플랜은 DAG, 실행은 레벨 동기화

planner는 DAG(`{id, tool, args, deps}`, `$N` 참조는 암묵적 deps)를 출력합니다.
여기서 **위상정렬 레벨**을 계산해 각 레벨을 LangGraph `Send` fan-out으로 실행합니다
(레벨 내 병렬, 레벨 간 배리어) — LangGraph의 관용적 방식입니다. LLM은 레벨
그룹핑을 절대 만들지 않고, 레벨은 파생됩니다. 이는 full eager DAG 스케줄링의
의도적 단순화이며, 자료구조는 DAG로 유지되어 나중에 planner를 건드리지 않고도
executor를 eager dispatch로 승격할 수 있습니다.

### 리플랜

LLMCompiler 방식: **joiner**가 플랜 전체 실행 후 한 번 돌며 `finish`(최종 답변)
또는 `replan`(이유)을 반환합니다. 그래프는 계획 라운드를 상한(`max_iterations`)으로
막고, task가 하나라도 에러나면 플랜을 **early-abort**합니다(남은 레벨은 건너뛰고
joiner가 판단). 리플랜 시 결과 네임스페이스는 새 플랜을 위해 리셋됩니다.

## `react` — 패턴

단일 reactive 루프: `reason`(LLM이 지금까지의 trajectory를 보고 다음 행동 결정:
tool 호출 또는 finish) ↔ `act`(tool 실행, 관찰을 scratchpad에 append). `finish`
또는 `max_steps` 상한에서 종료합니다. `plan_execute`와 달리 tool 에러는 abort하지
**않고** 관찰로 피드백되어 에이전트가 반응할 수 있게 합니다. 미리 짜는 플랜이 없고,
적응성은 루프 자체에서 나옵니다.

```
START ─▶ reason ─(route)─▶ act ─▶ reason ─▶ … ─▶ END   (finish / max_steps)
```

## 레이아웃

| 파일 | 역할 |
|---|---|
| `core/llm.py` *(공유)* | `LLMClient` 프로토콜; `FakeLLM`(테스트); `OpenAICompatLLM`(vLLM), `AnthropicLLM` |
| `core/tools.py` *(공유)* | `BaseTool`/`FunctionTool`/`AgentTool` + `ToolRegistry` — task가 라우팅되는 실행 단위 |
| `plan_execute/dag.py` | `Task`/`Plan`, 검증, `topological_levels`, `$N` 치환 (순수) |
| `plan_execute/orchestrator/planner.py` | *(판단)* query → DAG (JSON 스키마 강제) + 레벨 계산 |
| `plan_execute/orchestrator/joiner.py` | *(판단)* finish / replan 분류기 |
| `plan_execute/executor.py` | *(실행)* 해석된 task 하나를 registry로 실행 |
| `plan_execute/state.py` | `AgentState` + `results` merge reducer |
| `plan_execute/graph.py` | StateGraph 조립; `build_graph`, `arun`, `aresume` |
| `react/state.py` | `ReactState` + scratchpad append reducer |
| `react/policy.py` | 추론 스텝: trajectory → 다음 행동 (tool / finish) |
| `react/graph.py` | reason↔act 루프; `build_react_agent`, `arun`, `aresume` |
| `core/hitl.py` *(공유)* | HITL verdict 계약 (`approved`, `feedback_of`) |
| `core/checkpoint.py` *(공유)* | 로컬 SQLite checkpointer + thread 캡 pruner (최근 30개) |
| `run.py` | CLI 엔트리포인트 (`--type plan\|react`, `--demo`는 오프라인) |

## LLM / 추론

**OpenAI 호환 엔드포인트 뒤의 vLLM**을 전제로 설계됐습니다. `planner`/`joiner`는
`complete(...)`에 JSON 스키마를 넘기고, `OpenAICompatLLM`이 이를
`response_format: {type: json_schema}`(vLLM guided decoding)로 전달해 유효한 JSON을
받습니다.

> 그래프 전체가 테스트에서 `FakeLLM`으로 실행됩니다 — **테스트 스위트에는 실제
> 모델 호출도 네트워크도 없습니다.**

## 실행

```bash
conda activate stock-dataset
python -m agent.run --demo --type plan         # 오프라인 스모크 테스트 (FakeLLM)
python -m agent.run --demo --type react
python -m agent.run "your task" --type react \
    --base-url http://localhost:8000/v1 --model <served-model>   # 실제 vLLM
python -m pytest agent/tests -q                # 82개 테스트
```

## Tool & 서브에이전트

실행 단위는 `BaseTool`(`name`, `description`, `args_schema`, `run(**args)`)입니다.
평범한 callable은 `registry.register(...)` / `@registry.tool(...)`로 등록하고, 더
풍부한 tool은 `BaseTool` 서브클래스 인스턴스를 `registry.add(...)`로 추가합니다.
tool의 `description`/`args_schema`는 planner에게 광고되어 라우팅을 돕습니다.

**서브에이전트도 그냥 tool입니다**: `AgentTool`이 컴파일된 그래프를 감싸서 플랜이
이름으로 라우팅할 수 있게 하고, 그 안에서 자기만의 루프를 돌린 뒤 결과를
반환합니다(*agent-as-tool* 패턴). executor나 그래프를 건드리지 않고 planned
멀티에이전트를 추가하는 방법입니다.

## HITL (human-in-the-loop) + 상태 저장

`interrupt()` 기반. 일시정지에는 checkpointer가 필수라 상태 저장과 한 몸입니다.
게이트(시스템이 강제로 멈춤) 3종 + `ask_human`(모델이 자발적으로 멈춤) 1종:

- **plan 승인** — `build_graph(..., approve_plans=True, checkpointer=...)`:
  플랜 직후 멈추고 사람에게 플랜을 보여줍니다. verdict는 승인(`True`/`"approve"`/
  `{"approved": True}`), 수정(`{"plan": [...]}` — 검증 후 그대로 실행), 거절
  (`{"feedback": "..."}` — 피드백이 planner의 리플랜 채널로 들어가 다시 플랜).
  게이트가 **fan-out 이전**에 있어 Send 병렬 태스크 중간에 멈추지 않습니다.
- **level 리뷰** — `build_graph(..., approve_levels=True, ...)`: 병렬 레벨의
  태스크가 **전부 합류한 배리어**에서 멈추고 그 레벨의 결과를 보여줍니다.
  승인 → 다음 레벨, 거절+피드백 → 리플랜. 배리어는 단일 실행 지점이라
  동시 interrupt 문제가 없습니다.
- **tool 승인 (react)** — `registry.register(..., requires_approval=True)`로
  플래그된 tool은 실행 직전에 멈춥니다. 거절하면 tool은 실행되지 않고 거절
  사유가 observation으로 들어가 policy가 다음 행동을 정합니다.
- **`ask_human` (react)** — `core/hitl.py`의 tool. 등록해두면 **모델이** 필요할
  때 질문+선택지를 만들어 멈추고, 사람의 답이 tool의 반환값(observation)으로
  들어가 추론이 이어집니다. 구조적 게이트의 보완재.

```python
from agent.core.checkpoint import open_async_saver, open_saver, prune_threads
from agent.plan_execute.graph import arun, aresume, build_graph

saver = await open_async_saver("agent_state.sqlite")   # 그래프는 ainvoke로 돌므로 async saver
graph = build_graph(llm=llm, registry=reg, approve_plans=True, checkpointer=saver)

state = await arun(graph, "task", thread_id="req-1")   # plan_approval에서 멈춤
state["__interrupt__"][0].value                        # {"type": "plan_approval", "plan": ...}
state = await aresume(graph, True, thread_id="req-1")  # 승인 → 계속 실행
prune_threads(open_saver("agent_state.sqlite"), keep=30)  # 최근 30개 요청(thread)만 유지
```

콘솔 데모 (`run.py`의 driver 루프가 payload `type`별로 렌더링/입력/재개):

```bash
python -m agent.run --demo --type plan --approve-levels               # 레벨마다 멈춤
python -m agent.run --demo --type plan --approve-plans --state s.db  # SQLite 영속
```

저장 단위: 체크포인트는 super-step마다 쌓이지만, 캡은 **thread(요청 1건) 기준
최근 30개**이고 삭제는 thread 통째로만 합니다 — 반쯤 지워진 thread는 재개가
불가능하기 때문입니다. saver는 2종: `open_async_saver`(그래프 실행용 —
동기 `SqliteSaver`는 `ainvoke`를 거부), `open_saver`(pruning/조회용, 같은 스키마).

현재 제약: HITL은 **최상위 그래프 전용**입니다. `AgentTool`로 감싼 서브에이전트
내부의 interrupt는 부모로 전파/재개되지 않으므로, 서브에이전트에는 승인 게이트를
켜지 마세요. `ask_human`도 plan_execute의 병렬 레벨 안에서는 쓰지 마세요
(동시 다발 interrupt의 resume 매핑 미지원 — react 전용).

## 확장 (→ 뉴스→반응 에이전트)

1. `ToolRegistry`에 실제 tool 등록 (예: `fetch_article`, `get_price_window`,
   `extract_catalyst`, `predict_reaction`).
2. `PLANNER_SYSTEM` / `JOINER_SYSTEM` 프롬프트를 도메인에 맞게 조정.
3. `OpenAICompatLLM`을 vLLM 서버로 연결해 실행.

구조적 코어(`plan_execute/dag.py`, `plan_execute/graph.py`,
`plan_execute/state.py`)는 그대로 유지됩니다.
