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

패턴은 **표준 명칭으로 등록부에 등록**되어 있습니다 (`python -m agent.run --list-patterns`).
패턴별 업계 표준 명칭·출처·조합 규칙은 [`docs/PATTERNS.md`](docs/PATTERNS.md)에 있습니다.

## 패턴 목록 (등록부)

| 등록 이름 (별칭) | 표준 명칭 | 하는 일 |
|---|---|---|
| `react` | ReAct | 한 스텝씩: 추론 → 도구 1개 → 관찰 → … → 종료 |
| `plan_execute` (`plan`) | Plan-and-Execute + LLMCompiler | 도구 호출 DAG 계획 → 레벨 병렬 실행 → joiner 종료/재계획 |
| `prompt_chain` (`chain`) | Prompt chaining | 고정 단계(LLM 프롬프트 또는 도구) 순서 실행, 단계 사이 gate |
| `router` | Routing | 요청 분류(스키마로 경로 제한) → 경로 하나에 위임 |
| `parallel` (`parallelization`) | Parallelization | sectioning(섹션 병렬 → 이어붙임/LLM 종합) 또는 voting(n회 → 다수결) |
| `evaluator_optimizer` (`evaluator`) | Evaluator-optimizer | 생성 → 깨끗한 컨텍스트의 평가자가 기준 채점 → 수정, 상한까지 |
| `orchestrator_workers` (`orchestrator`) | Orchestrator-workers | 실행 중 하위 과제 결정 → 워커(기본 읽기 전용) 병렬 → 종합 또는 추가 라운드 |

**워크플로 패턴(아래 5개)의 위임 대상은 등록부의 도구**입니다. 서브에이전트는 `AgentTool`로 넣습니다.
그래서 승인 게이트·중첩 HITL·병렬 interrupt·읽기 전용·실패 전달·trace가 모든 패턴에 똑같이 적용됩니다.

```python
from agent.core.tools import AgentTool, ToolRegistry
from agent.patterns import registry as patterns

tools = ToolRegistry()
tools.add(AgentTool("researcher", patterns.get("react").build(llm=llm, registry=search_tools),
                    description="looks things up", read_only=True,
                    output_schema={"type": "object", "properties": {"price_usd": {"type": "number"}},
                                   "required": ["price_usd"]}))
router = patterns.get("router").build(llm=llm, registry=tools, routes={"researcher": "price questions"})
```

`AgentTool` 옵션:
- `read_only=True`: 자식(과 그 자식) 안에서 `writes=True` 도구를 거절 (single writer)
- `output_schema={...}`: 자식에게 그 형태의 `result`를 요청하고 그 객체를 반환. planner는 `returns:`를 보고
  `$1.price_usd`로 필드를 참조. 형태가 맞지 않으면 실패로 반환 (문장이 숫자 자리에 들어가지 않음)
- 호출 인자 `context=...`: 자식 프롬프트에 "Context from the caller"로 전달

## 공통 계약 (`core/contract.py`) — 패턴이 서로 끼워지는 이유

모든 패턴은 같은 입력으로 시작하고 같은 키로 끝납니다.

| | 키 |
|---|---|
| 입력 | `query`, `history`, `context`(선택: 호출자가 넘기는 컨텍스트), `iteration` |
| 출력 | `final`, `result`(선택: 구조화 결과), `status`, `error` |

`status`는 실행이 어떻게 끝났는지입니다: `ok`(패턴이 스스로 끝냄) / `error`(재시도 후에도
모델 출력을 쓸 수 없었음) / `capped`(스텝·재계획 상한에 걸림). `envelope(state)`가 어느
패턴의 결과든 이 형태로 정규화합니다.

`AgentTool`로 감싼 자식이 `ok`가 아닌 상태로 끝나면 부모는 실패 표식이 붙은 결과
(`{"error": "sub-agent 'x' ended with status=capped: …", "__failure__": true}`)를 받습니다.
`status` 키가 없는 기존(계약 이전) 그래프는 예전처럼 결과를 그대로 넘깁니다.

```python
from agent.patterns import registry as patterns
graph = patterns.get("react").build(llm=llm, registry=tools, max_steps=8)
graph = patterns.get("plan").build(llm=llm, registry=tools, checkpointer=saver)  # 별칭 가능
```

참고: react는 같은 `thread_id`로 다시 실행하면 이전 scratchpad가 누적됩니다
(thread = 요청 1건 규약).

## 컨텍스트 관리 (`core/context.py`, opt-in)

빌더에 `context_policy=ContextPolicy(...)`로 지정합니다. **react의 기본값은
`DEFAULT_REACT_POLICY = ContextPolicy(offload=True)`** 입니다 — 실모델 평가에서 잘라내기만 하면
미리보기 밖의 답을 읽지 않고 "없다"고 단정하거나 지어냈기 때문입니다 (`docs/baseline.md`).
`context_policy=None`이면 원래 프롬프트(1,500자 잘라내기, 읽기 도구 없음)입니다.
plan_execute는 기본이 `None`(결과 전체 표시)이고, 프롬프트가 컨텍스트를 넘으면 결과를 잘라 한 번 재시도합니다. 효과는 모델에 따라 다르므로 기본값은 모두 꺼져 있습니다.

| 옵션 | 기능 (표준 명칭) | 적용 |
|---|---|---|
| `offload=True` | 긴 관찰은 미리보기 + 포인터, `read_observation(step, offset)`로 이어 읽거나 `find="단어"`로 해당 위치로 점프 (just-in-time context). **react 기본** | react |
| `keep_recent=N` | 최근 N개 관찰만 전문, 이전 것은 한 줄 표시 (tool-result clearing) | react |
| `compact_at_chars=N` | 누적 스텝이 N자를 넘으면 오래된 스텝을 모델이 요약 (compaction) | react |
| `notes=True` | `notes_write(text)`, 메모는 매 프롬프트에 표시 (structured note-taking) | react |
| `skills=(Skill(...),)` | 스킬 이름·설명만 보여주고 `load_skill(name)`으로 본문 로드 (progressive disclosure). `load_skills(dir)`가 `<dir>/*/SKILL.md`를 읽음 | react |
| `tool_search_threshold=N`, `core_tools=(...)` | 도구가 N개를 넘으면 핵심 도구만 노출, `tool_search(query)`로 찾은 도구를 추가 노출 | react |
| `inline_chars=N` | 관찰·결과 한 건의 최대 노출 길이 (기본 1500) | react, plan_execute(planner·joiner의 결과 표시) |

`read_observation`, `notes_write`, `load_skill`, `tool_search`는 루프가 직접 처리하는
**하네스 도구**입니다(상태를 읽고 씀, 승인 게이트 없음). 같은 이름의 도구를 등록하면 빌드할 때 거부됩니다.

```python
from agent.core.context import ContextPolicy, load_skills
policy = ContextPolicy(offload=True, keep_recent=4, notes=True, skills=tuple(load_skills("skills/")))
graph = patterns.get("react").build(llm=llm, registry=tools, context_policy=policy)
```

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

도구 목록은 planner와 같은 포맷터(`ToolRegistry.format_for_prompt`)로 만들며 **args 스키마를 포함**합니다.
(2026-11-01 실모델 평가: 스키마가 없을 때 Qwen3.8-27B가 인자 이름을 추측해 9과제 중 3과제 실패 → 포함 후 0건.
자세한 수치는 `docs/baseline.md`.) 인자 이름이 틀리면 도구 에러에 기대 스키마가 붙어 돌아갑니다.

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
| `core/hitl.py` *(공유)* | HITL verdict 계약 (`approved`, `feedback_of`) + 공용 `arun_graph`/`aresume_graph` (다중 interrupt 재개) |
| `core/execution.py` *(공유)* | **모든 패턴이 거치는 단일 도구 실행 경로** — 승인 게이트, 실패 표식, interrupt 통과, trace |
| `core/trace.py` *(공유)* | 실행 기록 (JSONL, 중첩 서브에이전트는 `span`으로 구분). 기본 꺼짐 |
| `core/checkpoint.py` *(공유)* | 로컬 SQLite checkpointer + thread 캡 pruner (최근 30개) |
| `core/contract.py` *(공유)* | 공통 입출력 계약, `status`, `envelope()` |
| `core/context.py` *(공유)* | 컨텍스트 관리 옵션 (`ContextPolicy`, 하네스 도구, 스킬 로더) |
| `patterns/registry.py` | 패턴 등록부 (`PatternSpec`: 표준 명칭·빌더·메타데이터), `patterns/builtin.py`에서 7종 등록 |
| `patterns/workflows.py` | 워크플로 패턴 5종 (prompt_chain, router, parallel, evaluator_optimizer, orchestrator_workers) |
| `run.py` | CLI 엔트리포인트 (`--type <패턴 또는 별칭>`, `--list-patterns`, `--demo`는 오프라인, `--trace DIR`) |
| `examples/compose_matrix.py` | 조합 행렬: 등록된 모든 (부모, 자식) 패턴 쌍 × HITL·실패·계약 시나리오를 E2E 검증 |
| `docs/` | `PATTERNS.md`(용어·표준 명칭), `WORKPLAN.md`(작업 계획), `baseline.md`(검증 기록) |

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
python -m agent.run --demo-hitl --type plan    # 승인 필요 도구(publish)에서 멈추는 데모
python -m agent.run "hello agent" --demo --type router   # 워크플로 패턴 오프라인 데모 (chain/router/parallel/evaluator/orchestrator)
python -m agent.run --demo --type react --trace runs/   # 실행 기록(JSONL) 남기기
python -m agent.examples.compose_matrix        # 조합 행렬 (FakeLLM, 표 출력)
python -m agent.examples.evals.run_evals --base-url http://<IP>:8002/v1 \
    --model qwen3.8-27b-redhat-int4 \
    --sampling '{"chat_template_kwargs": {"enable_thinking": false}}'   # 실모델 평가
    # --suite main (일반 9과제 × 조건) | lo (큰 관찰 7과제) | wf (워크플로 패턴 8과제)
python -m pytest agent/tests -q                # 테스트 (조합 행렬 포함)
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
- **tool 승인 (모든 패턴)** — `registry.register(..., requires_approval=True)`로
  플래그된 tool은 실행 직전에 멈춥니다. 거절하면 tool은 실행되지 않고 거절
  사유(`{"denied": ...}`)가 결과로 들어갑니다.
  - react: 호출 1건마다 멈춤 (payload `tool_approval`).
  - plan_execute: 레벨을 내보내기 **전에** 그 레벨의 승인 필요 task를 모아
    **한 번만** 멈춤. 1건이면 `tool_approval`, 여러 건이면 `tool_approvals`
    (`items` 목록). 여러 건의 verdict는 전체 일괄(`True` 등) 또는 task별
    `{"decisions": {task_id: verdict}}`. 승인된 task는 실행되고, 거절된 task는
    거절 결과가 남으며, **다음 레벨은 실행되지 않고** joiner가 판단합니다.
- **`ask_human` (모든 패턴)** — `core/hitl.py`의 tool. 등록해두면 **모델이** 필요할
  때 질문+선택지를 만들어 멈추고, 사람의 답이 tool의 반환값으로 들어가 추론이
  이어집니다. 구조적 게이트의 보완재.

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

**중첩과 동시 interrupt**: `AgentTool`로 감싼 서브에이전트 안의 승인 게이트·
`ask_human`도 부모 그래프를 멈추고, 부모를 재개하면 자식이 멈춘 지점부터
이어집니다 (부모에만 checkpointer를 주면 됨). plan_execute의 병렬 레벨에서 여러
task가 동시에 멈추면 `state["__interrupt__"]`에 여러 개가 오고,
`aresume(graph, {interrupt.id: verdict, ...})`로 한 번에 재개합니다. 이 동작은
`examples/compose_matrix.py`가 패턴 쌍마다 검증합니다 (승인/거절/재시작/같은 자식
2회 호출/병렬 2건).

## 확장 (→ 뉴스→반응 에이전트)

1. `ToolRegistry`에 실제 tool 등록 (예: `fetch_article`, `get_price_window`,
   `extract_catalyst`, `predict_reaction`).
2. `PLANNER_SYSTEM` / `JOINER_SYSTEM` 프롬프트를 도메인에 맞게 조정.
3. `OpenAICompatLLM`을 vLLM 서버로 연결해 실행.

구조적 코어(`plan_execute/dag.py`, `plan_execute/graph.py`,
`plan_execute/state.py`)는 그대로 유지됩니다.
