# basic_agent 패턴 표준화 작업 계획

> 상태: **초안 v3 — 승인 전.** 아직 커밋·push 하지 않았다.
> 작성: 2026-11-01 KST · 범위: 기존 코드 개선 + 패턴 문서화·등록 + 신규 패턴
> 작업 브랜치: `feat/pattern-standardization` (로컬)

## 결정 기록
| 날짜 | 결정 |
|---|---|
| 2026-11-01 | 기존 코드도 개선 대상에 포함한다 |
| 2026-11-01 | **vendoring 사본(agentic_rag, news_reaction_agent)은 범위 밖.** 레퍼런스용 사본이므로, 나중에 그쪽 구조를 개선할 때 따로 반영한다. 이번 작업은 basic_agent 레포 한 곳에 집중한다 |
| 2026-11-01 | 실모델 검증은 전체 필수가 아니다. 구조 변경은 FakeLLM으로 검증하고, 행동 변경은 opt-in으로 넣는다 (원칙 5) |
| 2026-11-01 | FakeLLM으로 가능한 부분부터 진행한다. 실모델이 필요한 지점에 오면 사용자에게 알리고, 사용자가 vLLM을 띄워준다 |
| 2026-11-01 | (구현 중 결정) plan_execute 일괄 승인의 task별 판정은 `{"decisions": {task_id: verdict}}`로 감싼다. 감싸지 않은 `{task_id: ...}`는 LangGraph의 `{interrupt_id: ...}` 재개 매핑과 충돌함 [측정: 정수 키에서 LangGraph `TypeError`, `pregel/_loop.py:912`] |
| 2026-11-01 | (구현 중 결정) 레벨 안에서 일부 task가 거절되어도 **같은 레벨의 승인된 task는 실행**하고, **다음 레벨부터** early-abort 한다. 같은 레벨 task는 서로 독립이고, 사람이 task별로 판정했기 때문 |
| 2026-11-01 | 위 두 구현 중 결정을 사용자가 확인함. Phase 2~3을 FakeLLM으로 이어서 진행 (커밋은 사용자 요청 시) |

## 진행 현황
| Phase | 상태 | 비고 |
|---|---|---|
| 0 | **완료** | 0.3 실모델 평가: Qwen3.8-27B로 27회 × 2라운드. 20/27 → 26/27 (`docs/baseline.md`) |
| 1 | **완료** | 조합 행렬 20/42 → 47/47, 기존 테스트 97개 무수정 통과. 상세는 `docs/baseline.md` |
| 2 | **완료** | `docs/PATTERNS.md`. 학술 출처 6건은 arXiv에서 제목·날짜·초록 확인, LangChain 최신 분류(Subagents/Handoffs/Skills/Router) 반영 |
| 3 | **완료** | `core/contract.py`, `patterns/registry.py`, `run.py --list-patterns`. 행렬 62/62, 테스트 98 통과. 진행 중 형제 결함 D6 발견·수정 |
| 5 | **구조 완료** (opt-in) | `core/context.py` + react 통합, plan_execute는 결과 표시 제한만. 행렬 73/73. 정책 없을 때 프롬프트가 원본과 바이트 동일함을 확인. **효과(행동) 검증은 9B 필요** |
| 4 | 일부 완료 | **4A.1 (react args 스키마), 4A.3 (인자 오류에 기대 스키마)** — 실모델 근거로 기본값에 반영. 나머지 미착수 |
| 6, 7 | 미착수 | |

**2026-11-01 결정: react에서 `offload`(just-in-time 읽기)를 기본으로 켬** (사용자가 "더 고민해보고 맞는 쪽으로" 요청 → 추가 검증 후 결정)
- 근거: 큰 관찰 변형 7과제에서 기본(잘라내기)은 미리보기 밖 답을 **한 번도 읽지 않고** 4과제 모두 "없다"고 단정하거나 지어냄. offload는 인터페이스 수정 후 7/7. 일반 9과제에서 손해 0. 필요 없는 과제의 추가 비용은 호출당 약 60토큰
- 함께 바꾼 것: `read_observation`에 `find`(위치 점프)와 "다음 호출" 안내 추가 (실모델이 읽은 조각의 step 번호를 다시 읽는 실수를 함), D8·D9 수정, joiner 마지막 라운드 finish-only 스키마 + "정보가 없으면 없다고 끝내라" 지침
- 되돌리기: `build_react_agent(..., context_policy=None)`이면 원래 프롬프트

**Phase 3에서 정한 것**
- 자식 실패 전달: 계약을 아는 자식(`status` 키가 있음)이 `ok`가 아닌 상태로 끝나면 `AgentTool`이 실패 표식 결과를 반환한다. `status` 키가 없는 기존 그래프는 그대로 통과시킨다 (하위 호환)
- `status` 초기화는 호출자가 아니라 **패턴이 스스로** 한다 (react는 끝내지 않은 스텝마다, plan은 계획할 때마다). 호출자가 초기화를 빠뜨려 이전 실행의 `ok`가 남는 문제를 행렬에서 발견해 이렇게 바꿈
- react의 `history`·`context`는 **값이 있을 때만** 프롬프트에 들어간다 → 기본 프롬프트는 바뀌지 않음 (원칙 5의 `[프롬프트]` 영향 없음)
- 같은 thread 재실행 시 react scratchpad 누적: 막지 않고 문서화

**Phase 5에서 정한 것**
- 관찰값 저장소를 따로 두지 않는다. react는 이미 관찰 전문을 상태(=체크포인트)에 저장하고 프롬프트 표시만 잘랐으므로, `read_observation`은 상태에서 읽는다
- `read_observation`·`notes_write`·`load_skill`·`tool_search`는 루프가 직접 처리하는 하네스 도구 (등록부·승인 게이트를 거치지 않음, trace에는 `outcome=harness`로 기록)
- 계획(5.1)의 "요약 + 핸들"은 "앞부분 미리보기 + 포인터"로 구현. 요약을 만들려면 모델 호출이 더 필요하기 때문

**실모델 (2026-11-01 확인)**: 이 계획의 "9B"는 **Qwen3.8-27B-INT4**로 바꾼다 (사용자가 주로 쓰는 모델).
- 띄우는 법: `/mnt/d/workspace/customer-service-agent/scripts/serve.sh start 27b` (전체 스택) 또는 그 안의 27B `vllm serve` 줄만
- 접속: `--base-url http://<WSL IP>:8002/v1 --model qwen3.8-27b-redhat-int4 --sampling '{"chat_template_kwargs": {"enable_thinking": false}}'`
  - 서버가 `--reasoning-parser qwen3`로 떠서 사고 모드를 요청마다 꺼야 한다. 안 끄면 출력 토큰이 추론에 쓰여 JSON이 잘릴 수 있음 (customer-service-agent는 기본으로 끔). basic_agent는 `sampling`으로 끌 수 있음을 요청 본문으로 확인함 [측정]
- 제약: `max-model-len 8192` → **프롬프트 예산이 빡빡하다** (Phase 5의 `inline_chars`·clearing·compaction이 실제로 의미 있는 이유). `max-num-seqs 2` → 병렬 서브에이전트는 동시 2개까지만 실제로 병렬
- 지연: 짧은 답 3~4초, 첫 글자 약 2.8초 (customer-service-agent `docs/04`의 기록 인용, 직접 측정 아님)
- `~/models`에는 27B만 있음. 9B-FP8은 없음 [측정: `ls ~/models`]

**실모델이 필요한 지점 (사용자에게 요청할 것)**
- 0.3 실모델 기준선
- 1.4의 재시도 피드백 문구 `[프롬프트]`: 출력이 깨진 경우에만 동작하므로(이전에는 실행이 죽던 경로) 기본 동작이 바뀌는 범위는 작다. 9B 스모크는 0.3과 함께 돌린다
- Phase 5 `[행동]` 항목의 효과: 9B가 포인터를 보고 `read_observation`을 부르는지, compaction 요약이 필요한 사실을 보존하는지, `tool_search`로 도구를 찾는지. 확인 전까지 opt-in 유지

## 표기 규약
- **[측정]**: 2026-11-01 세션에서 실제로 실행해 확인한 것. 프로브 스크립트는 §1.3에 있다.
- **[코드]**: 코드를 읽고 판단했지만 실행하지는 않은 것.
- **[R#]**: 근거가 된 참고 자료. 목록과 확인 여부는 §5에 있다.

---

## 0. 목적

1. 에이전트 패턴과 용어를 업계 표준 명칭으로 정리한 문서(`docs/PATTERNS.md`)를 둔다.
2. 모든 패턴(기존 2종 + 신규)을 표준 명칭으로 **등록부(registry)**에 올린다.
3. **어떤 패턴이든 다른 패턴 안에 넣을 수 있는 성질**을 유지하고, 그 범위를 넓힌다. 범위를 넓힌다는 것은 컨텍스트 전달, 권한, HITL, 결과 형식까지 일관되게 맞추는 것이다.
4. 기존 코드에서 확인된 결함과 한계를 고친다.

---

## 1. 현황

### 1.1 기본 정보 [측정]
| 항목 | 값 |
|---|---|
| 원본 | `github.com/Taeksu-Kim/basic_agent` (public, 마지막 push 2026-07-18, HEAD `a15e99b`) |
| 로컬 위치 | `/mnt/d/workspace/basic_agent/agent/` (import 하려면 폴더명이 `agent`여야 함) |
| 테스트 | `pytest` **97 passed** (README에는 82개라고 적혀 있음) |
| 데모 | `--demo --type plan`, `--demo --type react` 정상 |
| 실제 모델 | vLLM `10.5.0.2:8000` 꺼져 있음. `anthropic` 패키지·API 키 없음 → **실모델 검증은 0건** |
| vendoring 사본 | `agentic_rag/agent`는 원본과 같음. `news_reaction_agent/agent`는 이전 버전. **이번 범위 밖** (결정 기록 참조) |

### 1.2 조합·HITL 동작 프로브 결과 [측정]
| # | 시나리오 | 결과 |
|---|---|---|
| P1 | plan_execute에서 `requires_approval=True` 도구 실행 | **승인 없이 실행됨.** interrupt가 걸리지 않음 |
| P2 | plan_execute에서 `ask_human` 실행 (task 1개) | **interrupt가 삼켜짐.** 결과가 `{"error": "GraphInterrupt: ..."}`로 기록되고 사람에게 묻지 않음 |
| P3a | plan ⊃ react (AgentTool) | 정상 (`{1: 'ABC'}`) |
| P3b | react ⊃ plan (AgentTool) | 정상 (`XYZ`) |
| P4 | 같은 레벨에 서브에이전트 3개, 각자 LLM 호출 1초 (동기) | 전체 1.01초 → **병렬로 실행됨** |
| P5 | react ⊃ react, 자식이 승인 필요 도구 호출 | 부모가 멈춤. 승인하면 도구 1회 실행, 거절하면 0회 실행. 자식 LLM 중복 호출 없음 → **중첩 HITL이 동작함.** README의 "HITL은 최상위 그래프 전용"은 이 경우에 맞지 않음 |
| P6 | planner가 JSON이 아닌 출력을 냄 | **실행 전체가 `PlanError`로 종료.** 같은 상황에서 react는 재시도 후 정상 종료 |
| P7 | planner가 없는 도구 이름을 냄 | 계획 검증을 통과하고 실행 때 `ToolError` 발생 → joiner가 판단 |

### 1.3 한계·결함 목록
| ID | 내용 | 근거 | 위치 |
|---|---|---|---|
| **D1** | plan_execute가 `requires_approval`을 무시함 (**안전 결함**) | [측정] P1 | `plan_execute/executor.py` |
| **D2** | plan_execute executor가 `GraphInterrupt`까지 잡아서 에러로 기록함. react의 `_run_tool`은 이 예외를 다시 던져서 정상인데, 같은 구조의 executor만 빠졌음 | [측정] P2 | `executor.py:24` ↔ `react/graph.py:35` |
| **D3** | planner·joiner 출력이 깨지면 실행 전체가 죽음 | [측정] P6 | `planner.py:103-104`, `joiner.py:64-65` |
| D4 | 계획 검증 단계에서 도구 이름이 등록부에 있는지 확인하지 않음 | [측정] P7 | `plan_execute/dag.py` `validate_plan` |
| D5 | `_has_error`가 `"error"` 키가 있는 dict면 무조건 실패로 봄. 정상 데이터에 `error` 필드가 있어도 early-abort 됨 | [코드] → [측정] 행렬 | `plan_execute/graph.py:34` |
| D7 | react 행동 스키마·joiner 판정 스키마가 `action`/`decision`만 필수. 실모델(Qwen3.8-27B)이 종료 시 답을 `final` 대신 `result`에 넣어 **답 없이 `status=ok`로 종료** (joiner도 같은 구조). → 스키마를 `anyOf` 두 형태로 나눠 종료 시 `final` 필수 (vLLM guided decoding에서 동작 확인) + `result`에만 답이 있으면 `final`로 복구 | [측정] 실모델 | `react/policy.py` `ACTION_SCHEMA`, `joiner.py` `DECISION_SCHEMA` |
| D8 | planner·joiner·도구 목록이 `json.dumps` 기본값(`ensure_ascii=True`)으로 직렬화 → 한글 결과가 `\uXXXX`로 바뀌어 **6,017자 → 26,703자**, 8,192 토큰 초과로 plan_execute 실행이 죽음 | [측정] 실모델 `lo_korean` | `joiner.py`, `planner.py`, `tools.py`, `execution.py`, `llm.py` |
| D9 | 컨텍스트 초과(HTTP 400)를 처리하지 않음 → 실행 전체가 죽음 (60,000자 결과) | [측정] 실모델 `lo_huge` | `core/llm.py`, planner·joiner·react policy |
| D6 | react policy가 `ActionError`만 잡음. `{not: json,,}`처럼 중괄호가 있는 비-JSON 출력이면 `JSONDecodeError`로 실행 전체가 죽음 (D3의 형제 결함, Phase 3 중 발견) | [측정] 원본에서 재현 | `react/policy.py` `reason` |
| L1 | ReAct 프롬프트에 `args_schema`가 빠짐. planner 프롬프트에는 있음 | [코드] | `react/policy.py:78` ↔ `planner.py:88` |
| L2 | 서브에이전트가 항상 `history: []`로 시작함. 부모 컨텍스트를 받지 못함 | [코드] | `core/tools.py:110` |
| L3 | 중첩 HITL 중 확인되지 않은 경우가 있음: plan⊃(승인 필요 자식), 병렬 자식 여럿이 동시에 interrupt, 같은 자식을 한 실행에서 두 번 호출 | [측정] P5 + 미확인 | — |
| L4 | 서브에이전트를 읽기 전용으로 제한할 수 없음 (single writer 강제 불가) | [코드] | `core/tools.py` |
| L5 | ReAct가 한 스텝에 행동을 하나만 냄 | [코드] | `react/policy.py` `ACTION_SCHEMA` |
| L6 | ReAct 관찰값을 1500자에서 잘라냄. 뒷부분을 다시 볼 수 없음 | [코드] | `react/policy.py:63` |
| L7 | 공통 계약이 문서에만 있음. 코드상 Protocol이나 등록부가 없음 | [코드] | — |
| L8 | 계약이 패턴마다 다름: react는 `history`를 무시하고, plan_execute에는 구조화된 `result` 채널이 없음 | [코드] | `react/state.py`, `plan_execute/state.py` |
| L9 | plan_execute는 결과 전체를 길이 제한 없이 joiner·planner 프롬프트에 넣음 (react와 정반대) | [코드] | `joiner.py:58`, `planner.py:81` |
| L10 | `LLMClient`가 단일 턴 `complete(system, user)`만 있음. 대화 메시지 목록, 네이티브 tool calling, 토큰 사용량 보고가 없음 | [코드] | `core/llm.py` |
| L11 | 실행 기록(trace)이 없음. 체크포인트만 있어서 토큰·지연·스텝 수를 측정할 수 없음 | [코드] | — |
| L12 | README가 낡음 (테스트 수, HITL 제약) | [측정] | `README.md` |

> D1·D2는 같은 근본원인에서 나온 형제 결함이다. 도구 실행 경로가 react와 plan_execute에 따로 구현되어 있다. 그래서 고칠 때 **도구 실행을 공통 함수 하나로 합친다** (§3 Phase 1).

---

## 2. 설계 원칙

1. **조합은 두 축으로 나눈다.**
   - **중첩(nesting)**: 패턴 안에 패턴을 넣는다. 모든 패턴이 공통 계약을 지키므로 `AgentTool`로 감싸면 어디든 들어간다.
   - **부착(mixin)**: 컨텍스트 관리나 권한 정책처럼 패턴과 무관한 기능은 패턴으로 만들지 않고, 어느 패턴에든 끼우는 부품으로 만든다.
2. **도구 실행 경로를 하나로 둔다.** 승인, 권한, 에러 처리, interrupt 전달, 결과 오프로드, trace 기록은 모두 `core/execution.py` 한 곳에서 처리한다. 패턴은 이 함수만 호출한다. D1·D2의 재발을 구조적으로 막기 위함이다.
3. **하위 호환을 지킨다.** `build_graph`, `build_react_agent`, `AgentTool`, `ainvoke` 계약은 시그니처를 바꾸지 않고 기능만 추가한다. 나중에 레퍼런스 사본들을 옮겨올 때 비용을 줄이기 위해서다. 호환 여부는 basic_agent 자체의 기존 테스트 97개와 데모가 **수정 없이** 통과하는지로 확인한다.
4. **모델에 대한 가정을 명시한다.** 패턴마다 "9B에서 쓸 만한지"를 기록한다. 하네스의 구성요소 하나하나는 모델의 한계에 대한 가정이고, 모델이 바뀌면 그 가정을 다시 검증해야 한다 [R3].
5. **검증은 E2E로 하고, 변경 종류에 따라 검증 수단을 나눈다.** 자동 테스트는 그래프·CLI 진입점 단위로만 추가하고, 기존 FakeLLM 테스트는 유지한다.
   FakeLLM이 검증할 수 있는 것은 "모델이 X를 내면 시스템이 Y를 한다"까지다. "모델이 실제로 X를 내는가"는 검증하지 못한다.
   - **구조 변경 `[구조]`**: 제어 흐름, 계약, 조합, 권한, 저장소 동작. **FakeLLM 조합 행렬로 검증한다.** 실모델이 필요 없다.
   - **행동 변경 `[행동]`**: 모델이 더 잘하게 만드는 것이 목적인 변경. **기본값을 꺼둔 옵션(opt-in)으로 넣는다.** 기본값을 켜거나 문서에 "효과 있음"이라고 적는 것은 실모델로 검증한 뒤에만 한다.
   - **기본 프롬프트 변경 `[프롬프트]`**: 옵션 없이 기본 동작이 바뀌는 변경. 머지 전에 9B로 `examples/evals/` 스모크 테스트를 한 번 돌린다(대규모 평가 아님). vLLM을 켤 수 없으면 해당 항목만 opt-in으로 넣는다.

---

## 3. 단계

> 순서를 v1에서 바꿨다. **확인된 안전 결함(D1~D3)을 문서 작업보다 먼저** 고친다. 각 단계는 PR 하나에 대응한다(§6).

### Phase 0 — 측정 인프라와 기준선 (기능 변경 없음)

| # | 작업 | 상세 | 근거 |
|---|---|---|---|
| 0.1 | `core/trace.py` 실행 기록 | 실행 1회당 JSONL 파일 하나. 이벤트 종류: `llm_call`(프롬프트 길이, 토큰 수가 있으면 토큰, 지연), `tool_call`(이름, 인자, 소요 시간, 성공 여부), `interrupt`, `resume`, `final`. 자식 에이전트 이벤트는 부모 trace에 `span` 경로로 중첩 기록. 끄면 오버헤드 0 | 세션 로그를 하네스 밖에 둠 [R5]. L11 |
| 0.2 | `examples/compose_matrix.py` | 모든 (부모, 자식) 패턴 쌍 × 시나리오를 FakeLLM으로 실행하고 결과를 표로 출력. 시나리오: 정상 / 자식 에러 / 자식이 `max_steps` 도달 / 빈 결과 / 자식 승인 필요(승인·거절) / 자식 `ask_human` / 같은 자식 2회 호출 / 병렬 자식 2개가 동시에 interrupt / 같은 thread로 재실행 / 체크포인트에서 재시작. **§1.2의 P1~P7을 상설 시나리오로 포함** | CLAUDE.md 검증 규율 3 |
| 0.3 | `examples/evals/` 실모델 과제 세트 | 9B로 돌릴 고정 과제 8~10개. 도구 인자 정확도, 다단계 조회, 큰 관찰값, 병렬로 할 수 있는 조회, 승인 필요 행동을 각각 포함. 지표: 성공 여부, 스텝 수, LLM 호출 수, 토큰, 인자 오류 수, 실행 시간 | 개선 전 task-completion eval 기준선을 먼저 확보 [R1] |
| 0.4 | `docs/baseline.md` | 0.2 행렬 결과와 0.3 실모델 기준선(vLLM이 켜지면) 기록 | — |

**완료 기준**: 행렬에서 D1·D2·D3가 **실패로 재현됨.** 실모델 기준선이 기록됨(vLLM 필요).

### Phase 1 — 안전·정합 결함 수정 (D1~D5)

| # | 작업 | 상세 |
|---|---|---|
| 1.1 | `core/execution.py` 공통 도구 실행 경로 | 지금 `react/graph.py`의 `_run_tool`과 `plan_execute/executor.py`의 `run_task`에 따로 있는 로직을 합친다. 담당: 등록부 조회 → 권한·승인 검사 → 실행 → 에러를 관찰값으로 변환(**`GraphBubbleUp`은 다시 던짐**) → trace 기록. 두 패턴 모두 이 함수만 호출 |
| 1.2 | plan_execute 승인 게이트 (D1) | 기존의 "fan-out 전에 게이트" 원칙을 따른다. 레벨을 내보내기(dispatch) 전에, 그 레벨에서 승인이 필요한 task들을 **interrupt 한 번으로 모아서** 묻는다. task별로 승인·거절을 받고, 거절된 task는 `{"denied": 사유}`를 결과로 기록해 joiner가 보게 한다. 병렬 실행 도중에는 멈추지 않음 |
| 1.3 | plan_execute 안에서 interrupt 전달 (D2) | 1.1로 해결. task 하나에서 `ask_human`이나 자식 승인이 동작하는지 확인. **병렬 task 여러 개가 동시에 interrupt하는 경우**는 LangGraph가 interrupt ID별로 재개 값을 매핑하는 방식을 지원한다 [R10]. `run.py` 드라이버와 `aresume`을 여러 interrupt를 처리하게 확장 |
| 1.4 | planner·joiner 파싱 실패 처리 (D3) | react policy와 같은 방식: 에러 내용을 피드백으로 넣어 재시도 N회 → 그래도 실패하면 실행을 죽이지 않고 `status="error"`로 종료 |
| 1.5 | 계획 검증에 도구 이름 확인 추가 (D4) | `validate_plan(plan, registry)`에서 없는 도구를 걸러내고 1.4의 재시도 채널로 planner에 되돌림 |
| 1.6 | 에러 표식 명시화 (D5) | 실행 실패는 전용 표식(예: `ToolFailure` 또는 `{"__error__": ...}`)으로 감싼다. `_has_error`는 이 표식만 실패로 본다 |
| 1.7 | 같은 패턴 grep (규율 2) | `except Exception`이 도구나 그래프 호출을 감싸는 곳을 전부 찾아 1.1 경로로 통일. `AgentTool.run` 포함 |

**완료 기준**: 행렬에서 P1(승인 게이트 동작), P2(`ask_human`으로 멈춤), P6(정상 종료), P7(재계획) 칸이 통과로 바뀜. plan⊃(승인 필요 자식), 병렬 interrupt 2개 칸이 통과. 기존 테스트 97개 통과.

### Phase 2 — 용어집 `docs/PATTERNS.md` (문서만)

- **구성**
  1. 개요: workflow와 agent의 구분 [R2], "agent = 루프 안에서 도구를 자율적으로 쓰는 LLM" 정의 [R6]
  2. 5개 축별 표
     - 축 1 제어 주체: Workflow / Agent
     - 축 2 계획 방식: ReAct / Plan-and-Execute / ReWOO / LLMCompiler / Reflexion
     - 축 3 토폴로지: Single-agent / Orchestrator-subagent (=Supervisor, Manager, agents-as-tools) / Handoff / Hierarchical / Network·Swarm
     - 축 4 행동 형식: Tool calling / Parallel tool calls / CodeAct / Programmatic tool calling
     - 축 5 컨텍스트 관리: Compaction / Tool-result clearing / Structured note-taking / Just-in-time context / Progressive disclosure / Context isolation
  3. Anthropic workflow 5종 표 (정의, 언제 쓰는지, 원문 예시) [R2]
  4. 용어집: harness, ACI(agent-computer interface), context engineering, single writer, verification subagent, effort scaling, telephone game 등
  5. **basic_agent 대응표**: 각 패턴의 상태(있음 / 계획 Phase N / 범위 밖), 등록 이름, 9B 적합성
  6. **조합 규칙**: 무엇을 무엇 안에 넣을 수 있는지, 제약은 무엇인지 (Phase 0 행렬 결과로 채움)
- **항목마다 적을 것**: 표준 명칭, 별칭, 한 줄 정의, 언제 쓰는지, 언제 쓰지 말지, 출처와 확인 여부, 9B 적합성, basic_agent 대응
- **완료 기준**: 출처를 확인하지 않은 항목은 "확인 안 함"으로 표시. 기존 2종이 정확한 명칭으로 매핑됨

### Phase 3 — 공통 계약과 등록부 (L7, L8)

| # | 작업 | 상세 |
|---|---|---|
| 3.1 | `core/contract.py` | `AgentGraph` Protocol (`ainvoke(state, config) -> state`). **입력**: `query`, `history`, `context`(부모가 넘긴 것, 선택), `iteration`. **출력 봉투**: `final: str`, `result: Any \| None`, `status: "ok" \| "capped" \| "error" \| "denied"`, `artifacts: list[handle]`. 키 이름은 상수로 정의 |
| 3.2 | 기존 패턴을 계약에 맞춤 (L8) | react가 `history`와 `context`를 프롬프트에 반영. plan_execute에 `result`·`status` 채널 추가. `max_steps`나 반복 상한에 걸려 끝나면 `status="capped"` |
| 3.3 | `patterns/registry.py` | `PatternSpec(name, std_name, aliases, axis, builder, config_schema, sources, weak_model_ok, notes)`, `register()` / `get()` / `list()`. 빌더 시그니처 통일: `build(llm, registry, **config) -> AgentGraph` |
| 3.4 | 기존 2종 등록 | `react` (std_name: ReAct), `plan_execute` (Plan-and-Execute + LLMCompiler). 기존 빌더 함수를 그대로 가리키기만 함 |
| 3.5 | `run.py` | `--type`을 등록부에서 찾음 (기존 `plan` 값은 별칭으로 유지). `--list-patterns` 추가. `--trace <dir>` 추가 |
| 3.6 | 호환성 확인 | 기존 테스트 97개와 `--demo --type plan/react`가 **테스트 코드 수정 없이** 통과하는지 확인 |

**완료 기준**: `--list-patterns`가 목록 출력. 행렬이 등록부를 순회해 자동으로 쌍을 생성. 기존 테스트·데모 무수정 통과.

### Phase 4 — ReAct·조합·LLM 계층 개선

**4A. ReAct 품질**
| # | 작업 | 상세 | 근거 |
|---|---|---|---|
| 4A.1 | 프롬프트에 `args_schema` 포함 (L1) | planner와 같은 형식 사용. 도구 설명 포맷터를 공통 함수 하나로 합침 (규율 2) | 도구 문서화가 ACI 핵심 [R2] |
| 4A.2 | 한 스텝 여러 행동 (L5) | `actions: [{tool, args}, ...]` 허용. 단일 `action`도 계속 받음. 독립 행동은 `gather`로 병렬 실행하고, 승인은 행동별로 (1.2와 같은 일괄 interrupt) | 병렬 tool call [R8] |
| 4A.3 | 고칠 수 있는 에러 메시지 | 잘못된 인자일 때 기대 스키마와 받은 값을 관찰값에 포함 → 모델이 다음 스텝에서 수정 가능 | 도움이 되는 에러 메시지 [R7] |
| 4A.4 | 실행 효과 측정 | 0.3 과제로 개선 전후 인자 오류 수와 성공률 비교 (vLLM 필요) | — |

**4B. 조합 (`AgentTool`)**
| # | 작업 | 상세 | 근거 |
|---|---|---|---|
| 4B.1 | 컨텍스트 전달 (L2) | `AgentTool(context="none" \| "last_n:K" \| "fields:[...]")`. 부모의 history나 결과 중 무엇을 넘길지 선택. 기본값은 `none` (하위 호환) | OpenAI `fork_turns` [R8] |
| 4B.2 | 위임 명세 표준화 | 자식 호출 인자에 선택 필드 추가: `objective`, `output_format`, `tool_guidance`, `boundaries`. 오케스트레이터 프롬프트에 위임 작성 지침 포함 | 서브에이전트에 목표, 출력 형식, 도구 지침, 경계가 필요 [R4] |
| 4B.3 | 읽기 전용 자식 (L4) | `BaseTool.writes: bool` 속성 추가. `AgentTool(read_only=True)`이면 자식이 쓰기 도구를 호출할 때 1.1 경로에서 거절 관찰값 반환 | single-threaded writes [R9] |
| 4B.4 | 결과 봉투 | 자식은 3.1의 출력 봉투로 반환. 결과가 크면 본문 대신 artifact 핸들을 반환 (5.1과 연결) | 공통 응답 스키마 [R1], 파일시스템으로 출력해 전달 손실 줄이기 [R4] |
| 4B.5 | 중첩 HITL 정리 (L3) | Phase 1 이후 남은 경우를 행렬로 확인: 같은 자식 2회 호출, 3단 중첩, 체크포인트에서 재시작. README의 제약 문구를 측정 결과로 교체 | [R10] |
| 4B.6 | 자식별 LLM 지정 | 자식마다 다른 `llm`을 쓰는 것은 이미 가능함. 문서화하고 예제 추가 (예: 9B 메인 + 더 큰 모델 자식 = Smart Friend 형태) | [R9] |

**4C. LLM 계층 (L10)**
| # | 작업 | 상세 |
|---|---|---|
| 4C.1 | `LLMClient.chat(messages, tools=None, schema=None)` 선택 메서드 | `complete`는 유지. `chat`을 구현한 클라이언트에서는 다중 턴과 네이티브 tool calling 사용 가능. 없으면 `complete`로 폴백 |
| 4C.2 | 사용량 보고 | 응답의 토큰 사용량을 trace에 기록 (0.1) |
| 4C.3 | vLLM 네이티브 tool calling 검증 | 9B 모델과 vLLM의 tool parser 설정으로 동작하는지 실측한 뒤 채택 여부 결정. **확인 안 함** |
| 4C.4 | `AnthropicLLM` 갱신 | 지금은 스키마를 시스템 프롬프트에 덧붙이는 방식. 현행 API의 구조화 출력·tool use 지원 여부와 기본 모델 ID를 **구현 시점에 공식 문서로 확인한 뒤** 반영 |

**완료 기준**: 행렬 4B 칸 전부 통과, 기존 테스트·데모 무수정 통과. 4A.4 전후 비교표는 vLLM이 켜진 경우에만 작성하고, 없으면 원칙 5에 따라 해당 항목을 opt-in으로 둔다.

### Phase 5 — 컨텍스트 관리 mixin (L6, L9)

| # | 작업 | 상세 | 근거 |
|---|---|---|---|
| 5.1 | 관찰값 저장소 + 핸들 | 임계치를 넘는 결과는 저장하고 `요약 + handle`만 컨텍스트에 넣음. `read_observation(handle, offset, length)` 도구 제공. react의 1500자 잘라내기와 plan_execute의 무제한 덤프(L9)를 **둘 다** 이걸로 대체 | 가벼운 식별자를 들고 필요할 때 로드 (just-in-time) [R6] |
| 5.2 | 오래된 도구 결과 지우기 | N스텝 이전의 관찰값은 핸들 요약으로 대체. 가장 저렴한 형태의 compaction | tool-result clearing [R6] |
| 5.3 | Compaction | 누적 컨텍스트가 임계치를 넘으면 앞 스텝들을 요약. 요약 프롬프트는 "빠뜨리지 않기(recall) 우선"으로 조정 | [R6] |
| 5.4 | 노트 도구 | `notes_write` / `notes_read`. 진행 상황과 결정을 컨텍스트 밖에 저장 | structured note-taking [R6], 진행 파일 [R11] |
| 5.5 | Progressive disclosure: 스킬 | `SKILL.md` 형식(agentskills.io 스펙, Anthropic·OpenAI 공통). 프롬프트에는 이름과 설명만 넣고 `load_skill(name)`으로 본문 로드 | [R1], [R8] |
| 5.6 | Progressive disclosure: 도구 | 도구가 임계치(설정값, 기본 20)를 넘으면 핵심 도구만 노출하고 `tool_search(query)`로 나머지를 로드. 도구는 API 단위가 아니라 의도 단위로 묶고 이름에 접두사(namespacing) 사용 권장 | 도구 정의 30~50개를 넘으면 선택 정확도 저하 [R1], namespacing [R7] |
| 5.7 | mixin 적용 방식 | `ContextPolicy` 객체 하나를 `build_*(..., context_policy=...)`로 주입. react의 프롬프트 생성, planner·joiner의 결과 렌더링이 모두 이 정책을 거침 | 원칙 1 |

**완료 기준**: 1500자를 넘는 관찰값이 필요한 과제에서 잘라내기로는 실패하고 5.1로는 성공. 같은 과제의 토큰 사용량을 기준선과 비교. 9B에서 compaction 요약 품질은 실측 후 채택 여부 결정.

### Phase 6 — 신규 패턴

모든 신규 패턴은 다음을 지킨다.
- 3.1 계약을 따르고 등록부에 등록한다.
- 부모로도, 자식으로도 행렬을 통과한다.
- 도구는 1.1 경로로만 실행한다.
- `--demo --type <name>`을 제공한다.

| # | 패턴 (std_name) | 구성 | 설정 | 9B 적합성 | 근거 |
|---|---|---|---|---|---|
| 6.1 | `prompt_chain` (Prompt chaining) | 고정 단계 목록. 각 단계는 프롬프트 또는 하위 패턴. 단계 사이에 프로그램 검사(gate) 선택 | `steps`, `gates` | 높음 | [R2] |
| 6.2 | `router` (Routing) | 분류 LLM이 enum 스키마로 경로를 고르고, 등록된 패턴이나 에이전트 하나로 보냄. 기본 경로 필수 | `routes: {name: (description, target)}`, `fallback` | 높음 (분류 작업) | [R2] |
| 6.3 | `parallel` (Parallelization: Sectioning / Voting) | Sectioning은 정해진 섹션을 자식들에게 나눠 병렬 실행 후 취합. Voting은 같은 과제를 N번 실행하고 다수결, 임계치, 또는 LLM 판정으로 집계 | `mode`, `sections` 또는 `n`, `aggregator` | 높음 | [R2] |
| 6.4 | `evaluator_optimizer` (Evaluator-optimizer) | 생성자(자식 패턴) → 평가자(깨끗한 컨텍스트, 명시적 기준, few-shot 보정) → 피드백 반영 반복. 통과하거나 상한에 도달하면 종료. 생성자에게도 평가 기준을 공유. **아무 패턴에나 씌우는 검증 래퍼로도 사용 가능** | `criteria`, `max_rounds`, `pass_threshold` | 중간 (평가 기준이 명확할 때만) | [R2], [R3] |
| 6.5 | `orchestrator_workers` (Orchestrator-workers, 동적) | 오케스트레이터가 실행 중에 하위 과제(4B.2 위임 명세)를 만들고, 등록된 패턴으로 워커를 띄워 결과를 종합. 노력 조절 규칙(쉬운 질문은 워커 1개)을 프롬프트에 포함. 워커는 기본 읽기 전용이고 결과는 핸들로 반환. **미리 짠 도구 호출 DAG로 고정 라우팅하는 `plan_execute`와는 구분** | `worker_patterns`, `max_workers`, `max_concurrency`, `effort_rules` | 낮음~중간 (위임 품질이 모델에 의존) | [R2], [R4], [R8] |

**완료 기준**: N×N 행렬이 전부 통과. 패턴마다 데모 동작. 9B로 패턴별 대표 과제 1개 이상 실행해 결과 기록.

### Phase 7 — 문서·배포
- README 갱신: 패턴 목록(등록부에서 생성), 조합 규칙, HITL 제약을 실측 기준으로, 테스트 수
- `PATTERNS.md` 대응표 상태 갱신, `CHANGELOG.md` 작성
- 레퍼런스 사본으로 옮길 때 참고할 **마이그레이션 노트** 작성 (깨지는 동작, 새 옵션, D1~D3 수정 내역). 사본 자체는 수정하지 않음

---

## 4. 범위 밖 (후보로만 기록)
| 항목 | 이유 |
|---|---|
| Handoff (대화 주도권 이전) | 사용자와 직접 대화하는 자식이 필요할 때만 권장됨 [R1]. 현재 그런 수요 없음 |
| Programmatic tool calling / CodeAct | 코드 실행 샌드박스가 필요함 [R8] |
| Network / Swarm | 근거 자료가 비권장 ("mostly a distraction") [R9] |
| 장기 실행 하네스 (기능 목록, 진행 파일, 세션 재시작) | 5.4 노트 도구 이후 필요하면 검토 [R11] |
| 에이전트 간 메시징 (`send_message`, `followup_task`) | 6.5 이후 필요하면 검토 [R8] |

---

## 5. 참고 자료

확인 여부: **원문** = 2026-11-01 세션에서 원문을 읽음 / **요약만** = 검색 요약만 봄

| ID | 자료 | 확인 |
|---|---|---|
| R1 | Anthropic, *Building agents: When to fold subagents into your main agent loop* (2026-10) — claude.com/resources/articles/building-agents-when-to-fold-subagents-into-your-main-agent-loop | 원문 |
| R1' | Anthropic, *When to use multi-agent systems (and when not to)* — claude.com/blog/building-multi-agent-systems-when-and-how-to-use-them | 원문 |
| R2 | Anthropic, *Building effective agents* (2024-12) — anthropic.com/engineering/building-effective-agents | 원문 |
| R3 | Anthropic, *Harness design for long-running apps* — anthropic.com/engineering/harness-design-long-running-apps | 원문 |
| R4 | Anthropic, *How we built our multi-agent research system* — anthropic.com/engineering/multi-agent-research-system | 원문 |
| R5 | Anthropic, *Scaling Managed Agents with stable interfaces* — anthropic.com/engineering/managed-agents | 원문 |
| R6 | Anthropic, *Effective context engineering for AI agents* — anthropic.com/engineering/effective-context-engineering-for-ai-agents | 원문 |
| R7 | Anthropic, *Writing effective tools for AI agents* — anthropic.com/engineering/writing-tools-for-agents | 원문 |
| R8 | OpenAI API docs: *Multi-agent*, *Programmatic tool calling*, *Skills*, *Agents* — developers.openai.com/api/docs/guides/… | 원문 |
| R8' | OpenAI, *A practical guide to building agents* (PDF) | **요약만** (PDF 파싱 실패) |
| R9 | Cognition, *Multi-Agents: What's Actually Working* (2026-04) — cognition.com/blog/multi-agents-working | 원문 |
| R10 | LangGraph docs, *Interrupts* (다중 interrupt 재개, 노드 재실행 규칙) — docs.langchain.com/oss/python/langgraph/interrupts | 원문 |
| R11 | Anthropic, *Effective harnesses for long-running agents* — anthropic.com/engineering/effective-harnesses-for-long-running-agents | 원문 |
| R12 | claude.dev, *The new rules of context engineering for Claude 5 generation models* (2026-07-24) | 원문 일부 (본문 세부는 확인 안 함) |
| — | ReAct(Yao 2022), Plan-and-Solve(Wang 2023), ReWOO(Xu 2023), LLMCompiler(Kim 2023), Reflexion(Shinn 2023), CodeAct(Wang 2024), LangChain Deep Agents | **확인 안 함** (학습 지식). Phase 2에서 원문 확인 후 표기 갱신 |

---

## 6. PR 분할과 결정 필요 사항

**PR 분할 (권장, 모두 브랜치에서 PR. main 직접 push 없음)**
| PR | 포함 | 비고 |
|---|---|---|
| PR1 | Phase 0 + Phase 1 | 결함 재현 → 수정을 한 PR에서 전후 비교로 보여줌 |
| PR2 | Phase 2 + Phase 3 | 문서와 계약 |
| PR3 | Phase 4 | ReAct·조합·LLM |
| PR4 | Phase 5 | 컨텍스트 mixin |
| PR5 | Phase 6 + Phase 7 | 신규 패턴, 문서 마무리 |

**항목별 검증 구분 (원칙 5)**
| 구분 | 항목 |
|---|---|
| `[구조]` FakeLLM으로 충분 | Phase 0 전체, Phase 1 전체(1.4는 재시도 프롬프트 문구만 `[프롬프트]`), Phase 3 (3.2의 프롬프트 반영 부분 제외), 4B.1~4B.6, 4C.1·4C.2, 5.1·5.2·5.4·5.7의 동작, 6장 각 패턴의 조합 가능 여부 |
| `[프롬프트]` 머지 전 9B 스모크 | 1.4 재시도 피드백 문구, 3.2 react의 history·context 반영, 4A.1 인자 스키마 추가, 4A.3 에러 메시지 형식 |
| `[행동]` opt-in, 실모델 검증 후 기본값 전환 | 4A.2 한 스텝 여러 행동, 4C.3 네이티브 tool calling, 5.3 compaction, 5.5·5.6 progressive disclosure, 6.4 evaluator의 판정 품질, 6.5 orchestrator의 위임 품질 |

**결정 필요**
1. **PR 방식과 분할**: 위 안으로 갈지
2. **실모델**: 9B vLLM을 켤 수 있는 시점. `[프롬프트]` 항목 머지와 `[행동]` 항목 기본값 전환에만 필요하다. 프런티어 API로도 비교할지 (키·패키지 필요)
3. **Phase 5와 6의 순서**: 현재 안은 5(컨텍스트)를 먼저. 9B에서 효과가 더 확실하다고 판단해서임
