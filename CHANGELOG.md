# Changelog

## Unreleased — 하네스: 재시도·검증 (2026-11-13)

검증: pytest 104/104, 조합 행렬 138/138, 어댑터 점검 17/17, docsmith pytest 14/14. `verify` 미지정 시 기존 프롬프트
438건 중 기존 424건이 순서·내용 그대로 (늘어난 14건은 신규 시나리오).

### Added
- `OpenAICompatLLM(max_retries=2, backoff=1.0)`: timeout·연결 오류·HTTP 429/5xx를 지수 백오프로 재시도
  (`Retry-After` 존중, 최대 30초). 다 실패하면 `LLMUnavailableError`. 다른 4xx·컨텍스트 초과는 재시도 안 함.
  측정: 이전에는 timeout 1회에 react·plan_execute 실행이 그대로 죽었음.
  SDK 기반 클라이언트(`AnyLLMClient`, `AnthropicLLM`)는 SDK가 같은 재시도를 함
- `LLMUnavailableError`는 `status=error`로 바꾸지 않고 던짐: 체크포인터가 있으면 `graph.ainvoke(None, config)`로
  마지막 단계부터 재개되고 끝난 도구는 다시 돌지 않음 (행렬 `llm_outage_resume` ×2)
- `verify=` (opt-in, react·plan_execute): 최종 답을 코드로 검사. 거절 시 react는 이유를 관찰값으로 돌려 다시
  추론, plan_execute는 그 이유로 재계획. 상한에서 거절되면 `status=error` (답은 `final`에 남음).
  `core/contract.py`의 `Verifier`, `verification_failed` (행렬 `verify` ×4)

### Fixed
- 자식 에이전트의 LLM 장애가 `execute_tool`에서 도구 실패로 기록되어, 부모가 그럴듯한 답으로 **`status=ok` 종료**하던
  문제 → `LLMUnavailableError`는 다시 던짐 (행렬 `llm_outage_in_child`)

## Unreleased — 리팩터링 (2026-11-13)

### Added
- **모든 패턴에 `system=`** (도메인 지침): plan_execute(planner·joiner), prompt_chain, router, parallel,
  evaluator_optimizer(생성·평가), orchestrator_workers(분해·종합). 내장 프롬프트 뒤에 덧붙음
  (`core/llm.with_guidance`, react와 같은 방식). `None`이면 프롬프트 변화 없음. 행렬 `system_guidance` ×7
- `agent/prompts.py`: 내장 프롬프트 16개를 한 파일로 (원래 모듈에서도 그대로 import 가능)

검증: pytest 100/100, 조합 행렬 131/131, 어댑터 점검 17/17, docsmith pytest 9/9 + `--demo`. 리팩터링 전후 FakeLLM 프롬프트 411건과 시나리오
관찰값이 바이트 단위로 같음 (신규 행·신규 시나리오 호출 제외). react `policy=None` 프롬프트 5종도 동일.

### Fixed
- 컨텍스트 초과(`ContextOverflowError`)로 workflow 패턴 5종(prompt_chain, router, parallel(llm 집계),
  evaluator_optimizer, orchestrator_workers)과 react compaction 요약이 **실행째 죽던 문제** → `status=error`로 종료.
  행렬 `wf_overflow` ×5, `ctx_overflow_compaction` 추가
- workflow 설정의 없는 도구 이름이 실행 중 `ToolError` 크래시 → **build 시점 `ValueError`** (행렬 `wf_unknown_tool` ×5)
- plan 승인에서 사람이 고친 plan이 잘못되면 크래시 → 거절로 처리, 오류를 재계획 사유로 planner에 전달
  (행렬 `bad_edited_plan`)
- `AnthropicLLM`: 컨텍스트 초과를 `ContextOverflowError`로 변환 (다른 클라이언트와 동일). 실제 API로는 확인 안 함
- `OpenAICompatLLM`: 초과 판정 문구를 `AnyLLMClient`와 같은 목록으로 (전에는 "maximum context length"만)
- `run.py --state`: 실행은 끝나도 aiosqlite 연결이 안 닫혀 **프로세스가 종료되지 않던 문제** → 연결 닫음
- `@registry.tool`이 `requires_approval`·`writes`·`output_schema`를 받지 못해 데코레이터 도구에 승인 게이트를 걸 수 없던 문제

### Changed (동작 변화 없음 — 구조만)
- 모델 출력 파싱·재시도 하나로: `core/llm.py`의 `extract_json`, `ask_parsed`, `OutputError`
  (react·planner·joiner·workflows에 있던 사본 4개씩 제거, `planner.retry_feedback` 제거)
- 계약 헬퍼 `core/contract.py`: `ContractState`(각 패턴 state의 기반), `ended_with`, `as_text`, `render_inputs`
  (plan_execute state에 `result` 채널이 생김)
- react: `policy=None` 렌더링 분기 제거 → `ContextPolicy()`로 렌더 (`_format_scratchpad`, `OBS_CHAR_LIMIT` 제거;
  상한 1500자는 `ContextPolicy.inline_chars`)
- harness 도구 등록부 `HARNESS_TOOLS` (`@_harness`로 추가) + 행렬 `harness_coverage`
- interrupt payload 타입 상수 `core/hitl.py`: `InterruptType`, `TOOL_APPROVAL` 등 5종
- `core/llm.py`: `DEFAULT_SAMPLING`, `with_schema_instruction` (클라이언트 2개씩의 사본 통합).
  `AnthropicLLM`의 스키마 지시 문구가 `AnyLLMClient`의 것으로 바뀜 (프롬프트 변화, 이 클래스 사용처는 0)
- `run.py`: `_amain`/`_run`의 중복 인자 목록 제거

## Unreleased — `feat/any-llm-adapter` (2026-11-01)
### Added
- `AnyLLMClient` (mozilla-ai any-llm, 선택 의존성 `any-llm-sdk==1.37.0` 고정): `structured=auto|native|prompt`,
  `extra_body`, 토큰 사용량, 컨텍스트 초과 → `ContextOverflowError`. CLI `--provider`, `--extra-body`
- `examples/llm_adapter_check.py`: 실제 호출 없이 SDK 경로 + 가짜 HTTP로 제공자 3종(vLLM 호환, OpenAI, Anthropic) 점검 (17/17)
### Changed
- 스키마의 `const`/`enum`에 `"type": "string"` 명시 (react 행동, joiner, router, orchestrator). 의미는 같고,
  `type` 없는 `const`를 Anthropic SDK가 거부하던 문제를 피함. **실제 vLLM guided decoding에서는 아직 재확인 안 함**

## Unreleased — `feat/pattern-standardization` (2026-11-01)

검증: FakeLLM 조합 행렬 111/111 (`examples/compose_matrix.py`), 실모델 Qwen3.8-27B-INT4 평가
(`examples/evals/run_evals.py`: main 9/9·9/9, lo 7/7, wf 8/8). 상세 `docs/baseline.md`.

### Added
- 패턴 등록부 `patterns/registry.py` + 표준 명칭 7종: `react`, `plan_execute`, `prompt_chain`, `router`,
  `parallel`, `evaluator_optimizer`, `orchestrator_workers` (`run.py --list-patterns`, `--type <이름|별칭>`)
- 공통 계약 `core/contract.py`: `status`(ok/error/capped), `envelope()`, 입력 `context`
- 단일 도구 실행 경로 `core/execution.py` (승인 게이트, 실패 표식 `__failure__`, interrupt 통과, 읽기 전용 스코프)
- `AgentTool(read_only=..., output_schema=...)`, 도구 `writes=` 플래그, `$N.field` / `$N[0]` 참조
- 컨텍스트 관리 `core/context.py`: `ContextPolicy` (offload + `read_observation(step, offset|find)`, clearing,
  compaction, notes, skills, tool search), 컨텍스트 초과 시 결과를 잘라 재시도
- 실행 기록 `core/trace.py` (JSONL, 중첩 span, 토큰 사용량), `run.py --trace`, `--sampling`, `--demo-hitl`
- 다중 interrupt 재개 `aresume(graph, {interrupt_id: verdict})`, plan_execute 레벨 일괄 승인
  (`tool_approvals`, `{"decisions": {task_id: verdict}}`)
- 문서 `docs/PATTERNS.md`(분류·용어·출처), `docs/WORKPLAN.md`, `docs/baseline.md`, `docs/MIGRATION.md`

### Changed (기본 동작이 바뀜)
- **react 기본 `context_policy=DEFAULT_REACT_POLICY`** (offload 켬): 긴 관찰은 미리보기 + `read_observation`.
  예전 프롬프트는 `context_policy=None`
- react 도구 목록에 **args 스키마 포함** (planner와 같은 포맷터)
- react 행동 / joiner 판정 스키마가 `anyOf` 두 형태 (종료 시 `final` 필수)
- joiner: 재계획 상한 라운드는 finish만 허용, "정보가 없으면 없다고 끝내라" 지침
- planner 시스템 프롬프트에 `$ID.field` 설명 추가
- 같은 레벨에서 일부 거절 시 승인된 형제 task는 실행, 다음 레벨부터 중단
- `AgentTool`: 계약을 아는 자식이 `ok`가 아니게 끝나면 실패 결과 반환

### Fixed
- D1 plan_execute가 `requires_approval`을 무시하고 실행 · D2 plan_execute가 `GraphInterrupt`를 삼킴
  (`ask_human`·중첩 승인 불가) · D3 planner/joiner 출력 파싱 실패 시 실행 전체 종료 · D4 없는 도구를 계획
  단계에서 못 거름 · D5 `error` 필드가 있는 정상 데이터를 실패로 판정 · D6 react에서 중괄호가 있는 비-JSON
  출력에 `JSONDecodeError`로 종료 · D7 답을 `result`에만 넣으면 빈 답으로 `ok` 종료 · D8 프롬프트 JSON의
  `ensure_ascii`로 한글이 `\uXXXX`가 되어 컨텍스트 초과 · D9 컨텍스트 초과(HTTP 400) 시 실행 전체 종료
