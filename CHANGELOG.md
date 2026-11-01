# Changelog

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
