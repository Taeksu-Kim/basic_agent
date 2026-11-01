# 마이그레이션 노트 (vendoring 사본 → 이 브랜치)

`agentic_rag/agent`, `news_reaction_agent/agent` 같은 vendoring 사본을 이 버전으로 옮길 때 확인할 것.
사본 자체는 이번 작업에서 수정하지 않았다 (사용자 결정, 2026-11-01).

## 그대로 동작하는 것
- `build_graph`, `build_react_agent`, `AgentTool`, `arun`, `aresume` 시그니처 (인자 추가만 있음)
- 기존 테스트 97개가 테스트 코드 수정 없이 통과

## 동작이 바뀌는 것 (확인 필요)
| 바뀐 점 | 영향 | 예전처럼 하려면 |
|---|---|---|
| react 기본이 offload | 프롬프트에 `read_observation` 도구 한 줄 + 긴 관찰 표시 문구가 바뀜. 이름이 같은 도구가 등록돼 있으면 빌드 시 `ValueError` | `context_policy=None` |
| react 도구 목록에 args 스키마 | 프롬프트가 길어짐 (실모델 평가에서 인자 오류 7→0) | 없음 (권장) |
| react·joiner 스키마 `anyOf` | 종료 시 `final` 필수. vLLM guided decoding에서 동작 확인 | — |
| `AgentTool` 자식 실패 | 계약을 아는 자식(`status` 키)이 `ok`가 아니면 `{"error": ..., "__failure__": True}` 반환 (예전: 빈 문자열/None) | 자식 그래프가 `status`를 안 쓰면 예전과 같음 |
| plan_execute 승인 | `requires_approval` 도구가 이제 실제로 멈춤 (예전: 승인 없이 실행) | — (안전 수정) |
| 에러 결과 형태 | `{"error": ..., "__failure__": True}` (키 추가) | `"error" in result` 검사는 그대로 동작 |
| 재개 API | 동시 interrupt 여러 개면 `aresume(graph, {interrupt_id: verdict})`. 레벨 일괄 승인은 `{"decisions": {task_id: verdict}}` | 단일 interrupt는 예전과 같음 |

## 실모델로 쓸 때
- Qwen3 계열 + `--reasoning-parser qwen3` 서버면 `OpenAICompatLLM(sampling={"chat_template_kwargs": {"enable_thinking": False}})`
- 컨텍스트가 짧은 서버(8k)면 react 기본(offload)과 plan_execute의 초과 재시도가 중요
