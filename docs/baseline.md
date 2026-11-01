# 기준선 — 조합 행렬 수정 전후

> 측정일 2026-11-01 KST · FakeLLM (네트워크 없음) · `python -m agent.examples.compose_matrix`
> 수정 전 = HEAD `a15e99b`, 수정 후 = 브랜치 `feat/pattern-standardization` (Phase 0~1)

## 요약
| | PASS / 전체 |
|---|---|
| 수정 전 | **20 / 42** |
| 수정 후 | **47 / 47** (신규 시나리오 5개 포함) |
| 기존 테스트 (`pytest`) | 97 → 98 passed (기존 97개는 **수정 없이** 통과, 조합 행렬 진입점 테스트 1개 추가) |

수정 전 22건 FAIL의 원인 분류
| 원인 | FAIL 수 | 수정 항목 |
|---|---|---|
| plan_execute가 `requires_approval`을 무시 (D1). 자식으로 들어가도 그대로 (react⊃plan에서 승인 없이 실행) | 7 | 1.1 / 1.2 |
| plan_execute executor가 `GraphInterrupt`를 삼킴 (D2). `ask_human`, 중첩 승인, 병렬 interrupt 모두 실패 | 12 | 1.1 / 1.3 |
| planner 출력이 깨지면 실행 전체가 죽음 (D3) | 1 | 1.4 |
| 없는 도구를 계획 단계에서 거르지 못함 (D4) | 1 | 1.5 |
| `"error"` 필드가 있는 정상 데이터를 실패로 판정 (D5) | 1 | 1.6 |

## 시나리오별 결과
| 시나리오 | 셀 | 수정 전 | 수정 후 | 항목 |
|---|---|---|---|---|
| approval_gate | react | PASS | PASS | 1.2 |
| ask_human | react | PASS | PASS | 1.1/1.3 |
| llm_garbage | react | PASS | PASS | 1.4 |
| garbage_then_valid | react | (신규) | PASS | 1.4 |
| approval_gate | plan | **FAIL** | PASS | 1.2 |
| ask_human | plan | **FAIL** | PASS | 1.1/1.3 |
| llm_garbage | plan | **FAIL** | PASS | 1.4 |
| garbage_then_valid | plan | (신규) | PASS | 1.4 |
| unknown_tool | plan | **FAIL** | PASS | 1.5 |
| error_field_in_data | plan | **FAIL** | PASS | 1.6 |
| batch_mixed_verdict | plan | (신규) | PASS | 1.2 |
| approval_shows_resolved_args | plan | (신규) | PASS | 1.2 |
| replan_resets_approval | plan | (신규) | PASS | 1.2 |
| normal / child_tool_error / child_llm_garbage | 4개 쌍 전부 | PASS | PASS | — |
| child_approval_approve / reject / restart | react⊃react | PASS | PASS | 1.1/1.2 |
| child_ask_human, same_child_twice | react⊃react | PASS | PASS | 1.1 |
| child_approval_approve / reject / restart | react⊃plan, plan⊃react, plan⊃plan | **FAIL** | PASS | 1.1/1.2 |
| child_ask_human | react⊃plan, plan⊃react, plan⊃plan | **FAIL** | PASS | 1.1/1.3 |
| same_child_twice | react⊃plan, plan⊃react, plan⊃plan | **FAIL** | PASS | 1.1/1.2 |
| parallel_interrupts | plan⊃react, plan⊃plan | **FAIL** | PASS | 1.3 |

## 실행 기록(trace) 검증
- CLI `--demo-hitl --trace` (plan, react): `run_start → llm_call → tool_call → interrupt → resume → approval → tool_call → llm_call → run_end`. 재개할 때 이벤트 중복 없음.
- 중첩 (plan ⊃ react×2, 병렬 승인 2건): 자식별 `llm_call` 2회(`span=["kid1"]`, `["kid2"]`), `approval` 각 1회, `danger` 실행 각 1회. 부모가 재개될 때 자식 이벤트 중복 없음.

## Phase 3 이후 (계약·등록부) — 62 / 62 PASS

추가된 시나리오 (전부 PASS)
| 시나리오 | 셀 | 확인 내용 |
|---|---|---|
| registry_coverage | — | 등록부의 모든 패턴(`plan_execute`, `react`)에 행렬 어댑터가 있음. 새 패턴을 등록하고 어댑터를 빠뜨리면 FAIL |
| llm_brace_garbage | react, plan | `{not: json,,}`처럼 중괄호는 있지만 JSON이 아닌 출력이 계속 나와도 `status=error`로 종료. **수정 전 react는 `JSONDecodeError`로 실행 전체가 죽었음** [측정: 원본 HEAD `a15e99b`] → 결함 D6 |
| status_contract | react, plan | 정상 종료는 `status=ok`, 상한 도달은 `status=capped` |
| status_not_leaked | react, plan | 같은 thread 2회차가 상한에 걸리면 `capped`. 호출자가 `status`를 초기화하지 않아도 됨 (패턴이 스스로 지움) |
| child_llm_garbage (강화) | 4개 쌍 | 부모가 자식 실패를 `{"error": "sub-agent 'kid' ended with status=error: …", "__failure__": true}`로 받음. 이전에는 빈 문자열이었음 |
| child_capped | 4개 쌍 | 자식이 상한에 걸리면 부모가 `status=capped` 실패를 받음 |
| context_to_child | 4개 쌍 | `AgentTool` 호출 인자의 `context`가 자식의 첫 프롬프트에 포함됨 |

## Phase 5 이후 (컨텍스트 관리, opt-in) — 73 / 73 PASS

**기본 동작 불변 확인**: `ContextPolicy`를 주지 않았을 때 react·planner·joiner 프롬프트 9종(빈 상태, 긴 관찰·한글·에러 결과, 정확히 1500자 경계, history, 재계획 결과)이 **원본 HEAD `a15e99b`와 바이트 단위로 동일** [측정].

| 시나리오 | 확인 내용 | 관찰 |
|---|---|---|
| ctx_offload | 3,506자 관찰의 2,990자 지점 NEEDLE: 미리보기에는 없음 → `read_observation(step=1, offset=2990)` 후 보임. **정책 없으면 끝까지 못 봄** | PASS |
| ctx_read_bad_args | 없는 step, 정수 아닌 step → 에러 관찰, 실행 계속 | PASS |
| ctx_clearing | `keep_recent=1`: 이전 2개 cleared, 최근 1개 전문 | PASS |
| ctx_compaction | 임계 초과 시 요약 1회, 마지막 프롬프트에 요약 + 최근 스텝, 오래된 원문 없음 | PASS |
| ctx_compaction_chain | 8스텝에서 요약 3회, 2회차부터 이전 요약을 입력으로 이어받음 | PASS |
| ctx_notes_survive_pause | 메모 → 승인 대기 → **그래프 재생성 후 재개** → 메모가 프롬프트에 남음 | PASS |
| ctx_skills | 처음엔 이름·설명만, `load_skill` 후 본문 | PASS |
| ctx_tool_search | 도구 26개 중 핵심 1개만 노출 → `tool_search("weather")` → 다음 프롬프트에 노출 → 실행 | PASS |
| ctx_name_clash | 하네스 도구 이름과 겹치는 도구 등록 → 빌드 시 거부 | PASS |
| ctx_nested_child | plan 부모 안의 정책 켠 react 자식이 뒷부분을 읽고 결과 반환 | PASS |
| ctx_plan_results_cap | joiner 프롬프트: 정책 262자 / 정책 없음 5,031자 | PASS |

실행 기록: 하네스 도구는 `tool_call outcome=harness`, 요약은 `llm_call caller=react.compaction`으로 구분되어 남음 [측정].

**실모델로만 확인 가능한 것 (아직 안 함)**: 9B가 포인터를 보고 실제로 `read_observation`을 부르는지, 요약이 필요한 사실을 보존하는지, `tool_search`로 필요한 도구를 찾는지. 이 기능들은 그때까지 opt-in으로 둔다.

## 실모델 평가 (WORKPLAN 0.3) — 2026-11-01

- 모델: **Qwen3.8-27B-INT4** (vLLM 0.24.0, `--max-model-len 8192 --max-num-seqs 2 --reasoning-parser qwen3`, `--gpu-memory-utilization 0.78`), thinking off, temperature 0
- 과제: `examples/evals/run_evals.py`의 9개 (가짜 도구, 정답 고정) × 조건 3개 (react / plan_execute / react+`ContextPolicy(offload, keep_recent=4, tool_search_threshold=10)`)
- **표본 크기: 과제·조건마다 1회.** temperature 0이라 재현성은 높지만, 아래 숫자는 경향이지 확정이 아니다

### 1회차 (수정 전): 20 / 27
| 조건 | 성공 | 도구 실패 | 실패 원인 [측정: trace] |
|---|---|---|---|
| react | 5/9 | 7 | **인자 이름 추측 6건**: `from_ccy`→`from_currency`/`from`/`source`, `qty`→`quantity`. react 프롬프트에 args 스키마가 없음 (L1). + 큰 관찰 1건 |
| plan_execute | 9/9 | 1 | (스키마를 받는 planner는 인자 이름 오류 0건) |
| react+policy | 6/9 | 6 | react와 같은 인자 이름 추측 |

추가로 실모델에서만 드러난 결함 **D7**: 종료 시 답을 `final`이 아니라 `result`에 넣어 **빈 답으로 `status=ok`**. 1회차 전에 스키마(`anyOf`)와 복구 로직으로 수정함.

### 2회차 (L1·4A.3 수정 후): 26 / 27
수정: react 프롬프트에 args 스키마 포함 (도구 목록 포맷터를 planner와 하나로 통일), 인자 오류 시 에러에 기대 스키마를 붙임.

| 조건 | 성공 1→2회차 | 도구 실패 | LLM 호출 | 실행 시간 합 | 프롬프트 토큰 합 |
|---|---|---|---|---|---|
| react | 5 → **8** | 7 → **0** | 28 → 24 | 83.9 → 61.0초 | 9,883 → 14,309 (+45%, 스키마 분량) |
| plan_execute | 9 → 9 | 1 → 1 | 22 → 22 | 76.6 → 74.8초 | 8,952 → 9,500 |
| react+policy | 6 → **9** | 6 → **0** | 31 → 28 | 96.8 → 71.2초 | 13,099 → 18,667 |

### 관찰
- **큰 관찰 (large_observation)**: react 기본은 1·2회차 모두 실패. 잘린 문서만 보고 "배당 $0.00, NVIDIA는 배당을 하지 않는다"고 **사전 지식으로 지어냄** (`…[truncated]` 표시를 무시). react+policy는 두 번 모두 모델이 포인터를 보고 **스스로 `read_observation`을 3회 호출**해 4,500자 지점까지 읽고 정답. 비용: 프롬프트 토큰 1,308 → 6,037 (과제 1건)
- **도구 31개 (many_tools)**: react+policy에서 `tool_search` 후 정답. 프롬프트 토큰 2,132(전체 노출) → 1,161
- **plan_execute 중첩**: planner가 서브에이전트의 **문장형 답을 `$1`로 숫자 인자에 그대로 치환**해 1회 실패 → 재계획으로 복구. 자식이 구조화 결과를 돌려줘야 해결됨 (4B.4 결과 봉투의 근거)
- **파싱 재시도**: 54회 실행 동안 0건. 1.4 재시도 문구는 실제로 발동하지 않아 효과는 미확인 (동작 이상도 없음)

## 큰 관찰 변형 과제 (react offload 기본값 결정 근거) — 2026-11-01

같은 모델·설정. `--suite lo` 7과제 × (react 기본 / react+offload / plan_execute), 과제·조건마다 1회.

| 과제 | 내용 | react 기본 | react+offload 1차 → 2차 | plan_execute 1차 → 2차 |
|---|---|---|---|---|
| lo_original | 답이 4,500자 지점 | FAIL (지어냄: "배당 $0") | OK (읽기 3회) → OK (1회) | OK → OK |
| lo_early | 답이 미리보기 안(400자) | OK | OK, **읽기 0회** → OK, 0회 | OK → OK |
| lo_absent | 답이 문서에 없음 | OK* (읽지 않고 "없다"고 단정) | OK (끝까지 3회 읽고 "없다") → OK (`find` 1회) | **FAIL** (같은 조회 3회 재계획 → capped, 빈 답) → OK |
| lo_split | 답이 2곳(2,000·4,200자) | FAIL ("없다" 단정) | **FAIL** (읽은 조각의 step 번호를 다시 읽음) → OK | OK → OK |
| lo_korean | 한글, 4,500자 지점 | FAIL ("없다" 단정) | OK (3회) → OK (1회) | **CRASH** (D8: 한글 이스케이프로 컨텍스트 초과) → OK |
| lo_deep | 12,000자 중 10,500자 | FAIL ("없다" 단정) | **FAIL** (6회 순차 읽기 → 스텝 상한) → OK (`find` 1회, 3스텝) | OK → OK |
| lo_huge | 60,000자, 답은 800자 | OK | OK, 읽기 0회 → OK | **CRASH** (D9: 컨텍스트 초과) → OK (잘라서 재시도) |
| **합계** | | **3/7** (2차도 동일) | 4/7 → **7/7** | 4/7 → **7/7** |

\* react 기본의 lo_absent 성공은 끝까지 확인한 결과가 아니라, 다른 3과제에서 오답을 낸 것과 같은 "미리보기만 보고 없다고 단정"한 결과다.

1차 → 2차 사이 수정: `read_observation`에 `find`와 다음 호출 안내 추가, D8(`ensure_ascii=False`), D9(컨텍스트 초과 시 잘라서 재시도 / react는 `status=error`), joiner 마지막 라운드 finish-only 스키마 + "없으면 없다고 끝내라" 지침.

토큰 (react+offload, 1차 → 2차): lo_original 6,037 → 2,648 · lo_korean 9,992 → 3,914 · lo_deep 13,590(capped) → 2,716.

### 기본값 변경 후 확인 (react, `DEFAULT_REACT_POLICY`)
| 묶음 | 결과 | 비고 |
|---|---|---|
| 일반 9과제 (`--suite main`) | **9/9** | 읽기 도구 호출은 `large_observation` 1건뿐 (불필요한 읽기 0). 추가 비용 예: single_lookup 1,036 → 1,140 토큰 (호출 2회) |
| 큰 관찰 7과제 (`--suite lo`) | **7/7** | react+offload 2차와 같은 결과 |

근거 강도: 과제·조건마다 1회(temperature 0). 16과제에서 회귀 0, 큰 관찰에서 3/7 → 7/7은 방향이 일관되지만, 과제 수가 적어 다른 도메인·모델로의 일반화는 확인하지 않음.

## 신규 워크플로 패턴 실모델 평가 (`--suite wf`) — 2026-11-01

같은 모델·설정. 과제마다 1회. 위임 대상은 react 서브에이전트(`price_agent`, 가짜 도구).

| 과제 | 패턴 | 1차 | 2차 | 비고 |
|---|---|---|---|---|
| router_fx / router_price | router | FAIL / FAIL | OK / OK | 1차: 모델이 **에이전트 경로에 구조화 args**(`amount`, `from_currency`)를 채워 `query`가 빠짐 → 에이전트 대상은 항상 `query`로 위임하게 수정 |
| parallel_sections_sum | parallel (sectioning + LLM 종합) | OK | OK | |
| parallel_voting | parallel (voting 3표) | OK | OK | |
| evaluator_summary | evaluator_optimizer | OK | OK | 1라운드 통과 |
| orchestrator_compare | orchestrator_workers | OK | OK | 워커 2개 + 종합 |
| chain_price_to_krw | prompt_chain (에이전트 → 숫자 추출 gate → 환전 도구) | OK | OK | |
| nested_structured | plan_execute + `AgentTool(output_schema)` | OK (도구 실패 1, 35.5초) | OK (실패 0, 17.2초) | 1차: planner가 `$1.price_usd` 대신 `$1` 전체를 넣음 — 시스템 프롬프트가 필드 참조 문법을 안 알려줌 → 추가 |
| **합계** | | **6/8** | **8/8** | |

planner 프롬프트 변경 후 plan_execute 일반 9과제 재확인: **9/9**. 대조: `output_schema` 없는 `nested_researcher`는 여전히 문장형 답을 숫자 인자에 넣어 1회 실패 후 복구.

## 남은 관찰
- 같은 `thread_id`로 react를 다시 실행하면 이전 scratchpad가 누적됨 [측정: 1회차 1스텝 → 2회차 2스텝]. plan_execute는 새 계획마다 결과를 비우므로 누적 없음 [측정]. **결정: 막지 않고 문서화** (thread = 요청 1건 규약. 누적을 대화 기억으로 쓰는 경우도 있음). `react/state.py`, README에 기재
- **실모델 기준선(0.3)**: 아직 없음. vLLM 필요
