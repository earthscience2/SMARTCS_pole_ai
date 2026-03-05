# 로그 규약

## 1) 로그 레벨
- `INFO`: 정상 흐름, 단계 시작/종료, 주요 결과 요약
- `WARN`: 비정상 가능성은 있으나 처리 계속 가능(대체 경로, 누락 파일 일부 등)
- `ERROR`: 기능 실패/중단, 재시도 또는 원인 분석 필요

## 2) 메시지 키워드 표준
- `APP_START`: 스크립트 시작
- `APP_END`: 스크립트 종료
- `DATA_LOAD`: 데이터 로드
- `DATA_SAVE`: 데이터 저장
- `MODEL_SELECT`: 베스트 모델 선택/교체
- `MODEL_TRAIN`: 모델 학습
- `MODEL_EVAL`: 모델 평가 결과
- `EVAL_STAGE`: 평가 단계 진입/완료
- `MODEL_EXPORT`: 모델 내보내기
- `DB_CONNECT`: DB 연결
- `DB_QUERY`: DB 조회/실행
- `FILE_IO`: 파일 입출력
- 미정의 키워드 사용 시 `GENERAL`로 기록

## 3) 메시지 포맷
- 기본 포맷: `[KEYWORD] message | key1=value1 key2=value2`
- 예시:
  - `[MODEL_SELECT] 베스트 모델 교체 | old_run=20260301_1010 new_run=20260305_2140`
  - `[EVAL_STAGE] 2차 검증 완료 | auc=0.9134 f1=0.8044`

## 4) 코드 사용 규칙
- 공통 로거 생성: `from logger import get_logger`
- 표준 이벤트 로깅: `from logger import log_event`
- 권장 호출:
  - `log_event(LOGGER, "INFO", "MODEL_SELECT", "베스트 모델 유지", run=run_name)`
  - `log_event(LOGGER, "WARN", "FILE_IO", "입력 파일 일부 누락", path=file_path)`
  - `log_event(LOGGER, "ERROR", "MODEL_EVAL", "평가 실패", reason=str(e))`
