# SMARTCS_Pole 프로젝트 개요

## 1. 목적

SMARTCS_Pole은 전주 계측 데이터를 기반으로 파단 위험을 예측하는 AI 파이프라인입니다.  
데이터 수집/정제부터 Light/Hard/MLP 모델 학습, 베스트 모델 관리, 배포 패키징까지 한 저장소에서 운영합니다.

## 2. 시스템 구성

- 데이터 준비: 원천 CSV를 정규화하고 학습 가능한 입력/라벨로 변환
- Light 모델: 파단/정상 이진 분류 중심의 1차 판단
- Hard 1차 모델: ROI 축(x/y/z)별 bbox 예측
- Hard 2차 모델: 1차 결과 기반 confidence head 보강
- MLP 모델: Light/Hard 출력 결합 최종 판정
- 결과 패키징: 모델/임계치/메타정보를 포함한 배포 패키지 생성

## 3. 디렉터리 개요

- `config/`: DB/환경 설정
- `main/1. make_data set/`: 데이터 수집·정제·ROI 편집
- `main/2. make_light_model/`: Light 학습/평가/베스트 관리
- `main/3. make_hard_model/`: Hard 1차/2차 학습/평가/베스트 관리
- `main/4. make_mlp_model/`: MLP 데이터 생성/학습/최종 베스트 구성
- `log/`: 공통 실행 로그

## 4. 베스트 모델 관리 원칙

- Light/Hard1/Hard2/MLP는 각각 `best_*` 디렉터리로 별칭 관리
- 교체/유지/스킵 이벤트를 히스토리 파일에 누적 기록
- 선택 기준과 성능 변화(delta)를 상세 로그로 보관

## 5. 로그 체계

- 공통 `logger.py` 기반으로 콘솔/파일 로그 포맷 통일
- 로그 접두어 표준:
- `[정보]`: 일반 진행 상황
- `[경고]`: 비치명 이슈
- `[오류]`: 실패/중단 원인

## 6. 실행 순서

상세 실행 순서는 `main/README.md`를 기준으로 관리합니다.

## 7. 참고 문서

- `main/README.md`: 전체 실행 흐름 요약
- `main/1. make_data set/README.md`: 데이터 단계
- `main/2. make_light_model/README.md`: Light 단계
- `main/3. make_hard_model/README.md`: Hard 단계
- `main/4. make_mlp_model/README.md`: MLP 단계
