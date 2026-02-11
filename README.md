# Telco Customer Churn Analysis

📊 Telco 고객 이탈 데이터를 기반으로 **고객 이탈 요인 탐색 → SQL/Python 기반 KPI 검증 → Streamlit 대시보드 시각화**까지 이어지는 프로젝트입니다.

---

## 프로젝트 구조

churn_project/
├─ app/
│ └─ churn_sql_streamlit.py # Streamlit 대시보드
├─ data/
│ └─ telco_churn.csv # 분석용 데이터
├─ notebooks/
│ ├─ pandas_eda.ipynb # 탐색적 데이터 분석 (EDA)
│ └─ sql_python_analysis.ipynb # SQL + Python KPI 분석
├─ .env # DB 접속 정보
├─ .gitignore
└─ requirements.txt

---

## 1️⃣ 프로젝트 개요

- **목적**: Telco 고객 데이터를 기반으로 **고객 이탈(Churn) 요인 탐색** 및 **고위험 세그먼트 식별**
- **데이터 범위**: 운영 DB(MySQL) 기준 전체 고객 데이터
- **분석 목적**
  1. 고객 이탈을 유발하는 핵심 변수 및 세그먼트 탐색
  2. SQL 기반 KPI 정의 및 Python을 활용한 검증
  3. Streamlit 대시보드를 통한 시각화 및 의사결정 지원

---

## 2️⃣ 분석 단계

### 2-1. 탐색적 데이터 분석 (EDA, `pandas_eda.ipynb`)
- 전체 고객 수, 이탈 고객 수, 이탈률 확인
- 타겟 변수 `Churn` 분포 확인 → 클래스 불균형 존재
- 단변량 분석
  - **Tenure**: 초기(0-5개월) 고객 이탈률 높음
  - **MonthlyCharges / TotalCharges**: 단기 이용 + 고비용 구조에서 이탈 집중
  - **Contract / InternetService**: Month-to-month, Fiber optic 고객군에서 이탈률 높음
- 변수 간 조합 분석 → 고위험 세그먼트 식별
  - Fiber optic × Month-to-month × 0–5개월 → 이탈률 최고

**핵심 인사이트**
- 초기 고객 및 Fiber optic 서비스 고객군 집중 관리 필요
- 단기 계약 고객의 장기 계약 전환 유도

---

### 2-2. SQL + Python 기반 KPI 분석 (`sql_python_analysis.ipynb`)
- **운영 DB 기준 KPI**
  - 전체 고객 수, 이탈 고객 수, 이탈률 산출
  - SQL 결과와 Pandas 재계산 결과 일치 확인
- **tenure_group 정의**
  - 0–5개월 / 6–11개월 / 12–23개월 / 24개월 이상
- 계약 유형별, 서비스 유형별, tenure 그룹별 이탈률 분석
- 핵심 세그먼트 도출
  - Fiber optic × Month-to-month × 0–5개월 → 최고 위험군
- Streamlit 대시보드용 최종 KPI 테이블 생성

---

### 2-3. Streamlit 대시보드 (`churn_sql_streamlit.py`)
- **목표**: SQL + Python 분석 결과를 시각화하고, 고위험 세그먼트 모니터링
- **주요 기능**
  1. 전체 고객 수 / 이탈률 KPI Overview
  2. Contract / InternetService / Tenure 별 이탈률 시각화
  3. 핵심 세그먼트(Contract × Tenure × Fiber optic) 히트맵
  4. Charges Analysis: 월 요금 분포 비교
  5. 인사이트 요약 및 액션 제안
  6. Appendix: SQL 쿼리 및 데이터 검증 결과
- **환경**
  - Python 3.9+, Streamlit, Pandas, Matplotlib
  - MySQL (CSV 대체 가능)
  - Linux / Windows 호환

---

## 3️⃣ 분석 결과 요약

| 구분 | 핵심 인사이트 |
|------|----------------|
| 신규 고객 | 0–5개월 초기 이탈 집중 → 온보딩 및 초기 관리 필요 |
| 요금 구조 | 단기 고객 이탈률 높음 → 고월요금 고객 할인·번들 제안 |
| 계약 전략 | Month-to-month → 장기 계약 유도 필요 |
| 서비스 유형 | Fiber optic 신규 고객 → 집중 관리 필요 |
| 핵심 세그먼트 | Fiber optic × Month-to-month × 0–5개월 → 최고 위험군, 약 75% 이탈률 |

---

## 4️⃣ 실행 방법

### 4-1. 환경 준비
```bash
pip install -r requirements.txt
```

### 4-2. CSV 기준 분석 (재현성 확보)
```bash
streamlit run app/churn_sql_streamlit.py
```

### 4-3. MySQL 연결 (실무용)
- .env 파일에 DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME 설정
- Streamlit에서 load_data_from_mysql() 함수 주석 해제 후 실행

---