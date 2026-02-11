import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.ticker import PercentFormatter
import os
from sqlalchemy import create_engine
from dotenv import load_dotenv

# =========================
# 한글 설정
# =========================
rcParams["font.family"] = "NanumGothic"
rcParams["axes.unicode_minus"] = False

# =========================
# Page Config
# =========================
st.set_page_config(
    page_title="Telco Churn Segment Analysis",
    layout="wide"
)

# ======================================================
# MySQL → Python Connection (환경변수 기반, 실무용 참고)
# ======================================================
load_dotenv()

def load_data_from_mysql():
    """
    실무에서는 MySQL에서 직접 데이터를 불러오는 방식으로 분석을 시작합니다.
    환경변수(DB_USER, DB_PASSWORD 등) 기반으로 연결하며,  
    TotalCharges 컬럼 타입 변환이 실패한 행은 NaN으로 처리됩니다.
    포폴에서는 CSV 기준으로 분석하는 것이 재현성 확보에 유리하여,  
    이 함수는 실행하지 않습니다.
    """
    engine = create_engine(
        f"mysql+pymysql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
        f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    )
    query = "SELECT * FROM telco_churn;"
    df = pd.read_sql(query, engine)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    return df

# ======================================================
# Load Data (CSV 배포용, 포폴용)
# ======================================================
@st.cache_data(ttl=0)
def load_data():
    df = pd.read_csv("../data/telco_churn.csv")
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna(subset=['TotalCharges']).reset_index(drop=True)
    return df

df = load_data()

# ======================================================
# Data Validation
# ======================================================
def validate_data(df):
    result = {}
    result['row_count'] = len(df)
    critical_cols = ['Churn', 'Contract', 'InternetService', 'tenure', 'MonthlyCharges']
    result['missing_values'] = df[critical_cols].isnull().sum()
    result['invalid_churn'] = (~df['Churn'].isin(['Yes', 'No'])).sum()
    result['negative_charges'] = (df['MonthlyCharges'] < 0).sum()
    return result

validation = validate_data(df)

# ======================================================
# Preprocessing
# ======================================================
def tenure_grouping(x):
    if x < 6:
        return '0-5개월'
    elif x < 12:
        return '6-11개월'
    elif x < 24:
        return '12-23개월'
    else:
        return '24개월 이상'

df['tenure_group'] = df['tenure'].apply(tenure_grouping)

# ======================================================
# Sidebar
# ======================================================
st.sidebar.title("📌 Telco Churn Dashboard")
st.sidebar.markdown("""
**목표**  
고객 이탈을 유발하는 핵심 세그먼트를 구조적으로 파악

**분석 흐름**  
1️⃣ 계약 구조  
2️⃣ 서비스 유형  
3️⃣ 이용 기간  
4️⃣ 핵심 위험군 도출

**환경**
- Linux  
- MySQL 
- Python / Streamlit
""")

menu = st.sidebar.radio(
    "페이지 선택",
    [
        'Overview',
        'Contract → Churn',
        'InternetService → Churn',
        'Tenure → Churn',
        'Core Segment',
        'Charges Analysis',
        'Insight',
        'Appendix (SQL & Validation)'
    ]
)

# ======================================================
# Overview
# ======================================================
if menu == 'Overview':
    st.title("📊 Telco Customer Churn Overview")
    st.caption("MySQL 기반 데이터 → Python 분석 → Streamlit 자동 리포트")

    churn_rate = (df['Churn'] == 'Yes').mean()

    col1, col2, col3 = st.columns(3)
    col1.metric("전체 고객 수", f"{len(df):,}")
    col2.metric("이탈 고객 수", f"{(df['Churn']=='Yes').sum():,}")
    col3.metric("이탈률", f"{churn_rate:.2%}")

    st.markdown("### ✅ 데이터 정합성 검증 결과")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Row 수", validation['row_count'])
    c2.metric("Churn 이상값", validation['invalid_churn'])
    c3.metric("요금 음수 값", validation['negative_charges'])
    c4.metric("핵심 컬럼 결측치", int(validation['missing_values'].sum()))

    st.markdown(f"""
➡ 전체 이탈률 {churn_rate:.2%}로, Month-to-month 계약과 Fiber optic 서비스 고객에서 상대적으로 높음  
➡ 단기 고객(0-5개월) 중심의 초기 관리 필요
""")
    st.markdown("""
※ MySQL Import 과정에서 TotalCharges 타입 변환 실패로 11건이 누락됨  
  → CSV 기준 분석으로 재현 가능성 확보
""")
    st.markdown("**인사이트:** 초기 고객 및 Fiber optic 서비스 중심의 이탈 관리 필요")

# ======================================================
# Contract → Churn
# ======================================================
elif menu == 'Contract → Churn':
    st.title("📌 Contract 유형별 이탈률")

    contract_churn = (
        df.groupby('Contract')['Churn']
        .apply(lambda x: (x == 'Yes').mean())
        .reset_index()
    )

    fig, ax = plt.subplots()
    ax.bar(contract_churn['Contract'], contract_churn['Churn'], color='skyblue')
    ax.set_ylabel('Churn Rate (%)')
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_title("Contract 유형별 고객 이탈률 (단위: %)")

    for i, row in contract_churn.iterrows():
        ax.text(i, row['Churn']+0.02, f"{row['Churn']:.2%}", ha='center')
    st.pyplot(fig)

    st.markdown("""
➡ Month-to-month 계약 고객 이탈률 약 40%로 가장 높음  
➡ 1년, 2년 계약 고객은 각각 약 10%, 3% 수준으로 안정적
""")
    st.markdown("**인사이트:** 단기 계약 고객 대상 장기 계약 유도 전략 필요")

# ======================================================
# InternetService → Churn
# ======================================================
elif menu == 'InternetService → Churn':
    st.title("📌 Internet Service 유형별 이탈률")

    internet_churn = (
        df.groupby('InternetService')['Churn']
        .apply(lambda x: (x == 'Yes').mean())
        .reset_index()
    )

    fig, ax = plt.subplots()
    ax.bar(internet_churn['InternetService'], internet_churn['Churn'], color='salmon')
    ax.set_ylabel('Churn Rate (%)')
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_title("서비스 유형별 고객 이탈률 (단위: %)")

    for i, row in internet_churn.iterrows():
        ax.text(i, row['Churn']+0.02, f"{row['Churn']:.2%}", ha='center')
    st.pyplot(fig)

    st.markdown("""
➡ Fiber optic 고객 이탈률 약 40%로 DSL/None보다 높음  
➡ InternetService 유형이 이탈에 큰 영향을 줌
""")
    st.markdown("**인사이트:** Fiber optic 서비스 고객 관리 집중 필요")

# ======================================================
# Tenure → Churn
# ======================================================
elif menu == 'Tenure → Churn':
    st.title("📌 이용 기간(Tenure)별 이탈률")

    tenure_churn = (
        df.groupby('tenure_group')['Churn']
        .apply(lambda x: (x == 'Yes').mean())
        .reindex(['0-5개월', '6-11개월', '12-23개월', '24개월 이상'])
    )

    fig, ax = plt.subplots()
    ax.plot(tenure_churn.index, tenure_churn.values, marker='o', color='green')
    ax.set_ylabel('Churn Rate (%)')
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_title("Tenure 그룹별 고객 이탈률 (단위: %)")

    for i, val in enumerate(tenure_churn.values):
        ax.text(i, val+0.02, f"{val:.2%}", ha='center')
    st.pyplot(fig)

    st.markdown("""
➡ 단기 고객일수록 이탈률 높음 (0-5개월 약 55%)  
➡ 장기 고객군(24개월 이상)은 약 15% 수준으로 안정적
""")
    st.markdown("**인사이트:** 단기 고객 초기 이탈 방지 전략 필요")

# ======================================================
# Core Segment
# ======================================================
elif menu == 'Core Segment':
    st.title("🔥 핵심 이탈 세그먼트 (Fiber optic 고객)")
    st.markdown("""
### 🔍 분석 흐름 (SQL 사고방식)
- **WHERE** : Fiber optic 고객 필터링  
- **GROUP BY** : Contract × Tenure  
- **집계** : 이탈률 기준 위험군 도출
➡ Month-to-month + 단기 고객이 최고 위험군
""")

    filtered = df[df['InternetService'] == 'Fiber optic']

    pivot_rate = filtered.pivot_table(
        values='Churn',
        index='Contract',
        columns='tenure_group',
        aggfunc=lambda x: (x == 'Yes').mean()
    ).reindex(columns=['0-5개월','6-11개월','12-23개월','24개월 이상'])

    pivot_count = filtered.pivot_table(
        values='Churn',
        index='Contract',
        columns='tenure_group',
        aggfunc='count'
    ).reindex(columns=['0-5개월','6-11개월','12-23개월','24개월 이상'])

    fig, ax = plt.subplots()
    cax = ax.imshow(pivot_rate.values, cmap='Reds', vmin=0, vmax=1)

    ax.set_xticks(range(len(pivot_rate.columns)))
    ax.set_xticklabels(pivot_rate.columns)
    ax.set_yticks(range(len(pivot_rate.index)))
    ax.set_yticklabels(pivot_rate.index)

    for i in range(len(pivot_rate.index)):
        for j in range(len(pivot_rate.columns)):
            rate = pivot_rate.iloc[i, j]
            count = pivot_count.iloc[i, j]
            color = 'white' if rate > 0.5 else 'black'
            ax.text(j, i, f"{rate:.2%}\n({count}명)", ha='center', va='center', color=color)

    ax.set_title("Fiber optic 고객: Contract × Tenure 그룹 이탈률 & 고객 수 (단위: %)")
    fig.colorbar(cax, ax=ax, format=PercentFormatter(1.0))
    st.pyplot(fig)

    st.markdown("""
➡ Month-to-month + 0-5개월 그룹 이탈률 약 75%, 총 575명  
➡ 장기 계약 고객은 안정적, 위험군 집중 관리 필요
""")
    st.markdown("**인사이트:** 위험군 고객 집중 관리로 초기 이탈 최소화 필요")

# ======================================================
# Charges Analysis
# ======================================================
elif menu == 'Charges Analysis':
    st.title("💰 매출 관점 고객 세그먼트 분석")

    tenure_order = ['0-5개월','6-11개월','12-23개월','24개월 이상']
    fig, ax = plt.subplots()
    colors = ['skyblue', 'salmon']
    labels = ['잔류 고객 (No)', '이탈 고객 (Yes)']

    for i, churn_status in enumerate(['No','Yes']):
        subset = df[df['Churn']==churn_status]
        data = [subset[subset['tenure_group']==tg]['MonthlyCharges'].values for tg in tenure_order]
        positions = [x + i*0.2 for x in range(len(tenure_order))]
        ax.boxplot(data, positions=positions, widths=0.15, patch_artist=True,
                   boxprops=dict(facecolor=colors[i]))

    ax.set_xticks([x+0.1 for x in range(len(tenure_order))])
    ax.set_xticklabels(tenure_order)
    ax.set_ylabel("Monthly Charges ($)")
    ax.set_title("Tenure 그룹별 월 요금 분포 (Churn 기준, 단위: $)")

    for color, label in zip(colors, labels):
        ax.plot([], [], color=color, label=label, linewidth=10)
    ax.legend(loc='upper right')

    for i, tg in enumerate(tenure_order):
        for j, churn_status in enumerate(['No','Yes']):
            subset = df[(df['tenure_group']==tg) & (df['Churn']==churn_status)]
            mean_val = subset['MonthlyCharges'].mean()
            ax.text(i + j*0.2, mean_val+0.5, f"{mean_val:.1f}", ha='center', color='black')

    st.pyplot(fig)

    st.markdown("""
➡ 0-5개월 그룹: **이탈 고객 중앙값 63.0, 잔류 고객 44.7**  
➡ 6-11개월 이후: **이탈 고객 요금이 잔류 고객보다 높음**  
➡ 장기 고객(24개월 이상): **잔류 고객 요금이 높아도 이탈률 낮음**
""")
    st.markdown("**인사이트:** 초기 고객 요금 민감도 고려한 관리 필요")

# ======================================================
# Insight
# ======================================================
elif menu == 'Insight':
    st.title("💡 최종 인사이트")
    st.markdown("""
**핵심 결론**
- 이탈은 요금보다 **계약 구조 + 서비스 유형**에서 발생
- Month-to-month + Fiber optic + 단기 고객이 최고 위험군
- 장기 고객군은 요금 민감도 낮음

**Action Item**
- 초기 고객 온보딩 및 이탈 방지 전력 필수
- 단기 계약 고객 대상 장기 계약 유도

**근거 데이터**
- Contract × Tenure 히트맵: Fiber optic 고객 위험군 확인  
- Charges Analysis 박스플롯: 단기 이탈 고객 요금 분포 확인
""")

# ======================================================
# Appendix
# ======================================================
elif menu == 'Appendix (SQL & Validation)':
    st.title("📎 Appendix: SQL & Data Validation")
    st.markdown("""
### 🔹 MySQL 기반 핵심 쿼리
```sql
SELECT
    Contract,
    CASE
        WHEN tenure < 6 THEN '0-5개월'
        WHEN tenure < 12 THEN '6-11개월'
        WHEN tenure < 24 THEN '12-23개월'
        ELSE '24개월 이상'
    END AS tenure_group,
    AVG(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS churn_rate
FROM telco_churn
WHERE InternetService = 'Fiber optic'
GROUP BY Contract, tenure_group;
🔹 KPI 정합성 검증

SQL 집계 결과와 Pandas 재계산 결과 비교

고객 수 / 이탈 고객 수 완전 일치

이탈률은 소수점 5자리 수준의 부동소수 오차만 존재

분석 해석에는 영향 없음
""")