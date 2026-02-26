import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.ticker import PercentFormatter
import os
from sqlalchemy import create_engine
from dotenv import load_dotenv
import platform
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve

# =========================
# 폰트 설정 (Streamlit Cloud 한글 깨짐 이슈로 영어만 사용)
# =========================
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
# GitHub raw URL로 CSV 불러오기
CSV_URL = "https://raw.githubusercontent.com/mars7421/telco-churn-analysis/main/data/cleaned_churn.csv"

@st.cache_data(ttl=3600)
def load_data():
    df = pd.read_csv(CSV_URL)
    # TotalCharges 숫자형 변환, 결측치 제거
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
        return '0-5months'
    elif x < 12:
        return '6-11months'
    elif x < 24:
        return '12-23months'
    else:
        return '24+ months'

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
2️⃣ 서비스 패턴  
3️⃣ 이용 기간  
4️⃣ 위험군 정의

**환경**
- Linux  
- jupyter Notebook  
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
        'Modeling',
        'Insights & Actions',
        'Appendix (SQL & Validation)'
    ]
)


# ======================================================
# Overview
# ======================================================
if menu == 'Overview':
    st.title("📊 Telco Customer Churn Overview")
    st.caption("MySQL 기반 KPI 정의 → Python 재현 → Streamlit 모니터링 리포트")

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
➡ 전체 이탈률 {churn_rate:.2%}
➡ Month-to-month 계약과 Fiber optic 서비스 고객에서 집중 발생  
""")

    st.markdown("""
### 🎯 Action
- 가입 초기(0-5개월) 고객 대상 온보딩 및 혜택 강화
- Month-to-month 고객의 장기 계약 전환 유도 필요
""")

    st.markdown("""
※ MySQL Import 과정에서 TotalCharges 타입 변환 실패로 11건이 누락됨  
  → CSV 기준 분석으로 재현 가능성 확보
""")


# ======================================================
# Contract → Churn
# ======================================================
elif menu == 'Contract → Churn':
    st.title("📌 Churn Rate by Contract Type")
    st.caption("Contract 유형별 이탈률")

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
    ax.set_title("Churn Rate by Contract Type (%)")

    for i, row in contract_churn.iterrows():
        ax.text(i, row['Churn']+0.02, f"{row['Churn']:.2%}", ha='center')
    st.pyplot(fig)

    st.markdown("""
➡ Month-to-month 계약 고객 이탈률 약 40%로 가장 높음  
➡ 1년, 2년 계약 고객은 각각 약 10%, 3% 수준으로 안정적
""")
    st.markdown("""
### 🎯 Action
- Month-to-month 고객 중 **0-5개월 구간 대상**
  → 장기 계약 전환 프로모션 적용
""")


# ======================================================
# InternetService → Churn
# ======================================================
elif menu == 'InternetService → Churn':
    st.title("📌 Churn Rate by Internet Service Type")
    st.caption("Internet Service 유형별 이탈률")

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
    ax.set_title("Churn Rate by Internet Service (%)")

    for i, row in internet_churn.iterrows():
        ax.text(i, row['Churn']+0.02, f"{row['Churn']:.2%}", ha='center')
    st.pyplot(fig)

    st.markdown("""
➡ Fiber optic 고객 이탈률 약 40%로 DSL/None보다 높음  
➡ InternetService 유형이 이탈에 큰 영향을 줌
""")
    st.markdown("""
### 🎯 Action
- Fiber optic 고객 대상
  - 초기 설치 경험 개선
  - 품질/불만 조기 대응 필요
""")


# ======================================================
# Tenure → Churn
# ======================================================
elif menu == 'Tenure → Churn':
    st.title("📌 Churn Rate by Tenure Group")
    st.caption("이용 기간(Tenure)별 이탈률")

    tenure_churn = (
        df.groupby('tenure_group')['Churn']
        .apply(lambda x: (x == 'Yes').mean())
        .reindex(['0-5months', '6-11months', '12-23months', '24+ months'])
    )

    fig, ax = plt.subplots()
    ax.plot(tenure_churn.index, tenure_churn.values, marker='o', color='green')
    ax.set_ylabel('Churn Rate (%)')
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_title("Churn Rate by Tenure Group (%)")

    for i, val in enumerate(tenure_churn.values):
        ax.text(i, val+0.02, f"{val:.2%}", ha='center')
    st.pyplot(fig)

    st.markdown("""
➡ 단기 고객일수록 이탈률 높음 (0-5개월 약 55%)  
➡ 장기 고객군(24개월 이상)은 약 15% 수준으로 안정적
""")
    st.markdown("""
### 🎯 Action
- 가입 후 **첫 0~5개월 집중 관리**
- 초기 요금/경험 개선 전략 필요
""")


# ======================================================
# Core Segment
# ======================================================
elif menu == 'Core Segment':
    st.title("🔥 Core Churn Segment (Fiber optic customers)")
    st.caption("핵심 이탈 세그먼트 (Fiber optic 고객)")
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
    ).reindex(columns=['0-5months','6-11months','12-23months','24+ months'])

    pivot_count = filtered.pivot_table(
        values='Churn',
        index='Contract',
        columns='tenure_group',
        aggfunc='count'
    ).reindex(columns=['0-5months','6-11months','12-23months','24+ months'])

    fig, ax = plt.subplots(figsize=(8, 5))
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
            ax.text(j, i, f"{rate:.2%}\n({count})", ha='center', va='center', color=color)

    fig.suptitle(
    "Fiber optic: Contract × Tenure Churn Rate & Customer Count",
    fontsize=13,
    y=0.98
    )

    fig.colorbar(cax, ax=ax, format=PercentFormatter(1.0), fraction=0.046, pad=0.04)
    st.pyplot(fig)

    st.markdown("""
➡ Month-to-month + 0-5개월 그룹 이탈률 약 75%, 총 575명  
➡ 장기 계약 고객은 안정적, 위험군 집중 관리 필요
""")
    st.markdown("""
### 🎯 Action
- 해당 그룹을 **최우선 관리 대상**으로 설정
- 할인 / 혜택 / 계약 전환 전략 집중 적용
""")


# ======================================================
# Charges Analysis
# ======================================================
elif menu == 'Charges Analysis':
    st.title("💰 Revenue Perspective Customer Analysis")
    st.caption("매출 관점 고객 세그먼트 분석")
    st.caption("※ 본 분석은 EDA에서 관찰된 요금 패턴을 KPI 관점에서 재확인하는 목적임")

    tenure_order = ['0-5months','6-11months','12-23months','24+ months']
    fig, ax = plt.subplots()
    colors = ['skyblue', 'salmon']
    labels = ['Retained (No)', 'Churned (Yes)']

    for i, churn_status in enumerate(['No','Yes']):
        subset = df[df['Churn']==churn_status]
        data = [subset[subset['tenure_group']==tg]['MonthlyCharges'].values for tg in tenure_order]
        positions = [x + i*0.2 for x in range(len(tenure_order))]
        ax.boxplot(data, positions=positions, widths=0.15, patch_artist=True,
                   boxprops=dict(facecolor=colors[i]))

    ax.set_xticks([x+0.1 for x in range(len(tenure_order))])
    ax.set_xticklabels(tenure_order)
    ax.set_ylabel("Monthly Charges ($)")
    ax.set_title("Monthly Charges by Tenure & Churn Status ($)")

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
    st.markdown("""
➡ 초기 고객은 요금 민감도가 높은 반면, 장기 고객은 요금이 높아도 이탈률이 낮음  
   즉, 요금 자체보다 '초기 경험'이 더 중요 
""")
    st.markdown("""
### 🎯 Action
- 초기 고객 대상 요금 부담 완화 (할인/프로모션)
- 장기 고객은 요금 인상 리스크 낮음
""")


# ======================================================
# Modeling
# ======================================================
elif menu == 'Modeling':
    st.title("🤖 Churn Prediction Modeling")
    st.caption("Logistic Regression / RandomForest 모델 성능 비교 + ROC Curve")

    # =========================
    # Modeling Pipeline
    # =========================
    @st.cache_resource
    def run_modeling(df):

        df_model = df.copy()
        df_model = df_model.drop(columns=['customerID', 'tenure_group'])
        df_model['Churn'] = df_model['Churn'].map({'Yes': 1, 'No': 0})

        X = df_model.drop(columns=['Churn'])
        y = df_model['Churn']

        cat_cols = X.select_dtypes(include='object').columns
        X_encoded = pd.get_dummies(X, columns=cat_cols, drop_first=True)

        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        # =========================
        # Logistic Regression
        # =========================
        log_model = LogisticRegression(max_iter=1000, random_state=42)
        log_model.fit(X_train, y_train)

        y_pred_log = log_model.predict(X_test)
        y_proba_log = log_model.predict_proba(X_test)[:, 1]

        # =========================
        # RandomForest
        # =========================
        rf_model = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
        rf_model.fit(X_train, y_train)

        y_pred_rf = rf_model.predict(X_test)
        y_proba_rf = rf_model.predict_proba(X_test)[:, 1]

        # =========================
        # Logistic (tuned)
        # =========================
        log_bal = LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=42
        )
        log_bal.fit(X_train, y_train)

        y_proba_bal = log_bal.predict_proba(X_test)[:, 1]
        y_pred_bal = (y_proba_bal >= 0.3).astype(int)

        # =========================
        # Metrics
        # =========================
        def get_metrics(y_true, y_pred, y_proba):
            report = classification_report(y_true, y_pred, output_dict=True)
            return {
                'accuracy': round(accuracy_score(y_true, y_pred), 2),
                'recall': round(report['1']['recall'], 2),
                'precision': round(report['1']['precision'], 2),
                'roc_auc': round(roc_auc_score(y_true, y_proba), 2)
            }

        log_metrics = get_metrics(y_test, y_pred_log, y_proba_log)
        rf_metrics = get_metrics(y_test, y_pred_rf, y_proba_rf)
        bal_metrics = get_metrics(y_test, y_pred_bal, y_proba_bal)

        # =========================
        # Risk Segmentation
        # =========================
        result_df = X_test.copy()
        result_df['Churn_Prob'] = y_proba_bal
        result_df['Churn_Pred'] = y_pred_bal

        def risk_group(p):
            if p >= 0.7:
                return "High Risk"
            elif p >= 0.4:
                return "Medium Risk"
            else:
                return "Low Risk"

        result_df['Risk_Group'] = result_df['Churn_Prob'].apply(risk_group)

        return {
            'log': log_metrics,
            'rf': rf_metrics,
            'bal': bal_metrics,
            'roc': {
                'y_test': y_test,
                'log': y_proba_log,
                'rf': y_proba_rf,
                'bal': y_proba_bal
            },
            'risk_df' : result_df
        }

    results = run_modeling(df)

    # =========================
    # 성능 비교 테이블
    # =========================
    st.markdown("### 📊 Model Performance Comparison")

    perf_df = pd.DataFrame([
        ['Logistic (baseline)', results['log']['accuracy'], results['log']['recall'], results['log']['precision'], results['log']['roc_auc']],
        ['RandomForest', results['rf']['accuracy'], results['rf']['recall'], results['rf']['precision'], results['rf']['roc_auc']],
        ['Logistic (tuned)', results['bal']['accuracy'], results['bal']['recall'], results['bal']['precision'], results['bal']['roc_auc']]
    ], columns=['모델', 'Accuracy', 'Recall', 'Precision', 'ROC-AUC'])

    st.dataframe(perf_df)

    # =========================
    # ROC Curve
    # =========================
    st.markdown("### 📈 ROC Curve Comparison")

    y_test = results['roc']['y_test']

    fpr_log, tpr_log, _ = roc_curve(y_test, results['roc']['log'])
    fpr_rf, tpr_rf, _ = roc_curve(y_test, results['roc']['rf'])
    fpr_bal, tpr_bal, _ = roc_curve(y_test, results['roc']['bal'])

    fig, ax = plt.subplots()
    ax.plot(fpr_log, tpr_log, label=f"Logistic (AUC={results['log']['roc_auc']:.2f})")
    ax.plot(fpr_rf, tpr_rf, label=f"RandomForest (AUC={results['rf']['roc_auc']:.2f})")
    ax.plot(fpr_bal, tpr_bal, label=f"Tuned Logistic (AUC={results['bal']['roc_auc']:.2f})")
    ax.plot([0, 1], [0, 1], 'k--')

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc='lower right')

    st.pyplot(fig)

    st.markdown("""
✅ ROC Curve를 통해 모델별 분류 성능을 직관적으로 비교 가능  
✅ Tuned Logistic 모델은 recall 향상으로 이탈 고객 탐지에 유리하며,  
   ROC-AUC 또한 baseline과 비슷해 모델 안정성 확인
""")

    # =========================
    # Threshold 전략
    # =========================
    st.markdown("### 🎯 Threshold Strategy")

    threshold_df = pd.DataFrame({
    "Threshold": [0.5, 0.4, 0.3],
    "Recall": [0.80, 0.87, 0.93],
    "Precision": [0.49, 0.45, 0.41]
})

    st.dataframe(threshold_df)

    st.markdown("""
    - 기본 threshold(0.5)에서는 이탈 고객을 충분히 탐지하지 못함  
    - threshold를 0.3으로 낮추면서 recall 개선 (≈0.93)  
    - 일부 precision 감소 trade-off 존재  

    👉 이탈 방지 관점에서 recall을 우선 기준으로 설정
    """)

    # =========================
    # Risk Segmentation
    # =========================
    st.markdown("### 🚨 Customer Risk Segmentation")
    st.caption("예측 확률 기반 고객 위험군 분류")

    risk_df = results['risk_df']
    risk_counts = risk_df['Risk_Group'].value_counts()

    col1, col2, col3 = st.columns(3)
    col1.metric("High Risk", risk_counts.get("High Risk", 0))
    col2.metric("Medium Risk", risk_counts.get("Medium Risk", 0))
    col3.metric("Low Risk", risk_counts.get("Low Risk", 0))

    st.bar_chart(risk_counts)

    # =========================
    # modeling insight
    # =========================    
    st.markdown("""
### 🎯 Action 연결
- 모델 기반으로 High Risk 고객 사전 탐지 가능
- 위험군 고객 대상 선제적 유지 전략 적용 가능
""")


# ======================================================
# Insights & Actions
# ======================================================
elif menu == 'Insights & Actions':
    st.title("💡 최종 인사이트 및 비즈니스 액션")

    st.markdown("""
### 1. KPI 기반 이탈 구조 (SQL 분석)
- 이탈은 **Month-to-month 계약 고객**에서 집중 발생
- 특히 **초기 고객(0–5개월)** 구간에서 이탈률이 가장 높음
- **Fiber optic 서비스 사용자**가 주요 고위험군으로 확인됨

👉 계약 유형 + 서비스 유형 + 초기 이용 구간의 조합이 핵심 이탈 구조

---

### 2. 세그먼트 기반 이탈 특성
- **Month-to-month × Fiber optic × 초기 고객**에서 최고 이탈률 발생
- 장기 계약(1년/2년) 고객은 동일 조건에서도 상대적으로 안정적

👉 단일 변수보다 **세그먼트 조합 단위에서 이탈 위험이 명확하게 구분됨**

---

### 3. 요금 변수 해석
- 초기 구간 이탈 고객의 월 요금이 상대적으로 높은 경향 존재
- 그러나 요금은 단독 원인이 아니라  
  **계약 구조 및 서비스 구성과 결합될 때 이탈을 강화하는 보조 요인**

👉 이탈의 본질은 요금이 아닌 **구조적 요인(계약 + 서비스)**

---

### 4. 모델링 적용 (Logistic Regression + Threshold 튜닝)
- 이탈 확률 예측 모델 구축 후 Threshold를 조정하여 **Recall 중심으로 최적화**
- 이탈 고객을 더 많이 사전에 탐지할 수 있도록 개선

👉 예측 정확도보다 **이탈 탐지 성능을 우선한 실무형 모델링**

---

### 5. 비즈니스 활용 전략
- 고위험 세그먼트 대상 선제적 대응 가능
  - 단기 계약 고객 → 장기 계약 전환 유도
  - Fiber optic 고객 → 초기 경험 개선 및 요금 정책 검토
  - 초기 고객 → 온보딩 및 관리 강화

👉 KPI 기반 세그먼트 정의 + 모델 예측을 결합하여  
**이탈 방지 전략을 구체적으로 실행 가능**
""")


# ======================================================
# Appendix
# ======================================================
elif menu == 'Appendix (SQL & Validation)':
    st.title("📎 Appendix: SAppendix: SQL KPI Definition & Validation")
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