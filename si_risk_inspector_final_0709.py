import os
import uuid
from pathlib import Path
from typing import List, Dict, Any
from urllib.parse import urlparse, unquote

import openai 
import pandas as pd
import streamlit as st
import altair as alt
#from dotenv import load_dotenv  #jsh
from azure.storage.blob import BlobServiceClient
import io
import time

#load_dotenv()  # jsh-Azure Web App에서는 환경변수 직접 사용, .env 로드 불필요

################################################################################
# 환경 설정 & 공통 유틸                                                             #
################################################################################

def require_api_key() -> None:
    """Ensure the Azure OpenAI API key 및 환경변수 존재; 없으면 에러."""
    if not os.getenv("OPENAI_API_KEY"):
        st.error("OPENAI_API_KEY 환경 변수가 설정되어 있지 않습니다.")
        st.stop()
    if not os.getenv("AZURE_ENDPOINT"):
        st.error("AZURE_ENDPOINT 환경 변수가 설정되어 있지 않습니다.")
        st.stop()
    if not os.getenv("OPENAI_API_TYPE"):
        st.error("OPENAI_API_TYPE 환경 변수가 설정되어 있지 않습니다.")
        st.stop()
    if not os.getenv("OPENAI_API_VERSION"):
        st.error("OPENAI_API_VERSION 환경 변수가 설정되어 있지 않습니다.")
        st.stop()
    openai.api_key = os.getenv("OPENAI_API_KEY")
    openai.azure_endpoint = os.getenv("AZURE_ENDPOINT")
    openai.api_type = os.getenv("OPENAI_API_TYPE")
    openai.api_version = os.getenv("OPENAI_API_VERSION")

DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME")

def gpt_single_call(prompt: str, model: str = None, temperature: float = 0.3) -> str:
    """단일 GPT 호출 래퍼 (캐시)."""
    require_api_key()
    if model is None:
        model = "gpt-4.1-mini"
    @st.cache_resource(show_spinner="GPT 분석 중...")
    def _cached_call(p: str, m: str, t: float) -> str:
        response = openai.chat.completions.create(
            model=m,
            temperature=t,
            messages=[{"role": "system", "content": "당신은 숙련된 SI 프로젝트 리스크 분석가입니다."},
                      {"role": "user", "content": p}]
        )
        return response.choices[0].message.content.strip()
    return _cached_call(prompt, model, temperature)


################################################################################
# GPT 프롬프트 & 파서                                                               #
################################################################################

PROMPT_TEMPLATE = """다음은 SI 프로젝트 회의록/이슈 내용이다. 각 항목별로 잠재 리스크 유형(category: 일정, 요구사항, 인력, 품질, 기타),\n리스크 내용(summary), 심각도(0~1 float), 대응 방안(action)을 JSON Lines 로 출력하라.\n### 입력 ###\n{chunk}\n### 출력 예시 ###\n{{\"category\": \"일정\", \"summary\": \"설계 산출물 지연\", \"score\": 0.7, \"action\": \"테크리드 리소스 추가\"}}"""

def parse_jsonl(raw: str) -> List[Dict[str, Any]]:
    """GPT JSON Lines 응답 파싱. 실패 시 빈 리스트 반환."""
    rows: List[Dict[str, Any]] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(pd.json_normalize(eval(line)).to_dict(orient="records")[0])
        except Exception:
            st.warning(f"JSON 파싱 실패: {line[:60]}")
    return rows


################################################################################
# 데이터 로드 & GPT 분석                                                             #
################################################################################

@st.cache_data(show_spinner="CSV 로드 중...")
def load_csvs(files: List[Any]) -> pd.DataFrame:
    if not files:
        return pd.DataFrame()
    dfs = [pd.read_csv(f) for f in files]
    return pd.concat(dfs, ignore_index=True)

# Azure Storage 연결 정보 (환경변수에서 읽음)
AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
AZURE_STORAGE_CONTAINER = os.getenv("AZURE_STORAGE_CONTAINER", "uploaded")

# 업로드 파일을 Azure Blob Storage에 저장하고 blob url 리스트 반환
def upload_files_to_azure(files):
    if not files:
        return []
    blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
    container_client = blob_service_client.get_container_client(AZURE_STORAGE_CONTAINER)
    try:
        container_client.create_container()
    except Exception:
        pass  # 이미 있으면 무시
    blob_urls = []
    for file in files:
        blob_client = container_client.get_blob_client(file.name)
        blob_client.upload_blob(file, overwrite=True)
        blob_urls.append(blob_client.url)

    time.sleep(1)  # 업로드 후 1초 대기

    return blob_urls

# Azure Blob에서 파일을 읽어 DataFrame으로 반환
def load_csvs_from_azure(blob_urls):
    #st.sidebar.write(f"blob_url: {blob_urls}")  #jsh
    if not blob_urls:
        return pd.DataFrame()
    blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
    dfs = []
    for url in blob_urls:
        # blob_name 추출 개선: 쿼리스트링 제거 및 경로 전체 사용 + URL 디코딩
        parsed_url = urlparse(url)
        blob_path = parsed_url.path.lstrip("/")
        if blob_path.startswith(f"{AZURE_STORAGE_CONTAINER}/"):
            blob_name = blob_path[len(f"{AZURE_STORAGE_CONTAINER}/"):]
        else:
            blob_name = blob_path
        blob_name = unquote(blob_name)  # URL 디코딩 추가
        #st.sidebar.write(f"blob_name: {blob_name}")  #jsh
        blob_client = blob_service_client.get_blob_client(AZURE_STORAGE_CONTAINER, blob_name)
        try:
            stream = blob_client.download_blob().readall()
            df = pd.read_csv(io.BytesIO(stream))
            dfs.append(df)
        except Exception as e:
            st.sidebar.error(f"[Blob 다운로드 오류] {blob_name}: {e}")
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

@st.cache_data(show_spinner="Risk 분석 중…")
def analyze_risks(df: pd.DataFrame, model: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    #st.sidebar.write(f"model2_1: {model}")  #jsh2
    all_results: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        text = str(row.get("text", ""))[:4000]  # GPT 토큰 제한 보호
        prompt = PROMPT_TEMPLATE.format(chunk=text)
        gpt_resp = gpt_single_call(prompt, model=model)

        #st.sidebar.write(f"model2_2: {model}")  #jsh2

        parsed = parse_jsonl(gpt_resp)
        for item in parsed:
            item["source_id"] = row.get("id", uuid.uuid4().hex)
            all_results.append(item)

    if not all_results:
        return pd.DataFrame()

    risk_df = pd.DataFrame(all_results)
    risk_df["score"] = pd.to_numeric(risk_df["score"], errors="coerce")
    return risk_df

################################################################################
# Streamlit UI                                                                      #
################################################################################

st.set_page_config(page_title="SI 프로젝트 리스크 탐지 AI", layout="wide", page_icon="★")
st.title("▣ SI 프로젝트 리스크 탐지 AI")

# ───────────────── Sidebar ─────────────────
st.sidebar.header("[데이터 로드]")
uploaded_files = st.sidebar.file_uploader("◎ 회의록·이슈 CSV 업로드", type=["csv"], accept_multiple_files=True)

# 업로드 파일을 Azure Blob Storage에 저장
blob_urls = upload_files_to_azure(uploaded_files) if uploaded_files else []

with st.sidebar.expander("♥ API 연결 (선택)"):
    st.text("Jira, DevOps API 연동은 TODO")

st.sidebar.header("[필터 & 옵션]")
score_threshold = st.sidebar.slider("◎ 리스크 점수 임계값", 0.0, 1.0, 0.5, 0.05)
                                    
# 모델명과 배포명 매핑
MODEL_DEPLOYMENT_MAP = {
    "gpt-4.1-mini": "dev-gpt-4.1-mini",  # 모델명: 배포명
}

# 모델명만 사용자에게 보여주고, 내부적으로 배포명으로 변환
model_names = list(MODEL_DEPLOYMENT_MAP.keys())
model_name_choice = st.sidebar.selectbox("사용 모델", model_names)
model_choice = MODEL_DEPLOYMENT_MAP[model_name_choice]

#st.sidebar.write(f"model_choice: {model_choice}")  #jsh

if st.sidebar.button("▶ 분석 실행"):
    st.session_state["run"] = True

# ───────────── 데이터 적재 ─────────────
raw_df = load_csvs_from_azure(blob_urls)

if "run" in st.session_state and st.session_state["run"]:
    risk_df = analyze_risks(raw_df, model=model_choice)
    # 임계값 필터링 적용 (score 컬럼 존재 여부 체크)
    if not risk_df.empty and "score" in risk_df.columns:
        risk_df = risk_df[risk_df["score"] >= score_threshold]
    else:
        st.warning("분석 결과가 없거나 'score' 컬럼이 존재하지 않습니다. GPT 응답/환경변수/모델 설정을 확인하세요.")
        risk_df = pd.DataFrame()
else:
    risk_df = pd.DataFrame()

# ───────────────── Main Tabs ─────────────────
overview_tab, insights_tab, logs_tab = st.tabs(["Overview", "Risk Insights", "Logs"])

with overview_tab:
    st.subheader("프로젝트 리스크 요약")
    col1, col2, col3 = st.columns(3)
    col1.metric("총 문서 수", len(raw_df))
    col2.metric("리스크 항목 수", len(risk_df))
    col3.metric("평균 리스크 점수", f"{risk_df['score'].mean():.2f}" if not risk_df.empty else "-")

    if not risk_df.empty:
        time_chart = (alt.Chart(risk_df.assign(idx=lambda x: x.index))
                      .mark_line(point=True)
                      .encode(x="idx", y="score", tooltip=["summary", "score"])
                      .interactive())
        st.altair_chart(time_chart, use_container_width=True)

with insights_tab:
    st.subheader("리스크 상세 인사이트")
    if risk_df.empty:
        st.info("먼저 사이드바에서 CSV를 업로드하고 '분석 실행'을 눌러주세요.")
    else:
        # 카테고리 분포 차트
        cat_chart = (alt.Chart(risk_df)
                      .mark_bar()
                      .encode(x="count()", y=alt.Y("category", sort="-x"), color="category")
                      .properties(height=300))
        st.altair_chart(cat_chart, use_container_width=True)

        # 상위 리스크 테이블
        st.write("### 고위험 리스크 Top 10")
        st.dataframe(risk_df.sort_values("score", ascending=False).head(10), use_container_width=True)

with logs_tab:
    st.subheader("원본 로그")
    if raw_df.empty:
        st.info("CSV를 업로드하면 원본 로그를 볼 수 있습니다.")
    else:
        keyword = st.text_input("키워드 검색")
        display_df = raw_df.copy()
        if keyword:
            display_df = display_df[display_df.apply(lambda row: keyword.lower() in str(row).lower(), axis=1)]
        st.dataframe(display_df, use_container_width=True)

################################################################################
# Footer                                                                            #
################################################################################

st.markdown("---")
st.caption("© 2025 SI Risk Inspector • Powered by Azure OpenAI & Streamlit")

# requirements: azure-storage-blob