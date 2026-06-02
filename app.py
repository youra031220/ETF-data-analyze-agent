# -*- coding: utf-8 -*-
import json
import os
import re
from io import BytesIO

import google.generativeai as genai
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from google.api_core import exceptions as google_exceptions
from streamlit.errors import StreamlitSecretNotFoundError
from PIL import Image


DEFAULT_EXCEL_PATH = "ETF 순매수 데이터_260529.xlsx"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_FALLBACK_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-flash-lite-latest",
    "gemini-flash-latest",
]
GEMINI_GENERATION_CONFIG = {
    "temperature": 0.45,
    "top_p": 0.9,
    "max_output_tokens": 8192,
}
ETF_BRAND_MAP = {
    "KODEX": "삼성자산운용",
    "TIGER": "미래에셋자산운용",
    "ACE": "한국투자신탁운용",
    "RISE": "KB자산운용",
    "PLUS": "한화자산운용",
    "SOL": "신한자산운용",
    "HANARO": "NH-Amundi자산운용",
    "KIWOOM": "키움투자자산운용",
}
BRAND_RULE_PROMPT = """
[ETF 브랜드 규칙]
KODEX = 삼성자산운용
TIGER = 미래에셋자산운용
ACE = 한국투자신탁운용
RISE = KB자산운용
PLUS = 한화자산운용
SOL = 신한자산운용
HANARO = NH-Amundi자산운용
KIWOOM = 키움투자자산운용

절대 브랜드와 운용사를 혼동하지 말 것.
TIGER ETF를 삼성자산운용 ETF라고 쓰면 안 된다.
ACE ETF를 삼성자산운용 ETF라고 쓰면 안 된다.
RISE ETF를 삼성자산운용 ETF라고 쓰면 안 된다.
SOL ETF를 삼성자산운용 ETF라고 쓰면 안 된다.

[삼성자산운용 제안 작성 규칙]
1. KODEX ETF는 직접 제안 가능
2. 타사 ETF(TIGER, ACE, RISE, PLUS, SOL, HANARO, KIWOOM 등)는 직접 홍보 대상으로 제안하지 말 것
3. 타사 ETF는 벤치마킹 또는 경쟁상품으로만 사용
4. 삼성자산운용 제안은 반드시 KODEX 관점에서 작성
5. 잘못된 예: TIGER 미국우주테크 마케팅 강화
6. 올바른 예: TIGER 미국우주테크에 자금이 유입되고 있으므로 삼성자산운용은 KODEX 미국우주항공 ETF의 노출을 강화할 필요가 있음
"""
BRAND_MANAGER_ERROR_PATTERNS = [
    "삼성자산운용의 TIGER",
    "삼성자산운용의 ACE",
    "삼성자산운용의 RISE",
    "삼성자산운용의 PLUS",
    "삼성자산운용의 SOL",
    "삼성자산운용의 HANARO",
    "삼성자산운용의 KIWOOM",
]


st.set_page_config(
    page_title="ETF Marketing Monitoring AI Agent",
    page_icon="📊",
    layout="wide",
)

st.markdown(
    """
    <style>
    .youra-credit {
        position: fixed;
        top: 3.15rem;
        left: 0.85rem;
        z-index: 999999;
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.02em;
        color: #f9fafb;
        background: #374151;
        border: 1px solid rgba(255, 255, 255, 0.16);
        border-radius: 6px;
        padding: 0.22rem 0.52rem;
        backdrop-filter: blur(6px);
        box-shadow: 0 3px 10px rgba(0, 0, 0, 0.12);
    }
    @media (prefers-color-scheme: dark) {
        .youra-credit {
            background: #374151;
        }
    }
    </style>
    <div class="youra-credit">Team 2</div>
    """,
    unsafe_allow_html=True,
)


def normalize_excel_source(file_source):
    if isinstance(file_source, tuple):
        _, file_bytes = file_source
        return BytesIO(file_bytes)
    return file_source


@st.cache_data(show_spinner=False)
def load_excel_data(file_sources):
    if not isinstance(file_sources, list):
        file_sources = [file_sources]

    all_dfs = []

    for file_source in file_sources:
        source = normalize_excel_source(file_source)
        all_dfs.extend(read_weekly_excel(source))

    if not all_dfs:
        return pd.DataFrame()

    data = pd.concat(all_dfs, ignore_index=True)
    data = data.replace("-", 0)

    exclude_cols = ["단축코드", "종목명", "week"]
    numeric_cols = [col for col in data.columns if col not in exclude_cols]

    for col in numeric_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0)

    if {"week", "종목명"}.issubset(data.columns):
        data = data.drop_duplicates(subset=["week", "종목명"], keep="last")

    classified = data["종목명"].apply(classify_etf)
    data["대표테마"] = classified.apply(lambda item: item["대표테마"])
    data["태그"] = classified.apply(lambda item: item["태그"])
    data["테마"] = data["대표테마"]
    return data.reset_index(drop=True)


def read_weekly_excel(file_source):
    xls = pd.ExcelFile(file_source)
    dfs = []

    for sheet in xls.sheet_names:
        if "참고" in str(sheet):
            continue

        temp = pd.read_excel(xls, sheet_name=sheet)
        temp["week"] = sheet
        dfs.append(temp)

    return dfs


THEME_PRIORITY = [
    "커버드콜",
    "레버리지",
    "인버스",
    "AI",
    "반도체",
    "빅테크",
    "2차전지",
    "전기차",
    "자율주행",
    "수소",
    "양자",
    "원자력",
    "방산",
    "전력",
    "로봇",
    "IT",
    "비만",
    "빅파마",
    "바이오",
    "조선",
    "자동차",
    "철강",
    "건설",
    "운송",
    "기계",
    "화학",
    "통신",
    "게임",
    "콘텐츠",
    "메타버스",
    "우주",
    "은행",
    "증권",
    "보험",
    "리츠",
    "채권",
    "금",
    "은",
    "원유",
    "원자재",
    "배당",
    "밸류업",
    "ESG",
    "소비재",
    "푸드",
    "뷰티",
    "브랜드",
    "그룹주",
    "가치주",
    "성장주",
    "모멘텀",
    "퀄리티",
    "저변동성",
    "멀티팩터",
    "인컴",
    "자산배분",
    "친환경",
    "탄소배출권",
    "머니마켓",
    "TDF",
    "S&P500",
    "나스닥",
    "KRX",
    "코스피",
    "코스닥",
    "미국",
    "중국",
    "일본",
    "인도",
    "베트남",
    "유럽",
    "독일",
    "멕시코",
    "아시아",
    "러시아",
    "선진국",
    "신흥국",
    "기타",
]


TAG_KEYWORDS = {
    "AI": ["AI", "인공지능", "생성형AI", "CHATGPT", "엔비디아", "NVIDIA"],
    "반도체": ["반도체", "HBM", "메모리"],
    "빅테크": [
        "구글",
        "애플",
        "엔비디아",
        "마이크로소프트",
        "테슬라",
        "NVIDIA",
        "GOOGLE",
        "APPLE",
        "MICROSOFT",
        "TESLA",
        "M7",
        "MAGNIFICENT7",
        "빅테크",
        "테크",
    ],
    "2차전지": ["2차전지", "배터리", "양극재", "음극재"],
    "전기차": ["전기차", "EV", "수소차"],
    "자율주행": ["자율주행", "AUTONOMOUS"],
    "수소": ["수소", "HYDROGEN"],
    "양자": ["양자", "QUANTUM", "양자컴퓨팅"],
    "원자력": ["원자력", "SMR"],
    "방산": ["방산", "디펜스"],
    "전력": ["전력", "전력망", "그리드"],
    "로봇": ["로봇"],
    "IT": ["IT", "TECHNOLOGY"],
    "바이오": ["바이오", "헬스케어", "제약"],
    "비만": ["비만", "OBESITY"],
    "빅파마": ["일라이릴리", "LILLY", "노보노디스크", "NOVO", "빅파마", "PHARMA"],
    "조선": ["조선", "해운"],
    "자동차": ["자동차", "AUTO"],
    "철강": ["철강", "STEEL"],
    "건설": ["건설"],
    "운송": ["운송", "물류"],
    "기계": ["기계"],
    "화학": ["화학"],
    "통신": ["통신", "TELECOM"],
    "게임": ["게임"],
    "콘텐츠": ["KPOP", "K-POP", "콘텐츠", "웹툰", "드라마", "엔터", "미디어"],
    "메타버스": ["메타버스", "METAVERSE"],
    "우주": ["우주", "항공"],
    "은행": ["은행"],
    "증권": ["증권"],
    "보험": ["보험"],
    "리츠": ["리츠", "REITS"],
    "채권": [
        "채권",
        "국채",
        "회사채",
        "KOFR",
        "통안채",
        "CD금리",
        "국고채",
        "국공채",
        "특수채",
        "물가채",
        "장기채",
        "중기채",
        "단기채",
    ],
    "금": ["금", "골드"],
    "은": ["은", "실버"],
    "원유": ["원유", "에너지"],
    "원자재": ["구리", "농산물", "원자재", "COMMODITY"],
    "배당": ["배당"],
    "커버드콜": ["커버드콜"],
    "밸류업": ["밸류업"],
    "ESG": ["ESG"],
    "소비재": ["소비재", "소비주", "필수소비재", "소비"],
    "푸드": ["푸드", "FOOD"],
    "뷰티": ["뷰티", "BEAUTY"],
    "브랜드": ["브랜드", "BRAND"],
    "그룹주": ["삼성그룹", "포스코그룹", "카카오그룹", "그룹주"],
    "가치주": ["가치", "VALUE"],
    "성장주": ["성장", "GROWTH"],
    "모멘텀": ["모멘텀", "MOMENTUM"],
    "퀄리티": ["퀄리티", "QUALITY"],
    "저변동성": ["저변동성", "최소변동성", "MINVOL", "LOWVOL"],
    "멀티팩터": ["멀티팩터", "MULTIFACTOR"],
    "인컴": ["인컴", "INCOME"],
    "자산배분": ["멀티에셋", "MULTIASSET", "자산배분", "ALLOCATION"],
    "친환경": ["친환경", "GREEN", "기후"],
    "탄소배출권": ["탄소", "배출권", "탄소중립", "NETZERO"],
    "머니마켓": ["머니마켓"],
    "TDF": ["TDF"],
    "레버리지": ["레버리지", "2X"],
    "인버스": ["인버스", "곱버스"],
    "S&P500": ["S&P"],
    "나스닥": ["나스닥"],
    "KRX": ["KRX100", "KRX300", "KTOP30"],
    "코스피": ["KOSPI", "200",  "200TR",
    "MSCI Korea",
    "Korea",
    "우량주",
    "블루칩",
    "Top10","200액티브" ,"코스피"],
    "코스닥": ["KOSDAQ", "코스닥"],
    "미국": [
        "미국",
        "US",
        "구글",
        "애플",
        "엔비디아",
        "마이크로소프트",
        "테슬라",
        "NVIDIA",
        "GOOGLE",
        "APPLE",
        "MICROSOFT",
        "TESLA",
        "M7",
        "MAGNIFICENT7",
    ],
    "중국": ["중국", "CHINA", "차이나", "항셍", "HANGSENG", "BYD", "샤오미"],
    "일본": ["일본", "JAPAN"],
    "인도": ["인도"],
    "베트남": ["베트남"],
    "유럽": ["유럽"],
    "독일": ["독일"],
    "멕시코": ["멕시코"],
    "아시아": ["아시아"],
    "러시아": ["러시아"],
    "선진국": ["선진국", "DEVELOPED"],
    "신흥국": ["신흥국", "EM"],
}


LATEST_ADDED_THEMES = {
    "가치주",
    "성장주",
    "모멘텀",
    "퀄리티",
    "저변동성",
    "멀티팩터",
    "KRX",
    "IT",
    "자동차",
    "철강",
    "건설",
    "운송",
    "기계",
    "화학",
    "통신",
    "자율주행",
    "수소",
    "양자",
    "비만",
    "빅파마",
    "푸드",
    "뷰티",
    "브랜드",
    "인컴",
    "자산배분",
    "아시아",
    "러시아",
    "선진국",
    "친환경",
    "탄소배출권",
}


def keyword_in_name(keyword, normalized_name):
    keyword = str(keyword).upper()
    if keyword == "200":
        return (
            "KOSPI200" in normalized_name
            or "코스피200" in normalized_name
            or "KODEX200" in normalized_name
            or "TIGER200" in normalized_name
            or "RISE200" in normalized_name
            or "KBSTAR200" in normalized_name
            or "ACE200" in normalized_name
            or normalized_name.endswith("200")
        )
    if keyword == "US":
        return (
            "미국" in normalized_name
            or "USA" in normalized_name
            or "S&P" in normalized_name
            or "나스닥" in normalized_name
            or "KODEXUS" in normalized_name
            or "TIGERUS" in normalized_name
            or "ACEUS" in normalized_name
            or "RISEUS" in normalized_name
            or "KBSTARUS" in normalized_name
            or "HANAROUS" in normalized_name
            or "SOLUS" in normalized_name
        )
    if keyword == "EM":
        return "신흥국" in normalized_name or "EM" in normalized_name
    if keyword == "EV":
        return (
            "전기차" in normalized_name
            or "수소차" in normalized_name
            or " EV" in str(normalized_name)
            or normalized_name.endswith("EV")
        )
    if keyword == "테크":
        china_context = (
            "중국" in normalized_name
            or "차이나" in normalized_name
            or "CHINA" in normalized_name
            or "항셍" in normalized_name
            or "HANGSENG" in normalized_name
        )
        return "테크" in normalized_name and not china_context
    if keyword == "IT":
        return (
            normalized_name.endswith("IT")
            or "KODEXIT" in normalized_name
            or "TIGERIT" in normalized_name
            or "ACEIT" in normalized_name
            or "RISEIT" in normalized_name
            or "KBSTARIT" in normalized_name
            or "HANAROIT" in normalized_name
            or "KOSEFIT" in normalized_name
            or "IT&" in normalized_name
            or "&IT" in normalized_name
        )
    return keyword in normalized_name


def add_related_tags(tags):
    tag_set = set(tags)

    if "자율주행" in tag_set:
        tag_set.update(["IT", "미국"])

    if "빅테크" in tag_set:
        tag_set.add("미국")

    if "비만" in tag_set or "빅파마" in tag_set:
        tag_set.add("바이오")

    return [theme for theme in THEME_PRIORITY if theme in tag_set and theme != "기타"]


def classify_etf_with_rules(name, excluded_themes=None):
    excluded_themes = excluded_themes or set()
    normalized_name = str(name).upper().replace(" ", "")
    tags = []

    for tag in THEME_PRIORITY:
        if tag == "기타" or tag in excluded_themes:
            continue

        keywords = TAG_KEYWORDS.get(tag, [])
        if any(keyword_in_name(keyword, normalized_name) for keyword in keywords):
            tags.append(tag)

    tags = add_related_tags(tags)
    tags = [tag for tag in tags if tag not in excluded_themes]

    representative_theme = "기타"
    for theme in THEME_PRIORITY:
        if theme in excluded_themes:
            continue
        if theme in tags:
            representative_theme = theme
            break

    if not tags:
        tags = ["기타"]

    return {
        "대표테마": representative_theme,
        "태그": tags,
    }


def classify_etf(name):
    return classify_etf_with_rules(name)


def classify_etf_before_latest_expansion(name):
    return classify_etf_with_rules(name, excluded_themes=LATEST_ADDED_THEMES)


def search_etf(keyword):
    result = df[df["종목명"].str.contains(str(keyword), case=False, na=False)]
    return result.sort_values("week")


def gr_plot_etf(etf_name, investor):
    temp = df[df["종목명"] == etf_name].copy()

    fig = px.line(
        temp,
        x="week",
        y=investor,
        markers=True,
        title=f"{etf_name} - {investor} 순매수 추세",
    )
    fig.update_layout(
        xaxis_title="주차",
        yaxis_title=f"{investor} 순매수",
        hovermode="x unified",
        template="plotly_white",
    )
    return fig


def plot_etf(etf_name, investor="개인"):
    return gr_plot_etf(etf_name, investor)


def top_etf(week, investor="개인", top_n=20):
    temp = df[df["week"] == week]
    return (
        temp[["종목명", investor]]
        .sort_values(investor, ascending=False)
        .head(int(top_n))
        .reset_index(drop=True)
    )


def rising_etf(current_week, prev_week, investor="개인", top_n=20):
    current = df[df["week"] == current_week][["종목명", investor]]
    prev = df[df["week"] == prev_week][["종목명", investor]]

    merged = current.merge(prev, on="종목명", how="outer", suffixes=("_current", "_prev"))
    merged = merged.fillna(0)
    merged["change"] = merged[f"{investor}_current"] - merged[f"{investor}_prev"]

    return merged.sort_values("change", ascending=False).head(int(top_n)).reset_index(drop=True)


def falling_etf(current_week, prev_week, investor="개인", top_n=20):
    current = df[df["week"] == current_week][["종목명", investor]]
    prev = df[df["week"] == prev_week][["종목명", investor]]

    merged = current.merge(prev, on="종목명", how="outer", suffixes=("_current", "_prev"))
    merged = merged.fillna(0)
    merged["change"] = merged[f"{investor}_current"] - merged[f"{investor}_prev"]

    return merged.sort_values("change", ascending=True).head(int(top_n)).reset_index(drop=True)


def new_entry_etf(current_week, prev_week):
    current = set(df[df["week"] == current_week]["종목명"])
    prev = set(df[df["week"] == prev_week]["종목명"])
    new_etfs = current - prev
    return pd.DataFrame({"종목명": sorted(list(new_etfs))})


def dropped_etf(current_week, prev_week):
    current = set(df[df["week"] == current_week]["종목명"])
    prev = set(df[df["week"] == prev_week]["종목명"])
    dropped = prev - current
    return pd.DataFrame({"종목명": sorted(list(dropped))})


def theme_analysis(week, investor="개인"):
    temp = (
        df[df["week"] == week]
        .groupby("대표테마")[investor]
        .sum()
        .reset_index()
        .sort_values(investor, ascending=False)
    )
    return temp.reset_index(drop=True)


def theme_bar(week, investor="개인"):
    temp = theme_analysis(week, investor)
    fig = px.bar(temp, x="대표테마", y=investor, title=f"{week} 대표테마별 순매수")
    fig.update_layout(template="plotly_white", xaxis_title="대표테마", yaxis_title=f"{investor} 순매수")
    return fig


def theme_change(current_week, prev_week, investor="개인"):
    current = df[df["week"] == current_week].groupby("대표테마")[investor].sum().reset_index()
    prev = df[df["week"] == prev_week].groupby("대표테마")[investor].sum().reset_index()

    merged = current.merge(prev, on="대표테마", how="outer", suffixes=("_current", "_prev"))
    merged = merged.fillna(0)
    merged["change"] = merged[f"{investor}_current"] - merged[f"{investor}_prev"]

    return merged.sort_values("change", ascending=False).reset_index(drop=True)


def theme_change_chart(current_week, prev_week, investor="개인"):
    temp = theme_change(current_week, prev_week, investor)
    fig = px.bar(temp, x="대표테마", y="change", title=f"{investor} 대표테마 로테이션")
    fig.update_layout(template="plotly_white", xaxis_title="대표테마", yaxis_title="전주 대비 변화")
    return fig


def tag_analysis(week, investor="개인"):
    temp = df[df["week"] == week][["태그", investor]].copy()
    temp = temp.explode("태그")
    temp = temp[temp["태그"].notna()]

    return (
        temp.groupby("태그")[investor]
        .sum()
        .reset_index()
        .rename(columns={"태그": "태그명", investor: "순매수합계"})
        .sort_values("순매수합계", ascending=False)
        .reset_index(drop=True)
    )


def tag_bar(week, investor="개인"):
    temp = tag_analysis(week, investor)
    fig = px.bar(temp, x="태그명", y="순매수합계", title=f"{week} 태그별 순매수")
    fig.update_layout(template="plotly_white", xaxis_title="태그", yaxis_title=f"{investor} 순매수")
    return fig


def classification_diagnostics():
    unique_etfs = df[["종목명", "대표테마", "태그"]].drop_duplicates(subset=["종목명"]).copy()
    total_count = len(unique_etfs)
    unique_etfs["이전대표테마"] = unique_etfs["종목명"].apply(
        lambda name: classify_etf_before_latest_expansion(name)["대표테마"]
    )
    theme_counts = (
        unique_etfs.groupby("대표테마")["종목명"]
        .nunique()
        .reset_index(name="ETF 수")
        .sort_values("ETF 수", ascending=False)
        .reset_index(drop=True)
    )
    others = unique_etfs[unique_etfs["대표테마"] == "기타"].sort_values("종목명").reset_index(drop=True)
    other_count = len(others)
    other_ratio = (other_count / total_count * 100) if total_count else 0
    previous_other_count = int((unique_etfs["이전대표테마"] == "기타").sum())
    previous_other_ratio = (previous_other_count / total_count * 100) if total_count else 0

    return theme_counts, others, total_count, other_count, other_ratio, previous_other_count, previous_other_ratio


def hot_theme(current_week, prev_week, investor="개인"):
    temp = theme_change(current_week, prev_week, investor)
    return temp.head(5)


def dead_theme(current_week, prev_week, investor="개인"):
    temp = theme_change(current_week, prev_week, investor)
    return temp.tail(5)


def make_bar_chart(data, x_col, y_col, title, ascending=True):
    chart_data = data.sort_values(y_col, ascending=ascending).copy()
    fig = px.bar(chart_data, x=y_col, y=x_col, orientation="h", title=title)
    fig.update_layout(template="plotly_white", xaxis_title=y_col, yaxis_title="", height=520)
    return fig


def configure_gemini():
    try:
        secret_key = st.secrets.get("GEMINI_API_KEY", None)
        model_name = st.secrets.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
    except StreamlitSecretNotFoundError:
        secret_key = None
        model_name = DEFAULT_GEMINI_MODEL

    api_key = secret_key or os.getenv("GEMINI_API_KEY")
    model_name = os.getenv("GEMINI_MODEL", model_name)
    if not api_key:
        return None

    genai.configure(api_key=api_key)
    model_names = [model_name] + [name for name in GEMINI_FALLBACK_MODELS if name != model_name]
    return [
        genai.GenerativeModel(
            name,
            generation_config=GEMINI_GENERATION_CONFIG,
        )
        for name in model_names
    ]


def gemini_error_message(exc):
    if isinstance(exc, google_exceptions.ResourceExhausted):
        return (
            "Gemini API 할당량을 초과했습니다. 잠시 후 다시 시도하거나, "
            "`GEMINI_MODEL`을 Flash 계열 모델로 변경하거나, Google AI Studio에서 "
            "결제/쿼터 설정을 확인하세요."
        )

    return f"Gemini API 호출 중 오류가 발생했습니다: {exc}"


def generate_with_gemini(contents):
    models = configure_gemini()
    if models is None:
        return "GEMINI_API_KEY가 설정되어 있지 않습니다. Streamlit secrets 또는 환경변수에 API 키를 설정하세요."

    last_error = None
    for model in models:
        try:
            response = model.generate_content(contents)
            return response.text
        except google_exceptions.NotFound as exc:
            last_error = exc
            continue
        except google_exceptions.ResourceExhausted as exc:
            last_error = exc
            continue
        except Exception as exc:
            return gemini_error_message(exc)

    return gemini_error_message(last_error)


def extract_json_object(text):
    if not text:
        return {}

    cleaned = str(text).strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not match:
        return {}

    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


def normalize_age_payload(payload):
    if not isinstance(payload, dict):
        return []

    if "age_groups" in payload and isinstance(payload["age_groups"], list):
        groups = payload["age_groups"]
    else:
        groups = [payload]

    rows = []
    for group in groups:
        if not isinstance(group, dict):
            continue

        age_group = str(group.get("age_group", "미확인"))
        etfs = group.get("etfs", [])
        if isinstance(etfs, str):
            etfs = [etfs]

        for etf_name in etfs:
            if etf_name:
                rows.append({"연령대": age_group, "추출ETF명": str(etf_name).strip()})

    return rows


def extract_age_etfs_from_image(image):
    prompt = """
이 이미지는 증권사 앱의 연령대별 ETF 인기 순위입니다.

이미지에 포함된 연령대별 ETF명을 추출하세요.
반드시 JSON만 출력하세요. Markdown, 설명 문장, 코드블록은 출력하지 마세요.

출력 형식:
{
  "age_groups": [
    {
      "age_group": "20대",
      "etfs": [
        "KODEX AI전력핵심설비",
        "ACE 미국빅테크7",
        "TIGER 미국나스닥100"
      ]
    }
  ]
}
"""

    raw_text = generate_with_gemini([prompt, image])
    payload = extract_json_object(raw_text)
    rows = normalize_age_payload(payload)
    return payload, rows, raw_text


def analyze_age_image(image):
    payload, _, raw_text = extract_age_etfs_from_image(image)
    if payload:
        return json.dumps(payload, ensure_ascii=False, indent=2)
    return raw_text


def normalize_etf_text(name):
    return re.sub(r"[^0-9A-Z가-힣]", "", str(name).upper())


def get_etf_brand_manager(etf_name):
    normalized_name = str(etf_name).upper().strip()

    for brand, manager in ETF_BRAND_MAP.items():
        if normalized_name.startswith(brand):
            return {
                "브랜드": brand,
                "운용사": manager,
            }

    return {
        "브랜드": "미확인",
        "운용사": "미확인",
    }


def validate_brand_manager_response(text):
    violations = [pattern for pattern in BRAND_MANAGER_ERROR_PATTERNS if pattern in str(text)]
    if not violations:
        return text

    warning = (
        "\n\n> ⚠️ ETF 브랜드-운용사 매핑 오류 감지: "
        + ", ".join(violations)
        + "\n> TIGER/ACE/RISE/PLUS/SOL/HANARO/KIWOOM은 삼성자산운용 ETF가 아닙니다. "
        "해당 문장은 KODEX 관점의 경쟁상품/벤치마킹 표현으로 재검토해야 합니다."
    )
    return str(text) + warning


def match_etf_name(etf_name):
    target = normalize_etf_text(etf_name)
    if not target:
        return None

    candidates = df["종목명"].dropna().drop_duplicates().tolist()
    normalized_candidates = [(name, normalize_etf_text(name)) for name in candidates]

    for name, normalized in normalized_candidates:
        if normalized == target:
            return name

    contains_matches = [
        name
        for name, normalized in normalized_candidates
        if target in normalized or normalized in target
    ]
    if contains_matches:
        return sorted(contains_matches, key=len)[0]

    token_matches = []
    for name, normalized in normalized_candidates:
        overlap = len(set(target) & set(normalized))
        if overlap >= max(4, int(len(target) * 0.5)):
            token_matches.append((overlap / max(len(set(target)), 1), name))

    if token_matches:
        return sorted(token_matches, reverse=True)[0][1]

    return None


def get_weekly_etf_value(etf_name, week, investor):
    if not etf_name:
        return 0

    temp = df[(df["week"] == week) & (df["종목명"] == etf_name)]
    if temp.empty or investor not in temp.columns:
        return 0
    return float(temp[investor].sum())


def analyze_age_etf_flow(extracted_etfs, current_week, previous_week, investor):
    rows = []
    for item in extracted_etfs:
        if isinstance(item, dict):
            age_group = item.get("연령대", item.get("age_group", "미확인"))
            extracted_name = item.get("추출ETF명", item.get("etf", item.get("ETF명", "")))
        else:
            age_group = "미확인"
            extracted_name = str(item)

        matched_name = match_etf_name(extracted_name)
        current_value = get_weekly_etf_value(matched_name, current_week, investor)
        previous_value = get_weekly_etf_value(matched_name, previous_week, investor)
        change = current_value - previous_value
        change_rate = np.nan
        if previous_value != 0:
            change_rate = change / abs(previous_value) * 100
        brand_info = get_etf_brand_manager(matched_name or extracted_name)

        rows.append(
            {
                "연령대": age_group,
                "추출ETF명": extracted_name,
                "매칭ETF명": matched_name or "",
                "브랜드": brand_info["브랜드"],
                "운용사": brand_info["운용사"],
                "현재주차": current_value,
                "비교주차": previous_value,
                "증감액": change,
                "증감률(%)": change_rate,
                "매칭상태": "매칭" if matched_name else "미매칭",
            }
        )

    return pd.DataFrame(rows)


def analyze_age_etf_themes(extracted_etfs, flow_df=None):
    rows = []
    flow_lookup = {}
    if flow_df is not None and not flow_df.empty:
        flow_lookup = {
            row["추출ETF명"]: row["매칭ETF명"]
            for _, row in flow_df.iterrows()
        }

    for item in extracted_etfs:
        if isinstance(item, dict):
            age_group = item.get("연령대", item.get("age_group", "미확인"))
            extracted_name = item.get("추출ETF명", item.get("etf", item.get("ETF명", "")))
        else:
            age_group = "미확인"
            extracted_name = str(item)

        matched_name = flow_lookup.get(extracted_name) or match_etf_name(extracted_name) or extracted_name
        classified = classify_etf(matched_name)
        rows.append(
            {
                "연령대": age_group,
                "ETF명": extracted_name,
                "분류기준명": matched_name,
                "대표테마": classified["대표테마"],
                "태그": ", ".join(classified["태그"]),
            }
        )

    return pd.DataFrame(rows)


def build_age_flow_summary(flow_df, theme_df):
    if flow_df is None or flow_df.empty:
        empty = pd.DataFrame()
        return {
            "유입상위": empty,
            "유출상위": empty,
            "관심자금일치": empty,
            "관심자금괴리": empty,
            "태그별변화": empty,
            "대표테마별변화": empty,
            "브랜드별변화": empty,
            "KODEX흐름": empty,
            "경쟁ETF흐름": empty,
        }

    base = flow_df.copy()
    base["증감액"] = pd.to_numeric(base["증감액"], errors="coerce").fillna(0)
    base["증감률(%)"] = pd.to_numeric(base["증감률(%)"], errors="coerce")

    theme_cols = ["ETF명", "분류기준명", "대표테마", "태그"]
    if theme_df is not None and not theme_df.empty and set(theme_cols).issubset(theme_df.columns):
        enriched = base.merge(
            theme_df[theme_cols],
            left_on=["추출ETF명", "매칭ETF명"],
            right_on=["ETF명", "분류기준명"],
            how="left",
        )
    else:
        enriched = base.copy()
        enriched["대표테마"] = ""
        enriched["태그"] = ""

    sort_cols = ["증감액"]
    inflow = enriched[enriched["증감액"] > 0].sort_values(sort_cols, ascending=False).head(10)
    outflow = enriched[enriched["증감액"] < 0].sort_values(sort_cols, ascending=True).head(10)
    aligned = inflow.copy()
    diverged = outflow.copy()

    tag_rows = []
    for _, row in enriched.iterrows():
        tags = [tag.strip() for tag in str(row.get("태그", "")).split(",") if tag.strip()]
        for tag in tags:
            tag_rows.append({"태그": tag, "증감액": row["증감액"]})

    if tag_rows:
        tag_change = (
            pd.DataFrame(tag_rows)
            .groupby("태그")["증감액"]
            .sum()
            .reset_index()
            .sort_values("증감액", ascending=False)
            .reset_index(drop=True)
        )
    else:
        tag_change = pd.DataFrame(columns=["태그", "증감액"])

    if "대표테마" in enriched.columns:
        theme_change_summary = (
            enriched.groupby("대표테마", dropna=False)["증감액"]
            .sum()
            .reset_index()
            .sort_values("증감액", ascending=False)
            .reset_index(drop=True)
        )
    else:
        theme_change_summary = pd.DataFrame(columns=["대표테마", "증감액"])

    if {"브랜드", "운용사"}.issubset(enriched.columns):
        brand_change = (
            enriched.groupby(["브랜드", "운용사"], dropna=False)["증감액"]
            .sum()
            .reset_index()
            .sort_values("증감액", ascending=False)
            .reset_index(drop=True)
        )
    else:
        brand_change = pd.DataFrame(columns=["브랜드", "운용사", "증감액"])

    kodex_flow = enriched[enriched.get("브랜드", "") == "KODEX"].sort_values("증감액", ascending=False)
    competitor_flow = enriched[
        (enriched.get("브랜드", "") != "KODEX") & (enriched.get("브랜드", "") != "미확인")
    ].sort_values("증감액", ascending=False)

    display_cols = [
        "연령대",
        "추출ETF명",
        "매칭ETF명",
        "브랜드",
        "운용사",
        "대표테마",
        "태그",
        "현재주차",
        "비교주차",
        "증감액",
        "증감률(%)",
    ]
    display_cols = [col for col in display_cols if col in enriched.columns]

    return {
        "유입상위": inflow[display_cols],
        "유출상위": outflow[display_cols],
        "관심자금일치": aligned[display_cols],
        "관심자금괴리": diverged[display_cols],
        "태그별변화": tag_change.head(15),
        "대표테마별변화": theme_change_summary.head(15),
        "브랜드별변화": brand_change,
        "KODEX흐름": kodex_flow[display_cols].head(15),
        "경쟁ETF흐름": competitor_flow[display_cols].head(15),
    }


def format_summary_table(summary, key):
    data = summary.get(key, pd.DataFrame())
    if data is None or data.empty:
        return "해당 없음"
    return data.to_string(index=False)


def add_brand_manager_columns(data, etf_col="종목명"):
    if data is None or data.empty or etf_col not in data.columns:
        return data

    result = data.copy()
    brand_info = result[etf_col].apply(get_etf_brand_manager)
    result["브랜드"] = brand_info.apply(lambda item: item["브랜드"])
    result["운용사"] = brand_info.apply(lambda item: item["운용사"])
    return result


def generate_age_integrated_insight(age_rows, flow_df, theme_df, current_week, previous_week, investor):
    flow_summary = build_age_flow_summary(flow_df, theme_df)
    prompt = f"""
당신은 ETF 마케팅 인텔리전스 리포트를 작성하는 시니어 애널리스트입니다.
증권사 앱의 연령대별 인기 ETF와 실제 ETF 순매수 데이터를 연결해, 마케팅 담당자가 바로 사용할 수 있는 고완성도 리포트를 작성하세요.

조건:
- 현재 주차: {current_week}
- 비교 주차: {previous_week}
- 투자주체: {investor}
- 관심과 실제 자금 유입을 반드시 구분
- 데이터에서 확인되는 내용과 가설을 구분
- 삼성자산운용 마케팅 시사점을 제안
- 단순 요약 금지. 각 섹션은 구체적 ETF명, 테마명, 증감 방향, 마케팅 해석을 포함
- 각 섹션은 최소 3문장 이상으로 작성하고, 핵심 섹션은 문단형으로 충분히 서술
- 숫자는 표의 값을 근거로 사용하되, 단위가 불명확하면 "데이터 기준" 또는 "순매수 변화"라고 표현
- 유입 상위 ETF와 유출 ETF를 구분해 설명
- 관심은 높지만 순매수가 감소한 ETF를 반드시 별도로 해석
- 관심과 실제 매수 흐름이 일치하는 테마와 괴리되는 테마를 모두 다룰 것
- 태그 기반으로 "미국 + AI", "AI + 반도체", "우주", "배당", "빅테크"처럼 복합 관심사를 해석
- 문체는 예시처럼 전문적이고 풍부한 리포트형 문장으로 작성
- 불필요하게 짧은 bullet만 나열하지 말고, 각 bullet 아래에 설명 문장을 붙일 것

{BRAND_RULE_PROMPT}

[연령대 인기 ETF]
{pd.DataFrame(age_rows).to_string(index=False) if age_rows else "추출 ETF 없음"}

[실제 순매수 변화]
{flow_df.to_string(index=False)}

[ETF 대표테마 및 태그]
{theme_df.to_string(index=False)}

[중간 분석 요약 - 유입 상위 ETF]
{format_summary_table(flow_summary, "유입상위")}

[중간 분석 요약 - 유출 상위 ETF]
{format_summary_table(flow_summary, "유출상위")}

[중간 분석 요약 - 관심과 자금유입 일치 ETF]
{format_summary_table(flow_summary, "관심자금일치")}

[중간 분석 요약 - 관심은 높지만 순매수 감소 ETF]
{format_summary_table(flow_summary, "관심자금괴리")}

[중간 분석 요약 - 태그별 순매수 변화 합계]
{format_summary_table(flow_summary, "태그별변화")}

[중간 분석 요약 - 대표테마별 순매수 변화 합계]
{format_summary_table(flow_summary, "대표테마별변화")}

[중간 분석 요약 - 브랜드/운용사별 순매수 변화 합계]
{format_summary_table(flow_summary, "브랜드별변화")}

[중간 분석 요약 - KODEX ETF 흐름]
{format_summary_table(flow_summary, "KODEX흐름")}

[중간 분석 요약 - 타사 경쟁 ETF 흐름]
{format_summary_table(flow_summary, "경쟁ETF흐름")}

리포트 작성 방식:
- 아래 헤더를 반드시 사용
- 각 헤더마다 구체적인 ETF 사례를 2개 이상 포함
- "유입 상위", "유출 상위", "관심과 자금이 일치하는 경우", "괴리가 존재하는 경우"를 명확히 구분
- 중간 분석 요약을 우선 근거로 사용하고, 원본 표는 보조 근거로 사용
- 삼성자산운용 제안은 최소 3개 액션 아이템으로 작성
- 삼성자산운용 제안은 반드시 KODEX 중심의 캠페인, 콘텐츠, 상품 포지셔닝, 경쟁 ETF 대응 관점으로 작성
- 타사 ETF는 "경쟁사 상품", "벤치마킹 대상", "시장 수요의 신호"로만 언급

## 연령대 인기 ETF
연령대별로 어떤 ETF와 테마가 관심을 받고 있는지 요약하세요. 단순 목록이 아니라, 해당 연령대의 투자 성향을 성장형, 지수형, 테마형, 인컴형 등으로 해석하세요.

## 실제 순매수 변화
유입 상위 ETF와 유출 ETF를 나누어 설명하세요. ETF별 증감 방향을 구체적으로 언급하고, 자금 유입이 집중된 테마와 자금 이탈이 발생한 테마를 해석하세요.
이 섹션에는 반드시 "주요 변화:" 문단을 먼저 작성하고, "급증"과 "감소"를 나누어 작성하세요.

## 관심과 자금유입 비교
인기 리스트에 있는 ETF 중 실제 순매수가 증가한 ETF와 감소한 ETF를 비교하세요. 관심과 자금이 일치하는 경우, 관심은 높지만 자금이 빠진 경우, 관심은 낮아 보이지만 자금 유입이 있는 경우를 구분하세요.
이 섹션은 반드시 번호 목록으로 작성하세요.
1. 연령대 인기 ETF 중 실제 자금 유입이 크게 발생한 ETF
2. 연령대 인기 ETF인데 순매수는 감소하거나 대규모 유출이 발생한 ETF
3. 연령대 관심과 실제 매수 사이 괴리

## 태그 기반 인사이트
대표테마가 아니라 태그 조합을 중심으로 해석하세요. 예: 미국 + AI, AI + 반도체, 우주, 빅테크, 배당, 나스닥, S&P500 등. 어떤 태그 조합이 실제 자금을 받고 있고, 어떤 태그는 관심만 높은지 설명하세요.
이 섹션은 "자금 유입을 받는 주요 테마"와 "관심만 높고 실제 매수는 둔화되거나 유출된 테마"로 나누세요.

## 삼성자산운용 제안
KODEX 관점에서 실행 가능한 마케팅 전략을 작성하세요. 경쟁사 ETF에 자금이 유입되었다면 해당 ETF를 직접 홍보하지 말고, 대응 가능한 KODEX 상품의 노출 강화, 콘텐츠 기획, 비교 포지셔닝, 장기 투자 교육 관점으로 제안하세요.
이 섹션은 최소 3개 전략으로 작성하고, 각 전략은 "현황 분석"과 "마케팅 전략 제안"을 포함하세요.

반드시 분석할 내용:
1. 연령대 인기 ETF 중 실제 자금 유입이 발생한 ETF
2. 연령대 인기 ETF인데 순매수는 감소한 ETF
3. 연령대 관심과 실제 매수 사이 괴리
4. 어떤 테마가 자금 유입을 받고 있는지
5. 어떤 테마는 관심만 높고 매수는 없는지
6. 삼성자산운용 마케팅 시사점

품질 기준:
- 최종 답변은 짧은 요약이 아니라 완성형 마케팅 리포트여야 함
- 전체 분량은 최소 1,200자 이상을 목표로 작성
- 섹션별로 데이터 해석과 마케팅 시사점을 연결
- "데이터 부족"이라고 끝내지 말고, 주어진 표 안에서 관찰 가능한 범위의 인사이트를 최대한 도출
"""

    return validate_brand_manager_response(generate_with_gemini(prompt))


def generate_ai_report(top_df, rising_df, theme_df, tag_df, rotation_df, investor, current_week, prev_week):
    top_report_df = add_brand_manager_columns(top_df, "종목명")
    rising_report_df = add_brand_manager_columns(rising_df, "종목명")
    prompt = f"""
당신은 ETF 마케팅 전략 애널리스트입니다.
아래 Dashboard 분석 결과를 바탕으로 주간 ETF 마케팅 리포트를 Markdown으로 작성하세요.

조건:
- 투자주체: {investor}
- 현재 주차: {current_week}
- 비교 주차: {prev_week}
- 마케팅 담당자가 바로 읽고 실행할 수 있는 완성형 리포트 문장으로 작성
- 과장하지 말고 데이터에서 관찰 가능한 내용과 가설을 구분
- 삼성자산운용 관점의 액션 아이디어 포함
- 대표테마와 태그 분석을 함께 활용해 미국, AI, 반도체, 배당 등 복합 선호를 해석
- AI·반도체 동반 강세, 미국 ETF 선호, 배당 수요 같은 태그 기반 인사이트를 우선 검토
- 단순 bullet 요약 금지. 각 핵심 항목은 2~4문장 이상으로 서술
- ETF명, 대표테마, 태그, 순매수 변화 방향을 연결해 구체적으로 해석
- 타사 ETF는 경쟁 신호나 벤치마킹 대상으로만 설명하고, 제안은 KODEX 중심으로 작성
- 최종 답변은 최소 1,000자 이상을 목표로 작성
- 섹션별로 "관찰된 데이터", "해석", "마케팅 액션"이 자연스럽게 드러나야 함

{BRAND_RULE_PROMPT}

[TOP ETF]
{top_report_df.to_string(index=False)}

[급등 ETF]
{rising_report_df.to_string(index=False)}

[테마 분석]
{theme_df.to_string(index=False)}

[태그 분석]
{tag_df.to_string(index=False)}

[테마 로테이션]
{rotation_df.to_string(index=False)}
"""

    return validate_brand_manager_response(generate_with_gemini(prompt))


def default_prev_week(weeks, current_week):
    if current_week in weeks:
        idx = weeks.index(current_week)
        if idx > 0:
            return weeks[idx - 1]
    return weeks[0] if weeks else None


def render_metric_row(current_week, prev_week, investor):
    col1, col2, col3, col4 = st.columns(4)
    total_current = df[df["week"] == current_week][investor].sum()
    total_prev = df[df["week"] == prev_week][investor].sum() if prev_week else 0
    total_change = total_current - total_prev
    etf_count = df[df["week"] == current_week]["종목명"].nunique()
    theme_count = df[df["week"] == current_week]["대표테마"].nunique()

    col1.metric("현재 주차 순매수", f"{total_current:,.0f}", f"{total_change:,.0f}")
    col2.metric("분석 ETF 수", f"{etf_count:,}")
    col3.metric("분석 테마 수", f"{theme_count:,}")
    col4.metric("투자주체", investor)


with st.sidebar:
    st.header("설정")

    uploaded_excels = st.file_uploader(
        "추가 ETF 순매수 엑셀 업로드",
        type=["xlsx", "xls"],
        accept_multiple_files=True,
        help="업로드한 파일은 폴더의 기본 엑셀 파일 뒤에 합쳐집니다. 같은 주차/ETF가 있으면 업로드 파일 값이 우선됩니다.",
    )

    excel_sources = []
    if os.path.exists(DEFAULT_EXCEL_PATH):
        excel_sources.append(DEFAULT_EXCEL_PATH)

    uploaded_excels = uploaded_excels or []

    for uploaded_excel in uploaded_excels:
        excel_sources.append((uploaded_excel.name, uploaded_excel.getvalue()))

    if excel_sources:
        excel_source = excel_sources
    else:
        excel_source = None

    if os.path.exists(DEFAULT_EXCEL_PATH):
        st.caption(f"기본 파일 포함: {DEFAULT_EXCEL_PATH}")
    if uploaded_excels:
        st.caption(f"추가 업로드 파일: {len(uploaded_excels)}개")


if excel_source is None:
    st.title("ETF Marketing Monitoring AI Agent")
    st.info("ETF 순매수 엑셀 파일을 사이드바에서 업로드하거나 앱 폴더에 기본 엑셀 파일을 배치하세요.")
    st.stop()


try:
    df = load_excel_data(excel_source)
except Exception as exc:
    st.title("ETF Marketing Monitoring AI Agent")
    st.error(f"데이터 로드 중 오류가 발생했습니다: {exc}")
    st.stop()


if df.empty:
    st.title("ETF Marketing Monitoring AI Agent")
    st.warning("분석 가능한 데이터가 없습니다.")
    st.stop()


weeks = list(df["week"].dropna().unique())
investor_list = [
    col
    for col in [
        "개인",
        "기관",
        "외국인",
        "금융투자",
        "보험",
        "투신",
        "사모",
        "은행",
        "기타금융",
        "연기금 등",
    ]
    if col in df.columns
]

if not investor_list:
    st.error("투자주체 컬럼을 찾을 수 없습니다.")
    st.stop()

with st.sidebar:
    selected_week = st.selectbox("주차 선택", weeks, index=len(weeks) - 1)
    selected_investor = st.selectbox("투자주체 선택", investor_list, index=0)


prev_default = default_prev_week(weeks, selected_week)

st.title("ETF Marketing Monitoring AI Agent")
render_metric_row(selected_week, prev_default, selected_investor)

tabs = st.tabs(
    [
        "ETF 검색",
        "ETF 추세",
        "TOP ETF",
        "급등 ETF",
        "급락 ETF",
        "신규 진입 ETF",
        "이탈 ETF",
        "테마 분석",
        "태그 분석",
        "테마 로테이션",
        "연령대 분석 AI",
        "AI 인사이트",
        "분류 진단",
    ]
)


with tabs[0]:
    st.subheader("ETF 검색")
    keyword = st.text_input("ETF명 검색", placeholder="예: AI반도체")
    if keyword:
        search_result = search_etf(keyword)
        st.dataframe(search_result, width="stretch", hide_index=True)
    else:
        st.dataframe(df.head(50), width="stretch", hide_index=True)


with tabs[1]:
    st.subheader("ETF 추세")
    etf_options = sorted(df["종목명"].dropna().unique())
    etf_name = st.selectbox("ETF 선택", etf_options)
    fig = gr_plot_etf(etf_name, selected_investor)
    st.plotly_chart(fig, width="stretch")
    st.dataframe(
        df[df["종목명"] == etf_name].sort_values("week"),
        width="stretch",
        hide_index=True,
    )


with tabs[2]:
    st.subheader("TOP ETF")
    col1, col2 = st.columns([2, 1])
    top_week = col1.selectbox("주차 선택", weeks, index=weeks.index(selected_week), key="top_week")
    top_n = col2.slider("TOP N 선택", 5, 50, 20, 1, key="top_n")
    top_result = top_etf(top_week, selected_investor, top_n)
    st.dataframe(top_result, width="stretch", hide_index=True)
    st.plotly_chart(
        make_bar_chart(top_result, "종목명", selected_investor, f"{top_week} TOP ETF", ascending=True),
        width="stretch",
    )


with tabs[3]:
    st.subheader("급등 ETF")
    col1, col2, col3 = st.columns([2, 2, 1])
    current_week = col1.selectbox("현재 주차", weeks, index=weeks.index(selected_week), key="rise_current")
    prev_week = col2.selectbox("비교 주차", weeks, index=weeks.index(prev_default), key="rise_prev")
    rise_n = col3.slider("TOP N", 5, 50, 20, 1, key="rise_n")
    rise_result = rising_etf(current_week, prev_week, selected_investor, rise_n)
    st.dataframe(rise_result, width="stretch", hide_index=True)
    st.plotly_chart(
        make_bar_chart(rise_result, "종목명", "change", "전주 대비 순매수 급등 ETF", ascending=True),
        width="stretch",
    )


with tabs[4]:
    st.subheader("급락 ETF")
    col1, col2, col3 = st.columns([2, 2, 1])
    current_week = col1.selectbox("현재 주차", weeks, index=weeks.index(selected_week), key="fall_current")
    prev_week = col2.selectbox("비교 주차", weeks, index=weeks.index(prev_default), key="fall_prev")
    fall_n = col3.slider("TOP N", 5, 50, 20, 1, key="fall_n")
    fall_result = falling_etf(current_week, prev_week, selected_investor, fall_n)
    st.dataframe(fall_result, width="stretch", hide_index=True)
    st.plotly_chart(
        make_bar_chart(fall_result, "종목명", "change", "전주 대비 순매수 급락 ETF", ascending=False),
        width="stretch",
    )


with tabs[5]:
    st.subheader("신규 진입 ETF")
    col1, col2 = st.columns(2)
    current_week = col1.selectbox("현재 주차", weeks, index=weeks.index(selected_week), key="new_current")
    prev_week = col2.selectbox("비교 주차", weeks, index=weeks.index(prev_default), key="new_prev")
    new_result = new_entry_etf(current_week, prev_week)
    st.dataframe(new_result, width="stretch", hide_index=True)


with tabs[6]:
    st.subheader("이탈 ETF")
    col1, col2 = st.columns(2)
    current_week = col1.selectbox("현재 주차", weeks, index=weeks.index(selected_week), key="drop_current")
    prev_week = col2.selectbox("비교 주차", weeks, index=weeks.index(prev_default), key="drop_prev")
    dropped_result = dropped_etf(current_week, prev_week)
    st.dataframe(dropped_result, width="stretch", hide_index=True)


with tabs[7]:
    st.subheader("테마 분석")
    col1, col2 = st.columns(2)
    theme_week = col1.selectbox("주차 선택", weeks, index=weeks.index(selected_week), key="theme_week")
    theme_investor = col2.selectbox(
        "투자주체 선택", investor_list, index=investor_list.index(selected_investor), key="theme_investor"
    )
    theme_result = theme_analysis(theme_week, theme_investor)
    st.dataframe(theme_result, width="stretch", hide_index=True)
    st.plotly_chart(theme_bar(theme_week, theme_investor), width="stretch")


with tabs[8]:
    st.subheader("태그 분석")
    col1, col2 = st.columns(2)
    tag_week = col1.selectbox("주차 선택", weeks, index=weeks.index(selected_week), key="tag_week")
    tag_investor = col2.selectbox(
        "투자주체 선택", investor_list, index=investor_list.index(selected_investor), key="tag_investor"
    )
    tag_result = tag_analysis(tag_week, tag_investor)
    st.dataframe(tag_result, width="stretch", hide_index=True)
    st.plotly_chart(tag_bar(tag_week, tag_investor), width="stretch")


with tabs[9]:
    st.subheader("테마 로테이션")
    col1, col2 = st.columns(2)
    current_week = col1.selectbox("현재 주차", weeks, index=weeks.index(selected_week), key="theme_current")
    prev_week = col2.selectbox("비교 주차", weeks, index=weeks.index(prev_default), key="theme_prev")
    rotation_result = theme_change(current_week, prev_week, selected_investor)
    st.dataframe(rotation_result, width="stretch", hide_index=True)
    st.plotly_chart(theme_change_chart(current_week, prev_week, selected_investor), width="stretch")


with tabs[10]:
    st.subheader("연령대 분석 AI")
    col1, col2, col3 = st.columns([2, 2, 1.5])
    age_current_week = col1.selectbox(
        "현재 주차",
        weeks,
        index=weeks.index(selected_week),
        key="age_current_week",
    )
    age_prev_week = col2.selectbox(
        "비교 주차",
        weeks,
        index=weeks.index(prev_default),
        key="age_prev_week",
    )
    age_investor = col3.selectbox(
        "투자주체",
        investor_list,
        index=investor_list.index(selected_investor),
        key="age_investor",
    )

    uploaded_image = st.file_uploader(
        "증권사 앱의 연령대별 ETF 인기 순위 스크린샷 업로드",
        type=["png", "jpg", "jpeg", "webp"],
    )

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, width="stretch")

        if st.button("Gemini Vision으로 분석", type="primary"):
            with st.spinner("이미지를 분석하는 중입니다."):
                payload, age_rows, raw_text = extract_age_etfs_from_image(image)

                if not age_rows:
                    st.warning("이미지에서 ETF 목록을 JSON으로 추출하지 못했습니다. Gemini 원문 응답을 확인하세요.")
                    st.code(raw_text)
                else:
                    extracted_df = pd.DataFrame(age_rows)
                    flow_df = analyze_age_etf_flow(
                        age_rows,
                        age_current_week,
                        age_prev_week,
                        age_investor,
                    )
                    age_theme_df = analyze_age_etf_themes(age_rows, flow_df)
                    age_summary = build_age_flow_summary(flow_df, age_theme_df)

                    st.markdown("#### 추출 ETF 목록")
                    st.dataframe(extracted_df, width="stretch", hide_index=True)

                    st.markdown("#### 📈 연령대 인기 ETF 실제 순매수 변화")
                    st.dataframe(flow_df, width="stretch", hide_index=True)

                    st.markdown("#### ETF 테마 분석")
                    st.dataframe(age_theme_df, width="stretch", hide_index=True)

                    with st.expander("중간 분석 요약", expanded=False):
                        st.caption("유입 상위 ETF")
                        st.dataframe(age_summary["유입상위"], width="stretch", hide_index=True)
                        st.caption("유출 상위 ETF")
                        st.dataframe(age_summary["유출상위"], width="stretch", hide_index=True)
                        st.caption("태그별 순매수 변화")
                        st.dataframe(age_summary["태그별변화"], width="stretch", hide_index=True)
                        st.caption("브랜드/운용사별 순매수 변화")
                        st.dataframe(age_summary["브랜드별변화"], width="stretch", hide_index=True)
                        st.caption("KODEX ETF 흐름")
                        st.dataframe(age_summary["KODEX흐름"], width="stretch", hide_index=True)
                        st.caption("타사 경쟁 ETF 흐름")
                        st.dataframe(age_summary["경쟁ETF흐름"], width="stretch", hide_index=True)

                    st.markdown("#### Gemini 통합 인사이트")
                    with st.spinner("연령대 관심과 실제 순매수 변화를 함께 분석하는 중입니다."):
                        st.markdown(
                            generate_age_integrated_insight(
                                age_rows,
                                flow_df,
                                age_theme_df,
                                age_current_week,
                                age_prev_week,
                                age_investor,
                            )
                        )


with tabs[11]:
    st.subheader("AI 인사이트")
    col1, col2, col3 = st.columns([2, 2, 1])
    report_current_week = col1.selectbox(
        "현재 주차", weeks, index=weeks.index(selected_week), key="report_current"
    )
    report_prev_week = col2.selectbox(
        "비교 주차", weeks, index=weeks.index(prev_default), key="report_prev"
    )
    report_top_n = col3.slider("분석 개수", 5, 30, 10, 1, key="report_top_n")

    report_top = top_etf(report_current_week, selected_investor, report_top_n)
    report_rising = rising_etf(report_current_week, report_prev_week, selected_investor, report_top_n)
    report_theme = theme_analysis(report_current_week, selected_investor)
    report_tag = tag_analysis(report_current_week, selected_investor)
    report_rotation = theme_change(report_current_week, report_prev_week, selected_investor)

    with st.expander("리포트 입력 데이터", expanded=False):
        st.caption("TOP ETF")
        st.dataframe(report_top, width="stretch", hide_index=True)
        st.caption("급등 ETF")
        st.dataframe(report_rising, width="stretch", hide_index=True)
        st.caption("테마 분석")
        st.dataframe(report_theme, width="stretch", hide_index=True)
        st.caption("태그 분석")
        st.dataframe(report_tag, width="stretch", hide_index=True)
        st.caption("테마 로테이션")
        st.dataframe(report_rotation, width="stretch", hide_index=True)

    if st.button("주간 ETF 마케팅 리포트 생성", type="primary"):
        with st.spinner("Gemini로 리포트를 생성하는 중입니다."):
            report = generate_ai_report(
                report_top,
                report_rising,
                report_theme,
                report_tag,
                report_rotation,
                selected_investor,
                report_current_week,
                report_prev_week,
            )
            st.markdown(report)
with tabs[12]:
    st.subheader("분류 진단")

    (
        theme_counts,
        others,
        total_count,
        other_count,
        other_ratio,
        previous_other_count,
        previous_other_ratio,
    ) = classification_diagnostics()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("전체 ETF 수", f"{total_count:,}")
    col2.metric("기타 ETF 수", f"{other_count:,}", f"{other_count - previous_other_count:+,}")
    col3.metric("기타 ETF 비중", f"{other_ratio:.1f}%", f"{other_ratio - previous_other_ratio:+.1f}%p")
    col4.metric("목표 기타 비중", "< 10%", "달성" if other_ratio < 10 else "미달")

    st.caption(
        f"이전 분류 기준 기타 비중: {previous_other_ratio:.1f}% "
        f"({previous_other_count:,}/{total_count:,}) | 개선 후 기타 비중: {other_ratio:.1f}% "
        f"({other_count:,}/{total_count:,})"
    )

    st.caption("대표테마 TOP20")
    st.dataframe(theme_counts.head(20), width="stretch", hide_index=True)

    st.caption("기타 ETF TOP100")
    st.dataframe(others.head(100), width="stretch", hide_index=True)

    csv = others.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "기타 ETF 다운로드",
        csv,
        "others.csv",
        "text/csv",
    )
