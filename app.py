import streamlit as st
import pandas as pd
from scipy import stats
import numpy as np

# ---------------------------------------------------------
# 1. ページ構成
# ---------------------------------------------------------
st.set_page_config(page_title="Bio-Stat Engine", layout="centered")
st.title("🧪 Bio-Stat Engine")
st.markdown("数値データを入力するだけで、最適な統計検定を自動選択し、P値とラベルを算出します。")

# ---------------------------------------------------------
# 2. データ入力セクション
# ---------------------------------------------------------
st.header("📂 データ入力")
col1, col2 = st.columns(2)

with col1:
    name1 = st.text_input("グループ 1 の名前", value="Control")
    input1 = st.text_area(f"{name1} の数値 (改行区切り)", value="100\n102\n98\n105\n95")

with col2:
    name2 = st.text_input("グループ 2 の名前", value="Target")
    input2 = st.text_area(f"{name2} の数値 (改行区切り)", value="80\n85\n78\n82\n88")

# 数値への変換処理
def parse_input(text):
    try:
        return [float(x.strip()) for x in text.replace(',', '\n').split('\n') if x.strip()]
    except:
        return []

data1 = parse_input(input1)
data2 = parse_input(input2)

# ---------------------------------------------------------
# 3. 解析ロジック
# ---------------------------------------------------------
if len(data1) > 2 and len(data2) > 2:
    st.divider()
    st.header("📊 解析結果")

    # A. 正規性の検定 (Shapiro-Wilk)
    _, p_shapiro1 = stats.shapiro(data1)
    _, p_shapiro2 = stats.shapiro(data2)
    is_normal = (p_shapiro1 > 0.05) and (p_shapiro2 > 0.05)

    # B. 等分散性の検定 (Levene)
    _, p_levene = stats.levene(data1, data2)
    is_equal_var = (p_levene > 0.05)

    # C. 検定の選択と実行
    test_name = ""
    p_value = 0.0

    if is_normal:
        if is_equal_var:
            test_name = "Student's t-test (対応なし・等分散)"
            _, p_value = stats.ttest_ind(data1, data2, equal_var=True)
        else:
            test_name = "Welch's t-test (対応なし・不等分散)"
            _, p_value = stats.ttest_ind(data1, data2, equal_var=False)
    else:
        test_name = "Mann-Whitney U-test (ノンパラメトリック)"
        _, p_value = stats.mannwhitneyu(data1, data2, alternative='two-sided')

    # D. 有意差ラベルの決定
    sig_label = ""
    if p_value < 0.001: sig_label = "***"
    elif p_value < 0.01: sig_label = "**"
    elif p_value < 0.05: sig_label = "*"
    else: sig_label = "ns"

    # ---------------------------------------------------------
    # 4. 結果表示
    # ---------------------------------------------------------
    res_col1, res_col2 = st.columns(2)
    with res_col1:
        st.metric("P-value", f"{p_value:.4f}")
        st.write(f"採用された検定: **{test_name}**")
    
    with res_col2:
        st.subheader("有意差ラベル")
        st.code(sig_label, language=None)
        st.caption("描画ツールの「有意差」欄にコピー＆ペーストしてください")

    # 詳細診断
    with st.expander("詳細な診断データ"):
        st.write(f"- {name1} 正規性 (p): {p_shapiro1:.4f}")
        st.write(f"- {name2} 正規性 (p): {p_shapiro2:.4f}")
        st.write(f"- 等分散性 (p): {p_levene:.4f}")
        st.info("p > 0.05 であれば、その前提条件（正規性・等分散性）を満たしていると判断されます。")

else:
    st.info("各グループに少なくとも3つ以上の数値を入力してください。")
