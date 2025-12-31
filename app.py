import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import scikit_posthocs as sp # pip install scikit-posthocs が必要です

# 1. ページ設定
st.set_page_config(page_title="Ultimate Stat Engine", layout="wide")
st.title("🔬 Ultimate Scientific Stat Engine")
st.markdown("データの性質（正規性・分散・群数）を自動診断し、論文に耐えうる最適な検定を自動実行します。")

# 2. グループ管理
if 'g_count' not in st.session_state: st.session_state.g_count = 3
c1, _ = st.columns([1, 4])
with c1:
    if st.button("＋ グループ追加"): st.session_state.g_count += 1
    if st.session_state.g_count > 2 and st.button("－ グループ削除"): st.session_state.g_count -= 1

# 3. データ入力欄
data_dict = {}
cols = st.columns(3)
for i in range(st.session_state.g_count):
    with cols[i % 3]:
        name = st.text_input(f"グループ {i+1} の名前", value=f"Group {i+1}", key=f"n{i}")
        raw = st.text_area(f"{name} の数値 (改行区切り)", key=f"d{i}", height=120)
        vals = [float(x.strip()) for x in raw.replace(',', '\n').split('\n') if x.strip()]
        if len(vals) >= 3: data_dict[name] = vals

# 4. 解析ロジック
if len(data_dict) >= 2:
    st.header("📊 解析結果と採用された科学的根拠")
    
    # --- 診断: 正規性と等分散性 ---
    all_normal = True
    shapiro_log = []
    for n, v in data_dict.items():
        _, p_s = stats.shapiro(v)
        all_normal &= (p_s > 0.05)
        shapiro_log.append(f"{n}(p={p_s:.4f})")
    
    _, p_lev = stats.levene(*data_dict.values())
    is_equal_var = (p_lev > 0.05)

    # --- 判定と実行 ---
    # A. 2群比較の場合
    if len(data_dict) == 2:
        gn = list(data_dict.keys())
        v1, v2 = data_dict[gn[0]], data_dict[gn[1]]
        if all_normal:
            method = "Student's t-test" if is_equal_var else "Welch's t-test"
            _, p = stats.ttest_ind(v1, v2, equal_var=is_equal_var)
        else:
            method = "Mann-Whitney U-test"
            _, p = stats.mannwhitneyu(v1, v2, alternative='two-sided')
        
        st.success(f"**採用手法: {method}**")
        p_disp = f"{p:.2e}" if p < 0.001 else f"{p:.4f}"
        st.metric("P-value", p_disp)

    # B. 3群以上の場合
    else:
        if all_normal and is_equal_var:
            method = "One-way ANOVA + Tukey's HSD"
            _, p_anova = stats.f_oneway(*data_dict.values())
            st.success(f"**採用手法: {method}**")
            p_a_disp = f"{p_anova:.2e}" if p_anova < 0.001 else f"{p_anova:.4f}"
            st.write(f"全体P値 (ANOVA): **{p_a_disp}**")
            
            if p_anova < 0.05:
                flat_data = [v for sub in data_dict.values() for v in sub]
                labels = [n for n, sub in data_dict.items() for _ in sub]
                res = pairwise_tukeyhsd(flat_data, labels)
                df_res = pd.DataFrame(data=res._results_table.data[1:], columns=res._results_table.data[0])
                st.table(df_res)
        else:
            method = "Kruskal-Wallis test (ノンパラメトリック)"
            _, p_kw = stats.kruskal(*data_dict.values())
            st.warning(f"**採用手法: {method}**")
            st.info("理由: データに正規性がない、または外れ値があるため、より頑健な手法を選択しました。")
            st.write(f"全体P値 (Kruskal-Wallis): **{p_kw:.4f}**")
            
            if p_kw < 0.05:
                st.write("各ペアの比較 (Dunn's test):")
                df_dunn = sp.posthoc_dunn(list(data_dict.values()), p_adjust='bonferroni')
                df_dunn.columns = df_dunn.index = data_dict.keys()
                st.table(df_dunn)

    # 診断ログ
    with st.expander("詳細な診断ログ (先生への説明用)"):
        st.write(f"・正規性判定: {', '.join(shapiro_log)}")
        st.write(f"・等分散性判定: p = {p_lev:.4f}")
else:
    st.info("2つ以上のグループに数値を入力してください。")
