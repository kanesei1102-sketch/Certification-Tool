import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# 1. ページ構成
st.set_page_config(page_title="Scientific Stat Engine", layout="wide")
st.title("🔬 Scientific Stat Engine")
st.markdown("データの性質を自動診断し、科学的に正しい統計手法を採択します。")

# 2. データ入力
if 'g_count' not in st.session_state: st.session_state.g_count = 3
c1, _ = st.columns([1, 4])
with c1:
    if st.button("＋ グループ追加"): st.session_state.g_count += 1
    if st.session_state.g_count > 2 and st.button("－ グループ削除"): st.session_state.g_count -= 1

st.divider()
data_dict = {}
cols = st.columns(3)
for i in range(st.session_state.g_count):
    with cols[i % 3]:
        name = st.text_input(f"Group {i+1} Name", value=f"Group {i+1}", key=f"n{i}")
        raw = st.text_area(f"{name} Data (数値)", key=f"d{i}", height=100)
        vals = [float(x.strip()) for x in raw.replace(',', '\n').split('\n') if x.strip()]
        if len(vals) >= 3: data_dict[name] = vals

# 3. 診断・解析エンジン
if len(data_dict) >= 2:
    st.header("📊 解析結果と採用理由")
    
    # --- A. 診断: 正規性と等分散性 ---
    all_normal = True
    shapiro_results = []
    for name, vals in data_dict.items():
        _, p_shap = stats.shapiro(vals)
        all_normal &= (p_shap > 0.05)
        shapiro_results.append(f"{name}(p={p_shap:.4f})")
    
    _, p_levene = stats.levene(*data_dict.values())
    is_equal_var = (p_levene > 0.05)

    # --- B. 検定の選択と実行 ---
    reason = ""
    if len(data_dict) == 2:
        # 2群比較
        names = list(data_dict.keys())
        v1, v2 = data_dict[names[0]], data_dict[names[1]]
        if all_normal:
            if is_equal_var:
                method = "Student's t-test"
                reason = "両群に正規性と等分散性が認められたため、標準的なt検定を採用しました。"
                _, p_val = stats.ttest_ind(v1, v2, equal_var=True)
            else:
                method = "Welch's t-test"
                reason = "正規性は認められましたが、分散が異なる（不等分散）ため、ウェルチのt検定を採用しました。"
                _, p_val = stats.ttest_ind(v1, v2, equal_var=False)
        else:
            method = "Mann-Whitney U-test"
            reason = "データが正規分布に従わないため、ノンパラメトリック検定を採用しました。"
            _, p_val = stats.mannwhitneyu(v1, v2, alternative='two-sided')
        
        # 結果表示
        p_disp = f"{p_val:.2e}" if p_val < 0.001 else f"{p_val:.4f}"
        st.success(f"**採用手法: {method}**")
        st.info(f"**理由:** {reason}")
        st.metric("P-value", p_disp)

    else:
        # 3群以上 (ANOVA + Tukey)
        if all_normal and is_equal_var:
            method = "One-way ANOVA + Tukey's HSD"
            reason = "全群の正規性と等分散性が確認されたため、分散分析および多重比較（Tukey法）を採用しました。"
            f_stat, p_anova = stats.f_oneway(*data_dict.values())
            st.success(f"**採用手法: {method}**")
            st.info(f"**理由:** {reason}")
            
            p_a_disp = f"{p_anova:.2e}" if p_anova < 0.001 else f"{p_anova:.4f}"
            st.write(f"ANOVA全体 P値: **{p_a_disp}**")
            
            if p_anova < 0.05:
                flat_data = [v for sub in data_dict.values() for v in sub]
                labels = [n for n, sub in data_dict.items() for _ in sub]
                tukey = pairwise_tukeyhsd(flat_data, labels)
                df_t = pd.DataFrame(data=tukey._results_table.data[1:], columns=tukey._results_table.data[0])
                st.table(df_t)
        else:
            st.error("3群以上の非正規データまたは不等分散データの解析は、より複雑な補正が必要です。現状は正規分布のみ対応しています。")

    # 診断ログの表示
    with st.expander("統計診断ログ (先生への説明用)"):
        st.write(f"- 正規性判定 (p > 0.05で合格): {', '.join(shapiro_results)}")
        st.write(f"- 等分散性判定 (p > 0.05で合格): p = {p_levene:.4f}")
else:
    st.info("各群3つ以上の数値を入力してください。")
