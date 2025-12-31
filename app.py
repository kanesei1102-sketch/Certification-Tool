import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import scikit_posthocs as sp # 3群以上のノンパラ比較用

# --- 有意差ラベル判定用関数 ---
def get_sig_label(p):
    if p < 0.001: return "***"
    if p < 0.01: return "**"
    if p < 0.05: return "*"
    return "ns"

# 1. ページ構成
with st.sidebar:
        # 既存の入力フォームなどのコード...
        st.write("---") # 区切り線
        st.markdown("""
        ### 【Notice / ご案内】
        This tool is a beta version. If you plan to use results from this tool in your publications or conference presentations, **please contact the developer (Seiji Kaneko) in advance.**

        本ツールは現在開発中のベータ版です。論文掲載や学会発表等に使用される際は、**事前に開発者（金子）まで必ず一報ください。**

        👉 **[Contact & Feedback Form / 連絡窓口](https://forms.gle/xgNscMi3KFfWcuZ1A)**

        We will provide guidance on validation support and proper acknowledgments/co-authorship.
        バリデーションのサポートや、謝辞・共著の記載についてご案内させていただきます。
        """)
st.set_page_config(page_title="Ultimate Stat Engine", layout="wide")
st.title("🔬 Ultimate Scientific Stat Engine")
st.markdown("データの性質を自動診断し、最適な検定と報告用レポートを作成します。")

# 2. グループ管理
if 'g_count' not in st.session_state: st.session_state.g_count = 3
c1, _ = st.columns([1, 4])
with c1:
    if st.button("＋ グループ追加"): st.session_state.g_count += 1
    if st.session_state.g_count > 2 and st.button("－ グループ削除"): st.session_state.g_count -= 1

st.divider()

# 3. データ入力
data_dict = {}
cols = st.columns(3)
for i in range(st.session_state.g_count):
    with cols[i % 3]:
        name = st.text_input(f"Group {i+1} Name", value=f"Group {i+1}", key=f"n{i}")
        raw = st.text_area(f"{name} の数値 (改行区切り)", key=f"d{i}", height=120)
        vals = [float(x.strip()) for x in raw.replace(',', '\n').split('\n') if x.strip()]
        if len(vals) >= 3: data_dict[name] = vals

# 4. 解析エンジン
if len(data_dict) >= 2:
    st.header("📊 解析結果と採用理由")
    
    # --- 診断: 正規性と等分散性 ---
    all_normal = True
    shapiro_log = []
    for n, v in data_dict.items():
        _, p_s = stats.shapiro(v)
        all_normal &= (p_s > 0.05)
        shapiro_log.append(f"{n}(p={p_s:.4f})")
    
    _, p_lev = stats.levene(*data_dict.values())
    is_equal_var = (p_lev > 0.05)

    # 初期化
    method = ""
    p_final = 0.0
    p_disp = ""

    # A. 2群比較
    if len(data_dict) == 2:
        gn = list(data_dict.keys())
        v1, v2 = data_dict[gn[0]], data_dict[gn[1]]
        if all_normal:
            method = "Student's t-test" if is_equal_var else "Welch's t-test"
            _, p_final = stats.ttest_ind(v1, v2, equal_var=is_equal_var)
        else:
            method = "Mann-Whitney U-test"
            _, p_final = stats.mannwhitneyu(v1, v2, alternative='two-sided')
        
        st.success(f"**採用手法: {method}**")
        p_disp = f"{p_final:.2e}" if p_final < 0.001 else f"{p_final:.4f}"
        st.metric("P-value", p_disp)

    # B. 3群以上
    else:
        if all_normal and is_equal_var:
            method = "One-way ANOVA + Tukey's HSD"
            _, p_anova = stats.f_oneway(*data_dict.values())
            p_final = p_anova
            st.success(f"**採用手法: {method}**")
            p_disp = f"{p_anova:.2e}" if p_anova < 0.001 else f"{p_anova:.4f}"
            st.write(f"全体P値 (ANOVA): **{p_disp}**")
            
            if p_anova < 0.05:
                flat_data = [v for sub in data_dict.values() for v in sub]
                labels = [n for n, sub in data_dict.items() for _ in sub]
                res = pairwise_tukeyhsd(flat_data, labels)
                df_res = pd.DataFrame(data=res._results_table.data[1:], columns=res._results_table.data[0])
                st.table(df_res)
        else:
            method = "Kruskal-Wallis test (ノンパラメトリック)"
            _, p_kw = stats.kruskal(*data_dict.values())
            p_final = p_kw
            st.warning(f"**採用手法: {method}**")
            p_disp = f"{p_kw:.4f}"
            st.write(f"全体P値 (Kruskal-Wallis): **{p_disp}**")
            
            if p_kw < 0.05:
                st.write("各ペアの比較 (Dunn's test):")
                df_dunn = sp.posthoc_dunn(list(data_dict.values()), p_adjust='bonferroni')
                df_dunn.columns = df_dunn.index = data_dict.keys()
                st.table(df_dunn)

    # --- 診断ログの表示 ---
    with st.expander("詳細な診断ログ (先生への説明用)"):
        st.write(f"・正規性判定: {', '.join(shapiro_log)}")
        st.write(f"・等分散性判定: p = {p_lev:.4f}")

    # --- 5. 初心者でもわかる報告用レポート作成 ---
    st.divider()
    st.header("📝 そのまま使える報告用レポート")
    
    if all_normal and is_equal_var:
        easy_reason = "データの分布が偏っておらず、バラツキも均一だったため、最も標準的で精度の高い『t検定/ANOVA』を選択しました。"
    elif not all_normal:
        easy_reason = "データに極端な偏りや外れ値が見られたため、数値の大小関係（順位）を重視する、外れ値に強い『ノンパラメトリック検定』を選択しました。"
    else:
        easy_reason = "データのバラツキが群の間で異なっていたため、その差を補正して計算する『Welchの方法』を選択しました。"

    is_significant = (p_final < 0.05)
    result_summary = "【有意差あり】グループ間に、偶然とは言い切れない明らかな差が見つかりました。" if is_significant else "【有意差なし】グループ間の差は、誤差の範囲内である可能性が高いです。"

    report_text = f"""
【解析報告書：{", ".join(data_dict.keys())} の比較】

1. この解析で何を確認したか：
   各グループの数値の平均に、意味のある「違い」があるかどうかを調べました。

2. どの方法で調べたか（その理由）：
   採用した手法：{method}
   理由：{easy_reason}
   ※ 闇雲に計算するのではなく、データの形（正規性）やバラツキ（等分散性）を事前にチェックした上で、最も科学的に妥当な手順を選んでいます。

3. 解析の結果：
   判定：{result_summary}
   全体のP値：{p_disp}
   （※P値が0.05より小さければ、統計学的に「差がある」と判断します）

4. 個別の違い（多重比較）：
   {"3群以上の比較のため、各ペアを総当たりで調べ、厳しい基準で有意差を判定しました。詳細は結果表のラベルを確認してください。" if len(data_dict) > 2 else "2つのグループを直接比較しました。"}

5. 結論：
   解析の結果、今回のデータからは統計学的な裏付けが得られました。この内容に基づき、有意差ラベルを付与したグラフを作成してください。
    """
    
    st.text_area("主査への説明やスライドのメモにコピペしてください", value=report_text, height=400)

    # --- 6. ダウンロードボタン ---
    st.download_button(
        label="レポートをテキストファイルとして保存",
        data=report_text,
        file_name="statistical_report.txt",
        mime="text/plain"
    )

else:
    st.info("解析を始めるには、各グループに3つ以上の数値を入力してください。")
    # --- 画面の最下部に免責事項を表示 ---
    st.divider() # 区切り線
    st.caption("【免責事項 / Disclaimer】")
    st.caption("""
    本ツールは統計学的判断および解析の補助を目的としています。
    計算には信頼性の高いライブラリを使用していますが、最終的な解釈および結論については、
    利用者が専門的知見に基づいて判断してください。

    This tool is for assistive purposes. Final interpretations and conclusions 
    should be made by the user based on professional expertise.
    """)
