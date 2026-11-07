"""Demo DSS hỗ trợ quyết định – tái hiện & cải tiến Zolbanin et al. (2026). Giao diện Streamlit.
Phần xử lý nằm trong core.py (sinh tự động từ demo/src/demo_dss.py)."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Demo DSS tái phạm", page_icon="⚖️", layout="wide")


from pathlib import Path


@st.cache_resource(show_spinner="Đang tải dữ liệu và huấn luyện mô hình chính… (lần đầu khoảng 1–2 phút)")
def load_core():
    import core          # chỉ tải dữ liệu + huấn luyện RF chính; phân tích nặng tính khi cần
    return core


C = load_core()


@st.cache_resource(show_spinner=False)
def precomputed():
    p = Path(__file__).with_name("precomputed.json")
    return C.load_results(p) if p.exists() else {}


@st.cache_resource(show_spinner="Đang tính kết quả cho trang này… (chỉ lần đầu; có thể mất vài phút)")
def _compute(name):
    return getattr(C, "compute_" + name)()


def R(name):
    """Lấy kết quả phân tích: ưu tiên precomputed.json, nếu không có thì tính một lần rồi giữ trong bộ nhớ."""
    pre = precomputed()
    return pre[name] if name in pre else _compute(name)
RED, BLUE, GREY, PURPLE = "#d62728", "#1f77b4", "#7f7f7f", "#6c5ce7"
TEST_IDS = list(C.X_test.index[:300])
WHATIF = [f for f in ["percent_days_employed", "jobs_per_year", "program_unexcusedabsences",
                      "drugtests_meth_positive", "residence_changes", "delinquency_reports"] if f in C.FEATS]
ATTR = {"Chủng tộc": "race", "Giới tính": "gender", "Nhóm tuổi": "age_group"}


def label_of(i):
    return f"Hồ sơ #{i} — thực tế: {C.LABEL[int(C.y.loc[i])]}"


def risk_cal(row):
    return float(C.p_cal_of(row.to_frame().T)[0])


@st.cache_data(show_spinner=False)
def explain(i):
    xdf = C.X_test.loc[i].to_frame().T[C.FEATS]
    p_raw = float(C.rf.predict_proba(xdf)[:, 1][0])
    contribs = pd.Series(C.shap_pos(xdf)[0], index=C.FEATS).sort_values(key=np.abs, ascending=False)
    return p_raw, contribs


@st.cache_data(show_spinner=False)
def counterfactual(i):
    return C.counterfactual(C.X_test.loc[i])[0]


def clean(ax):
    ax.spines[["top", "right"]].set_visible(False)


def shap_fig(contribs, k=10):
    c = contribs.head(k)[::-1]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.barh([C.vn(f) for f in c.index], c.values, color=[RED if v > 0 else BLUE for v in c.values])
    ax.axvline(0, color=GREY, lw=0.8)
    ax.set_xlabel("Giá trị SHAP  (đỏ = tăng rủi ro, xanh = giảm rủi ro)")
    clean(ax); fig.tight_layout()
    return fig


def side_by_side(demo_df, paper_df, title_demo="Demo (NIJ – Georgia)", title_paper="Bài báo (Oklahoma)"):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**{title_demo}**")
        st.dataframe(demo_df.round(3), use_container_width=True)
    with c2:
        st.markdown(f"**{title_paper}**")
        st.dataframe(paper_df.round(3), use_container_width=True)


# ================= Thanh bên =================
with st.sidebar:
    st.markdown("## ⚖️ Demo DSS tái phạm")
    mode = st.radio("Chế độ", ["📄 Bản gốc (theo bài báo)", "🚀 Bản cải tiến (gap của nhóm)"])
    st.divider()
    st.caption("Zolbanin et al., *Designing transparent, equitable, and efficient decision support systems for "
               "drug courts using machine learning*, Decision Support Systems, vol. 204, 2026.")
    st.markdown("**Dữ liệu:** NIJ Recidivism Forecasting Challenge (Georgia) – thay cho dữ liệu Oklahoma không công khai; "
                "COMPAS (Broward County, FL) dùng cho kiểm chứng G6.")
    st.metric("Số hồ sơ NIJ", f"{len(C.X):,}")
    st.metric("Tỉ lệ tái phạm 3 năm", f"{C.y.mean():.1%}")

# ================= BẢN GỐC =================
if mode.startswith("📄"):
    st.title("📄 Bản gốc — tái hiện DSS theo bài báo")
    st.caption("Pipeline của bài báo: Random Forest (+ undersampling) → điểm rủi ro → SHAP → khuyến nghị can thiệp. "
               "Module GenAI (OpenAI o1-preview) được mô phỏng bằng thư viện can thiệp ánh xạ từ các yếu tố có SHAP cao nhất.")
    page = st.radio("Trang", ["Dashboard DSS", "Hiệu năng mô hình (Table 5)", "Tích hợp đa nguồn (Table 6)",
                              "Công bằng (Table 7–8)", "Biến quan trọng (Fig. 4)", "Đánh giá người dùng"],
                    horizontal=True, label_visibility="collapsed")
    st.divider()

    if page == "Dashboard DSS":
        i = st.selectbox("Nhập / chọn mã người tham gia", TEST_IDS, format_func=label_of, key="pid_o")
        p_raw, contribs = explain(i)
        row = C.X_test.loc[i]
        high = p_raw >= 0.5
        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown("##### Tầng 1 – Phân loại nhanh (System 1)")
            st.metric("Điểm rủi ro tái phạm 3 năm", f"{p_raw:.0%}")
            (st.error if high else st.success)(f"Mức rủi ro: **{'CAO' if high else 'THẤP'}**")
        with c2:
            st.markdown("##### Tầng 2 – Giải thích SHAP (System 2)")
            st.pyplot(shap_fig(contribs))
        st.markdown("##### Khuyến nghị can thiệp cá nhân hoá")
        recs = C.paper_recommendations(row, contribs)
        if len(recs):
            st.dataframe(recs, use_container_width=True, hide_index=True)
        else:
            st.write("Không có yếu tố nào làm tăng rủi ro đáng kể.")
        st.caption("Bài báo gửi giá trị biến + điểm SHAP + từ điển dữ liệu cho LLM và yêu cầu tối đa 3 can thiệp. "
                   "Demo dùng cùng logic (các yếu tố có SHAP dương lớn nhất, kèm giá trị thực) nhưng sinh nội dung từ thư viện can thiệp cố định.")

    elif page == "Hiệu năng mô hình (Table 5)":
        t5 = R("t5")
        st.markdown("So sánh ba thuật toán của bài báo, cùng huấn luyện trên dữ liệu đã undersampling, ngưỡng 0.5.")
        side_by_side(t5, C.PAPER_T5)
        fig, ax = plt.subplots(figsize=(8, 3.4)); w = 0.25; xs = np.arange(len(t5.columns))
        for k, (name, r) in enumerate(t5.iterrows()):
            ax.bar(xs + (k - 1) * w, r.values, w, label=name)
        ax.set_xticks(xs); ax.set_xticklabels(t5.columns); ax.set_ylim(0, 1); ax.legend(fontsize=8); clean(ax)
        fig.tight_layout(); st.pyplot(fig)
        best = t5["Recall"].idxmax()
        st.info(f"Bài báo chọn RF vì recall cao nhất. Trên dữ liệu NIJ, mô hình có recall cao nhất là **{best}**.")

    elif page == "Tích hợp đa nguồn (Table 6)":
        t6 = R("t6")
        st.markdown("Bỏ lần lượt từng nhóm nguồn dữ liệu rồi huấn luyện lại. Hiệu năng giảm càng nhiều thì nhóm đó càng quan trọng (mục tiêu O1).")
        side_by_side(t6, C.PAPER_T6)
        drop = (t6.loc["Tất cả biến", "AUC"] - t6["AUC"]).drop("Tất cả biến")
        fig, ax = plt.subplots(figsize=(7, 3)); ax.barh(drop.index[::-1], drop.values[::-1], color=PURPLE)
        ax.set_xlabel("Mức giảm AUC khi bỏ nhóm"); clean(ax); fig.tight_layout(); st.pyplot(fig)
        with st.expander("Cách ánh xạ biến NIJ vào 5 nhóm của bài báo"):
            st.dataframe(pd.DataFrame({"Biến": [C.vn(f) for f in C.FEATS], "Nhóm": [C.CATEGORY[f] for f in C.FEATS]}),
                         use_container_width=True, hide_index=True)

    elif page == "Công bằng (Table 7–8)":
        st.markdown("Đúng cách của bài báo: so **tỉ trọng mỗi nhóm chủng tộc trong số ca FN/FP** với tỉ trọng của nhóm đó "
                    "trong tập test (kiểm định hai tỉ lệ, mức ý nghĩa 5%). p-value < 0.05 nghĩa là có chênh lệch.")
        ft = C.paper_fairness_table(C.y_test, C.p_test_raw, C.meta_test["race"])
        st.dataframe(ft.round(3), use_container_width=True, hide_index=True)
        st.caption("Bài báo: không có khác biệt ý nghĩa ở FN; với FP, nhóm 'Other' p = 0.045 và 'Unknown' (n = 5) p < 0.001.")

    elif page == "Biến quan trọng (Fig. 4)":
        st.markdown("Biến quan trọng toàn cục: làm nhiễu từng biến và đo mức giảm AUC (tương tự phân tích độ nhạy của bài báo).")
        imp = R("perm_imp").head(10)[::-1]
        fig, ax = plt.subplots(figsize=(8, 4)); ax.barh([C.vn(f) for f in imp.index], imp.values, color=PURPLE)
        ax.set_xlabel("Mức giảm AUC"); clean(ax); fig.tight_layout(); st.pyplot(fig)
        st.caption("Bài báo (Fig. 4): top biến gồm số năm tù nếu thất bại, số lần chế tài, số lần khen thưởng, tuổi lúc bị bắt, "
                   "số lần bị bắt trước đó, thu nhập từ việc làm, thời gian chờ, quyền nuôi con.")

    else:
        st.markdown("Kết quả khảo sát 42 chuyên gia **do bài báo báo cáo** (Table 9–11). Phần này không tái hiện được bằng dữ liệu công khai.")
        survey = pd.DataFrame({
            "Chiều đánh giá": ["Relevance", "Actionability", "Interpretability", "Fairness", "Transparency", "Trustworthiness", "Usability"],
            "Mean": [4.23, 4.18, 4.12, 4.05, 3.95, 3.89, 3.78],
            "SD": [0.61, 0.52, 0.58, 0.82, 0.74, 0.71, 0.58],
            "Cronbach α": [0.785, 0.712, 0.847, 0.823, 0.801, 0.798, 0.689],
            "% người ≥ 4": ["85.4%", "82.9%", "78.0%", "73.2%", "68.3%", "65.9%", "58.5%"],
            "Mức đạt": ["Đạt đầy đủ"] * 4 + ["Đạt phần lớn"] * 3})
        st.dataframe(survey, use_container_width=True, hide_index=True)
        fig, ax = plt.subplots(figsize=(7, 3.2))
        ax.barh(survey["Chiều đánh giá"][::-1], survey["Mean"][::-1], color=PURPLE)
        ax.axvline(3, color=GREY, ls="--", lw=0.8); ax.set_xlim(1, 5); ax.set_xlabel("Điểm trung bình (Likert 1–5; đường đứt = trung lập)")
        clean(ax); fig.tight_layout(); st.pyplot(fig)
        st.caption("Đánh giá chuyên gia (n = 2): độ tự tin +9,7 điểm (p < 0.001); thời gian quyết định −11,8% (không có ý nghĩa thống kê).")

# ================= BẢN CẢI TIẾN =================
else:
    st.title("🚀 Bản cải tiến — khắc phục các gap của bài báo")
    page = st.radio("Trang", ["G1·G2·G3 – Hồ sơ nâng cao", "G1 – What-if", "G3 – Độ bất định",
                              "G4 – Công bằng chuẩn", "G5·G6 – Rò rỉ & khái quát hoá"],
                    horizontal=True, label_visibility="collapsed")
    st.divider()

    if page.startswith("G1·G2"):
        a, b = st.columns([3, 2])
        i = a.selectbox("Chọn hồ sơ (tập test)", TEST_IDS, format_func=label_of, key="pid_e")
        alpha = b.slider("α conformal (1 − độ tin cậy)", 0.05, 0.30, float(C.ALPHA), 0.05, key="alpha_e")
        p_raw, contribs = explain(i)
        p_cal = float(C.iso.predict([p_raw])[0])
        pset = sorted(C.conformal_set(p_cal, C.qhat(alpha)))
        changes = counterfactual(i)
        row = C.X_test.loc[i]
        m1, m2, m3 = st.columns(3)
        m1.metric("Rủi ro đã hiệu chỉnh (G3)", f"{p_cal:.0%}")
        m2.metric("Điểm thô RF (bản gốc)", f"{p_raw:.0%}", help="Bị lệch do undersampling")
        m3.metric(f"Tập conformal ({1 - alpha:.0%})", "{" + ", ".join(C.LABEL[s] for s in pset) + "}")
        if len(pset) == 2:
            st.warning("⚠️ Mô hình **không đủ chắc chắn** với hồ sơ này → đề nghị hội đồng xem xét thủ công.")
        else:
            st.success("✅ Mô hình đủ chắc chắn ở mức tin cậy đã chọn.")
        left, right = st.columns(2)
        with left:
            st.markdown("#### Giải thích SHAP")
            st.pyplot(shap_fig(contribs))
        with right:
            st.markdown("#### G2 – Narrative có kiểm chứng")
            text, claims = C.template_narrative(row, p_cal, set(pset), contribs, changes)
            st.markdown(text)
            score, ftab = C.faithfulness(claims, contribs)
            st.info(f"Faithfulness: **{score:.0%}** khẳng định khớp với SHAP.")
            with st.expander("Bảng kiểm chứng từng khẳng định"):
                st.dataframe(ftab, use_container_width=True, hide_index=True)
        st.markdown("#### G1 – Kế hoạch can thiệp phản thực tế")
        st.caption("Chỉ thay đổi các biến có thể can thiệp, có ràng buộc chiều thay đổi. Đây là phản thực tế của mô hình, không phải quan hệ nhân quả.")
        if changes:
            st.dataframe(pd.DataFrame([{"Biến can thiệp": C.vn(f), "Hiện tại": C.fmt(x0), "Đề xuất": C.fmt(x1),
                                        "Rủi ro trước": f"{C.iso.predict([q0])[0]:.0%}",
                                        "Rủi ro sau": f"{C.iso.predict([q1])[0]:.0%}"}
                                       for f, x0, x1, q0, q1 in changes]), use_container_width=True, hide_index=True)
        else:
            st.write("Không cần, hoặc không tìm được thay đổi khả thi trên các biến can thiệp được.")

    elif page == "G1 – What-if":
        st.markdown("Kéo thanh trượt các biến **có thể can thiệp** để xem rủi ro (đã hiệu chỉnh) thay đổi thế nào.")
        j = st.selectbox("Chọn hồ sơ", TEST_IDS, format_func=label_of, key="pid_wi")
        base = C.X_test.loc[j]; new = base.copy(); cols = st.columns(2)
        for k, f in enumerate(WHATIF):
            lo, hi = float(C.X_train[f].min()), float(C.X_train[f].max())
            step = 0.01 if hi <= 1 else 1.0
            new[f] = cols[k % 2].slider(C.vn(f), lo, hi, float(np.clip(base[f], lo, hi)), step, key=f"wi_{j}_{f}")
        p0, p1 = risk_cal(base), risk_cal(new)
        c1, c2 = st.columns(2)
        c1.metric("Rủi ro ban đầu", f"{p0:.0%}")
        c2.metric("Rủi ro sau thay đổi", f"{p1:.0%}", f"{(p1 - p0) * 100:+.1f} điểm %", delta_color="inverse")

    elif page == "G3 – Độ bất định":
        al = st.slider("α conformal", 0.05, 0.30, float(C.ALPHA), 0.05, key="alpha4")
        q, cover, amb = C.conformal_report(al)
        c1, c2, c3 = st.columns(3)
        c1.metric("Độ tin cậy mục tiêu", f"{1 - al:.0%}")
        c2.metric("Độ phủ thực tế (test)", f"{cover:.1%}")
        c3.metric("Hồ sơ gắn cờ 'không chắc chắn'", f"{amb:.1%}")
        b1, b2 = st.columns(2)
        b1.metric("Brier – bản gốc", f"{C.res_base.iloc[1]['Brier']:.3f}")
        b2.metric("Brier – sau hiệu chỉnh", f"{C.res_base.iloc[2]['Brier']:.3f}", delta_color="inverse",
                  delta=f"{C.res_base.iloc[2]['Brier'] - C.res_base.iloc[1]['Brier']:+.3f}")
        fig, ax = plt.subplots(figsize=(6, 4.6)); bins = np.linspace(0, 1, 11)
        for p, lab in [(C.p_test_raw, "RF + undersampling (bản gốc)"), (C.p_test_cal, "Sau hiệu chỉnh isotonic")]:
            bi = np.clip(np.digitize(p, bins) - 1, 0, 9)
            ks = [k for k in range(10) if (bi == k).sum() > 20]
            ax.plot([p[bi == k].mean() for k in ks], [C.y_test.values[bi == k].mean() for k in ks], marker="o", label=lab)
        ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Hiệu chỉnh hoàn hảo")
        ax.set_xlabel("Xác suất dự đoán"); ax.set_ylabel("Tỉ lệ tái phạm thực tế"); ax.legend(); clean(ax); fig.tight_layout()
        st.pyplot(fig, use_container_width=False)

    elif page == "G4 – Công bằng chuẩn":
        attr = st.radio("Thuộc tính nhạy cảm", list(ATTR), horizontal=True)
        g = C.meta_test[ATTR[attr]]
        gm = C.group_metrics(C.y_test, C.p_test_raw, g)
        st.metric("Equalized-odds difference", f"{C.eod(gm):.3f}", help="0 = công bằng tuyệt đối")
        st.markdown("**FPR/FNR trong từng nhóm** (bài báo chỉ so tỉ trọng nhóm trong FP/FN, và chỉ theo chủng tộc)")
        st.dataframe(gm.round(3), use_container_width=True, hide_index=True)
        fig, ax = plt.subplots(figsize=(7, 3.6)); w = 0.38; xs = np.arange(len(gm))
        ax.bar(xs - w / 2, gm["FPR"], w, label="FPR (bị gắn nhầm rủi ro cao)", color=RED)
        ax.bar(xs + w / 2, gm["FNR"], w, label="FNR (bỏ sót người tái phạm)", color=BLUE)
        ax.set_xticks(xs); ax.set_xticklabels(gm["Nhóm"], rotation=15); ax.legend(); clean(ax); fig.tight_layout()
        st.pyplot(fig)

    else:
        st.subheader("G5 — Biến lúc tiếp nhận vs biến phát sinh trong chương trình")
        st.dataframe(R("g5").round(3), use_container_width=True)
        st.caption("Nếu hiệu năng giảm mạnh khi chỉ dùng biến lúc tiếp nhận, con số của mô hình 'tất cả biến' "
                   "không phản ánh năng lực dự đoán tại thời điểm ra quyết định đầu vào.")
        st.subheader("G6 — Cùng một pipeline trên nhiều bộ dữ liệu")
        st.dataframe(R("g6").round(3), use_container_width=True, hide_index=True)
        st.subheader("G6 — 5-fold cross-validation trên NIJ (bài báo chỉ dùng 1 lần holdout)")
        st.dataframe(R("cv").round(3), use_container_width=True)
