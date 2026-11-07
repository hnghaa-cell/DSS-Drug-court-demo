# core.py - sinh tự động từ demo/src/demo_dss.py bằng build_streamlit.ps1. Không sửa trực tiếp.

import re, sys, json, math, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (accuracy_score, recall_score, precision_score,
                             f1_score, roc_auc_score, brier_score_loss)
warnings.filterwarnings("ignore")
SEED = 42
IN_COLAB = "google.colab" in sys.modules
ALPHA = 0.10          # mức lỗi conformal mặc định -> độ tin cậy 90%
pd.set_option("display.float_format", lambda v: f"{v:.3f}")

# Portal gốc data.ojp.usdoj.gov đã ngừng hoạt động -> dùng nhiều nguồn dự phòng, thử lần lượt
NIJ_URLS = [
    "https://nij.ojp.gov/sites/nij/files/media/document/nij-challenge2021_full_dataset.csv",   # bản chính thức trên trang NIJ
    "https://raw.githubusercontent.com/MattCon70/NIJ_National_Institute_of_Justice_Recidivism_Challenge-Analysis/main/NIJ_s_Recidivism_Challenge_Full_Dataset.csv",  # bản sao GitHub
    "https://data.ojp.usdoj.gov/api/views/ynf5-u8nk/rows.csv?accessType=DOWNLOAD",              # portal cũ
]
COMPAS_URLS = ["https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv"]

def load_csv(urls, name, required=True):
    for url in urls:
        try:
            df = pd.read_csv(url, storage_options={"User-Agent": "Mozilla/5.0"})
            print(f"Đã tải {name}: {df.shape[0]:,} dòng × {df.shape[1]} cột  (nguồn: {url})")
            return df
        except Exception as e:
            print(f"Không tải được {name} từ {url}: {e}")
    if not required:
        return None
    if not IN_COLAB:
        raise RuntimeError(f"Không tải được {name} từ mọi nguồn.")
    print("→ Hãy upload file CSV thủ công:")
    from google.colab import files
    up = files.upload()
    return pd.read_csv(list(up)[0])

nij_raw = load_csv(NIJ_URLS, "NIJ Recidivism Challenge")
compas_raw = load_csv(COMPAS_URLS, "COMPAS (ProPublica)", required=False)

def to_num(s):
    """Chuyển cột sang số nếu được (Yes/No, True/False, '3 or more', số). Trả None nếu không phải cột số."""
    if s.dtype == bool:
        return s.astype(int)
    if s.dtype != object:
        return s
    x = s.astype(str).str.strip()
    low = x.str.lower()
    valid = low[low != "nan"]
    BIN = {"true": 1, "yes": 1, "1": 1, "false": 0, "no": 0, "0": 0}
    if len(valid) and set(valid.unique()) <= set(BIN):
        return low.map(BIN)
    m = x.str.extract(r"^(\d+(?:\.\d+)?)\s*(?:or more)?$")[0]
    if len(valid) and m.notna().sum() >= 0.95 * len(valid):
        return pd.to_numeric(m, errors="coerce")
    return None

def first_num(v):
    m = re.search(r"\d+", str(v))
    return float(m.group()) if m else np.nan

def prison_years(v):
    t = str(v).lower()
    if "less" in t: return 0
    if t.startswith("1"): return 1
    if "greater" in t: return 2
    if "more" in t: return 3
    return np.nan

ORDINAL = {
    "prison_years": prison_years,
    "education_level": lambda v: {"less than hs diploma": 0, "high school diploma": 1,
                                  "at least some college": 2}.get(str(v).lower(), np.nan),
    "supervision_level_first": lambda v: {"standard": 0, "high": 1,
                                          "specialized": 2}.get(str(v).lower(), np.nan),
    "age_at_release": first_num,
}

def norm_cols(df):
    df = df.copy()
    df.columns = [re.sub(r"[^0-9a-z]+", "_", c.lower()).strip("_") for c in df.columns]
    return df

def prep_nij(raw):
    df = norm_cols(raw)
    y = to_num(df["recidivism_within_3years"]).astype(int)
    meta = pd.DataFrame({
        "race": df["race"].astype(str).str.upper().str.strip(),
        "gender": df["gender"].astype(str).str.upper().str.strip().map({"M": "Nam", "F": "Nữ"}).fillna("Khác"),
        "age_group": df["age_at_release"].astype(str),
    })
    drop = [c for c in df.columns if c.startswith("recidivism_")
            or c in ("id", "training_sample", "residence_puma", "race")]
    df = df.drop(columns=drop)
    X = pd.DataFrame(index=df.index)
    for c in df.columns:
        if c in ORDINAL:
            X[c] = df[c].map(ORDINAL[c]).astype(float)
        elif c == "gender":
            X["gioi_tinh_nam"] = (meta["gender"] == "Nam").astype(int)
        else:
            v = to_num(df[c])
            if v is not None:
                X[c] = v.astype(float)
            else:
                d = pd.get_dummies(df[c].astype(str), prefix=c).astype(int)
                if d.shape[1] <= 15:
                    X = pd.concat([X, d], axis=1)
    X = X.fillna(X.median(numeric_only=True)).astype(float)
    return X.reset_index(drop=True), y.reset_index(drop=True), meta.reset_index(drop=True)

X, y, meta = prep_nij(nij_raw)
FEATS = list(X.columns)
print(f"NIJ sau xử lý: {X.shape[0]:,} người, {X.shape[1]} biến; tỉ lệ tái phạm 3 năm = {y.mean():.1%}")

# Nhóm biến PHÁT SINH TRONG THỜI GIAN GIÁM SÁT (tương tự 'Program Metrics' của bài báo)
IN_PROGRAM_PREFIX = ("violations_", "drugtests_", "drug_tests_", "avg_days_per_drugtest",
                     "percent_days_employed", "jobs_per_year", "employment_exempt",
                     "program_attendances", "program_unexcusedabsences",
                     "residence_changes", "delinquency_reports")
IN_PROG = [c for c in FEATS if c.startswith(IN_PROGRAM_PREFIX)]
ADMISSION = [c for c in FEATS if c not in IN_PROG]
print(f"Biến lúc tiếp nhận: {len(ADMISSION)} | Biến trong chương trình: {len(IN_PROG)}")

VN = {
    "age_at_release": "Tuổi lúc ra tù (nhóm)", "gioi_tinh_nam": "Giới tính nam",
    "gang_affiliated": "Liên quan băng nhóm", "supervision_risk_score_first": "Điểm rủi ro giám sát ban đầu",
    "supervision_level_first": "Mức giám sát ban đầu", "education_level": "Trình độ học vấn",
    "dependents": "Số người phụ thuộc", "prison_years": "Số năm ở tù",
    "prior_arrest_episodes_felony": "Số lần bị bắt (trọng tội)", "prior_arrest_episodes_misd": "Số lần bị bắt (tiểu hình)",
    "prior_arrest_episodes_violent": "Số lần bị bắt (bạo lực)", "prior_arrest_episodes_property": "Số lần bị bắt (tài sản)",
    "prior_arrest_episodes_drug": "Số lần bị bắt (ma tuý)", "prior_conviction_episodes_felony": "Số lần bị kết án (trọng tội)",
    "prior_conviction_episodes_misd": "Số lần bị kết án (tiểu hình)", "prior_conviction_episodes_drug": "Số lần bị kết án (ma tuý)",
    "prior_revocations_parole": "Số lần bị thu hồi tạm tha", "prior_revocations_probation": "Số lần bị thu hồi án treo",
    "condition_mh_sa": "Điều kiện: điều trị tâm thần/nghiện", "condition_cog_ed": "Điều kiện: giáo dục nhận thức",
    "condition_other": "Điều kiện khác", "violations_instruction": "Vi phạm chỉ dẫn",
    "violations_failtoreport": "Không trình diện", "violations_movewithoutpermission": "Chuyển chỗ ở không phép",
    "violations_electronicmonitoring": "Vi phạm giám sát điện tử", "delinquency_reports": "Số báo cáo vi phạm",
    "program_attendances": "Số buổi tham gia chương trình", "program_unexcusedabsences": "Số buổi vắng không phép",
    "residence_changes": "Số lần đổi chỗ ở", "avg_days_per_drugtest": "Số ngày TB giữa 2 lần xét nghiệm",
    "drugtests_thc_positive": "Tỉ lệ XN dương tính cần sa", "drugtests_cocaine_positive": "Tỉ lệ XN dương tính cocaine",
    "drugtests_meth_positive": "Tỉ lệ XN dương tính meth", "drugtests_other_positive": "Tỉ lệ XN dương tính chất khác",
    "percent_days_employed": "Tỉ lệ ngày có việc làm", "jobs_per_year": "Số việc làm / năm",
    "employment_exempt": "Được miễn yêu cầu việc làm",
}
def vn(f): return VN.get(f, f.replace("_", " "))
def fmt(v):
    v = float(v)
    return f"{int(v)}" if v.is_integer() else f"{v:.2f}"

def make_rf(n=300):
    return RandomForestClassifier(n_estimators=n, min_samples_leaf=3, max_depth=14,
                                  max_features="sqrt", n_jobs=-1, random_state=SEED)

def undersample(Xd, yd, seed=SEED):
    r = np.random.default_rng(seed)
    i1, i0 = yd[yd == 1].index.values, yd[yd == 0].index.values
    n = min(len(i1), len(i0))
    keep = np.concatenate([r.choice(i1, n, replace=False), r.choice(i0, n, replace=False)])
    return Xd.loc[keep], yd.loc[keep]

def metrics(yt, p, thr=0.5):
    yt = np.asarray(yt); yh = (np.asarray(p) >= thr).astype(int)
    return {"Accuracy": accuracy_score(yt, yh), "Recall": recall_score(yt, yh),
            "Precision": precision_score(yt, yh, zero_division=0), "F1": f1_score(yt, yh),
            "AUC": roc_auc_score(yt, p), "Brier": brier_score_loss(yt, p)}

idx = np.arange(len(X))
tr_idx, rest = train_test_split(idx, test_size=0.4, stratify=y, random_state=SEED)
iso_idx, rest2 = train_test_split(rest, test_size=0.75, stratify=y.iloc[rest], random_state=SEED)
conf_idx, te_idx = train_test_split(rest2, test_size=2/3, stratify=y.iloc[rest2], random_state=SEED)
X_train, y_train = X.loc[tr_idx], y.loc[tr_idx]
X_test, y_test, meta_test = X.loc[te_idx], y.loc[te_idx], meta.loc[te_idx]

Xu, yu = undersample(X_train, y_train)
rf = make_rf().fit(Xu, yu)
p_test_raw = rf.predict_proba(X_test)[:, 1]

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.inspection import permutation_importance
from scipy.stats import norm

def fit_eval(cols, n=300):
    Xa, ya = undersample(X_train[cols], y_train)
    m = make_rf(n).fit(Xa, ya)
    return metrics(y_test, m.predict_proba(X_test[cols])[:, 1])

# --- Table 5: ba mô hình ---
gbt = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=SEED).fit(Xu, yu)
mlp = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300,
                                                    early_stopping=True, random_state=SEED)).fit(Xu, yu)
res_t5 = pd.DataFrame({
    "Random Forest (RF)": metrics(y_test, p_test_raw),
    "Gradient Boosted Trees (GBT)": metrics(y_test, gbt.predict_proba(X_test)[:, 1]),
    "MLP-ANN": metrics(y_test, mlp.predict_proba(X_test)[:, 1]),
}).T.drop(columns="Brier")
PAPER_T5 = pd.DataFrame({
    "Random Forest (RF)": {"Accuracy": 0.857, "Recall": 0.869, "Precision": 0.656, "F1": 0.7476, "AUC": 0.928},
    "Gradient Boosted Trees (GBT)": {"Accuracy": 0.850, "Recall": 0.861, "Precision": 0.644, "F1": 0.7369, "AUC": 0.926},
    "MLP-ANN": {"Accuracy": 0.853, "Recall": 0.848, "Precision": 0.672, "F1": 0.7497, "AUC": 0.917},
}).T
print("Table 5 (demo, NIJ):\n", res_t5)

# --- Table 6: ablation theo nhóm nguồn dữ liệu (ánh xạ biến NIJ sang 5 nhóm của bài báo) ---
def category_of(c):
    if c in ("age_at_release", "gioi_tinh_nam"): return "Nhân khẩu học"
    if c in ("education_level", "dependents", "percent_days_employed", "jobs_per_year",
             "employment_exempt", "residence_changes"): return "Kinh tế – xã hội"
    if c.startswith(("prior_", "prison_")): return "Tiền án"
    if c.startswith(("drugtests_", "drug_tests_", "avg_days_per_drugtest", "gang_affiliated")): return "Hành vi"
    return "Chỉ số chương trình"
CATEGORY = {c: category_of(c) for c in FEATS}
CATS = ["Nhân khẩu học", "Kinh tế – xã hội", "Tiền án", "Chỉ số chương trình", "Hành vi"]
t6 = {"Tất cả biến": metrics(y_test, p_test_raw)}
for cat in CATS:
    t6[f"Bỏ nhóm {cat}"] = fit_eval([c for c in FEATS if CATEGORY[c] != cat], n=200)
res_t6 = pd.DataFrame(t6).T[["Accuracy", "Recall", "F1", "AUC"]]
PAPER_T6 = pd.DataFrame({
    "Tất cả biến": [0.857, 0.869, 0.748, 0.928], "Bỏ nhóm Nhân khẩu học": [0.828, 0.826, 0.710, 0.905],
    "Bỏ nhóm Kinh tế – xã hội": [0.827, 0.833, 0.711, 0.910], "Bỏ nhóm Tiền án": [0.820, 0.836, 0.703, 0.911],
    "Bỏ nhóm Chỉ số chương trình": [0.723, 0.743, 0.597, 0.794], "Bỏ nhóm Hành vi": [0.829, 0.813, 0.717, 0.913],
}, index=["Accuracy", "Recall", "F1", "AUC"]).T
print("Table 6 (demo, NIJ):\n", res_t6)

# --- Table 7–8: công bằng theo cách của bài báo, có p-value ---
def _ptest(k1, n1, k2, n2):
    if n1 == 0 or n2 == 0: return np.nan
    p1, p2 = k1 / n1, k2 / n2; pool = (k1 + k2) / (n1 + n2)
    se = math.sqrt(pool * (1 - pool) * (1 / n1 + 1 / n2)) if 0 < pool < 1 else 0
    return float(2 * (1 - norm.cdf(abs(p2 - p1) / se))) if se > 0 else 1.0

def paper_fairness_table(yt, p, groups, thr=0.5):
    yt = np.asarray(yt); yh = (np.asarray(p) >= thr).astype(int); groups = np.asarray(groups)
    fn, fp = (yt == 1) & (yh == 0), (yt == 0) & (yh == 1); N, NFN, NFP = len(yt), fn.sum(), fp.sum(); rows = []
    for g in sorted(pd.unique(groups)):
        m = groups == g
        rows.append({"Nhóm": g, "# test": int(m.sum()), "% test": m.mean(),
                     "# FN": int((m & fn).sum()), "% FN": (m & fn).sum() / max(NFN, 1),
                     "p-value FN": _ptest(m.sum(), N, (m & fn).sum(), NFN),
                     "# FP": int((m & fp).sum()), "% FP": (m & fp).sum() / max(NFP, 1),
                     "p-value FP": _ptest(m.sum(), N, (m & fp).sum(), NFP)})
    return pd.DataFrame(rows)
print(paper_fairness_table(y_test, p_test_raw, meta_test["race"]))

# --- Fig. 4: biến quan trọng toàn cục (làm nhiễu từng biến, đo mức giảm AUC) ---
_sub = X_test.sample(min(2000, len(X_test)), random_state=SEED)
_pi = permutation_importance(rf, _sub, y_test.loc[_sub.index], scoring="roc_auc",
                             n_repeats=3, random_state=SEED, n_jobs=-1)
PERM_IMP = pd.Series(_pi.importances_mean, index=FEATS).sort_values(ascending=False)

# --- Module khuyến nghị: mô phỏng đầu ra GenAI của bài báo bằng thư viện can thiệp ---
INTERVENTIONS = [
    (("percent_days_employed", "jobs_per_year", "employment_exempt"),
     "Kết nối dịch vụ việc làm / đào tạo nghề; sắp xếp lịch giám sát linh hoạt để giữ việc làm."),
    (("drugtests_", "avg_days_per_drugtest"),
     "Tăng tần suất xét nghiệm và chuyển sang điều trị nghiện chuyên sâu cho chất tương ứng."),
    (("program_unexcusedabsences", "program_attendances"),
     "Nhắc lịch, hỗ trợ đi lại và tư vấn tạo động lực để tăng mức tham gia chương trình."),
    (("violations_", "delinquency_reports"),
     "Họp rà soát tuân thủ; áp dụng chế tài tăng dần kết hợp khen thưởng khi tiến bộ."),
    (("residence_changes",), "Hỗ trợ nhà ở ổn định hoặc nhà ở chuyển tiếp."),
    (("prior_", "prison_"), "Giám sát chặt hơn giai đoạn đầu; tham gia trị liệu nhận thức – hành vi (CBT) giảm tái phạm."),
    (("age_at_release",), "Chương trình cố vấn (mentoring) và kỹ năng sống cho người trẻ."),
    (("gang_affiliated",), "Chương trình can thiệp giúp rời bỏ băng nhóm."),
    (("supervision_",), "Giữ mức giám sát tương xứng với mức rủi ro đã đánh giá."),
    (("condition_",), "Bảo đảm tiếp cận dịch vụ điều trị sức khoẻ tâm thần / nghiện theo điều kiện giám sát."),
    (("education_level",), "Hỗ trợ học bổ túc / lấy bằng tương đương phổ thông."),
    (("dependents",), "Hỗ trợ gia đình và chăm sóc trẻ để giảm áp lực."),
]
def intervention_for(f):
    for keys, text in INTERVENTIONS:
        if f.startswith(keys): return text
    return "Đưa yếu tố này vào thảo luận trong buổi họp nhóm để xác định hỗ trợ phù hợp."

def paper_recommendations(row, contribs, n=3, k=8):
    """Như prompt của bài báo: xét các yếu tố có |SHAP| cao nhất, kết hợp giá trị thực, đề xuất tối đa n can thiệp."""
    recs, used = [], set()
    for f, v in contribs.head(k).items():
        if v <= 0: continue
        text = intervention_for(f)
        if text in used: continue
        used.add(text)
        recs.append({"Yếu tố": vn(f), "Giá trị": fmt(row[f]), "SHAP": round(float(v), 3), "Can thiệp đề xuất": text})
        if len(recs) == n: break
    return pd.DataFrame(recs)

iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
iso.fit(rf.predict_proba(X.loc[iso_idx])[:, 1], y.loc[iso_idx])
def p_cal_of(Xd): return iso.predict(rf.predict_proba(Xd[FEATS])[:, 1])
p_test_cal = iso.predict(p_test_raw)

p_conf = p_cal_of(X.loc[conf_idx]); y_conf = y.loc[conf_idx].values
CONF_SCORES = np.where(y_conf == 1, 1 - p_conf, p_conf)      # điểm bất phù hợp (LAC)
def qhat(alpha):
    n = len(CONF_SCORES)
    return float(np.quantile(CONF_SCORES, min(1, math.ceil((n + 1) * (1 - alpha)) / n), method="higher"))
def conformal_set(p, q):
    s = set()
    if 1 - p <= q: s.add(1)
    if p <= q: s.add(0)
    if not s: s.add(int(p >= 0.5))
    return s
LABEL = {0: "Không tái phạm", 1: "Tái phạm"}

def conformal_report(alpha):
    q = qhat(alpha)
    sets = [conformal_set(p, q) for p in p_test_cal]
    cover = np.mean([yt in s for yt, s in zip(y_test.values, sets)])
    amb = np.mean([len(s) == 2 for s in sets])
    return q, cover, amb

q, cov, amb = conformal_report(ALPHA)
print(f"Conformal α={ALPHA}: độ phủ thực tế trên test = {cov:.1%} (mục tiêu ≥ {1-ALPHA:.0%}); "
      f"{amb:.1%} hồ sơ được gắn cờ 'không chắc chắn'")

res_base = pd.DataFrame({
    "Bài báo — RF (Oklahoma, Tbl 5)": {"Accuracy": 0.857, "Recall": 0.869, "Precision": 0.656, "F1": 0.7476, "AUC": 0.928, "Brier": np.nan},
    "Demo — RF + undersampling (tái hiện)": metrics(y_test, p_test_raw),
    "Demo — RF + hiệu chỉnh isotonic": metrics(y_test, p_test_cal),
}).T
res_base

explainer = shap.TreeExplainer(rf)
BASE_VALUE = float(np.atleast_1d(explainer.expected_value)[-1])
def shap_pos(Xd):
    sv = explainer.shap_values(Xd[FEATS], check_additivity=False)
    if isinstance(sv, list): sv = sv[1]
    sv = np.asarray(sv)
    if sv.ndim == 3: sv = sv[:, :, 1]
    return sv

sv_glob = shap_pos(X_test.iloc[:200])
GLOBAL_IMP = pd.Series(np.abs(sv_glob).mean(0), index=FEATS).sort_values(ascending=False)
GLOBAL_IMP.head(10).rename(index=vn)

ACTIONABLE = {   # +1: chỉ được tăng, -1: chỉ được giảm
    "percent_days_employed": +1, "jobs_per_year": +1, "program_attendances": +1,
    "program_unexcusedabsences": -1, "residence_changes": -1, "delinquency_reports": -1,
    "drugtests_thc_positive": -1, "drugtests_cocaine_positive": -1,
    "drugtests_meth_positive": -1, "drugtests_other_positive": -1,
    "violations_instruction": -1, "violations_failtoreport": -1,
    "violations_movewithoutpermission": -1, "violations_electronicmonitoring": -1,
    "condition_mh_sa": +1, "condition_cog_ed": +1,
}
ACTIONABLE = {k: v for k, v in ACTIONABLE.items() if k in FEATS}
CANDS = {f: np.unique(np.quantile(X_train[f], np.linspace(0, 1, 11))) for f in ACTIONABLE}
SCALE = {f: (X_train[f].std() or 1.0) for f in ACTIONABLE}

def counterfactual(row, target=0.5, max_changes=3):
    x = row.copy(); changes = []
    p = float(rf.predict_proba(x.to_frame().T[FEATS])[:, 1][0])
    for _ in range(max_changes):
        if p < target: break
        cand_rows, info = [], []
        for f, d in ACTIONABLE.items():
            if f in [c[0] for c in changes]: continue
            for v in CANDS[f]:
                if (v - x[f]) * d <= 0: continue
                x2 = x.copy(); x2[f] = v
                cand_rows.append(x2); info.append((f, v, abs(v - x[f]) / SCALE[f]))
        if not cand_rows: break
        ps = rf.predict_proba(pd.DataFrame(cand_rows)[FEATS])[:, 1]
        gains = [(p - p2) / (cost + 0.25) for p2, (_, _, cost) in zip(ps, info)]
        j = int(np.argmax(gains))
        if ps[j] >= p: break
        f, v, _ = info[j]
        changes.append((f, x[f], v, p, float(ps[j])))
        x[f] = v; p = float(ps[j])
    return changes, p

def faithfulness(claims, contribs, k=5):
    top = contribs.head(k); rows = []
    for f, s in claims:
        in_top = f in top.index
        ok = in_top and np.sign(top[f]) == s
        rows.append({"Biến được nhắc": f"{vn(f)} ({f})", "Có trong top-SHAP?": "Có" if in_top else "KHÔNG (bịa / ngoài top)",
                     "Chiều tác động": "Đúng" if ok else "Sai"})
    score = float(np.mean([r["Chiều tác động"] == "Đúng" for r in rows])) if rows else 0.0
    return score, pd.DataFrame(rows)

def template_narrative(row, p_cal, pset, contribs, changes, k=3):
    lvl = "CAO" if p_cal >= 0.6 else ("TRUNG BÌNH" if p_cal >= 0.4 else "THẤP")
    L = [f"**Rủi ro tái phạm trong 3 năm (đã hiệu chỉnh): {p_cal:.0%} — mức {lvl}.**"]
    if len(pset) == 2:
        L.append("⚠️ Mô hình **không đủ chắc chắn** (tập conformal gồm cả hai khả năng) → đề nghị hội đồng xem xét thủ công.")
    L.append("Các yếu tố ảnh hưởng mạnh nhất:")
    claims = []
    for f, v in contribs.head(k).items():
        L.append(f"- *{vn(f)}* = {fmt(row[f])} → **{'làm TĂNG' if v > 0 else 'làm GIẢM'}** rủi ro (SHAP {v:+.3f})")
        claims.append((f, int(np.sign(v))))
    if changes:
        L.append("Gợi ý can thiệp (phản thực tế của mô hình, *không phải quan hệ nhân quả*):")
        for f, a, b, p0, p1 in changes:
            L.append(f"- {'Tăng' if b > a else 'Giảm'} *{vn(f)}* từ {fmt(a)} → {fmt(b)}")
    return "\n".join(L), claims

USE_LLM = False   # đổi thành True nếu đã bật GPU (Runtime → Change runtime type → T4 GPU)
llm = None
if USE_LLM:
    import torch
    from transformers import pipeline
    llm = pipeline("text-generation", model="Qwen/Qwen2.5-1.5B-Instruct",
                   torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                   device_map="auto")
    print("Đã nạp LLM cục bộ.")

def llm_explain(row, contribs, k=5):
    facts = "\n".join(f"- {f} ({vn(f)}): giá trị = {fmt(row[f])}, SHAP = {v:+.3f}"
                      for f, v in contribs.head(k).items())
    prompt = ("Bạn là trợ lý cho cán bộ quản lý hồ sơ người đang chấp hành giám sát sau khi ra tù. "
              "Dưới đây là các yếu tố ảnh hưởng nhiều nhất tới rủi ro tái phạm của một người "
              "(SHAP dương = làm tăng rủi ro, SHAP âm = làm giảm rủi ro):\n" + facts +
              "\nHãy đề xuất tối đa 3 can thiệp. Chỉ trả lời bằng JSON dạng list: "
              '[{"feature": "<tên biến gốc, phần trước dấu ngoặc>", "huong": "tang" hoặc "giam", '
              '"can_thiep": "<đề xuất ngắn bằng tiếng Việt>"}]. Trường "huong" là chiều tác động của yếu tố đó lên rủi ro.')
    out = llm([{"role": "user", "content": prompt}], max_new_tokens=400, do_sample=False)
    text = out[0]["generated_text"][-1]["content"]
    try:
        items = json.loads(text[text.index("["): text.rindex("]") + 1])
    except Exception:
        items = []
    claims = [(str(it.get("feature", "")).strip(), 1 if str(it.get("huong", "")).lower().startswith("t") else -1)
              for it in items if isinstance(it, dict)]
    return text, items, claims

def group_metrics(yt, p, groups, thr=0.5, min_n=30):
    yt = np.asarray(yt); yh = (np.asarray(p) >= thr).astype(int); groups = np.asarray(groups); rows = []
    for g in sorted(pd.unique(groups)):
        m = groups == g
        if m.sum() < min_n: continue
        t, h = yt[m], yh[m]
        fp = ((t == 0) & (h == 1)).sum(); tn = ((t == 0) & (h == 0)).sum()
        fn = ((t == 1) & (h == 0)).sum(); tp = ((t == 1) & (h == 1)).sum()
        rows.append({"Nhóm": g, "N": int(m.sum()), "Tỉ lệ tái phạm thực": t.mean(),
                     "Tỉ lệ bị gắn rủi ro cao": h.mean(),
                     "FPR": fp / max(fp + tn, 1), "FNR": fn / max(fn + tp, 1)})
    return pd.DataFrame(rows)

def paper_style(yt, p, groups, thr=0.5):
    """Cách bài báo làm (Tbl 7-8): so tỉ trọng nhóm trong số FP/FN với tỉ trọng nhóm trong test."""
    yt = np.asarray(yt); yh = (np.asarray(p) >= thr).astype(int); groups = np.asarray(groups)
    fp, fn = (yt == 0) & (yh == 1), (yt == 1) & (yh == 0); rows = []
    for g in sorted(pd.unique(groups)):
        m = groups == g
        rows.append({"Nhóm": g, "% trong test": m.mean(), "% trong FP": (m & fp).sum() / max(fp.sum(), 1),
                     "% trong FN": (m & fn).sum() / max(fn.sum(), 1)})
    return pd.DataFrame(rows)

def eod(gm):
    return float(max(gm["FPR"].max() - gm["FPR"].min(), gm["FNR"].max() - gm["FNR"].min())) if len(gm) > 1 else np.nan

for a in ["race", "gender", "age_group"]:
    print(f"Equalized-odds difference theo {a}: {eod(group_metrics(y_test, p_test_raw, meta_test[a])):.3f}")

res_g5 = pd.DataFrame({
    "Tất cả biến (như bài báo)": metrics(y_test, p_test_raw),
    "Chỉ biến lúc tiếp nhận (dùng được khi ra quyết định đầu vào)": fit_eval(ADMISSION),
    "Chỉ biến trong chương trình": fit_eval(IN_PROG),
}).T.drop(columns="Brier")
print(res_g5)

def cv_scores(Xd, yd, k=5):
    skf = StratifiedKFold(k, shuffle=True, random_state=SEED); out = []
    for tr, te in skf.split(Xd, yd):
        Xa, ya = undersample(Xd.iloc[tr], yd.iloc[tr])
        m = make_rf(200).fit(Xa, ya)
        out.append(metrics(yd.iloc[te], m.predict_proba(Xd.iloc[te])[:, 1]))
    d = pd.DataFrame(out).drop(columns="Brier")
    return d.mean(), d.std()

cv_mean, cv_std = cv_scores(X, y)
res_cv = pd.DataFrame({"Trung bình": cv_mean, "Độ lệch chuẩn": cv_std,
                       "KTC 95% (±)": 1.96 * cv_std / np.sqrt(5)}).T
print("\n5-fold CV trên NIJ:\n", res_cv)

def prep_compas(raw):
    d = raw[(raw.days_b_screening_arrest <= 30) & (raw.days_b_screening_arrest >= -30) &
            (raw.is_recid != -1) & (raw.c_charge_degree != "O") & (raw.score_text != "N/A")]
    Xc = pd.DataFrame({"age": d.age, "priors_count": d.priors_count, "juv_fel_count": d.juv_fel_count,
                       "juv_misd_count": d.juv_misd_count, "juv_other_count": d.juv_other_count,
                       "charge_felony": (d.c_charge_degree == "F").astype(int),
                       "gioi_tinh_nam": (d.sex == "Male").astype(int)}).astype(float)
    mc = pd.DataFrame({"race": d.race.astype(str), "gender": d.sex.astype(str), "age_group": d.age_cat.astype(str)})
    return Xc.reset_index(drop=True), d.two_year_recid.astype(int).reset_index(drop=True), mc.reset_index(drop=True)

def run_pipeline(Xd, yd, md, name):
    Xtr, Xte, ytr, yte, mtr, mte = train_test_split(Xd, yd, md, test_size=0.25, stratify=yd, random_state=SEED)
    Xa, ya = undersample(Xtr, ytr)
    p = make_rf().fit(Xa, ya).predict_proba(Xte)[:, 1]
    r = {"Bộ dữ liệu": name, "N": len(Xd), "Tỉ lệ tái phạm": yd.mean(), **metrics(yte, p)}
    r["EOD theo chủng tộc"] = eod(group_metrics(yte, p, mte["race"]))
    return r

rows_g6 = [{"Bộ dữ liệu": "Bài báo — Oklahoma drug courts (Tbl 5)", "N": 35711, "Tỉ lệ tái phạm": 0.24,
            "Accuracy": 0.857, "Recall": 0.869, "Precision": 0.656, "F1": 0.7476, "AUC": 0.928}]
rows_g6.append(run_pipeline(X, y, meta, "NIJ — Georgia, tái phạm 3 năm"))
if compas_raw is not None:
    Xc, yc, mc = prep_compas(compas_raw)
    rows_g6.append(run_pipeline(Xc, yc, mc, "COMPAS — Broward FL, tái phạm 2 năm"))
res_g6 = pd.DataFrame(rows_g6).drop(columns="Brier", errors="ignore")
res_g6

