"""Preprocess the two public UCI datasets into fully categorical integer tables.

BRFSS 2015 (UCI #891, CDC Diabetes Health Indicators): 22 variables, 72 categories.
Diabetes 130-US hospitals (UCI #296): first encounter per patient, encounters ending
in death or hospice removed, 27 variables, 111 categories.

Output: data/<name>.npz with X (n x d int8 codes), K (categories per variable),
names (variable names), levels (category labels), outcome (index of downstream outcome).
"""
import sys, json
import os, zipfile
import numpy as np
import pandas as pd

DATA = sys.argv[1] if len(sys.argv) > 1 else "data"


def encode(df, order=None):
    names, K, levels, cols = [], [], [], []
    for c in df.columns:
        s = df[c]
        if order and c in order:
            cats = order[c]
        else:
            cats = sorted(s.unique().tolist(), key=lambda v: (str(type(v)), v))
        mp = {v: i for i, v in enumerate(cats)}
        assert set(s.unique()) <= set(cats), (c, set(s.unique()) - set(cats))
        cols.append(s.map(mp).to_numpy().astype(np.int8))
        names.append(c); K.append(len(cats)); levels.append([str(v) for v in cats])
    return np.stack(cols, 1), np.array(K), names, levels


def brfss():
    src = f"{DATA}/raw/uci891_cdc_diabetes_health_indicators.csv"
    d = pd.read_csv(src if os.path.exists(src) else f"{DATA}/brfss.csv").drop(columns=["ID"])
    bmi = d["BMI"]
    d["BMI"] = np.select([bmi < 18.5, bmi < 25, bmi < 30], ["1_under", "2_normal", "3_over"], "4_obese")
    for c in ["MentHlth", "PhysHlth"]:
        v = d[c]
        d[c] = np.select([v == 0, v <= 13], ["0", "1-13"], "14-30")
    order = {"BMI": ["1_under", "2_normal", "3_over", "4_obese"],
             "MentHlth": ["0", "1-13", "14-30"], "PhysHlth": ["0", "1-13", "14-30"]}
    X, K, names, levels = encode(d, order)
    return X, K, names, levels, names.index("Diabetes_binary")


def icd_group(code):
    if code == "?" or pd.isna(code):
        return "Other"
    if code.startswith("V") or code.startswith("E"):
        return "Other"
    if code.startswith("250"):
        return "Diabetes"
    x = float(code)
    if 390 <= x <= 459 or int(x) == 785: return "Circulatory"
    if 460 <= x <= 519 or int(x) == 786: return "Respiratory"
    if 520 <= x <= 579 or int(x) == 787: return "Digestive"
    if 800 <= x <= 999: return "Injury"
    if 710 <= x <= 739: return "Musculoskeletal"
    if 580 <= x <= 629 or int(x) == 788: return "Genitourinary"
    if 140 <= x <= 239: return "Neoplasms"
    return "Other"


def cut(v, edges, labels):
    return pd.cut(v, bins=edges, labels=labels, right=True).astype(str)


def diabetes130():
    src = f"{DATA}/raw/uci296_diabetes_130us.zip"
    if os.path.exists(src):
        with zipfile.ZipFile(src) as z, z.open("diabetic_data.csv") as fh:
            d = pd.read_csv(fh, na_values=[], keep_default_na=False)
    else:
        d = pd.read_csv(f"{DATA}/diabetic_data.csv", na_values=[], keep_default_na=False)
    d = d.sort_values("encounter_id").drop_duplicates("patient_nbr", keep="first")
    d = d[~d["discharge_disposition_id"].isin([11, 13, 14, 19, 20, 21])]   # death / hospice
    d = d[d["gender"] != "Unknown/Invalid"]
    o = pd.DataFrame(index=d.index)
    o["race"] = d["race"].replace({"?": "Unknown"})
    o["gender"] = d["gender"]
    o["age"] = d["age"]
    o["admission_type"] = d["admission_type_id"].map({1: "Emergency", 2: "Urgent", 3: "Elective"}).fillna("Other")
    o["admission_source"] = d["admission_source_id"].map({1: "Referral", 2: "Referral", 3: "Referral", 7: "ER"}).fillna("Other")
    o["time_in_hospital"] = cut(d["time_in_hospital"], [0, 2, 4, 7, 14], ["1-2", "3-4", "5-7", "8-14"])
    o["primary_diag"] = d["diag_1"].map(icd_group)
    o["num_lab_procedures"] = cut(d["num_lab_procedures"], [0, 30, 45, 60, 200], ["1-30", "31-45", "46-60", ">60"])
    o["num_procedures"] = cut(d["num_procedures"], [-1, 0, 1, 2, 10], ["0", "1", "2", "3+"])
    o["num_medications"] = cut(d["num_medications"], [0, 10, 15, 20, 30, 200], ["1-10", "11-15", "16-20", "21-30", ">30"])
    for c in ["number_outpatient", "number_emergency", "number_inpatient"]:
        o[c] = cut(d[c], [-1, 0, 1, 200], ["0", "1", "2+"])
    o["number_diagnoses"] = cut(d["number_diagnoses"], [0, 4, 6, 8, 20], ["1-4", "5-6", "7-8", "9+"])
    o["max_glu_serum"] = d["max_glu_serum"].replace({"": "None"})
    o["A1Cresult"] = d["A1Cresult"].replace({"": "None"})
    for c in ["metformin", "glipizide", "glyburide", "pioglitazone", "rosiglitazone", "glimepiride", "insulin"]:
        o[c] = d[c]
    o["change"] = d["change"]
    o["diabetesMed"] = d["diabetesMed"]
    disp = d["discharge_disposition_id"]
    o["discharge"] = np.select([disp == 1, disp.isin([6, 8]), disp.isin([2, 3, 4, 5, 9, 10, 15, 22, 23, 24, 27, 28, 29, 30])],
                               ["Home", "HomeHealth", "Transfer"], "Other")
    o["readmitted"] = d["readmitted"]
    med = ["No", "Steady", "Up", "Down"]
    order = {"age": sorted(o["age"].unique()),
             "time_in_hospital": ["1-2", "3-4", "5-7", "8-14"],
             "num_lab_procedures": ["1-30", "31-45", "46-60", ">60"],
             "num_procedures": ["0", "1", "2", "3+"],
             "num_medications": ["1-10", "11-15", "16-20", "21-30", ">30"],
             "number_outpatient": ["0", "1", "2+"], "number_emergency": ["0", "1", "2+"],
             "number_inpatient": ["0", "1", "2+"], "number_diagnoses": ["1-4", "5-6", "7-8", "9+"],
             "max_glu_serum": ["None", "Norm", ">200", ">300"], "A1Cresult": ["None", "Norm", ">7", ">8"],
             "readmitted": ["NO", ">30", "<30"], **{c: med for c in ["metformin", "glipizide", "glyburide",
                                                                      "pioglitazone", "rosiglitazone", "glimepiride", "insulin"]}}
    X, K, names, levels = encode(o, order)
    return X, K, names, levels, names.index("readmitted")


def summary(name, X, K, names):
    print(f"{name}: n={X.shape[0]}, d={X.shape[1]}, categories={K.sum()}, binary={int((K==2).sum())}, "
          f">=4 cats={int((K>=4).sum())}, complete={not np.any(X < 0)}")


if __name__ == "__main__":
    for name, fn in [("brfss", brfss), ("diabetes", diabetes130)]:
        X, K, names, levels, outcome = fn()
        summary(name, X, K, names)
        np.savez_compressed(f"{DATA}/{name}.npz", X=X, K=K, outcome=outcome)
        with open(f"{DATA}/{name}_meta.json", "w") as f:
            json.dump({"names": names, "levels": levels, "K": K.tolist(), "outcome": names[outcome]}, f, indent=1)
