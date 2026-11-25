"""Detectors compared in the project. Every model exposes the same tiny API:
    fit(entries)           entries = [dict(df=features, ops=graph operators, y=labels, mask=labelled rows)]
    predict(df, ops)  ->   risk score in [0,1] for every account
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from sklearn.neural_network import MLPClassifier #SIGN-style GNN ke neural network part mein use hota hai
from sklearn.preprocessing import StandardScaler
from .features import feature_cols #Project ki feature selection function -> the three group: amt,str,tmp (amount,structure,time)


#multiple datasets ko combine karke ek training dataset banata hai.
def _stack(entries, cols=None):
    X, y = [], []
    for e in entries:
        m = e.get("mask") #Mask = Select karna ki dataset ki kaunsi rows training mein use hongi, Agar mask hai → sirf selected rows use karo.
        df = e["df"] if m is None else e["df"][m] #Agar mask nahi hai → poora DataFrame use karo.
        yy = np.asarray(e["y"])
        X.append(df if cols is None else df[cols]) #Selected features ko X list mein add karta hai.
        y.append(yy if m is None else yy[m]) #Corresponding labels ko y list mein add karta hai.
    return pd.concat(X), np.concatenate(y) #Saare feature DataFrames ko combine karta hai.


#Tabular ML models ko handle karne wali class (Random Forest, XGBoost)
class TabModel:
    """Gradient boosting / random forest on one feature group (amount | structure | all)."""
    def __init__(self, group="all", kind="xgb", seed=0):
        self.group, self.kind, self.seed = group, kind, seed

    #means yeh group use karke yeh (kind) model banao

    #Training starts
    def fit(self, entries):
        X, y = _stack(entries) #Data combine karta ha
        self.cols = feature_cols(X, self.group) #Required features select karta hai
        pos = max(1, int(y.sum())) #1 = Fraud / laundering, 0 = Legitimate, ensures pos zero na ho, pos = Dataset mein positive (fraud) examples ki count.
        if self.kind == "xgb":
            self.m = xgb.XGBClassifier(n_estimators=300,  #300 boosting trees
                                       max_depth=4, 
                                       learning_rate=0.05,  #Har boosting step ka contribution control karta hai
                                       subsample=0.85,
                                       colsample_bytree=0.85, 
                                       reg_lambda=2,  #Regularization control karta hai, taaki model unnecessarily complex na ho aur overfitting kam ho.
                                       scale_pos_weight=(len(y) - pos) / pos,
                                       tree_method="hist", 
                                       n_jobs=1, 
                                       random_state=self.seed, 
                                       eval_metric="logloss")
        else:
            self.m = RandomForestClassifier(n_estimators=300, 
                                            min_samples_leaf=2, 
                                            class_weight="balanced_subsample",
                                            n_jobs=1, 
                                            random_state=self.seed)
        self.m.fit(X[self.cols].values, y)
        return self

    #basically, Dono models same TabModel class ke through manage ho rahe hain, bas algorithm different hai.

    def predict(self, df, ops=None):
        return self.m.predict_proba(df[self.cols].values)[:, 1] #har account ke liye dono classes ki probability deta hai

    #Ye function explain karta hai ki model ne kisi account ko suspicious kyun classify kiya.
    def contribs(self, df):
        """Exact TreeSHAP values (xgboost built-in). Returns DataFrame [features..., bias]."""
        dm = xgb.DMatrix(df[self.cols].values, feature_names=self.cols)
        c = self.m.get_booster().predict(dm, pred_contribs=True)
        return pd.DataFrame(c, index=df.index, columns=self.cols + ["bias"]) #har feature ka contribution batati hai ki usne laundering risk badhaya ya kam kiya.

    #model khud complex patterns learn nahi karta, balki pehle se defined rules check karta hai, rules like:- 
    # Bahut fast transactions
    # Multiple accounts se ek account mein money aana
    # Money ka quickly forward hona
    # Unusually large transfers

class RuleModel:
    """Transparent bank-style rules (the baseline): short fast cycle, convergence burst, dispersal burst,
    quick forwarding, large transfer. Cut-offs are 95th percentiles of the training data."""
    def fit(self, entries):
        X, _ = _stack(entries)
        q = lambda c, p=0.95: float(X[c].quantile(p))
        # kisi feature ki 95th percentile value calculate karta hai.
        # training data mein us feature ki ek high-value boundary nikal raha hai, jise baad mein suspicious activity identify karne ke liye use karenge
        self.t = {"in": max(3.0, q("tmp_joined_in_burst")), "out": max(3.0, q("tmp_joined_out_burst")),
                  "fwd": q("tmp_fwd_fast_frac"), "amt": q("amt_out_max")}
        return self

    # in → unusually high incoming transactions.
    # out → unusually high outgoing transactions.
    # fwd → kitni quickly money forward ho raha hai.
    # amt → maximum outgoing amount.

    # Ye function naye accounts ka data lekar check karega ki woh suspicious hain ya nahi
    def predict(self, df, ops=None):
        s = (2.0 * (df["str_wcycle_6"] > 0) + 1.5 * (df["tmp_joined_in_burst"] >= self.t["in"]) +
             1.5 * (df["tmp_joined_out_burst"] >= self.t["out"]) + 1.0 * (df["tmp_fwd_fast_frac"] >= self.t["fwd"]) +
             2.0 * (df["amt_out_max"] >= self.t["amt"])) #suspicion score
        return (s / 8.0).to_numpy() #total suspicion score ko 8 se divide karke 0–1 ke beech convert karta hai.

    # str_wcycle_6 → account kisi cycle ka part hai?
    # tmp_joined_in_burst → bahut saare incoming transactions hain?
    # tmp_joined_out_burst → bahut saare outgoing transactions hain?
    # tmp_fwd_fast_frac → money quickly forward ho raha hai?
    # amt_out_max → outgoing amount unusually high hai?

#Ye sirf individual account ke features nahi dekhta, balki uske connected accounts aur unke transaction patterns ko bhi consider karta hai.
class SignGNN:
    """Graph neural network in the SIGN family (Frasca et al. 2020): message passing is pre-computed
    (mean over out-, in- and 2-hop undirected neighbourhoods of node features) and a neural network
    (MLP) is trained on the concatenated multi-hop representation. Inductive: works on unseen graphs."""

    def __init__(self, group="structure", seed=0):
        self.group, self.seed = group, seed

    #Ye function account ke features ke saath uske neighbouring accounts ki information bhi combine karta hai.
    def _z(self, df, ops):
        X = df[self.cols].to_numpy(float)
        X = np.sign(X) * np.log1p(np.abs(X))
        return np.hstack([X, ops["out"] @ X, ops["in"] @ X, ops["und"] @ (ops["und"] @ X)])

    def fit(self, entries): 
        #model decide kar raha hai ki training ke liye kaunse features use karne hain.
        # Model har dataset se training information collect kar raha hai. Training ke liye required data collect ho raha hai.
        # Ab model ke paas combined training data ready hai. 
        # model ka data normalize/standardize ho raha hai. 
        # Model normal aur suspicious accounts ko alag identify kar raha hai. 
        # then calculate karna ki suspicious accounts ko training mein kitni baar repeat karna hai as Dataset mein normal accounts zyada aur suspicious accounts kam ho sakte hain. Is imbalance ko handle karna.

        self.cols = feature_cols(entries[0]["df"], self.group)
        Z, Y = [], []
        for e in entries:
            z = self._z(e["df"], e["ops"]); y = np.asarray(e["y"])
            m = e.get("mask")
            Z.append(z if m is None else z[m]); Y.append(y if m is None else y[m])
        Z, Y = np.vstack(Z), np.concatenate(Y)
        self.sc = StandardScaler().fit(Z)
        Z = self.sc.transform(Z)
        pos, neg = np.flatnonzero(Y == 1), np.flatnonzero(Y == 0)
        rep = max(1, int(round(len(neg) / max(1, len(pos)) / 3)))
        idx = np.concatenate([neg] + [pos] * rep)
        self.m = MLPClassifier(hidden_layer_sizes=(96, 48), 
                               alpha=1e-3, early_stopping=True, n_iter_no_change=15,
                               max_iter=300, 
                               random_state=self.seed).fit(Z[idx], Y[idx])
        return self

    # hidden_layer_sizes=(96, 48) → 2 hidden layers.
    # early_stopping=True → performance improve na hone par training rok sakta hai
    # max_iter=300 → maximum 300 iterations.
    # .fit() → model ko training data deta hai.

    def predict(self, df, ops):
        return self.m.predict_proba(self.sc.transform(self._z(df, ops)))[:, 1] 
    #SIGN-GNN bhi har account ka risk score deta hai.

#Ye function model ke predictions ke liye best cut-off value find karta hai.
#decide karta hai ki kis risk score ke upar account ko suspicious classify karna chahiye
# y → actual labels.
# p → model ke predicted risk scores.
# different threshold values generate karta hai jinhe model test karega.

def best_threshold(y, p):
    cand = np.unique(np.concatenate([np.linspace(0.02, 0.98, 49), np.quantile(p, np.linspace(0.5, 0.995, 60))]))
    best, bt = -1, 0.5
    for t in cand:
        pr = p >= t 
        #Score t ya usse zyada → suspicious (True), Score t se kam → normal (False)
        tp = (pr & (y == 1)).sum()
        f1 = 2 * tp / max(1, pr.sum() + (y == 1).sum())
        if f1 > best:
            best, bt = f1, float(t)
    return bt


# y → actual labels.
# p → model ke risk scores.

def metrics(y, p, thr):
    y = np.asarray(y); pr = p >= thr
    tp = int((pr & (y == 1)).sum()); fp = int((pr & (y == 0)).sum()); fn = int((~pr & (y == 1)).sum())
    prec = tp / max(1, tp + fp); rec = tp / max(1, tp + fn)
    return {"ap": float(average_precision_score(y, p)) if y.sum() else float("nan"),
            "precision": prec, "recall": rec, "f1": 2 * prec * rec / max(1e-12, prec + rec),
            "tp": tp, "fp": fp, "fn": fn}
