"""Loads the IBM AMLSim CSVs (Kaggle) into the uniform schema used everywhere:
transactions -> src, dst, amount, ts, is_fraud, alert_id."""
import numpy as np
import pandas as pd
from .config import RAW_DIR


def load_amlsim():
    tx = pd.read_csv(RAW_DIR / "transactions.csv")
    acc = pd.read_csv(RAW_DIR / "accounts.csv")
    alerts = pd.read_csv(RAW_DIR / "alerts.csv")
    tx = tx.rename(columns={"TX_ID": "tx_id", "SENDER_ACCOUNT_ID": "src", "RECEIVER_ACCOUNT_ID": "dst",
                            "TX_AMOUNT": "amount", "TIMESTAMP": "ts", "IS_FRAUD": "is_fraud",
                            "ALERT_ID": "alert_id"})
    tx["is_fraud"] = tx["is_fraud"].astype(bool)
    acc = acc.rename(columns={"ACCOUNT_ID": "account_id", "INIT_BALANCE": "init_balance",
                              "IS_FRAUD": "is_fraud"})
    acc["is_fraud"] = acc["is_fraud"].astype(bool)
    alerts = alerts.rename(columns={"ALERT_ID": "alert_id", "ALERT_TYPE": "alert_type",
                                    "SENDER_ACCOUNT_ID": "src", "RECEIVER_ACCOUNT_ID": "dst"})
    return tx, acc, alerts


def amlsim_groups_and_typology(tx, acc, alerts):
    """Ring-aware grouping so CV never splits one laundering ring across train and test.
    Fraud accounts are grouped by connected components of the *fraud* edges; every other
    account is its own group. Typology comes from alerts.csv (used for evaluation only)."""
    n = len(acc)
    pos = pd.Index(acc.account_id).get_indexer
    parent = np.arange(n)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    fr = tx[tx.is_fraud]
    for s, d in zip(pos(fr.src), pos(fr.dst)):
        a, b = find(s), find(d)
        if a != b:
            parent[a] = b
    groups = np.array([find(i) for i in range(n)])
    typ = pd.Series("none", index=acc.account_id)
    for atype, sub in alerts.groupby("alert_type"):
        ids = pd.unique(np.concatenate([sub.src.values, sub.dst.values]))
        typ.loc[typ.index.intersection(ids)] = atype
    typ[(acc.set_index("account_id").is_fraud) & (typ == "none")] = "unknown"
    return groups, typ.values
