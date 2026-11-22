"""Loads the IBM AMLSim CSVs (Kaggle) into the uniform schema used everywhere:
transactions -> src, dst, amount, ts, is_fraud, alert_id.

The loader validates its input and fails loudly with a readable message, because a malformed
file would otherwise produce silently wrong features (e.g. an unknown account id maps to -1
and quietly lands on the last account)."""
import numpy as np
import pandas as pd
from .config import RAW_DIR


class DataValidationError(ValueError):
    """Raised when an AMLSim CSV is missing, has the wrong columns or inconsistent contents."""


REQUIRED = {
    "transactions.csv": ["TX_ID", "SENDER_ACCOUNT_ID", "RECEIVER_ACCOUNT_ID", "TX_AMOUNT", "TIMESTAMP",
                         "IS_FRAUD", "ALERT_ID"],
    "accounts.csv": ["ACCOUNT_ID", "INIT_BALANCE", "IS_FRAUD"],
    "alerts.csv": ["ALERT_ID", "ALERT_TYPE", "SENDER_ACCOUNT_ID", "RECEIVER_ACCOUNT_ID"],
}


def _read(name, raw_dir):
    path = raw_dir / name
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download the IBM AMLSim dataset from Kaggle and place "
            f"transactions.csv, accounts.csv and alerts.csv in {raw_dir} (see data/raw/README.txt).")
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED[name] if c not in df.columns]
    if missing:
        raise DataValidationError(f"{name}: missing column(s) {missing}. Found: {list(df.columns)}")
    return df


def _to_bool(s, where):
    """Robust bool conversion (astype(bool) would turn the string 'False' into True)."""
    if s.dtype == bool:
        return s
    m = {"true": True, "false": False, "1": True, "0": False}
    out = s.astype(str).str.strip().str.lower().map(m)
    if out.isna().any():
        raise DataValidationError(f"{where}: IS_FRAUD must be True/False or 1/0.")
    return out.astype(bool)


def validate(tx, acc):
    """Consistency checks on the renamed frames. Raises DataValidationError."""
    if acc.account_id.isna().any():
        raise DataValidationError("accounts.csv: ACCOUNT_ID contains empty values.")
    if acc.account_id.duplicated().any():
        raise DataValidationError(f"accounts.csv: {int(acc.account_id.duplicated().sum())} duplicate ACCOUNT_ID value(s).")
    for c in ("src", "dst", "amount", "ts"):
        if tx[c].isna().any():
            raise DataValidationError(f"transactions.csv: {int(tx[c].isna().sum())} empty value(s) in {c!r}.")
    for c in ("amount", "ts"):
        if not pd.api.types.is_numeric_dtype(tx[c]):
            raise DataValidationError(f"transactions.csv: column {c!r} must be numeric.")
    if (tx.amount < 0).any():
        raise DataValidationError("transactions.csv: negative TX_AMOUNT found.")
    known = pd.Index(acc.account_id)
    unknown = ~(tx.src.isin(known) & tx.dst.isin(known))
    if unknown.any():
        raise DataValidationError(f"transactions.csv: {int(unknown.sum())} transfer(s) reference an account "
                                  f"that is not in accounts.csv.")


def load_amlsim(raw_dir=None):
    raw_dir = RAW_DIR if raw_dir is None else raw_dir
    tx = _read("transactions.csv", raw_dir)
    acc = _read("accounts.csv", raw_dir)
    alerts = _read("alerts.csv", raw_dir)
    tx = tx.rename(columns={"TX_ID": "tx_id", "SENDER_ACCOUNT_ID": "src", "RECEIVER_ACCOUNT_ID": "dst",
                            "TX_AMOUNT": "amount", "TIMESTAMP": "ts", "IS_FRAUD": "is_fraud",
                            "ALERT_ID": "alert_id"})
    tx["is_fraud"] = _to_bool(tx["is_fraud"], "transactions.csv")
    acc = acc.rename(columns={"ACCOUNT_ID": "account_id", "INIT_BALANCE": "init_balance",
                              "IS_FRAUD": "is_fraud"})
    acc["is_fraud"] = _to_bool(acc["is_fraud"], "accounts.csv")
    alerts = alerts.rename(columns={"ALERT_ID": "alert_id", "ALERT_TYPE": "alert_type",
                                    "SENDER_ACCOUNT_ID": "src", "RECEIVER_ACCOUNT_ID": "dst"})
    validate(tx, acc)
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