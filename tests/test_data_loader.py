import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd, pytest
from src.data_loader import load_amlsim, DataValidationError

TX = {"TX_ID": [1, 2], "SENDER_ACCOUNT_ID": [10, 11], "RECEIVER_ACCOUNT_ID": [11, 12],
      "TX_AMOUNT": [100.0, 50.0], "TIMESTAMP": [1, 2], "IS_FRAUD": [False, True], "ALERT_ID": [-1, 5]}
ACC = {"ACCOUNT_ID": [10, 11, 12], "INIT_BALANCE": [1000.0, 500.0, 200.0], "IS_FRAUD": [False, True, True]}
ALERTS = {"ALERT_ID": [5], "ALERT_TYPE": ["cycle"], "SENDER_ACCOUNT_ID": [11], "RECEIVER_ACCOUNT_ID": [12]}


def write(d, tx=TX, acc=ACC, alerts=ALERTS):
    pd.DataFrame(tx).to_csv(d / "transactions.csv", index=False)
    pd.DataFrame(acc).to_csv(d / "accounts.csv", index=False)
    pd.DataFrame(alerts).to_csv(d / "alerts.csv", index=False)
    return d


def test_loads_valid_data(tmp_path):
    tx, acc, alerts = load_amlsim(write(tmp_path))
    assert list(tx.columns[:5]) == ["tx_id", "src", "dst", "amount", "ts"]
    assert tx.is_fraud.dtype == bool and acc.is_fraud.dtype == bool
    assert len(tx) == 2 and len(acc) == 3 and "alert_type" in alerts.columns


def test_missing_file_gives_helpful_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="Kaggle"):
        load_amlsim(tmp_path)


def test_missing_column_is_reported(tmp_path):
    bad = {k: v for k, v in TX.items() if k != "TX_AMOUNT"}
    with pytest.raises(DataValidationError, match="TX_AMOUNT"):
        load_amlsim(write(tmp_path, tx=bad))


def test_unknown_account_is_rejected(tmp_path):
    bad = dict(TX, RECEIVER_ACCOUNT_ID=[11, 999])
    with pytest.raises(DataValidationError, match="not in accounts.csv"):
        load_amlsim(write(tmp_path, tx=bad))


def test_duplicate_accounts_rejected(tmp_path):
    bad = dict(ACC, ACCOUNT_ID=[10, 10, 12])
    with pytest.raises(DataValidationError, match="duplicate"):
        load_amlsim(write(tmp_path, acc=bad))


def test_negative_amount_rejected(tmp_path):
    with pytest.raises(DataValidationError, match="negative"):
        load_amlsim(write(tmp_path, tx=dict(TX, TX_AMOUNT=[-5.0, 50.0])))


def test_string_false_is_not_treated_as_true(tmp_path):
    tx, _, _ = load_amlsim(write(tmp_path, tx=dict(TX, IS_FRAUD=["False", "True"])))
    assert tx.is_fraud.tolist() == [False, True]