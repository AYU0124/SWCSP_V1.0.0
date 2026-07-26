import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


DEFAULT_N_STRATA_BINS = 5


def read_data(file):
    data = pd.read_excel(file, 0)
    data = data.iloc[:, :]
    return data


def _build_regression_strata(y, n_bins=DEFAULT_N_STRATA_BINS):
    """Create target bins for stratified regression sampling."""
    y_series = pd.Series(np.asarray(y).ravel()).reset_index(drop=True)

    if y_series.nunique() < 2:
        raise ValueError("The target must contain at least two unique values.")

    max_bins = int(min(max(2, n_bins), y_series.nunique()))

    for current_bins in range(max_bins, 1, -1):
        try:
            strata = pd.qcut(
                y_series,
                q=current_bins,
                labels=False,
                duplicates="drop",
            )
            strata = pd.Series(strata, dtype="Int64")
            if strata.nunique(dropna=True) >= 2:
                return strata.astype(int), int(strata.nunique())
        except Exception:
            continue

    strata = pd.cut(y_series, bins=2, labels=False, include_lowest=True)
    strata = pd.Series(strata, dtype="Int64")
    return strata.astype(int), int(strata.nunique())


def _summarize_split(y_all, y_train, y_test, train_bins, test_bins):
    rows = []
    all_series = pd.Series(np.asarray(y_all).ravel())
    train_series = pd.Series(np.asarray(y_train).ravel())
    test_series = pd.Series(np.asarray(y_test).ravel())

    for bin_id in sorted(pd.unique(pd.concat([train_bins, test_bins], axis=0))):
        bin_id = int(bin_id)
        train_mask = train_bins == bin_id
        test_mask = test_bins == bin_id

        rows.append({
            "Stratum": bin_id + 1,
            "Train_Count": int(train_mask.sum()),
            "Test_Count": int(test_mask.sum()),
            "Total_Count": int(train_mask.sum() + test_mask.sum()),
            "Train_Ratio": float(train_mask.mean()),
            "Test_Ratio": float(test_mask.mean()),
            "Train_Target_Min": float(train_series.loc[train_mask].min()) if train_mask.any() else np.nan,
            "Train_Target_Max": float(train_series.loc[train_mask].max()) if train_mask.any() else np.nan,
            "Test_Target_Min": float(test_series.loc[test_mask].min()) if test_mask.any() else np.nan,
            "Test_Target_Max": float(test_series.loc[test_mask].max()) if test_mask.any() else np.nan,
        })

    rows.append({
        "Stratum": "ALL",
        "Train_Count": int(len(train_series)),
        "Test_Count": int(len(test_series)),
        "Total_Count": int(len(all_series)),
        "Train_Ratio": float(len(train_series) / len(all_series)),
        "Test_Ratio": float(len(test_series) / len(all_series)),
        "Train_Target_Min": float(train_series.min()),
        "Train_Target_Max": float(train_series.max()),
        "Test_Target_Min": float(test_series.min()),
        "Test_Target_Max": float(test_series.max()),
    })
    return pd.DataFrame(rows)


def load_and_preprocess_data(
    file,
    test_size=0.2,
    split_random_state=23,
    n_strata_bins=DEFAULT_N_STRATA_BINS,
    use_stratified_regression_split=True,
):
    data = read_data(file)
    data_df = pd.DataFrame(data, columns=data.columns)

    X_raw = data_df.iloc[:, :-1].copy().reset_index(drop=True)
    y = pd.Series(data_df.iloc[:, -1].copy()).reset_index(drop=True)

    if use_stratified_regression_split:
        y_strata, actual_n_bins = _build_regression_strata(y, n_bins=n_strata_bins)
        X_train_raw, X_test_raw, y_train, y_test, strata_train, strata_test = train_test_split(
            X_raw,
            y,
            y_strata,
            test_size=test_size,
            random_state=split_random_state,
            stratify=y_strata,
        )
        strata_train = pd.Series(strata_train).reset_index(drop=True).astype(int)
        strata_test = pd.Series(strata_test).reset_index(drop=True).astype(int)
    else:
        actual_n_bins = 1
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X_raw,
            y,
            test_size=test_size,
            random_state=split_random_state,
        )
        strata_train = pd.Series(np.zeros(len(y_train), dtype=int))
        strata_test = pd.Series(np.zeros(len(y_test), dtype=int))

    X_train_raw = pd.DataFrame(X_train_raw).reset_index(drop=True)
    X_test_raw = pd.DataFrame(X_test_raw).reset_index(drop=True)
    y_train = pd.Series(np.asarray(y_train).ravel()).reset_index(drop=True)
    y_test = pd.Series(np.asarray(y_test).ravel()).reset_index(drop=True)

    split_summary_df = _summarize_split(
        y_all=y,
        y_train=y_train,
        y_test=y_test,
        train_bins=strata_train,
        test_bins=strata_test,
    )

    return {
        "X_train_raw": X_train_raw,
        "X_test_raw": X_test_raw,
        "y_train": y_train,
        "y_test": y_test,
        "train_strata": strata_train,
        "test_strata": strata_test,
        "split_random_state": split_random_state,
        "test_size": test_size,
        "use_stratified_regression_split": use_stratified_regression_split,
        "requested_n_strata_bins": n_strata_bins,
        "actual_n_strata_bins": actual_n_bins,
        "split_summary_df": split_summary_df,
        "n_total_samples": int(len(data_df)),
        "feature_names": list(X_raw.columns),
        "target_name": str(data_df.columns[-1]),
    }


class StratifiedRegressionKFold:
    """Stratified K-fold splitter for continuous regression targets."""
    def __init__(
        self,
        n_splits=10,
        shuffle=True,
        random_state=26,
        n_strata_bins=DEFAULT_N_STRATA_BINS,
    ):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.n_strata_bins = n_strata_bins

    def split(self, X, y, groups=None):
        y_series = pd.Series(np.asarray(y).ravel()).reset_index(drop=True)
        y_strata, _ = _build_regression_strata(y_series, n_bins=self.n_strata_bins)

        skf = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=self.shuffle,
            random_state=self.random_state,
        )

        X_dummy = np.zeros(len(y_series))
        for train_idx, valid_idx in skf.split(X_dummy, y_strata):
            yield train_idx, valid_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


def make_stratified_regression_cv(
    n_splits=10,
    shuffle=True,
    random_state=26,
    n_strata_bins=DEFAULT_N_STRATA_BINS,
):
    return StratifiedRegressionKFold(
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state,
        n_strata_bins=n_strata_bins,
    )
