import argparse
import numpy as np
from azureml.core import Workspace, Experiment, Environment, Dataset, Run
import pandas as pd
from tsfresh import extract_features, select_features
from tsfresh.utilities.dataframe_functions import roll_time_series, impute
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score, confusion_matrix, classification_report,
    recall_score, accuracy_score, f1_score, roc_auc_score
)
from label_datafn import label_data
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

# Setup command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--stop_loss', nargs='+', type=float, required=True)
parser.add_argument('--take_profit', nargs='+', type=float, required=True)
parser.add_argument('--dataset_name', type=str, required=True)
parser.add_argument('--n_estimators', type=int, required=True)
parser.add_argument('--class_weight_0', type=float, required=True)
parser.add_argument('--class_weight_1', type=float, required=True)
parser.add_argument('--max_features', type=str, required=False)  # 'None' for no limit
parser.add_argument('--random_state', type=int, required=True)
parser.add_argument('--n_components', type=int, required=True)
parser.add_argument('--threshold', type=float, required=True)
parser.add_argument('--fdr_level', type=float, required=True)
parser.add_argument('--highcorr', type=float, required=True)    # Probability threshold
args = parser.parse_args()

run = Run.get_context()
ws = run.experiment.workspace

# Load dataset
dataset = Dataset.get_by_name(ws, args.dataset_name)
df = dataset.to_pandas_dataframe()

symbol = ""

# Ensure take_profit > stop_loss and apply labeling
for sl, tp in zip(args.stop_loss, args.take_profit):
    if tp < sl:
        print(f"Adjusting take_profit for stop_loss {sl} to ensure it's greater.")
        tp = float(sl) + 1.0
    label_data(df, [sl], [tp], 80, symbol, False)

def add_lag_features(df, lags):
    for lag in lags:
        df[f'open_lag_{lag}'] = df['open'].shift(lag)
        df[f'close_lag_{lag}'] = df['close'].shift(lag)
        df[f'high_lag_{lag}'] = df['high'].shift(lag)
        df[f'low_lag_{lag}'] = df['low'].shift(lag)
    return df

def extract_rolling_features(df, signal, symbol, max_shift=20, min_shift=5):
    df_melted = df[['time', signal]].copy()
    df_melted["Symbols"] = symbol
    df_rolled = roll_time_series(df_melted, column_id="Symbols", column_sort="time",
                                 max_timeshift=max_shift, min_timeshift=min_shift)
    X = extract_features(df_rolled.drop("Symbols", axis=1),
                         column_id="id", column_sort="time", column_value=signal,
                         impute_function=impute, show_warnings=False)
    X = X.set_index(X.index.map(lambda x: x[1]), drop=True)
    X.index.name = "time"
    return X.dropna()

df.drop(columns=['s_flag'], inplace=True)

# Optionally add lag features from the same day
df = add_lag_features(df, lags=[1, 2, 3, 4, 5])

df['time'] = pd.to_datetime(df['time'])

X1  = extract_rolling_features(df, 'WILLR_15',  symbol)
X2  = extract_rolling_features(df, 'WILLR_42',  symbol)
X3  = extract_rolling_features(df, 'RSI_14',    symbol)
X4  = extract_rolling_features(df, 'MACD_hist', symbol)
X5  = extract_rolling_features(df, 'EMA_9',     symbol)
X6  = extract_rolling_features(df, 'EMA_21',    symbol)
X7  = extract_rolling_features(df, 'EMA_50',    symbol)
X9  = extract_rolling_features(df, 'RSI_9',     symbol)
X10 = extract_rolling_features(df, 'RSI_21',    symbol)
X11 = extract_rolling_features(df, 'WILLR_23',  symbol)
X12 = extract_rolling_features(df, 'WILLR_145', symbol)
X13 = extract_rolling_features(df, 'SAR',       symbol)
X14 = extract_rolling_features(df, 'BB_width',  symbol)
X15 = extract_rolling_features(df, 'MACD_signal', symbol)
X16 = extract_rolling_features(df, 'CCI_14',    symbol)

X_tsfresh = pd.concat(
    [X1, X2, X3, X4, X5, X6, X7, X9, X10, X11, X12, X13, X14, X15, X16],
    axis=1, join='inner'
).dropna()

# ================================
# 6) MERGE TSFRESH FEATURES WITH MAIN DF
# ================================
df = df.set_index(pd.to_datetime(df['time']))
df.drop(columns=['time'], inplace=True)

X = X_tsfresh[X_tsfresh.index.isin(df.index)]
X = pd.concat([df, X], axis=1, join='inner')

# Now X has open, high, low, close, indicators, b_flag, plus TSFresh features.

# ================================
# 7) SHIFT FEATURES FOR OPEN-OF-DAY
# ================================
X_df = X.copy()

target_col = 'b_flag'
all_cols  = [c for c in X_df.columns if c != target_col]
X_df = X_df[all_cols + [target_col]]

# SHIFT all features (except b_flag) by +1 row
feature_cols = all_cols
X_df[feature_cols] = X_df[feature_cols].shift(1)

# Drop rows with NaNs from shifting
X_df.dropna(inplace=True)

# ================================
# 8) TSFRESH FEATURE SELECTION
# (Apply it to the SHIFTED DataFrame)
# ================================
X_df = select_features(X_df, X_df[target_col], fdr_level = args.fdr_level)
X_df = X_df[[col for col in X_df if col != target_col] + [target_col]]

# ================================
# 9) CORRELATION FILTER
# ================================
corr_matrix = X_df.corr().abs()
upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
high_corr_features = [column for column in upper_triangle.columns if any(upper_triangle[column] > args.highcorr)]

# Avoid removing the target_col if it appears in high_corr_features
high_corr_features = [f for f in high_corr_features if f != target_col]

X_df.drop(columns=high_corr_features, inplace=True, errors='ignore')
def custom_time_series_split(X_df, n_splits, test_size=0.1):
    total_samples = len(X_df)
    test_samples = int(total_samples * test_size)
    print("Total Samples:", total_samples)
    print("Test Samples:", test_samples)

    split_indices = [total_samples - (n_splits - i) * test_samples for i in range(n_splits)]
    print("Split Indices:", split_indices)

    train_index_1 = X_df.index[:split_indices[0]]
    test_index_1 = X_df.index[split_indices[0]:]
    print("Fold 1 Train:", train_index_1, "Test:", test_index_1)
    yield train_index_1, test_index_1

    for i in range(1, n_splits):
        train_index = X_df.index[:split_indices[i]]
        if i == n_splits - 1:
            test_index = X_df.index[split_indices[i]:]
        else:
            test_index = X_df.index[split_indices[i]:len(X_df)]
        print("Fold", i+1, "Train:", train_index)
        print("Test:", test_index)
        yield train_index, test_index

tscv = custom_time_series_split(X_df, n_splits=5) 

overall_sum_fp = 0
overall_sum_tp = 0
overall_trades = 0
BreakEvenRatio = float(sl / (sl + tp))
run.log('BreakEvenRatio', BreakEvenRatio)
split_num = 1

for train_index, test_index in tscv:
    train_data = X_df.loc[train_index]
    test_data = X_df.loc[test_index]
    print("Train Data Shape:", train_data.shape)
    print("Test Data Shape:", test_data.shape)
    
    x_train = train_data.iloc[:, :-1].values
    y_train = train_data['b_flag'].values
    x_test = test_data.iloc[:, :-1].values
    y_test = test_data['b_flag'].values

    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)

    pca = PCA(n_components=args.n_components, svd_solver='randomized', random_state=args.random_state)
    x_train = pca.fit_transform(x_train)
    x_test = pca.transform(x_test)

    class_weight = {0: args.class_weight_0, 1: args.class_weight_1}
    if args.max_features == 'None':
        args.max_features = None

    rf_classifier_yf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        class_weight=class_weight,
        max_features=args.max_features,
        random_state=args.random_state
    )

    rf_classifier_yf.fit(x_train, y_train)
    y_pred_proba = rf_classifier_yf.predict_proba(x_test)[:, 1]

    # Use custom threshold from args.threshold
    y_pred = (y_pred_proba > args.threshold).astype(int)

    precision = precision_score(y_test, y_pred)
    conf_mat = confusion_matrix(y_test, y_pred)
    false_positives = conf_mat[0][1]
    true_positives = conf_mat[1][1]

    no_of_trades = false_positives + true_positives
    overall_sum_fp += false_positives
    overall_sum_tp += true_positives
    overall_trades += no_of_trades

    run.log('adj_takeprofit',tp)
    run.log('no_of_trades', no_of_trades)
    run.log('false_positives', false_positives)
    run.log('true_positives', true_positives)
    run.log(f'precision_split_{split_num}', precision)
    
    if BreakEvenRatio > precision:
        print(f"Early stopping: BreakEvenRatio {BreakEvenRatio} is greater than precision {precision}")
        run.log('early_stopping_triggered', True)
        run.log('break_even_ratio', BreakEvenRatio)
        run.log('precision', precision)
        break    

    if split_num == 4:
        run.log('WIN/LOSS-Diff_4', round(100*(precision-BreakEvenRatio),2))
    
    if split_num == 5:
        run.log('WIN/LOSS-Diff_5', round(100*(precision-BreakEvenRatio),2))

    split_num += 1

print('Overall False Positives:', overall_sum_fp)
print('Overall True Positives:', overall_sum_tp)
if (overall_sum_fp + overall_sum_tp) > 0:
    overall_precision = overall_sum_tp / (overall_sum_fp + overall_sum_tp)
    print('Overall Precision:', overall_precision)
    run.log('overall_precision', overall_precision)
else:
    print('No positive predictions made.')
    run.log('overall_precision', 0)

run.log('overall_trades', overall_trades)
