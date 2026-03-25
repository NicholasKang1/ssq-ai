import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, Callback
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import MinMaxScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import random
from copy import deepcopy
import warnings
import os
from datetime import datetime
import json
import itertools
import plotly.express as px
import plotly.graph_objects as go
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader

# 可选库
try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# 自动更新模块
try:
    import data_fetcher
    DATA_FETCHER_AVAILABLE = True
except ImportError:
    DATA_FETCHER_AVAILABLE = False

try:
    import auto_train
    AUTO_TRAIN_AVAILABLE = True
except ImportError:
    AUTO_TRAIN_AVAILABLE = False

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# --------------------------
# 配置参数
# --------------------------
RED_MIN, RED_MAX = 1, 33
BLUE_MIN, BLUE_MAX = 1, 16
RED_BALL_COUNT = 6
INITIAL_CAPITAL = 1000
BET_PER_NOTE = 2
MONTE_CARLO_COUNT = 10000
GENETIC_POPULATION = 20
GENETIC_GENERATIONS = 5

BONUS_CONFIG = {
    1: 10000000,
    2: 500000,
    3: 3000,
    4: 200,
    5: 10,
    6: 5,
    0: 0
}

# ==================== 特征名称 ====================
def get_feature_names(enhanced=True):
    base_names = [f'red_{i+1}' for i in range(33)]
    if not enhanced:
        return base_names
    extra_names = ['sum_red', 'odd_ratio', 'big_ratio', 'consecutive', 'ac']
    trans_names = [f'trans_{i+1}' for i in range(33)]
    return base_names + extra_names + trans_names

# --------------------------
# 数据加载与预处理
# --------------------------
@st.cache_data
def load_history_data():
    try:
        df = pd.read_csv('data/history.csv')
        red_cols = [f'red{i}' for i in range(1, 7)]
        for col in red_cols + ['blue']:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
        for col in red_cols:
            df.loc[df[col] < RED_MIN, col] = RED_MIN
            df.loc[df[col] > RED_MAX, col] = RED_MAX
        df.loc[df['blue'] < BLUE_MIN, 'blue'] = BLUE_MIN
        df.loc[df['blue'] > BLUE_MAX, 'blue'] = BLUE_MAX
        df['sum_red'] = df[red_cols].sum(axis=1)
        if 'issue' in df.columns:
            df = df.sort_values('issue').reset_index(drop=True)
        return df
    except FileNotFoundError:
        st.error("未找到历史数据文件，请在data目录下创建history.csv文件！")
        st.stop()
    except Exception as e:
        st.error(f"数据加载错误: {str(e)}")
        st.stop()

def check_data_quality(df, check_continuity=False):
    warnings_list = []
    red_cols = [f'red{i}' for i in range(1,7)]
    for col in red_cols:
        if (df[col] < RED_MIN).any() or (df[col] > RED_MAX).any():
            warnings_list.append(f"{col} 超出范围")
    if (df['blue'] < BLUE_MIN).any() or (df['blue'] > BLUE_MAX).any():
        warnings_list.append("蓝球超出范围")
    if df['issue'].duplicated().any():
        warnings_list.append("存在重复期号")
    if check_continuity and 'issue' in df.columns:
        issues = df['issue'].values
        for i in range(1, len(issues)):
            prev = issues[i-1]
            curr = issues[i]
            expected = prev + 1
            if prev % 1000 == 153 and curr % 1000 == 1 and curr // 1000 == prev // 1000 + 1:
                continue
            if curr != expected:
                warnings_list.append(f"期号不连续：{prev} 到 {curr}")
    return warnings_list

def preprocess_data_raw(df):
    red_cols = [f'red{i}' for i in range(1, 7)]
    return df[red_cols].values

def preprocess_data(df):
    red_cols = [f'red{i}' for i in range(1, 7)]
    red_data = df[red_cols].values
    ball_matrix = np.zeros((len(df), 33))
    for i in range(len(df)):
        for ball in red_data[i]:
            ball_matrix[i, ball-1] = 1
    return ball_matrix

def compute_transition_matrix(df):
    red_cols = [f'red{i}' for i in range(1, 7)]
    red_data = df[red_cols].values
    trans_mat = np.zeros((33, 33))
    for i in range(1, len(df)):
        prev_balls = red_data[i-1]
        curr_balls = red_data[i]
        for prev in prev_balls:
            for curr in curr_balls:
                trans_mat[prev-1, curr-1] += 1
    row_sums = trans_mat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    trans_mat = trans_mat / row_sums
    return trans_mat

def compute_transition_features(df, trans_mat):
    red_cols = [f'red{i}' for i in range(1, 7)]
    red_data = df[red_cols].values
    n = len(df)
    trans_features = np.zeros((n, 33))
    for i in range(1, n):
        prev_balls = red_data[i-1]
        for j in range(1, 34):
            probs = [trans_mat[prev-1, j-1] for prev in prev_balls]
            trans_features[i, j-1] = np.mean(probs)
    trans_features[0] = trans_features[1] if n > 1 else 0
    return trans_features

def compute_extra_features(df, window=10):
    n = len(df)
    extra_list = []
    for i in range(n):
        if i == 0:
            extra_list.append([0,0,0,0,0])
        else:
            hist = df.iloc[max(0, i-window):i]
            if len(hist) == 0:
                extra_list.append([0,0,0,0,0])
            else:
                anderson = hist['sum_red'].mean()
                odd_counts = []
                big_counts = []
                cons_counts = []
                ac_values = []
                for _, row in hist.iterrows():
                    reds = [row[f'red{j}'] for j in range(1,7)]
                    odd_counts.append(sum(1 for r in reds if r % 2 == 1))
                    big_counts.append(sum(1 for r in reds if r > 16))
                    cons = 0
                    for j in range(5):
                        if reds[j+1] - reds[j] == 1:
                            cons += 1
                    cons_counts.append(cons)
                    ac_values.append(max(reds) - min(reds))
                odd_ratio = np.mean(odd_counts) / 6.0
                big_ratio = np.mean(big_counts) / 6.0
                consecutive = np.mean(cons_counts) / 5.0
                ac = np.mean(ac_values) / 32.0
                extra_list.append([anderson/200.0, odd_ratio, big_ratio, consecutive, ac])
    return np.array(extra_list)

def preprocess_data_enhanced(df):
    base = preprocess_data(df)
    extra = compute_extra_features(df)
    trans_mat = compute_transition_matrix(df)
    trans_features = compute_transition_features(df, trans_mat)
    return np.concatenate([base, extra, trans_features], axis=1)

# --------------------------
# 模型构建（多标签分类）
# --------------------------
def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(33, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.0005, clipnorm=1.0), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def attention_pooling(inputs):
    attention_weights = Dense(1, activation='tanh')(inputs)
    attention_weights = tf.nn.softmax(attention_weights, axis=1)
    weighted = inputs * attention_weights
    pooled = tf.reduce_sum(weighted, axis=1)
    return pooled

def build_enhanced_lstm_model(input_shape, lstm_units=128, num_layers=2, dropout=0.2, learning_rate=0.0005, use_l2=False):
    inputs = Input(shape=input_shape)
    x = inputs
    for i in range(num_layers):
        return_seq = (i < num_layers - 1)
        kernel_reg = l2(0.001) if use_l2 else None
        x = Bidirectional(LSTM(lstm_units, return_sequences=return_seq, dropout=dropout, kernel_regularizer=kernel_reg))(x)
        if return_seq:
            x = Dropout(dropout)(x)
    if num_layers == 1 or (num_layers > 1 and not return_seq):
        x = Dense(32, activation='relu')(x)
    else:
        x = attention_pooling(x)
        x = Dropout(dropout)(x)
    outputs = Dense(33, activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=outputs)
    optimizer = Adam(learning_rate=learning_rate, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

# ==================== 蓝球预测 ====================
def train_blue_model(df):
    blue_series = df['blue'].values
    counts = np.bincount(blue_series, minlength=17)[1:]
    prob = counts / len(blue_series)
    return prob

def predict_blue_frequency(history_df, method='frequency'):
    """基于历史频率返回最可能的蓝球"""
    counts = history_df['blue'].value_counts()
    if method == 'frequency':
        return counts.idxmax()
    elif method == 'hot':
        recent = history_df.tail(50)['blue'].value_counts()
        return recent.idxmax() if not recent.empty else 1
    elif method == 'cold':
        recent = history_df.tail(50)['blue'].value_counts()
        all_nums = set(range(1,17))
        appeared = set(recent.index)
        cold = list(all_nums - appeared)
        if cold:
            return random.choice(cold)
        else:
            return 1
    else:
        return random.randint(1,16)

def backtest_blue_accuracy(history_df, method='frequency', n_periods=100):
    """回测蓝球预测准确率"""
    if len(history_df) < n_periods + 1:
        n_periods = len(history_df) - 1
    if n_periods <= 0:
        return 0.0
    test_df = history_df.tail(n_periods + 1).reset_index(drop=True)
    correct = 0
    for i in range(1, len(test_df)):
        train = test_df.iloc[:i]
        pred = predict_blue_frequency(train, method)
        actual = test_df.iloc[i]['blue']
        if pred == actual:
            correct += 1
    return correct / n_periods

# ==================== 进度回调（增强版：显示 accuracy）====================
class StreamlitProgressCallback(Callback):
    def __init__(self, progress_bar, status_text, epochs):
        super().__init__()
        self.progress_bar = progress_bar
        self.status_text = status_text
        self.epochs = epochs

    def on_epoch_end(self, epoch, logs=None):
        self.progress_bar.progress((epoch + 1) / self.epochs)
        acc = logs.get('accuracy', logs.get('acc', 0))
        val_acc = logs.get('val_accuracy', logs.get('val_acc', 0))
        self.status_text.text(
            f"Epoch {epoch+1}/{self.epochs} - loss: {logs['loss']:.4f} - acc: {acc:.4f} - val_loss: {logs['val_loss']:.4f} - val_acc: {val_acc:.4f}"
        )

# ==================== 多模型训练（多标签）====================
def train_models(data, look_back=10, enhanced=False, use_xgboost=False, use_rf=False, use_lgb=False,
                 lstm_units=128, num_layers=2, dropout=0.2, lstm_weight=1.0,
                 xgb_ensemble_size=1, xgb_params=None, use_stacking=False,
                 learning_rate=0.0005, epochs=300, patience=80, use_l2=False,
                 progress_bar=None, status_text=None):
    # 检查标签是否为二值0/1
    y_check = data[:10, :33]
    st.write(f"标签检查: min={y_check.min():.4f}, max={y_check.max():.4f}, mean={y_check.mean():.4f}")
    if y_check.min() < 0 or y_check.max() > 1:
        st.error("错误：标签不是0/1！请检查数据预处理。")
        return None, None, None, None, None, None
    # 如果值不是严格0/1，但接近，可以强制二值化
    if y_check.min() >= 0 and y_check.max() <= 1 and not np.all(np.isin(y_check, [0,1])):
        st.warning("标签有小数，将强制二值化（>0.5为1）")
        data[:, :33] = (data[:, :33] > 0.5).astype(np.float32)

    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)

    X_seq, y = [], []
    for i in range(look_back, len(data_scaled)):
        X_seq.append(data_scaled[i-look_back:i, :])
        y.append(data[i, :33])  # 目标取原始 one‑hot (0/1)
    X_seq = np.array(X_seq)
    y = np.array(y)

    if len(X_seq) == 0:
        return None, None, None, None, None, None

    split = int(len(X_seq) * 0.8)
    X_train_seq, X_val_seq = X_seq[:split], X_seq[split:]
    y_train, y_val = y[:split], y[split:]

    lstm_model = None
    history = None
    if lstm_weight > 0:
        if enhanced:
            lstm_model = build_enhanced_lstm_model((X_train_seq.shape[1], X_train_seq.shape[2]),
                                                    lstm_units, num_layers, dropout, learning_rate, use_l2)
        else:
            lstm_model = build_lstm_model((X_train_seq.shape[1], X_train_seq.shape[2]))

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-6),
            ModelCheckpoint('models/best_temp.h5', monitor='val_loss', save_best_only=True)
        ]
        if progress_bar is not None and status_text is not None:
            callbacks.append(StreamlitProgressCallback(progress_bar, status_text, epochs))

        history = lstm_model.fit(
            X_train_seq, y_train,
            validation_data=(X_val_seq, y_val),
            epochs=epochs,
            batch_size=32,
            callbacks=callbacks,
            verbose=0
        )

        lstm_model = tf.keras.models.load_model('models/best_temp.h5')
        if os.path.exists('models/best_temp.h5'):
            os.remove('models/best_temp.h5')
    else:
        if progress_bar is not None:
            progress_bar.empty()
        if status_text is not None:
            status_text.text("跳过LSTM训练")

    X_train_flat = X_train_seq.reshape(X_train_seq.shape[0], -1)
    X_val_flat = X_val_seq.reshape(X_val_seq.shape[0], -1)

    models_dict = {}
    base_models = []

    if use_xgboost:
        xgb_models = []
        default_params = {
            'n_estimators': 100,
            'max_depth': 4,
            'learning_rate': 0.05,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0
        }
        if xgb_params is None:
            xgb_params = default_params
        for i in range(xgb_ensemble_size):
            model = xgb.XGBRegressor(
                n_estimators=xgb_params.get('n_estimators', 100),
                max_depth=xgb_params.get('max_depth', 4),
                learning_rate=xgb_params.get('learning_rate', 0.05),
                reg_alpha=xgb_params.get('reg_alpha', 0.1),
                reg_lambda=xgb_params.get('reg_lambda', 1.0),
                random_state=42 + i
            )
            multi_model = MultiOutputRegressor(model)
            multi_model.fit(X_train_flat, y_train)
            xgb_models.append(multi_model)
            base_models.append(('xgb_'+str(i), multi_model))
        def predict_xgb(x):
            x_flat = x.reshape(1, -1)
            all_probs = []
            for m in xgb_models:
                prob = m.predict(x_flat)[0]
                prob = np.clip(prob, 0, None)
                all_probs.append(prob)
            return np.mean(all_probs, axis=0)
        models_dict['xgb'] = predict_xgb
        st.info(f"XGBoost集成训练完成 ({xgb_ensemble_size}个模型)")

    if use_rf:
        rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        rf_multi = MultiOutputRegressor(rf_model)
        rf_multi.fit(X_train_flat, y_train)
        base_models.append(('rf', rf_multi))
        def predict_rf(x):
            x_flat = x.reshape(1, -1)
            prob = rf_multi.predict(x_flat)[0]
            prob = np.clip(prob, 0, None)
            return prob
        models_dict['rf'] = predict_rf
        st.info("随机森林训练完成")

    if use_lgb and LGB_AVAILABLE:
        lgb_model = lgb.LGBMRegressor(n_estimators=100, max_depth=6, learning_rate=0.05, random_state=42)
        lgb_multi = MultiOutputRegressor(lgb_model)
        lgb_multi.fit(X_train_flat, y_train)
        base_models.append(('lgb', lgb_multi))
        def predict_lgb(x):
            x_flat = x.reshape(1, -1)
            prob = lgb_multi.predict(x_flat)[0]
            prob = np.clip(prob, 0, None)
            return prob
        models_dict['lgb'] = predict_lgb
        st.info("LightGBM训练完成")

    stacker = None
    if use_stacking and len(base_models) >= 1:
        X_val_meta = []
        for name, model in base_models:
            preds = model.predict(X_val_flat)
            X_val_meta.append(preds)
        X_val_meta = np.concatenate(X_val_meta, axis=1)
        stacker = MultiOutputRegressor(LinearRegression())
        stacker.fit(X_val_meta, y_val)
        st.info("堆叠融合模型训练完成")

    return lstm_model, scaler, X_seq, models_dict, history, stacker

def predict_ball_probability(lstm_model, scaler, X_seq, models_dict=None, use_models=None, weight_lstm=0.5, stacker=None, base_models_list=None):
    last_sequence = X_seq[-1].reshape(1, X_seq.shape[1], X_seq.shape[2])
    last_flat = last_sequence.reshape(1, -1)

    lstm_prob = None
    if lstm_model is not None:
        lstm_prob = lstm_model.predict(last_sequence, verbose=0)[0]  # sigmoid 输出
        # 确保是一维数组
        lstm_prob = np.asarray(lstm_prob).flatten()

    model_probs = []
    if models_dict:
        for name, pred_func in models_dict.items():
            if use_models is None or name in use_models:
                prob = pred_func(last_flat)
                prob = np.asarray(prob).flatten()
                model_probs.append(prob)

    if stacker is not None and base_models_list:
        X_meta = []
        for name, model in base_models_list:
            pred = model.predict(last_flat)[0]
            pred = np.asarray(pred).flatten()
            X_meta.append(pred)
        X_meta = np.concatenate(X_meta).reshape(1, -1)
        stack_prob = stacker.predict(X_meta)[0]
        stack_prob = np.clip(stack_prob, 0, None)
        return stack_prob

    if len(model_probs) == 0:
        if lstm_prob is not None:
            return lstm_prob
        else:
            return np.ones(33) / 33

    if lstm_prob is not None:
        all_probs = [lstm_prob] + model_probs
        weights = [weight_lstm] + [(1-weight_lstm)/len(model_probs)] * len(model_probs)
        combined = np.average(all_probs, axis=0, weights=weights)
    else:
        combined = np.mean(model_probs, axis=0)

    combined = np.clip(combined, 0, None)
    return combined

# ==================== 模型版本管理 ====================
def save_model_version(model_type, params, model_path):
    records_file = 'models/model_records.json'
    record = {
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'type': model_type,
        'params': params,
        'path': model_path
    }
    if os.path.exists(records_file):
        with open(records_file, 'r') as f:
            records = json.load(f)
    else:
        records = []
    records.append(record)
    with open(records_file, 'w') as f:
        json.dump(records, f, indent=2)

def load_model_versions():
    records_file = 'models/model_records.json'
    if os.path.exists(records_file):
        with open(records_file, 'r') as f:
            return json.load(f)
    return []

# ==================== 自动调参 ====================
def auto_tune_lstm(X_train, y_train, X_val, y_val, param_grid, progress_callback=None):
    best_score = float('inf')
    best_params = None
    best_model = None
    total = len(list(itertools.product(*param_grid.values())))
    count = 0
    for units in param_grid['units']:
        for layers in param_grid['layers']:
            for dropout in param_grid['dropout']:
                count += 1
                if progress_callback:
                    progress_callback(count / total, f"测试 units={units}, layers={layers}, dropout={dropout}")
                model = build_enhanced_lstm_model((X_train.shape[1], X_train.shape[2]), units, layers, dropout)
                model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
                val_loss = model.evaluate(X_val, y_val, verbose=0)
                if val_loss < best_score:
                    best_score = val_loss
                    best_params = {'units': units, 'layers': layers, 'dropout': dropout}
                    best_model = model
    return best_model, best_params, best_score

def auto_tune_xgboost(X_train_flat, y_train, X_val_flat, y_val, param_grid, progress_callback=None):
    best_score = float('inf')
    best_params = None
    best_model = None
    total = len(list(itertools.product(*param_grid.values())))
    count = 0
    for n_est in param_grid['n_estimators']:
        for depth in param_grid['max_depth']:
            for lr in param_grid['learning_rate']:
                count += 1
                if progress_callback:
                    progress_callback(count / total, f"测试 n_est={n_est}, depth={depth}, lr={lr}")
                model = xgb.XGBRegressor(n_estimators=n_est, max_depth=depth, learning_rate=lr, random_state=42)
                multi = MultiOutputRegressor(model)
                multi.fit(X_train_flat, y_train)
                pred = multi.predict(X_val_flat)
                mse = np.mean((pred - y_val)**2)
                if mse < best_score:
                    best_score = mse
                    best_params = {'n_estimators': n_est, 'max_depth': depth, 'learning_rate': lr}
                    best_model = multi
    return best_model, best_params, best_score

def auto_tune_rf(X_train_flat, y_train, X_val_flat, y_val, param_grid, progress_callback=None):
    best_score = float('inf')
    best_params = None
    best_model = None
    total = len(list(itertools.product(*param_grid.values())))
    count = 0
    for n_est in param_grid['n_estimators']:
        for depth in param_grid['max_depth']:
            count += 1
            if progress_callback:
                progress_callback(count / total, f"测试 n_est={n_est}, depth={depth}")
            model = RandomForestRegressor(n_estimators=n_est, max_depth=depth, random_state=42, n_jobs=-1)
            multi = MultiOutputRegressor(model)
            multi.fit(X_train_flat, y_train)
            pred = multi.predict(X_val_flat)
            mse = np.mean((pred - y_val)**2)
            if mse < best_score:
                best_score = mse
                best_params = {'n_estimators': n_est, 'max_depth': depth}
                best_model = multi
    return best_model, best_params, best_score

# ==================== 历史回测相关 ====================
def predict_reds_from_prob(prob, n=6):
    prob = np.asarray(prob).flatten()
    top_indices = np.argsort(prob)[-n:][::-1]
    return sorted(top_indices + 1)

# ==================== PDF报告生成 ====================
def generate_pdf_report(ball_prob, blue_prob, recommended_notes, history_df, filename="report.pdf"):
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.lib.utils import ImageReader
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # 确保 ball_prob 是一维数组
    ball_prob = np.asarray(ball_prob).flatten()
    if len(ball_prob) != 33:
        if len(ball_prob) > 33:
            ball_prob = ball_prob[:33]
        else:
            ball_prob = np.pad(ball_prob, (0, 33 - len(ball_prob)), 'constant')

    c = canvas.Canvas(filename, pagesize=A4)
    width, height = A4

    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height-50, "AI双色球量化分析报告")
    c.setFont("Helvetica", 12)
    c.drawString(50, height-80, f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height-120, "红球概率排名 (Top 10)")
    c.setFont("Helvetica", 10)
    top10 = np.argsort(ball_prob)[-10:][::-1] + 1
    y_pos = height-150
    for i, ball in enumerate(top10):
        c.drawString(70, y_pos - i*15, f"{ball}: {ball_prob[ball-1]:.4f}")

    c.setFont("Helvetica-Bold", 14)
    c.drawString(300, height-120, "蓝球概率分布 (Top 5)")
    c.setFont("Helvetica", 10)
    top5_blue = np.argsort(blue_prob)[-5:][::-1] + 1
    y_pos = height-150
    for i, ball in enumerate(top5_blue):
        c.drawString(320, y_pos - i*15, f"{ball}: {blue_prob[ball-1]:.4f}")

    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height-250, "推荐号码 (Top 5)")
    c.setFont("Helvetica", 10)
    y_pos = height-280
    for i, note in enumerate(recommended_notes[:5]):
        red_str = ' '.join([f'{x:02d}' for x in note['red']])
        c.drawString(70, y_pos - i*15, f"{i+1}: {red_str} + {note['blue']:02d}")

    fig, ax = plt.subplots(figsize=(4, 3))
    prob_matrix = np.zeros((6, 6))
    for i in range(33):
        row = i // 6
        col = i % 6
        prob_matrix[row, col] = ball_prob[i]
    sns.heatmap(prob_matrix, annot=True, fmt='.3f', cmap='Reds', ax=ax, cbar=False)
    ax.set_title('红球概率热力图')
    img_path = 'temp_heatmap.png'
    plt.savefig(img_path, dpi=150, bbox_inches='tight')
    plt.close()
    c.drawImage(ImageReader(img_path), 50, height-500, width=300, height=200)
    os.remove(img_path)
    c.save()
    return filename

def generate_html_report(ball_prob, blue_prob, recommended_notes, history_df):
    ball_prob = np.asarray(ball_prob).flatten()
    if len(ball_prob) != 33:
        if len(ball_prob) > 33:
            ball_prob = ball_prob[:33]
        else:
            ball_prob = np.pad(ball_prob, (0, 33 - len(ball_prob)), 'constant')
    html = f"""
    <html>
    <head><title>AI双色球分析报告</title></head>
    <body>
    <h1>AI双色球分析报告</h1>
    <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    <h2>红球概率排名</h2>
    <ul>
    """
    top10 = np.argsort(ball_prob)[-10:][::-1] + 1
    for i, ball in enumerate(top10):
        html += f"<li>{ball}: {ball_prob[ball-1]:.4f}</li>"
    html += "</ul>"
    html += "<h2>推荐号码 (TOP 5)</h2><ul>"
    for i, note in enumerate(recommended_notes[:5]):
        html += f"<li>{note['red']} + {note['blue']}</li>"
    html += "</ul></body></html>"
    return html

# --------------------------
# 蒙特卡洛生成（最终强化版）
# --------------------------
def monte_carlo_generate(ball_prob, n=MONTE_CARLO_COUNT):
    """蒙特卡洛模拟生成组合，ball_prob 为红球概率向量"""
    # 确保 ball_prob 是一维数组且长度正确
    if ball_prob is None:
        ball_prob = np.ones(33) / 33
    ball_prob = np.asarray(ball_prob).flatten()
    if len(ball_prob) != 33:
        ball_prob = np.ones(33) / 33
    # 确保概率非负且归一化
    prob = np.maximum(ball_prob, 0)  # 将负值截断为0
    total = np.sum(prob)
    if total <= 0:
        # 如果所有概率为0，则使用均匀分布
        prob = np.ones(33) / 33
    else:
        prob = prob / total

    combinations = []
    for _ in range(n):
        red_balls = np.random.choice(
            range(RED_MIN, RED_MAX+1),
            size=6,
            replace=False,
            p=prob
        )
        blue_ball = random.randint(BLUE_MIN, BLUE_MAX)
        red_balls = sorted(red_balls)
        combinations.append({'red': red_balls, 'blue': blue_ball})
    return combinations

# ==================== 推荐方式多样化 ====================
def generate_by_hot_cold(history_df, n=20):
    red_cols = [f'red{i}' for i in range(1,7)]
    all_reds = []
    for col in red_cols:
        all_reds.extend(history_df[col].tolist())
    red_counts = pd.Series(all_reds).value_counts()
    hot_pool = red_counts.head(15).index.tolist()
    cold_pool = red_counts.tail(15).index.tolist()
    recommendations = []
    for _ in range(n):
        n_hot = random.randint(3,4)
        n_cold = 6 - n_hot
        reds = random.sample(hot_pool, n_hot) + random.sample(cold_pool, n_cold)
        reds = sorted(reds)
        blue = random.randint(1,16)
        recommendations.append({'red': reds, 'blue': blue})
    return recommendations

def generate_by_missing(history_df, n=20):
    red_cols = [f'red{i}' for i in range(1,7)]
    last_100 = history_df.tail(100)
    appeared = set()
    for col in red_cols:
        appeared.update(last_100[col].tolist())
    missing = [i for i in range(1,34) if i not in appeared]
    recommendations = []
    for _ in range(n):
        if len(missing) >= 6:
            reds = random.sample(missing, 6)
        else:
            reds = missing + random.sample([i for i in range(1,34) if i not in missing], 6-len(missing))
        reds = sorted(reds)
        blue = random.randint(1,16)
        recommendations.append({'red': reds, 'blue': blue})
    return recommendations

def generate_random(n=20):
    recommendations = []
    for _ in range(n):
        reds = sorted(random.sample(range(1,34), 6))
        blue = random.randint(1,16)
        recommendations.append({'red': reds, 'blue': blue})
    return recommendations

# --------------------------
# 遗传算法
# --------------------------
def calculate_score(combination):
    red_balls = combination['red']
    score = 0
    sum_red = sum(red_balls)
    if 80 <= sum_red <= 140:
        score += 20
    else:
        score += max(0, 20 - abs(sum_red - 110) * 0.5)
    odd_count = sum(1 for ball in red_balls if ball % 2 == 1)
    if 2 <= odd_count <= 4:
        score += 20
    else:
        score += max(0, 20 - abs(odd_count - 3) * 5)
    big_count = sum(1 for ball in red_balls if ball > 16)
    if 2 <= big_count <= 4:
        score += 20
    else:
        score += max(0, 20 - abs(big_count - 3) * 5)
    consecutive_count = 0
    for i in range(len(red_balls)-1):
        if red_balls[i+1] - red_balls[i] == 1:
            consecutive_count += 1
    if consecutive_count <= 2:
        score += 20
    else:
        score += max(0, 20 - (consecutive_count - 2) * 5)
    return score

def select_best(combinations, n=GENETIC_POPULATION):
    scored = [(comb, calculate_score(comb)) for comb in combinations]
    scored_sorted = sorted(scored, key=lambda x: x[1], reverse=True)
    return [x[0] for x in scored_sorted[:n]]

def crossover(parent1, parent2):
    cross_point = random.randint(1, 5)
    child_red = sorted(list(set(
        parent1['red'][:cross_point] + parent2['red'][cross_point:]
    )))
    while len(child_red) < 6:
        ball = random.randint(RED_MIN, RED_MAX)
        if ball not in child_red:
            child_red.append(ball)
    while len(child_red) > 6:
        child_red.pop(random.randint(0, len(child_red)-1))
    child_red = sorted(child_red)
    child_blue = random.choice([parent1['blue'], parent2['blue']])
    return {'red': child_red, 'blue': child_blue}

def mutate(combination, mutation_rate=0.1):
    mutated = deepcopy(combination)
    for i in range(6):
        if random.random() < mutation_rate:
            new_ball = random.randint(RED_MIN, RED_MAX)
            while new_ball in mutated['red']:
                new_ball = random.randint(RED_MIN, RED_MAX)
            mutated['red'][i] = new_ball
            mutated['red'] = sorted(mutated['red'])
    if random.random() < mutation_rate:
        mutated['blue'] = random.randint(BLUE_MIN, BLUE_MAX)
    return mutated

def genetic_algorithm(initial_population, generations=GENETIC_GENERATIONS):
    population = initial_population
    for gen in range(generations):
        parents = select_best(population)
        offspring = []
        while len(offspring) < MONTE_CARLO_COUNT - GENETIC_POPULATION:
            p1 = random.choice(parents)
            p2 = random.choice(parents)
            child = crossover(p1, p2)
            child = mutate(child)
            offspring.append(child)
        population = parents + offspring
    return select_best(population)

# --------------------------
# 回测模拟
# --------------------------
def calculate_bonus(predicted, actual):
    red_hits = len(set(predicted['red']) & set(actual['red']))
    blue_hit = 1 if predicted['blue'] == actual['blue'] else 0
    if red_hits == 6 and blue_hit == 1:
        return BONUS_CONFIG[1]
    elif red_hits == 6:
        return BONUS_CONFIG[2]
    elif red_hits == 5 and blue_hit == 1:
        return BONUS_CONFIG[3]
    elif red_hits == 5 or (red_hits == 4 and blue_hit == 1):
        return BONUS_CONFIG[4]
    elif red_hits == 4 or (red_hits == 3 and blue_hit == 1):
        return BONUS_CONFIG[5]
    elif blue_hit == 1:
        return BONUS_CONFIG[6]
    else:
        return BONUS_CONFIG[0]

def backtest_simulation_strategy(recommended_notes, history_df, n_periods=50, strategy='fixed', stop_loss=None):
    test_data = history_df.tail(n_periods).reset_index(drop=True)
    capital = [INITIAL_CAPITAL]
    total_bets = 0
    total_bonus = 0
    bet_size = BET_PER_NOTE * len(recommended_notes)
    consecutive_loss = 0

    for idx, row in test_data.iterrows():
        actual = {
            'red': [row[f'red{i}'] for i in range(1, 7)],
            'blue': row['blue']
        }
        if strategy == 'fixed':
            stake = bet_size
        elif strategy == 'martingale':
            stake = bet_size * (2 ** consecutive_loss)
            if stake > capital[-1]:
                stake = capital[-1]
        elif strategy == 'stop_loss':
            stake = bet_size
            if stop_loss and capital[-1] < stop_loss:
                break
        else:
            stake = bet_size

        total_bets += stake
        period_bonus = 0
        for note in recommended_notes:
            period_bonus += calculate_bonus(note, actual)
        total_bonus += period_bonus

        if period_bonus > 0:
            consecutive_loss = 0
        else:
            consecutive_loss += 1

        current_capital = capital[-1] - stake + period_bonus
        capital.append(max(0, current_capital))
        if current_capital <= 0:
            break

    total_return = (capital[-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    max_drawdown = min(
        [(capital[i] - max(capital[:i+1])) / max(capital[:i+1]) * 100 for i in range(len(capital))]
    ) if max(capital) > 0 else 0
    return {
        'capital_curve': capital,
        'total_return': total_return,
        'max_drawdown': max_drawdown,
        'total_bets': total_bets,
        'total_bonus': total_bonus,
        'final_capital': capital[-1]
    }

# ==================== 蓝球分析函数 ====================
def get_blue_stats(df):
    blue_counts = df['blue'].value_counts().sort_index()
    blue_probs = blue_counts / len(df)
    stats_df = pd.DataFrame({
        '蓝球号码': range(BLUE_MIN, BLUE_MAX+1),
        '出现次数': [blue_counts.get(i, 0) for i in range(BLUE_MIN, BLUE_MAX+1)],
        '出现概率': [blue_probs.get(i, 0) for i in range(BLUE_MIN, BLUE_MAX+1)]
    })
    return stats_df

def plot_blue_trend(df):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df.index, df['blue'], marker='o', markersize=3, linewidth=1, color='blue', alpha=0.7)
    ax.axhline(y=np.mean(df['blue']), color='r', linestyle='--', alpha=0.7, label='平均值')
    ax.set_xlabel('开奖期数')
    ax.set_ylabel('蓝球号码')
    ax.set_title('蓝球走势图')
    ax.set_ylim(BLUE_MIN-0.5, BLUE_MAX+0.5)
    ax.set_yticks(range(BLUE_MIN, BLUE_MAX+1, 2))
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig

# --------------------------
# 可视化模块（修复数组形状）
# --------------------------
def plot_heatmap(ball_prob):
    # 确保 ball_prob 是一维数组且长度为33
    ball_prob = np.asarray(ball_prob).flatten()
    if len(ball_prob) != 33:
        if len(ball_prob) > 33:
            ball_prob = ball_prob[:33]
        else:
            ball_prob = np.pad(ball_prob, (0, 33 - len(ball_prob)), 'constant')
    prob_matrix = np.zeros((6, 6))
    for i in range(33):
        row = i // 6
        col = i % 6
        prob_matrix[row, col] = ball_prob[i]
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(prob_matrix, annot=True, fmt='.3f', cmap='Reds', ax=ax,
                cbar_kws={'label': '出现概率'})
    row_labels = [f'{i*6+1}-{min(i*6+6,33)}' for i in range(6)]
    col_labels = [str(i+1) for i in range(6)]
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)
    ax.set_xlabel('列内序号')
    ax.set_ylabel('红球区间')
    ax.set_title('红球概率热力图')
    return fig

def plot_sum_trend(df):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df.index, df['sum_red'], marker='o', markersize=4, linewidth=1)
    ax.axhline(y=80, color='r', linestyle='--', alpha=0.7, label='下限(80)')
    ax.axhline(y=140, color='r', linestyle='--', alpha=0.7, label='上限(140)')
    ax.axhline(y=110, color='g', linestyle='--', alpha=0.7, label='最优值(110)')
    ax.set_xlabel('开奖期数')
    ax.set_ylabel('红球和值')
    ax.set_title('历史红球和值走势图')
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig

def plot_capital_curve(capital_data):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(capital_data, marker='o', markersize=4, linewidth=2, color='blue')
    ax.axhline(y=INITIAL_CAPITAL, color='r', linestyle='--', alpha=0.7, label='初始资金')
    ax.set_xlabel('模拟期数')
    ax.set_ylabel('资金金额 (元)')
    ax.set_title('模拟资金曲线')
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig

# ==================== 特征重要性绘图 ====================
def plot_feature_importance(models_dict, feature_names):
    if 'xgb' not in models_dict:
        return None
    try:
        importances = models_dict['xgb'].__self__.estimators_[0].feature_importances_
        df = pd.DataFrame({'特征': feature_names, '重要性': importances})
        df = df.sort_values('重要性', ascending=False).head(20)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=df, x='重要性', y='特征', ax=ax)
        ax.set_title('Top 20 特征重要性 (XGBoost)')
        return fig
    except Exception as e:
        st.warning(f"特征重要性绘图失败: {e}")
        return None

# ==================== 3D散点图 ====================
def plot_3d_prob(ball_prob):
    ball_prob = np.asarray(ball_prob).flatten()
    if len(ball_prob) != 33:
        if len(ball_prob) > 33:
            ball_prob = ball_prob[:33]
        else:
            ball_prob = np.pad(ball_prob, (0, 33 - len(ball_prob)), 'constant')
    nums = np.arange(1,34)
    odd_even = (nums % 2)
    section = (nums - 1) // 11
    sizes = ball_prob * 1000
    fig = px.scatter_3d(
        x=nums, y=odd_even, z=section,
        size=sizes,
        color=nums,
        text=nums,
        title='红球概率3D分布',
        labels={'x': '号码', 'y': '奇偶性', 'z': '区间'}
    )
    return fig

# ==================== SHAP解释 ====================
def explain_with_shap(model, X_sample, feature_names):
    if not SHAP_AVAILABLE:
        st.warning("请安装shap库以使用此功能")
        return None
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
        return fig
    except Exception as e:
        st.warning(f"SHAP解释失败: {e}")
        return None

# --------------------------
# 预训练模型加载（修复反序列化错误，并设置默认值）
# --------------------------
@st.cache_resource
def load_pretrained_model():
    model_path = 'models/latest.h5'
    if os.path.exists(model_path):
        # 自定义对象映射，解决反序列化问题
        custom_objects = {
            'mse': tf.keras.losses.MeanSquaredError(),
            'accuracy': tf.keras.metrics.BinaryAccuracy(),
            'binary_crossentropy': tf.keras.losses.BinaryCrossentropy()
        }
        try:
            model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
            # 获取模型输入特征数
            input_shape = model.input_shape
            if len(input_shape) == 3:
                features = input_shape[2]
                st.session_state['model_input_features'] = features
                st.session_state['model_input_timesteps'] = input_shape[1] if input_shape[1] is not None else 10
                st.sidebar.success(f"✅ 已加载预训练模型 (输入特征数: {features})")
            else:
                st.sidebar.warning("模型输入形状异常，将使用默认特征数33")
                st.session_state['model_input_features'] = 33
                st.session_state['model_input_timesteps'] = 10
            return model
        except Exception as e:
            st.sidebar.warning(f"正常加载失败，尝试 compile=False 加载: {e}")
            try:
                model = tf.keras.models.load_model(model_path, compile=False, custom_objects=custom_objects)
                # 重新编译模型
                model.compile(optimizer=Adam(learning_rate=0.0005, clipnorm=1.0), loss='binary_crossentropy', metrics=['accuracy'])
                input_shape = model.input_shape
                if len(input_shape) == 3:
                    features = input_shape[2]
                    st.session_state['model_input_features'] = features
                    st.session_state['model_input_timesteps'] = input_shape[1] if input_shape[1] is not None else 10
                    st.sidebar.success(f"✅ 已用兼容模式加载模型 (输入特征数: {features})")
                else:
                    st.session_state['model_input_features'] = 33
                    st.session_state['model_input_timesteps'] = 10
                return model
            except Exception as e2:
                st.sidebar.error(f"模型加载失败: {e2}")
                st.session_state['model_input_features'] = 33  # 即使失败也设置默认值
                st.session_state['model_input_timesteps'] = 10
                return None
    else:
        st.sidebar.info("ℹ️ 暂无预训练模型")
        st.session_state['model_input_features'] = 33  # 设置默认值
        st.session_state['model_input_timesteps'] = 10
        return None
    # --------------------------
# Streamlit 页面布局
# --------------------------
def main():
    st.set_page_config(
        page_title="AI双色球量化分析系统",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # ========== 用户登录验证（可选，若不需要可注释此段）==========
    # if "authenticated" not in st.session_state:
    #     st.session_state.authenticated = False
    # if not st.session_state.authenticated:
    #     st.title("🔐 用户登录")
    #     try:
    #         users = st.secrets["users"]
    #     except:
    #         st.error("系统配置错误，请联系管理员。")
    #         st.stop()
    #     username = st.text_input("用户名")
    #     password = st.text_input("密码", type="password")
    #     if st.button("登录"):
    #         if username in users and users[username] == password:
    #             st.session_state.authenticated = True
    #             st.success("登录成功！")
    #             st.rerun()
    #         else:
    #             st.error("用户名或密码错误")
    #     st.stop()
    # ========== 登录验证结束 ==========

    st.title("🎯 AI双色球量化分析系统 (终极增强版)")
    st.markdown("---")

    with st.spinner("加载历史数据..."):
        history_df = load_history_data()

    if 'enable_continuity_check' not in st.session_state:
        st.session_state.enable_continuity_check = False
    if 'model_input_features' not in st.session_state:
        st.session_state['model_input_features'] = None

    with st.sidebar:
        st.header("⚙️ 参数设置")
        bet_amount = st.number_input("每注投注金额 (元)", value=BET_PER_NOTE, min_value=1, max_value=100)
        mc_count = st.slider("蒙特卡洛模拟注数", 1000, 50000, MONTE_CARLO_COUNT, step=1000)
        ga_gens = st.slider("遗传算法迭代代数", 1, 20, GENETIC_GENERATIONS)
        backtest_periods = st.slider("回测期数", 10, 200, 50)

        st.markdown("---")
        st.header("🧠 模型优化选项")
        use_enhanced = st.checkbox("启用增强模型（双向LSTM+注意力+特征工程）", value=True)
        if use_enhanced:
            lstm_units = st.selectbox("LSTM单元数", [32, 64, 128, 256], index=2)
            num_layers = st.selectbox("LSTM层数", [1, 2, 3], index=1)
            dropout = st.slider("Dropout比率", 0.1, 0.6, 0.2, step=0.05)
            learning_rate = st.slider("学习率 (LSTM)", 0.0001, 0.01, 0.0005, step=0.0001, format="%.4f")
            use_l2 = st.checkbox("启用L2正则化 (推荐)", value=True)
        else:
            lstm_units = 64
            num_layers = 1
            dropout = 0.4
            learning_rate = 0.0005
            use_l2 = False

        st.subheader("多模型融合")
        use_xgboost = st.checkbox("启用XGBoost", value=True)
        use_rf = st.checkbox("启用随机森林", value=False)
        use_lgb = st.checkbox("启用LightGBM", value=False) if LGB_AVAILABLE else st.checkbox("启用LightGBM", value=False, disabled=True)
        lstm_weight = st.slider("LSTM权重 (其他模型均分剩余权重)", 0.0, 1.0, 0.5, 0.1)
        use_stacking = st.checkbox("启用模型堆叠 (Stacking)", value=False)

        st.subheader("推荐方式")
        rec_method = st.radio("选择推荐算法",
                               ['遗传算法优化', '冷热号推荐', '遗漏值推荐', '完全随机'],
                               index=0)

        with st.expander("🚀 高级优化选项"):
            enable_progress = st.checkbox("显示训练进度", value=True)
            enable_feature_importance = st.checkbox("显示特征重要性", value=True)
            enable_versioning = st.checkbox("自动保存模型版本", value=True)
            enable_3d_plot = st.checkbox("显示3D概率分布图", value=False)
            enable_shap = st.checkbox("启用SHAP解释", value=False)
            force_retrain = st.checkbox("强制重新训练模型（忽略已有模型）", value=False)

            epochs = st.slider("最大训练轮数 (Epochs)", 50, 500, 300, step=50)
            patience = st.slider("早停耐心值 (Patience)", 20, 120, 80, step=10)
            xgb_ensemble_size = st.number_input("XGBoost集成数量", min_value=1, max_value=10, value=1)

            st.subheader("XGBoost参数")
            xgb_n_estimators = st.slider("树的数量 (n_estimators)", 50, 300, 100, 10)
            xgb_max_depth = st.slider("最大深度 (max_depth)", 3, 10, 4, 1)
            xgb_learning_rate = st.slider("学习率 (learning_rate)", 0.01, 0.3, 0.05, 0.01)
            xgb_reg_alpha = st.slider("L1正则化 (reg_alpha)", 0.0, 1.0, 0.1, 0.05)
            xgb_reg_lambda = st.slider("L2正则化 (reg_lambda)", 0.1, 2.0, 1.0, 0.1)

            st.session_state.enable_continuity_check = st.checkbox("检查期号连续性", value=False)

            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔍 自动调参 (LSTM)"):
                    if not use_enhanced:
                        st.warning("请先启用增强模型")
                    else:
                        with st.spinner("正在进行自动调参..."):
                            data_matrix = preprocess_data_enhanced(history_df)
                            scaler = MinMaxScaler()
                            data_scaled = scaler.fit_transform(data_matrix)
                            X_seq, y = [], []
                            for i in range(10, len(data_scaled)):
                                X_seq.append(data_scaled[i-10:i, :])
                                y.append(data_matrix[i, :33])
                            X_seq = np.array(X_seq)
                            y = np.array(y)
                            split = int(len(X_seq) * 0.8)
                            X_train, X_val = X_seq[:split], X_seq[split:]
                            y_train, y_val = y[:split], y[split:]
                            param_grid = {
                                'units': [32, 64, 128],
                                'layers': [1, 2, 3],
                                'dropout': [0.2, 0.3, 0.4, 0.5]
                            }
                            progress_bar = st.progress(0)
                            status = st.empty()
                            def progress_callback(frac, msg):
                                progress_bar.progress(frac)
                                status.text(msg)
                            best_model, best_params, best_score = auto_tune_lstm(X_train, y_train, X_val, y_val, param_grid, progress_callback)
                            st.success(f"最佳参数: {best_params}, 验证损失: {best_score:.4f}")
                            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                            model_path = f'models/lstm_best_{timestamp}.h5'
                            best_model.save(model_path)
                            st.info(f"最佳模型已保存至 {model_path}")

            with col2:
                if st.button("🔍 自动调参 (XGBoost)"):
                    with st.spinner("正在进行XGBoost自动调参..."):
                        data_matrix = preprocess_data_enhanced(history_df) if use_enhanced else preprocess_data(history_df)
                        scaler = MinMaxScaler()
                        data_scaled = scaler.fit_transform(data_matrix)
                        X_seq, y = [], []
                        for i in range(10, len(data_scaled)):
                            X_seq.append(data_scaled[i-10:i, :])
                            y.append(data_matrix[i, :33])
                        X_seq = np.array(X_seq)
                        y = np.array(y)
                        split = int(len(X_seq) * 0.8)
                        X_train, X_val = X_seq[:split], X_seq[split:]
                        y_train, y_val = y[:split], y[split:]
                        X_train_flat = X_train.reshape(X_train.shape[0], -1)
                        X_val_flat = X_val.reshape(X_val.shape[0], -1)
                        param_grid = {
                            'n_estimators': [50, 100, 150],
                            'max_depth': [3, 4, 5, 6],
                            'learning_rate': [0.03, 0.05, 0.07]
                        }
                        progress_bar = st.progress(0)
                        status = st.empty()
                        def progress_callback(frac, msg):
                            progress_bar.progress(frac)
                            status.text(msg)
                        best_model, best_params, best_score = auto_tune_xgboost(X_train_flat, y_train, X_val_flat, y_val, param_grid, progress_callback)
                        st.success(f"最佳XGBoost参数: {best_params}, MSE: {best_score:.4f}")

            if use_rf:
                if st.button("🔍 自动调参 (随机森林)"):
                    with st.spinner("正在进行随机森林自动调参..."):
                        data_matrix = preprocess_data_enhanced(history_df) if use_enhanced else preprocess_data(history_df)
                        scaler = MinMaxScaler()
                        data_scaled = scaler.fit_transform(data_matrix)
                        X_seq, y = [], []
                        for i in range(10, len(data_scaled)):
                            X_seq.append(data_scaled[i-10:i, :])
                            y.append(data_matrix[i, :33])
                        X_seq = np.array(X_seq)
                        y = np.array(y)
                        split = int(len(X_seq) * 0.8)
                        X_train, X_val = X_seq[:split], X_seq[split:]
                        y_train, y_val = y[:split], y[split:]
                        X_train_flat = X_train.reshape(X_train.shape[0], -1)
                        X_val_flat = X_val.reshape(X_val.shape[0], -1)
                        param_grid = {
                            'n_estimators': [50, 100, 150],
                            'max_depth': [5, 10, 15]
                        }
                        progress_bar = st.progress(0)
                        status = st.empty()
                        def progress_callback(frac, msg):
                            progress_bar.progress(frac)
                            status.text(msg)
                        best_model, best_params, best_score = auto_tune_rf(X_train_flat, y_train, X_val_flat, y_val, param_grid, progress_callback)
                        st.success(f"最佳随机森林参数: {best_params}, MSE: {best_score:.4f}")

            if st.button("📊 运行历史回测 (最近{}期)".format(backtest_periods)):
                if ('lstm_model' not in st.session_state and not st.session_state.get('models_dict')):
                    st.warning("请先训练模型（点击『生成AI推荐号码』）")
                else:
                    with st.spinner("正在进行历史回测..."):
                        try:
                            look_back = st.session_state.get('look_back', 10)
                            if use_enhanced:
                                data_matrix = preprocess_data_enhanced(history_df)
                            else:
                                data_matrix = preprocess_data(history_df)
                            if 'scaler' not in st.session_state:
                                st.error("未找到scaler，请先训练模型")
                                st.stop()
                            scaler = st.session_state['scaler']
                            data_scaled = scaler.transform(data_matrix)
                            X_seq, y_onehot = [], []
                            for i in range(look_back, len(data_scaled)):
                                X_seq.append(data_scaled[i-look_back:i, :])
                                y_onehot.append(data_matrix[i, :33])
                            X_seq = np.array(X_seq)
                            y_onehot = np.array(y_onehot)
                            test_size = min(backtest_periods, len(X_seq))
                            X_test = X_seq[-test_size:]

                            actual_issues = history_df['issue'].values[-test_size:]
                            actual_reds_list = []
                            actual_blues = []
                            for i in range(-test_size, 0):
                                row = history_df.iloc[i]
                                actual_reds = [row[f'red{j}'] for j in range(1,7)]
                                actual_blues.append(row['blue'])
                                actual_reds_list.append(actual_reds)

                            blue_prob = train_blue_model(history_df)

                            results = []
                            for idx in range(len(X_test)):
                                X_input = X_test[idx].reshape(1, look_back, X_test.shape[2])
                                red_prob = predict_ball_probability(
                                    st.session_state.get('lstm_model'),
                                    None, X_input,
                                    models_dict=st.session_state.get('models_dict'),
                                    use_models=['xgb','rf','lgb'] if any([use_xgboost, use_rf, use_lgb]) else None,
                                    weight_lstm=lstm_weight
                                )
                                pred_reds = predict_reds_from_prob(red_prob)
                                actual_reds = actual_reds_list[idx]
                                red_hits = len(set(pred_reds) & set(actual_reds))
                                pred_blue = np.argmax(blue_prob) + 1
                                actual_blue = actual_blues[idx]
                                blue_hit = 1 if pred_blue == actual_blue else 0
                                results.append({
                                    '期号': actual_issues[idx],
                                    '预测红球': pred_reds,
                                    '实际红球': actual_reds,
                                    '红球命中': red_hits,
                                    '预测蓝球': pred_blue,
                                    '实际蓝球': actual_blue,
                                    '蓝球命中': blue_hit
                                })

                            df_results = pd.DataFrame(results)
                            avg_red_hits = df_results['红球命中'].mean()
                            total_blue_hits = df_results['蓝球命中'].sum()
                            blue_accuracy = total_blue_hits / len(df_results)

                            st.success("回测完成！")
                            col1, col2, col3 = st.columns(3)
                            col1.metric("平均命中红球个数", f"{avg_red_hits:.2f}")
                            col2.metric("蓝球命中次数", f"{total_blue_hits}/{len(df_results)}")
                            col3.metric("蓝球命中率", f"{blue_accuracy:.2%}")

                            fig, ax = plt.subplots(figsize=(10, 4))
                            hit_counts = df_results['红球命中'].value_counts().sort_index()
                            sns.barplot(x=hit_counts.index, y=hit_counts.values, ax=ax)
                            ax.set_xlabel('命中红球个数')
                            ax.set_ylabel('期数')
                            ax.set_title('红球命中分布')
                            st.pyplot(fig, width='stretch')

                            with st.expander("查看每期详细数据"):
                                display_df = df_results.copy()
                                display_df['预测红球'] = display_df['预测红球'].apply(lambda x: ' '.join([f'{i:02d}' for i in x]))
                                display_df['实际红球'] = display_df['实际红球'].apply(lambda x: ' '.join([f'{i:02d}' for i in x]))
                                st.dataframe(display_df, width='stretch')

                        except Exception as e:
                            st.error(f"回测过程中出错: {e}")
                            import traceback
                            st.code(traceback.format_exc())

            if st.button("📥 导出PDF报告"):
                if 'ball_prob' not in st.session_state or 'recommended_notes' not in st.session_state:
                    st.warning("请先生成推荐号码")
                else:
                    try:
                        filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                        generate_pdf_report(st.session_state.ball_prob, st.session_state.blue_prob,
                                            st.session_state.recommended_notes, history_df, filename)
                        with open(filename, "rb") as f:
                            st.download_button("点击下载PDF报告", f, file_name=filename)
                    except Exception as e:
                        st.error(f"生成PDF失败: {e}")

        st.markdown("---")
        st.header("🔄 自动更新")
        data_path = 'data/history.csv'
        if os.path.exists(data_path):
            mod_time = datetime.fromtimestamp(os.path.getmtime(data_path))
            st.caption(f"数据最后更新: {mod_time.strftime('%Y-%m-%d %H:%M')}")

        if DATA_FETCHER_AVAILABLE:
            if st.button("🔄 更新历史数据"):
                with st.spinner("正在抓取最新数据..."):
                    success = data_fetcher.update_history_csv()
                if success:
                    st.success("数据更新成功！页面将重新加载。")
                    st.rerun()
                else:
                    st.error("数据更新失败，请检查网络或 data_fetcher.py")
        else:
            st.warning("未找到 data_fetcher.py，无法自动更新数据")

        model_path = 'models/latest.h5'
        if os.path.exists(model_path):
            st.caption(f"预训练模型: 存在")
        else:
            st.caption(f"预训练模型: 不存在")

        if AUTO_TRAIN_AVAILABLE:
            if st.button("🧠 立即训练新模型"):
                with st.spinner("正在训练模型（这可能需要几分钟）..."):
                    trainer = auto_train.LotteryAutoTrainer()
                    trainer.auto_retrain_if_needed(check_interval_days=0)
                st.success("模型训练完成！页面将重新加载。")
                st.rerun()
        else:
            st.warning("未找到 auto_train.py，无法自动训练模型")

        if enable_versioning:
            records = load_model_versions()
            if records:
                version_options = [f"{r['timestamp']} - {r['type']}" for r in records]
                selected_version = st.selectbox("加载历史模型", version_options)
                if st.button("加载选中模型"):
                    idx = version_options.index(selected_version)
                    record = records[idx]
                    if record['type'] == 'lstm':
                        model = load_model(record['path'])
                        st.session_state['loaded_lstm'] = model
                        st.success(f"已加载LSTM模型: {record['timestamp']}")
                    else:
                        st.warning("非LSTM模型暂不支持加载")

        st.markdown("---")
        generate_btn = st.button("🚀 生成AI推荐号码", type="primary")

    # 数据质量监控
    warnings_list = check_data_quality(history_df, check_continuity=st.session_state.enable_continuity_check)
    if warnings_list:
        for w in warnings_list:
            st.warning(f"数据质量警告: {w}")
    else:
        st.success("数据质量检查通过")

    # 标签页
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 数据概览", "🤖 AI预测", "🎲 推荐号码", "📈 资金模拟", "📋 历史数据", "🔵 蓝球专报"
    ])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("红球和值走势图")
            sum_fig = plot_sum_trend(history_df)
            st.pyplot(sum_fig, width='stretch')
        with col2:
            st.subheader("红球数据统计")
            red_cols = [f'red{i}' for i in range(1, 7)]
            all_red_balls = []
            for col in red_cols:
                all_red_balls.extend(history_df[col].tolist())
            red_counts = pd.Series(all_red_balls).value_counts().sort_index()
            red_stats_df = pd.DataFrame({
                '球号': range(RED_MIN, RED_MAX+1),
                '出现次数': [red_counts.get(i, 0) for i in range(RED_MIN, RED_MAX+1)],
                '出现概率': [red_counts.get(i, 0)/len(history_df) for i in range(RED_MIN, RED_MAX+1)]
            })
            st.dataframe(red_stats_df, width='stretch')

        st.markdown("---")
        st.subheader("蓝球分析")
        col3, col4 = st.columns(2)
        with col3:
            st.subheader("蓝球走势图")
            blue_trend_fig = plot_blue_trend(history_df)
            st.pyplot(blue_trend_fig, width='stretch')
        with col4:
            st.subheader("蓝球统计")
            blue_stats_df = get_blue_stats(history_df)
            st.dataframe(blue_stats_df, width='stretch')

    # ---------- 生成推荐 ----------
    recommended_notes = None
    ball_prob = None
    blue_prob = None

    if generate_btn:
        progress_bar = st.progress(0) if enable_progress else None
        status_text = st.empty() if enable_progress else None

        # 判断是否使用预训练模型
        pretrained_model = None
        use_pretrained = False

        if not force_retrain and os.path.exists('models/latest.h5'):
            with st.spinner("加载预训练模型..."):
                pretrained_model = load_pretrained_model()
                if pretrained_model is not None:
                    use_pretrained = True
                    st.success("已加载预训练模型，可直接预测。")
                else:
                    st.warning("预训练模型加载失败，将重新训练。")
                    use_pretrained = False

        if use_pretrained:
            with st.spinner("使用预训练模型预测概率..."):
                model_features = st.session_state.get('model_input_features', 33)
                model_timesteps = st.session_state.get('model_input_timesteps', 10)

                if model_features == 6:
                    data_matrix = preprocess_data_raw(history_df)
                    st.info("使用原始6维特征（红球号码）进行预测")
                elif model_features == 33:
                    data_matrix = preprocess_data(history_df)
                    st.info("使用33维one-hot特征进行预测")
                elif model_features == 71:
                    data_matrix = preprocess_data_enhanced(history_df)
                    st.info("使用71维增强特征进行预测")
                else:
                    st.error(f"不支持的模型输入特征数: {model_features}，请使用6、33或71维模型")
                    st.stop()

                scaler = MinMaxScaler(feature_range=(0, 1))
                data_scaled = scaler.fit_transform(data_matrix)
                look_back = model_timesteps
                X_seq = []
                for i in range(look_back, len(data_scaled)):
                    X_seq.append(data_scaled[i-look_back:i, :])
                if len(X_seq) == 0:
                    st.error("数据量不足以构造序列，请增加历史数据")
                    st.stop()
                X_seq = np.array(X_seq)
                last_sequence = X_seq[-1].reshape(1, X_seq.shape[1], X_seq.shape[2])
                lstm_prob = pretrained_model.predict(last_sequence, verbose=0)[0]
                ball_prob = np.asarray(lstm_prob).flatten()
                st.session_state['scaler'] = scaler
                st.session_state['look_back'] = look_back
                st.session_state['lstm_model'] = pretrained_model
        else:
            with st.spinner("训练AI模型..."):
                if use_enhanced:
                    data_matrix = preprocess_data_enhanced(history_df)
                    st.info(f"使用增强特征，特征维度: {data_matrix.shape[1]}")
                else:
                    data_matrix = preprocess_data(history_df)

                if len(data_matrix) > 10:
                    xgb_params = {
                        'n_estimators': xgb_n_estimators,
                        'max_depth': xgb_max_depth,
                        'learning_rate': xgb_learning_rate,
                        'reg_alpha': xgb_reg_alpha,
                        'reg_lambda': xgb_reg_lambda
                    }
                    lstm_model, scaler, X_seq, models_dict, history, stacker = train_models(
                        data_matrix,
                        look_back=10,
                        enhanced=use_enhanced,
                        use_xgboost=use_xgboost,
                        use_rf=use_rf,
                        use_lgb=use_lgb,
                        lstm_units=lstm_units,
                        num_layers=num_layers,
                        dropout=dropout,
                        lstm_weight=lstm_weight,
                        xgb_ensemble_size=xgb_ensemble_size,
                        xgb_params=xgb_params,
                        use_stacking=use_stacking,
                        learning_rate=learning_rate,
                        epochs=epochs,
                        patience=patience,
                        use_l2=use_l2,
                        progress_bar=progress_bar,
                        status_text=status_text
                    )
                    if history is not None:
                        st.session_state['training_history'] = history.history
                    else:
                        if 'training_history' in st.session_state:
                            del st.session_state['training_history']

                    if lstm_model is not None or models_dict:
                        ball_prob = predict_ball_probability(
                            lstm_model, scaler, X_seq,
                            models_dict=models_dict,
                            use_models=['xgb','rf','lgb'] if any([use_xgboost, use_rf, use_lgb]) else None,
                            weight_lstm=lstm_weight,
                            stacker=stacker,
                            base_models_list=None
                        )
                        st.session_state['lstm_model'] = lstm_model
                        st.session_state['models_dict'] = models_dict
                        st.session_state['scaler'] = scaler
                        st.session_state['look_back'] = 10

                        if lstm_model is not None:
                            os.makedirs('models', exist_ok=True)
                            lstm_model.save('models/latest.h5')
                            st.info("新模型已保存为 models/latest.h5，下次可直接加载。")

                        if enable_versioning:
                            if lstm_model is not None:
                                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                                model_path = f'models/lstm_{timestamp}.h5'
                                lstm_model.save(model_path)
                                params = {'units': lstm_units, 'layers': num_layers, 'dropout': dropout, 'learning_rate': learning_rate, 'use_l2': use_l2}
                                save_model_version('lstm', params, model_path)
                    else:
                        st.error("模型训练失败，数据不足？")
                        ball_prob = np.ones(33) / 33
                else:
                    ball_prob = np.ones(33) / 33
                    st.warning("历史数据不足，使用均匀概率分布！")

        blue_prob = train_blue_model(history_df)
        st.session_state['blue_prob'] = blue_prob

        if rec_method == '遗传算法优化':
            with st.spinner("蒙特卡洛模拟生成组合..."):
                mc_combinations = monte_carlo_generate(ball_prob, n=mc_count)
            with st.spinner("遗传算法优化..."):
                recommended_notes = genetic_algorithm(mc_combinations, generations=ga_gens)
        elif rec_method == '冷热号推荐':
            recommended_notes = generate_by_hot_cold(history_df, n=20)
        elif rec_method == '遗漏值推荐':
            recommended_notes = generate_by_missing(history_df, n=20)
        else:
            recommended_notes = generate_random(n=20)

        st.session_state['ball_prob'] = ball_prob
        st.session_state['recommended_notes'] = recommended_notes

        st.success("✅ AI推荐号码生成完成！")

    with tab2:
        if ball_prob is not None:
            st.subheader("红球概率热力图")
            heatmap_fig = plot_heatmap(ball_prob)
            st.pyplot(heatmap_fig, width='stretch')

            if 'training_history' in st.session_state:
                with st.expander("📉 查看训练曲线 (loss & val_loss)"):
                    history_dict = st.session_state['training_history']
                    loss_df = pd.DataFrame({
                        'epoch': range(1, len(history_dict['loss'])+1),
                        'loss': history_dict['loss'],
                        'val_loss': history_dict['val_loss']
                    })
                    st.line_chart(loss_df.set_index('epoch')[['loss', 'val_loss']])
            else:
                if use_xgboost and lstm_weight <= 0:
                    st.info("当前使用纯XGBoost模型，训练过程无loss曲线显示。")

            if enable_feature_importance and st.session_state.get('models_dict'):
                with st.expander("📊 特征重要性 (XGBoost)"):
                    feature_names = get_feature_names(enhanced=use_enhanced)
                    fig = plot_feature_importance(st.session_state['models_dict'], feature_names)
                    if fig:
                        st.pyplot(fig, width='stretch')

            if enable_3d_plot:
                with st.expander("📊 3D概率分布图"):
                    fig_3d = plot_3d_prob(ball_prob)
                    st.plotly_chart(fig_3d, width='stretch')

            if enable_shap and SHAP_AVAILABLE and 'models_dict' in st.session_state and 'xgb' in st.session_state['models_dict']:
                with st.expander("🔍 SHAP模型解释"):
                    if st.button("生成SHAP力图"):
                        feature_names = get_feature_names(enhanced=use_enhanced)
                        X_sample = st.session_state.get('X_seq')[-1].reshape(1, -1) if 'X_seq' in st.session_state else None
                        if X_sample is not None:
                            shap_fig = explain_with_shap(st.session_state['models_dict']['xgb'].__self__, X_sample, feature_names)
                            if shap_fig:
                                st.pyplot(shap_fig, width='stretch')

            st.subheader("红球概率排名")
            prob_ranking = pd.DataFrame({
                '红球号码': range(RED_MIN, RED_MAX+1),
                '出现概率': ball_prob,
                '排名': np.argsort(np.argsort(-ball_prob)) + 1
            }).sort_values('出现概率', ascending=False)
            st.dataframe(prob_ranking, width='stretch')

            if blue_prob is not None:
                st.subheader("蓝球概率分布")
                blue_df = pd.DataFrame({
                    '蓝球号码': range(BLUE_MIN, BLUE_MAX+1),
                    '出现概率': blue_prob
                }).sort_values('出现概率', ascending=False)
                st.dataframe(blue_df, width='stretch')
        else:
            st.info("请点击左侧的『生成AI推荐号码』按钮开始分析")

    with tab3:
        if recommended_notes is not None:
            st.subheader("AI推荐号码 (TOP 20)")
            notes_data = []
            for i, note in enumerate(recommended_notes, 1):
                red_str = ' '.join([f'{x:02d}' for x in note['red']])
                blue_str = f'{note["blue"]:02d}'
                score = calculate_score(note)
                notes_data.append({
                    '序号': i,
                    '红球': red_str,
                    '蓝球': blue_str,
                    '综合评分': round(score, 2)
                })
            notes_df = pd.DataFrame(notes_data)
            st.dataframe(notes_df, width='stretch')
            csv = notes_df.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="📥 下载推荐号码 (CSV)",
                data=csv,
                file_name="ai_lottery_recommendations.csv",
                mime="text/csv"
            )
        else:
            st.info("请点击左侧的『生成AI推荐号码』按钮获取推荐号码")

    with tab4:
        if recommended_notes is not None:
            strategy = st.selectbox("选择投注策略", ['fixed', 'martingale', 'stop_loss'])
            stop_loss_val = None
            if strategy == 'stop_loss':
                stop_loss_val = st.number_input("止损线 (元)", value=500, min_value=100, max_value=INITIAL_CAPITAL)
            if st.button("运行资金模拟"):
                with st.spinner("运行资金模拟..."):
                    backtest_result = backtest_simulation_strategy(
                        recommended_notes,
                        history_df,
                        n_periods=backtest_periods,
                        strategy=strategy,
                        stop_loss=stop_loss_val
                    )
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.subheader("模拟资金曲线")
                    capital_fig = plot_capital_curve(backtest_result['capital_curve'])
                    st.pyplot(capital_fig, width='stretch')
                with col2:
                    st.subheader("模拟结果统计")
                    st.metric("初始资金", f"¥{INITIAL_CAPITAL:,.0f}")
                    st.metric("最终资金", f"¥{backtest_result['final_capital']:,.0f}")
                    st.metric("总收益率", f"{backtest_result['total_return']:.2f}%")
                    st.metric("最大回撤", f"{backtest_result['max_drawdown']:.2f}%")
                    st.metric("总投注额", f"¥{backtest_result['total_bets']:,.0f}")
                    st.metric("总奖金", f"¥{backtest_result['total_bonus']:,.0f}")
        else:
            st.info("请先生成推荐号码后查看资金模拟结果")

    with tab5:
        st.subheader("历史开奖数据")
        st.dataframe(history_df, width='stretch')
        csv = history_df.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            label="📥 下载历史数据",
            data=csv,
            file_name="lottery_history.csv",
            mime="text/csv"
        )

    # ==================== 蓝球专报标签页 ====================
    with tab6:
        st.header("🔵 蓝球深度分析")
        st.markdown("本模块独立分析蓝球号码，展示多种预测方法的历史准确率，助您理性参考。")

        st.subheader("蓝球走势图")
        st.pyplot(plot_blue_trend(history_df), width='stretch')

        st.subheader("蓝球频率统计")
        st.dataframe(get_blue_stats(history_df), width='stretch')

        st.subheader("蓝球预测方法准确率对比（回测最近100期）")
        methods = {
            '频率法（全历史）': 'frequency',
            '热号法（最近50期）': 'hot',
            '冷号法（最近50期）': 'cold',
            '随机法': 'random'
        }
        accuracy_data = []
        for name, method in methods.items():
            acc = backtest_blue_accuracy(history_df, method=method, n_periods=100)
            accuracy_data.append({'方法': name, '准确率': acc * 100})
        acc_df = pd.DataFrame(accuracy_data)
        acc_df['准确率'] = acc_df['准确率'].round(2)

        col_acc1, col_acc2 = st.columns([1, 2])
        with col_acc1:
            st.dataframe(acc_df, width='stretch')
        with col_acc2:
            fig, ax = plt.subplots(figsize=(6, 4))
            bars = ax.barh(acc_df['方法'], acc_df['准确率'], color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
            ax.set_xlabel('准确率 (%)')
            ax.set_title('蓝球预测方法历史回测准确率')
            for bar in bars:
                width = bar.get_width()
                ax.text(width + 0.5, bar.get_y() + bar.get_height()/2, f'{width:.1f}%', va='center')
            st.pyplot(fig, width='stretch')

        st.subheader("🎯 当前蓝球推荐")
        st.markdown("**基于频率法的蓝球推荐**（历史出现次数最多）")
        current_blue = predict_blue_frequency(history_df, method='frequency')
        st.success(f"⭐ 推荐蓝球：**{current_blue:02d}**")
        st.caption("注：蓝球完全随机，任何预测方法均无法保证准确性，请理性参考。")

        st.info("""
        **说明**：
        - 准确率基于历史回测，使用前一期及以前的数据预测下一期，滚动计算。
        - 随机法准确率理论上约为 6.25%，实际回测可能略有波动。
        - 蓝球预测难度远高于红球，任何方法都无法稳定达到高准确率。
        - 本系统仅用于学习研究，不构成投注建议。
        """)

    st.markdown("---")
    st.warning("""
    ⚠️ 免责声明：
    1. 本系统仅为量化分析学习演示，不构成任何购彩建议
    2. 彩票开奖号码完全随机，任何预测都不具有科学依据
    3. 理性购彩，量力而行
    """)

if __name__ == "__main__":
    main()