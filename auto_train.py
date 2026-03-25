import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
import os
from datetime import datetime
import json

class LotteryAutoTrainer:
    def __init__(self, data_path='data/history.csv', model_dir='models'):
        self.data_path = data_path
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
    def load_data(self):
        df = pd.read_csv(self.data_path)
        red_cols = ['red1','red2','red3','red4','red5','red6']
        for col in red_cols + ['blue']:
            if col not in df.columns:
                raise ValueError(f"缺少必要的列: {col}")
        data = df[red_cols].values
        return data, df
    
    def prepare_sequences(self, data, seq_length=10):
        X, y = [], []
        for i in range(seq_length, len(data)):
            X.append(data[i-seq_length:i])
            y.append(data[i])
        return np.array(X), np.array(y)
    
    def build_model(self, seq_length, n_features):
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=(seq_length, n_features)),
            Dropout(0.2),
            LSTM(64, return_sequences=False),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(n_features)
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model
    
    def train(self, epochs=50, seq_length=10):
        print(f"[{datetime.now()}] 开始训练...")
        data, df = self.load_data()
        data = data / 33.0
        
        X, y = self.prepare_sequences(data, seq_length)
        if len(X) == 0:
            print("数据不足，无法训练")
            return None
        
        split = int(len(X) * 0.8)
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]
        
        model = self.build_model(seq_length, 6)
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=16,
            verbose=1,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
            ]
        )
        
        val_loss = min(history.history['val_loss'])
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = os.path.join(self.model_dir, f'model_{timestamp}_loss{val_loss:.4f}.h5')
        model.save(model_path)
        print(f"模型已保存至: {model_path}")
        
        # 更新 latest.h5
        latest_link = os.path.join(self.model_dir, 'latest.h5')
        if os.path.exists(latest_link):
            os.remove(latest_link)
        import shutil
        shutil.copy2(model_path, latest_link)
        print(f"已更新最新模型: {latest_link}")
        
        print(f"[{datetime.now()}] 训练完成")
        return model_path
    
    def auto_retrain_if_needed(self, check_interval_days=7):
        records_file = os.path.join(self.model_dir, 'training_records.json')
        if not os.path.exists(records_file):
            print("没有训练记录，开始首次训练...")
            return self.train()
        
        with open(records_file, 'r') as f:
            records = json.load(f)
        if not records:
            return self.train()
        
        last_train = datetime.strptime(records[-1]['timestamp'], '%Y%m%d_%H%M%S')
        days_since = (datetime.now() - last_train).days
        if days_since >= check_interval_days:
            print(f"上次训练是在 {days_since} 天前，开始重新训练...")
            return self.train()
        else:
            print(f"距离上次训练还有 {check_interval_days - days_since} 天，暂不训练")
            return None

if __name__ == "__main__":
    trainer = LotteryAutoTrainer()
    trainer.auto_retrain_if_needed(check_interval_days=0)  # 强制训练