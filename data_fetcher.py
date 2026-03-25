import requests
import pandas as pd
from datetime import datetime

def fetch_latest_data():
    """
    抓取最新的双色球开奖数据
    """
    url = "https://datachart.500.com/ssq/history/newinc/history.php"
    params = {'start': '24001', 'end': '26001'}
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

    try:
        print("正在抓取最新数据...")
        response = requests.get(url, params=params, headers=headers, timeout=15)
        response.encoding = 'utf-8'
        
        tables = pd.read_html(response.text, header=0)
        df_raw = tables[0]

        # 数据清洗：删除全为空的行
        df_raw = df_raw.dropna(how='all')

        if df_raw.shape[1] < 8:
            print("表格列数不足，可能页面结构变化")
            return None

        # 提取有效列
        df_clean = pd.DataFrame()
        df_clean['issue'] = df_raw.iloc[:, 0]          # 期号
        for i in range(6):
            df_clean[f'red{i+1}'] = df_raw.iloc[:, i+1]  # 红球1-6
        df_clean['blue'] = df_raw.iloc[:, 7]            # 蓝球
        if df_raw.shape[1] > 8:
            df_clean['date'] = df_raw.iloc[:, -1]

        # 转换数据类型，并过滤无效行
        for col in ['red1','red2','red3','red4','red5','red6','blue']:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        df_clean = df_clean.dropna(subset=['red1','red2','red3','red4','red5','red6','blue'])

        # 转换为整数
        for col in ['red1','red2','red3','red4','red5','red6','blue']:
            df_clean[col] = df_clean[col].astype(int)

        # 确保 issue 列为整数
        df_clean['issue'] = pd.to_numeric(df_clean['issue'], errors='coerce').astype('Int64')
        df_clean = df_clean.dropna(subset=['issue'])

        # 按期号降序排序
        df_clean = df_clean.sort_values('issue', ascending=False).reset_index(drop=True)

        print(f"成功抓取 {len(df_clean)} 条记录")
        return df_clean

    except Exception as e:
        print(f"抓取失败: {e}")
        return None

def update_history_csv():
    """更新 data/history.csv 文件"""
    new_data = fetch_latest_data()
    if new_data is None:
        return False

    try:
        existing = pd.read_csv('data/history.csv')
    except FileNotFoundError:
        existing = pd.DataFrame()

    # 统一数据类型为整数
    for df in [new_data, existing]:
        if not df.empty and 'issue' in df.columns:
            df['issue'] = pd.to_numeric(df['issue'], errors='coerce').astype('Int64')
        for col in ['red1','red2','red3','red4','red5','red6','blue']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')

    if not existing.empty:
        combined = pd.concat([existing, new_data]).drop_duplicates(
            subset=['issue'], keep='last'
        ).sort_values('issue', ascending=False).reset_index(drop=True)
    else:
        combined = new_data

    final_cols = ['red1','red2','red3','red4','red5','red6','blue']
    if 'issue' in combined.columns:
        final_cols = ['issue'] + final_cols

    combined = combined[final_cols]
    combined.to_csv('data/history.csv', index=False, encoding='utf-8-sig')
    print(f"更新完成！现有 {len(combined)} 条记录")
    return True

if __name__ == "__main__":
    update_history_csv()