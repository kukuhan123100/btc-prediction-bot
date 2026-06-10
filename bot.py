# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import ta
import joblib
import schedule
import time
import requests
from datetime import datetime, timedelta
from luno_python.client import Client

# ====================== 请修改这里 ======================
TELEGRAM_TOKEN = "8649270491:AAHGHf22Sk2VUoUzRehoSJZslL1YFW1dB1w"
CHAT_ID = "2074757970"

API_KEY_ID = "hpxstz48qndp3"
API_KEY_SECRET = "iqIfLA0fzUbLmQWd0RhlUTcCKAGqI4rxzHQCFmTbNwg"
# =======================================================

client = Client(api_key_id=API_KEY_ID, api_key_secret=API_KEY_SECRET)

model = joblib.load('best_xgb_24h_prediction.pkl')
top15_features = pd.read_csv('top15_features.csv')['feature'].tolist()


def send_telegram(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message, "parse_mode": "HTML"}
    try:
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        print("Telegram 发送失败:", e)


def add_indicators(df):
    df['returns'] = df['close'].pct_change()
    df['rsi_14'] = ta.momentum.rsi(df['close'], 14)
    df['sma_25'] = ta.trend.sma_indicator(df['close'], 25)
    df['macd'] = ta.trend.MACD(df['close']).macd()
    df['stoch_k'] = ta.momentum.stoch(df['high'], df['low'], df['close'], window=14)
    df['cci_20'] = ta.trend.cci(df['high'], df['low'], df['close'], window=20)
    return df


def get_latest_features():
    try:
        timeframes = {
            3600: 90,      # 1h
            14400: 180,    # 4h
            28800: 180,    # 8h
            86400: 365,    # 24h
            259200: 730,   # 3d
            604800: 1095   # 7d
        }

        all_dfs = {}
        for dur, days in timeframes.items():
            since = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
            candles = client.get_candles(pair="XBTMYR", since=since, duration=dur)
            df = pd.DataFrame(candles['candles'])
            df.rename(columns={'timestamp':'time','open':'open','high':'high',
                               'low':'low','close':'close','volume':'volume'}, inplace=True)
            df['time'] = pd.to_datetime(df['time'], unit='ms')
            df = df.sort_values('time').reset_index(drop=True)
            df[['open','high','low','close','volume']] = df[['open','high','low','close','volume']].astype(float)
            df = add_indicators(df)
            all_dfs[dur] = df

        # 以 4h 为基准
        df_4h = all_dfs[14400].copy()
        df = df_4h.set_index('time')

        # 重采样（包含 3d 和 7d）
        for dur in [3600, 28800, 86400, 259200, 604800]:
            if dur in all_dfs:
                if dur == 259200:      # 3d
                    suffix = "_3d"
                elif dur == 604800:    # 7d
                    suffix = "_7d"
                else:
                    suffix = f"_{dur//3600}h"

                res = all_dfs[dur].set_index('time').resample('4h').agg({
                    'rsi_14': 'mean',
                    'returns': 'mean',
                    'sma_25': 'last',
                    'macd': 'last',
                    'stoch_k': 'mean',
                    'cci_20': 'mean'
                }).add_suffix(suffix)
                df = df.join(res, how='left')

        df = df.ffill().bfill().dropna().reset_index()

        missing = [col for col in top15_features if col not in df.columns]
        if missing:
            print("缺少以下特征列:", missing)
            return None

        return df.iloc[-1:][top15_features]

    except Exception as e:
        print("获取数据失败:", e)
        return None


def predict_and_notify():
    print(f"\n[{datetime.now()}] 开始预测...")
    try:
        latest = get_latest_features()
        if latest is None:
            return

        proba = model.predict_proba(latest)[0]
        pred = model.predict(latest)[0]

        direction = "上涨" if pred == 1 else "下跌/平"
        confidence = round(max(proba) * 100, 1)

        message = (
            f"📊 <b>BTC/MYR 24小时预测</b>\n"
            f"时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
            f"预测方向：<b>{direction}</b>\n"
            f"置信度：<b>{confidence}%</b>\n\n"
            f"上涨概率：{round(proba[1]*100, 1)}%\n"
            f"下跌概率：{round(proba[0]*100, 1)}%"
        )

        send_telegram(message)
        print("预测完成，已发送 Telegram")

    except Exception as e:
        error_msg = f"❌ 预测出错：{str(e)}"
        print(error_msg)
        send_telegram(error_msg)


# ====================== 定时任务 ======================
schedule.every(4).hours.do(predict_and_notify)

print("✅ 24小时预测 Bot 已启动！每 4 小时运行一次")
predict_and_notify()

while True:
    schedule.run_pending()
    time.sleep(60)