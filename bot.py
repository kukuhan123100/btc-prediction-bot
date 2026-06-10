# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import ta
import joblib
import requests
from datetime import datetime, timedelta
from luno_python.client import Client

# ====================== Secrets ======================
TELEGRAM_TOKEN = "${{ secrets.TELEGRAM_TOKEN }}"
CHAT_ID = "${{ secrets.CHAT_ID }}"
API_KEY_ID = "${{ secrets.LUNO_API_KEY_ID }}"
API_KEY_SECRET = "${{ secrets.LUNO_API_KEY_SECRET }}"
# =====================================================

client = Client(api_key_id=API_KEY_ID, api_key_secret=API_KEY_SECRET)

# 加载两个模型
model_24h = joblib.load('best_xgb_24h_prediction.pkl')
model_4h  = joblib.load('best_xgb_optuna_top15_final.pkl')

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

        df_4h = all_dfs[14400].copy()
        df = df_4h.set_index('time')

        for dur in [3600, 28800, 86400, 259200, 604800]:
            if dur in all_dfs:
                if dur == 259200:
                    suffix = "_3d"
                elif dur == 604800:
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
    print(f"\n[{datetime.now()}] 开始双模型预测...")

    try:
        latest = get_latest_features()
        if latest is None:
            return

        # ===== 24h 模型 =====
        proba_24h = model_24h.predict_proba(latest)[0]
        pred_24h = model_24h.predict(latest)[0]
        dir_24h = "上涨" if pred_24h == 1 else "下跌/平"

        # ===== 4h 模型 =====
        proba_4h = model_4h.predict_proba(latest)[0]
        pred_4h = model_4h.predict(latest)[0]
        dir_4h = "上涨" if pred_4h == 1 else "下跌/平"

        message = (
            f"📊 <b>BTC/MYR 双模型预测</b>\n"
            f"时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
            f"【24小时预测】\n"
            f"方向：<b>{dir_24h}</b>（置信度 {round(max(proba_24h)*100,1)}%）\n"
            f"上涨概率：{round(proba_24h[1]*100,1)}%\n\n"
            f"【4小时预测】\n"
            f"方向：<b>{dir_4h}</b>（置信度 {round(max(proba_4h)*100,1)}%）\n"
            f"上涨概率：{round(proba_4h[1]*100,1)}%"
        )

        send_telegram(message)
        print("双模型预测完成，已发送 Telegram")

    except Exception as e:
        error_msg = f"❌ 预测出错：{str(e)}"
        print(error_msg)
        send_telegram(error_msg)


predict_and_notify()