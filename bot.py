"""
Bybit ФЬЮЧЕРСНЫЙ бот (Linear USDT Perpetual)
============================================
- Без мартингейла
- Размер позиции = f(score) — чем выше скор, тем больше риск
- Поддержка/сопротивление учитываются при входе и постановке TP/SL
- Bybit AI сигнал (опционально) корректирует скор
- TP и SL ставятся нативно в ордере (futures поддерживают!)
- Только LINEAR рынок (USDT perpetual) — нет проблем с setLeverage

ИСПРАВЛЕНИЯ v2:
- S/R: кластеризация уровней (группировка пиков в ±0.5%), убраны шумовые уровни
- S/R: минимум 3 касания для near_support / near_resistance (было 2)
- SR_PROXIMITY_PCT: 0.3% (было 0.5%) — только реально близкие уровни
- Фильтр near_resistance в главном цикле: блокирует только dist < 0.15% (было любое)
- S/R: анализируем только последние SR_PERIOD свечей (не весь датафрейм)
"""

import os
import time
import json
import logging
import requests
import ccxt
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

# ================== НАСТРОЙКИ ==================
SYMBOLS = [
    # ── Мегакэп / высокая ликвидность ──────────────────────────
    "BTC/USDT:USDT",   "ETH/USDT:USDT",   "BNB/USDT:USDT",
    "XRP/USDT:USDT",   "SOL/USDT:USDT",   "ADA/USDT:USDT",
    "TRX/USDT:USDT",   "TON/USDT:USDT",   "AVAX/USDT:USDT",
    "DOT/USDT:USDT",   "LTC/USDT:USDT",   "BCH/USDT:USDT",
    "ATOM/USDT:USDT",  "XLM/USDT:USDT",   "NEAR/USDT:USDT",

    # ── Мемкоины (высокая волатильность) ───────────────────────
    "DOGE/USDT:USDT",  "SHIB/USDT:USDT",  "PEPE/USDT:USDT",
    "FLOKI/USDT:USDT", "BONK/USDT:USDT",  "WIF/USDT:USDT",
    "MEME/USDT:USDT",  "BOME/USDT:USDT",  "DOGS/USDT:USDT",
    "NEIRO/USDT:USDT", "PNUT/USDT:USDT",  "ACT/USDT:USDT",
    "POPCAT/USDT:USDT","TURBO/USDT:USDT", "BRETT/USDT:USDT",

    # ── AI / DePIN нарратив ─────────────────────────────────────
    "FET/USDT:USDT",   "RENDER/USDT:USDT","TAO/USDT:USDT",
    "WLD/USDT:USDT",   "ARKM/USDT:USDT",  "AGIX/USDT:USDT",
    "IO/USDT:USDT",    "ONDO/USDT:USDT",  "VIRTUAL/USDT:USDT",
    "AI16Z/USDT:USDT",

    # ── DeFi / DEX ──────────────────────────────────────────────
    "UNI/USDT:USDT",   "AAVE/USDT:USDT",  "CRV/USDT:USDT",
    "DYDX/USDT:USDT",  "JUP/USDT:USDT",   "PENDLE/USDT:USDT",
    "GMX/USDT:USDT",   "LDO/USDT:USDT",

    # ── L2 / экосистема ─────────────────────────────────────────
    "ARB/USDT:USDT",   "OP/USDT:USDT",    "MATIC/USDT:USDT",
    "STX/USDT:USDT",   "IMX/USDT:USDT",   "STRK/USDT:USDT",
    "ZK/USDT:USDT",    "MANTA/USDT:USDT",

    # ── Gaming / NFT / Metaverse ────────────────────────────────
    "AXS/USDT:USDT",   "SAND/USDT:USDT",  "MANA/USDT:USDT",
    "GALA/USDT:USDT",  "ENJ/USDT:USDT",   "ILV/USDT:USDT",
    "PIXEL/USDT:USDT", "PORTAL/USDT:USDT",

    # ── Инфраструктура / прочее ─────────────────────────────────
    "LINK/USDT:USDT",  "GRT/USDT:USDT",   "FIL/USDT:USDT",
    "ICP/USDT:USDT",   "RUNE/USDT:USDT",  "INJ/USDT:USDT",
    "SUI/USDT:USDT",   "APT/USDT:USDT",   "SEI/USDT:USDT",
    "TIA/USDT:USDT",   "PYTH/USDT:USDT",  "JTO/USDT:USDT",
    "W/USDT:USDT",     "ENA/USDT:USDT",   "EIGEN/USDT:USDT",
    "HBAR/USDT:USDT",  "VET/USDT:USDT",   "ALGO/USDT:USDT",
    "IOTA/USDT:USDT",  "EOS/USDT:USDT",   "XTZ/USDT:USDT",
    "THETA/USDT:USDT", "FLOW/USDT:USDT",  "KSM/USDT:USDT",
    "CHZ/USDT:USDT",   "MASK/USDT:USDT",  "1INCH/USDT:USDT",
    "COMP/USDT:USDT",  "ZRO/USDT:USDT",   "NOT/USDT:USDT",
    "HMSTR/USDT:USDT", "CATI/USDT:USDT",
]

LEVERAGE           = 3           # плечо (рекомендую 2-3x для консервативной работы)
TIMEFRAME_TA       = "5m"
TIMEFRAME_TREND    = "1h"
TIMEFRAME_MID      = "15m"
SCAN_INTERVAL      = 300         # секунд между сканированиями
MIN_SCORE          = 55          # порог входа

# Риск: базовый % от баланса на сделку (без учёта скора)
BASE_RISK_PCT      = 1.5         # 1.5% баланса при скоре=55
MAX_RISK_PCT       = 4.0         # 4.0% при скоре=100
# Формула: risk_pct = BASE + (MAX - BASE) * (score - MIN) / (100 - MIN)

TP_PERCENT         = 1.5         # % движения в прибыль (на фьючерсах с плечом реальнее)
SL_PERCENT         = 1.0         # % стоп-лосс
TRADE_MAX_LIFETIME = 7200        # 2 часа максимум в позиции

# ── Трейлинг после безубытка ──────────────────────────────────
TRAILING_STEP_PCT   = 0.3        # подтягиваем SL каждые 0.3% роста цены
TRAILING_OFFSET_PCT = 0.4        # SL на 0.4% ниже пика

MIN_BALANCE        = 5.0
REPORT_INTERVAL    = 1800
STATE_FILE         = "state_futures.json"

# Bybit комиссия на фьючерсах (taker)
BYBIT_FEE          = 0.00055     # 0.055% taker

# Уровни поддержки/сопротивления
SR_PERIOD          = 100         # свечей для поиска S/R (только последние N)
SR_PROXIMITY_PCT   = 0.3         # ИСПРАВЛЕНО: было 0.5% → 0.3% (строже)
SR_MIN_TOUCHES     = 3           # ИСПРАВЛЕНО: было 2 → минимум 3 касания для "реального" уровня
SR_CLUSTER_TOL     = 0.005       # 0.5% — допуск кластеризации (уровни в ±0.5% = один кластер)
SR_BLOCK_DIST_PCT  = 0.15        # блокируем вход только если resistance ближе этого % (было любое)

# ================== ЛОГИРОВАНИЕ ==================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%d.%m.%Y %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("bot_futures.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# ================== БИРЖА (LINEAR FUTURES) ==================
exchange = ccxt.bybit({
    "apiKey":          os.getenv("BYBIT_API_KEY"),
    "secret":          os.getenv("BYBIT_API_SECRET"),
    "enableRateLimit": True,
    "options": {
        "defaultType": "linear",   # ← КЛЮЧЕВОЕ: linear = USDT perpetual futures
    },
})

# ================== СТАТИСТИКА ==================
TRADES_FILE    = "trades_history.json"
ANALYTICS_FILE = "analytics_report.json"

stats = {
    "запусков":        0,
    "сделок_всего":    0,
    "тейкпрофит":      0,
    "стоплосс":        0,
    "таймаут":         0,
    "прибыль_usdt":    0.0,
    "убыток_usdt":     0.0,
    "депозит_старт":   0.0,
    "старт_время":     "",
    "последний_отчёт": 0.0,
}

# ================== ЖУРНАЛ СДЕЛОК ==================
def загрузить_историю() -> list:
    if not os.path.exists(TRADES_FILE):
        return []
    try:
        with open(TRADES_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return []

def сохранить_сделку(запись: dict):
    история = загрузить_историю()
    история.append(запись)
    try:
        with open(TRADES_FILE, "w", encoding="utf-8") as f:
            json.dump(история, f, ensure_ascii=False, indent=2)
    except Exception as e:
        log.warning(f"Не удалось сохранить сделку в журнал: {e}")

# ================== ПОСТ-ТРЕЙД АНАЛИЗ ==================
def пост_трейд_анализ(запись: dict):
    r      = запись["результат"]
    sym    = запись["symbol"]
    score  = запись["score"]
    det    = запись.get("details", {})
    pnl    = запись.get("pnl_usdt", 0)
    dur    = запись.get("duration_min", 0)
    знак   = "✅" if r == "tp" else ("❌" if r == "sl" else "⏰")

    log.info("")
    log.info("━" * 60)
    log.info(f"  📋 ПОСТ-ТРЕЙД АНАЛИЗ: {sym.split(':')[0]}")
    log.info(f"  Результат: {знак} {r.upper()}   P&L: {pnl:+.4f} USDT   "
             f"Длительность: {dur:.1f} мин")
    log.info(f"  Скор входа: {score}/100")
    log.info("  ─── Показания индикаторов на входе ───────────────────")

    индикаторы = [
        ("RSI 5m",       det.get("rsi", "?"),          lambda v: 25 <= float(v) <= 42 if str(v).replace('.','').lstrip('-').isdigit() else False),
        ("RSI 1h",       det.get("rsi_1h", "?"),        lambda v: float(v) < 55 if str(v).replace('.','').lstrip('-').isdigit() else False),
        ("MACD",         det.get("macd", "?"),           lambda v: v == "бычий"),
        ("Range Filter", det.get("range_filter", "?"),   lambda v: v == "вверх"),
        ("Supertrend 5m",det.get("supertrend", "?"),     lambda v: v == "вверх"),
        ("Supertrend 15m",det.get("supertrend_15m","?"), lambda v: v == "вверх"),
        ("Hull MA",      det.get("hull", "?"),           lambda v: v == "вверх"),
        ("Тренд 1h",     det.get("тренд_1h", "?"),       lambda v: v == "бычий"),
        ("Тренд 15m",    det.get("тренд_15m", "?"),      lambda v: v == "бычий"),
        ("ADX",          det.get("adx", "?"),            lambda v: float(v) > 25 if str(v).replace('.','').lstrip('-').isdigit() else False),
        ("Stoch K",      det.get("stoch_k", "?"),        lambda v: float(v) < 25 if str(v).replace('.','').lstrip('-').isdigit() else False),
        ("Объём ratio",  det.get("объём_ratio", "?"),    lambda v: float(v) > 1.5 if str(v).replace('.','').lstrip('-').isdigit() else False),
        ("VWAP dev%",    det.get("vwap_dev", "?"),       lambda v: -3 <= float(v) <= -0.3 if str(v).replace('.','').lstrip('-').isdigit() else False),
        ("SR сигнал",    det.get("sr_signal", "?"),      lambda v: "поддержки" in str(v)),
    ]

    for название, значение, хороший in индикаторы:
        try:
            хор = хороший(значение)
        except:
            хор = False
        иконка = "🟢" if хор else "🔴"
        log.info(f"    {иконка}  {название:16s} = {значение}")

    if det.get("свечи_3red"):
        log.info("    🔴  Штраф: 3 красных свечи подряд")

    log.info("━" * 60)
    log.info("")

# ================== СТАТИСТИКА ПО ИНСТРУМЕНТАМ ==================
def аналитика_по_инструментам():
    история = загрузить_историю()
    if len(история) < 2:
        log.info("  📊 Аналитика: недостаточно сделок (нужно минимум 2)")
        return

    по_символам: dict = {}
    for сд in история:
        sym = сд.get("symbol", "?").split(":")[0]
        if sym not in по_символам:
            по_символам[sym] = {"tp": 0, "sl": 0, "timeout": 0,
                                 "pnl": 0.0, "scores": [], "dur": []}
        r = сд.get("результат", "")
        if r == "tp":
            по_символам[sym]["tp"] += 1
        elif r == "sl":
            по_символам[sym]["sl"] += 1
        else:
            по_символам[sym]["timeout"] += 1
        по_символам[sym]["pnl"]    += сд.get("pnl_usdt", 0)
        по_символам[sym]["scores"].append(сд.get("score", 0))
        по_символам[sym]["dur"].append(сд.get("duration_min", 0))

    индик_стат: dict = {}
    инд_ключи = ["rsi", "rsi_1h", "macd", "range_filter", "supertrend",
                  "supertrend_15m", "hull", "тренд_1h", "тренд_15m",
                  "adx", "stoch_k", "объём_ratio", "vwap_dev", "sr_signal"]

    for сд in история:
        det = сд.get("details", {})
        победа = 1 if сд.get("результат") == "tp" else 0
        for к in инд_ключи:
            if к not in индик_стат:
                индик_стат[к] = {"bull_wins": 0, "bull_total": 0,
                                   "bear_wins": 0, "bear_total": 0}
            v = det.get(к)
            if v is None:
                continue
            бычий = False
            try:
                if к == "rsi":        бычий = 25 <= float(v) <= 42
                elif к == "rsi_1h":   бычий = float(v) < 55
                elif к == "macd":     бычий = v == "бычий"
                elif к in ("range_filter",): бычий = v == "вверх"
                elif к in ("supertrend", "supertrend_15m", "hull"): бычий = v == "вверх"
                elif к in ("тренд_1h", "тренд_15m"): бычий = v == "бычий"
                elif к == "adx":      бычий = float(v) > 25
                elif к == "stoch_k":  бычий = float(v) < 25
                elif к == "объём_ratio": бычий = float(v) > 1.5
                elif к == "vwap_dev": бычий = -3 <= float(v) <= -0.3
                elif к == "sr_signal": бычий = "поддержки" in str(v)
            except:
                pass

            if бычий:
                индик_стат[к]["bull_total"] += 1
                индик_стат[к]["bull_wins"]  += победа
            else:
                индик_стат[к]["bear_total"] += 1
                индик_стат[к]["bear_wins"]  += победа

    log.info("")
    log.info("=" * 70)
    log.info("  📊  АНАЛИТИКА ПО ИНСТРУМЕНТАМ")
    log.info(f"  Всего сделок в истории: {len(история)}")
    log.info("  ─" * 35)
    log.info(f"  {'Символ':<16} {'Сделок':>6}  {'TP':>4}  {'SL':>4}  "
             f"{'WR%':>6}  {'P&L':>8}  {'Ср.скор':>7}  {'Ср.мин':>7}")
    log.info("  " + "─" * 65)

    сорт = sorted(по_символам.items(), key=lambda x: x[1]["pnl"], reverse=True)
    for sym, d in сорт:
        всего = d["tp"] + d["sl"] + d["timeout"]
        wr    = d["tp"] / всего * 100 if всего > 0 else 0
        ср_sc = sum(d["scores"]) / len(d["scores"]) if d["scores"] else 0
        ср_мин= sum(d["dur"])    / len(d["dur"])    if d["dur"]    else 0
        знак  = "+" if d["pnl"] >= 0 else ""
        log.info(
            f"  {sym:<16} {всего:>6}  {d['tp']:>4}  {d['sl']:>4}  "
            f"{wr:>5.1f}%  {знак}{d['pnl']:>7.4f}U  "
            f"{ср_sc:>6.1f}  {ср_мин:>6.1f}м"
        )

    log.info("")
    log.info("  📈  ЭФФЕКТИВНОСТЬ ИНДИКАТОРОВ")
    log.info(f"  {'Индикатор':<18} {'🟢Бычий WR%':>11}  {'n':>4}  "
             f"{'🔴Медвежий WR%':>14}  {'n':>4}  {'Разница':>8}")
    log.info("  " + "─" * 65)

    инд_сорт = sorted(
        индик_стат.items(),
        key=lambda x: (
            x[1]["bull_wins"] / x[1]["bull_total"] if x[1]["bull_total"] > 0 else 0
        ) - (
            x[1]["bear_wins"] / x[1]["bear_total"] if x[1]["bear_total"] > 0 else 0
        ),
        reverse=True
    )
    for инд, d in инд_сорт:
        b_wr = d["bull_wins"] / d["bull_total"] * 100 if d["bull_total"] > 0 else 0
        r_wr = d["bear_wins"] / d["bear_total"] * 100 if d["bear_total"] > 0 else 0
        diff = b_wr - r_wr
        знак = "▲" if diff > 5 else ("▼" if diff < -5 else "≈")
        log.info(
            f"  {инд:<18}  {b_wr:>9.1f}%  {d['bull_total']:>4}  "
            f"{r_wr:>12.1f}%  {d['bear_total']:>4}  {знак}{diff:>+7.1f}%"
        )

    log.info("")
    log.info("  💡 РЕКОМЕНДАЦИИ (на основе истории):")
    for инд, d in инд_сорт[:3]:
        b_wr = d["bull_wins"] / d["bull_total"] * 100 if d["bull_total"] > 0 else 0
        r_wr = d["bear_wins"] / d["bear_total"] * 100 if d["bear_total"] > 0 else 0
        if b_wr - r_wr > 10 and d["bull_total"] >= 3:
            log.info(f"    ✅ {инд}: бычий сигнал даёт WR {b_wr:.1f}% — важный фильтр")
    for инд, d in list(reversed(инд_сорт))[:2]:
        b_wr = d["bull_wins"] / d["bull_total"] * 100 if d["bull_total"] > 0 else 0
        r_wr = d["bear_wins"] / d["bear_total"] * 100 if d["bear_total"] > 0 else 0
        if r_wr - b_wr > 10 and d["bear_total"] >= 3:
            log.info(f"    ⚠️  {инд}: слабый предиктор — рассмотреть снижение веса")

    log.info("=" * 70)
    log.info("")

    отчёт = {
        "сформирован":      datetime.now().strftime("%d.%m.%Y %H:%M:%S"),
        "всего_сделок":     len(история),
        "по_символам":      {k: {kk: vv for kk, vv in v.items() if kk != "scores" and kk != "dur"}
                              for k, v in по_символам.items()},
        "эффективность_индикаторов": {
            к: {
                "bull_winrate_pct": round(d["bull_wins"]/d["bull_total"]*100, 1) if d["bull_total"] > 0 else None,
                "bull_n":    d["bull_total"],
                "bear_winrate_pct": round(d["bear_wins"]/d["bear_total"]*100, 1) if d["bear_total"] > 0 else None,
                "bear_n":    d["bear_total"],
            }
            for к, d in индик_стат.items()
        },
    }
    try:
        with open(ANALYTICS_FILE, "w", encoding="utf-8") as f:
            json.dump(отчёт, f, ensure_ascii=False, indent=2)
        log.info(f"  💾 Отчёт сохранён в {ANALYTICS_FILE}")
    except Exception as e:
        log.warning(f"  Не удалось сохранить аналитику: {e}")

def сохранить_состояние():
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
    except Exception as e:
        log.warning(f"Не удалось сохранить состояние: {e}")

def загрузить_состояние():
    global stats
    if not os.path.exists(STATE_FILE):
        return False
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            saved = json.load(f)
        for key in stats:
            if key in saved:
                stats[key] = saved[key]
        log.info(f"Состояние восстановлено из {STATE_FILE}")
        return True
    except Exception as e:
        log.warning(f"Не удалось загрузить состояние: {e}")
        return False

# ================== БАЛАНС И ПОЗИЦИИ ==================
def баланс_usdt() -> float:
    try:
        b = exchange.fetch_balance({"type": "linear"})
        return float(b.get("USDT", {}).get("free", 0.0))
    except Exception as e:
        log.warning(f"Ошибка получения баланса: {e}")
        return 0.0

def получить_позиции() -> list:
    try:
        positions = exchange.fetch_positions()
        return [p for p in positions if float(p.get("contracts", 0) or 0) > 0]
    except Exception as e:
        log.warning(f"Ошибка получения позиций: {e}")
        return []

def закрыть_все_позиции():
    log.info("  🔒 Закрытие всех открытых позиций...")
    try:
        positions = получить_позиции()
        for pos in positions:
            sym = pos["symbol"]
            side = pos["side"]
            qty  = abs(float(pos["contracts"] or 0))
            if qty <= 0:
                continue
            close_side = "sell" if side == "long" else "buy"
            try:
                exchange.create_market_order(sym, close_side, qty, params={"reduceOnly": True})
                log.info(f"    Закрыта позиция {sym} {side} qty={qty}")
            except Exception as e:
                log.warning(f"    Не удалось закрыть {sym}: {e}")
    except Exception as e:
        log.warning(f"  Ошибка закрытия позиций: {e}")

def отменить_все_ордера():
    log.info("  🗑️  Отмена всех открытых ордеров...")
    try:
        orders = exchange.fetch_open_orders()
        for o in orders:
            try:
                exchange.cancel_order(o["id"], o["symbol"])
            except Exception as e:
                log.warning(f"    Не удалось отменить ордер {o['id']}: {e}")
    except Exception as e:
        log.warning(f"  Ошибка отмены ордеров: {e}")

def установить_плечо(symbol: str, leverage: int):
    try:
        exchange.set_leverage(leverage, symbol, params={"buyLeverage": leverage, "sellLeverage": leverage})
    except Exception as e:
        log.warning(f"  Не удалось установить плечо {symbol}: {e}")

# ================== ИНДИКАТОРЫ ==================
def _ema(s, span):
    return s.ewm(span=span, adjust=False).mean()

def _rma(s, span):
    return s.ewm(alpha=1/span, adjust=False).mean()

def calc_rsi(close, period=14):
    d = close.diff()
    gain = d.clip(lower=0)
    loss = (-d).clip(lower=0)
    avg_g = _rma(gain, period)
    avg_l = _rma(loss, period)
    rs = avg_g / avg_l.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def calc_macd(close, fast=12, slow=26, signal=9):
    ml = _ema(close, fast) - _ema(close, slow)
    sl = _ema(ml, signal)
    return ml, sl, ml - sl

def calc_atr(df, period=14):
    hi, lo, pc = df["h"], df["l"], df["c"].shift(1)
    tr = pd.concat([hi - lo, (hi - pc).abs(), (lo - pc).abs()], axis=1).max(axis=1)
    return _rma(tr, period)

def calc_supertrend(df, period=10, mult=3.0):
    atr = calc_atr(df, period)
    hl2 = (df["h"] + df["l"]) / 2
    ub = (hl2 + mult * atr).copy()
    lb = (hl2 - mult * atr).copy()
    trend = pd.Series(1, index=df.index)
    for i in range(1, len(df)):
        c, pc = df["c"].iloc[i], df["c"].iloc[i-1]
        pu, pl, pt = ub.iloc[i-1], lb.iloc[i-1], trend.iloc[i-1]
        ub.iloc[i] = ub.iloc[i] if ub.iloc[i] < pu or pc > pu else pu
        lb.iloc[i] = lb.iloc[i] if lb.iloc[i] > pl or pc < pl else pl
        if pt == 1 and c < lb.iloc[i]:
            trend.iloc[i] = -1
        elif pt == -1 and c > ub.iloc[i]:
            trend.iloc[i] = 1
        else:
            trend.iloc[i] = pt
    return trend == 1, trend == -1

def calc_stochastic(df, k=14, d=3, smooth=3):
    lo = df["l"].rolling(k).min()
    hi = df["h"].rolling(k).max()
    ks = (100 * (df["c"] - lo) / (hi - lo + 1e-10)).rolling(smooth).mean()
    return ks, ks.rolling(d).mean()

def calc_hull(close, period=55):
    hma = _ema(2 * _ema(close, period//2) - _ema(close, period), int(np.sqrt(period)))
    return hma > hma.shift(2), hma < hma.shift(2)

def calc_adx(df, period=14):
    atr = calc_atr(df, period)
    pdm = (df["h"] - df["h"].shift(1)).clip(lower=0)
    mdm = (df["l"].shift(1) - df["l"]).clip(lower=0)
    pdm = pdm.where(pdm >= mdm, 0)
    mdm = mdm.where(mdm >= pdm, 0)
    pdi = 100 * _rma(pdm, period) / atr.replace(0, np.nan)
    mdi = 100 * _rma(mdm, period) / atr.replace(0, np.nan)
    adx = _rma(100 * (pdi - mdi).abs() / (pdi + mdi + 1e-10), period)
    return adx, pdi, mdi

def calc_vwap_deviation(df, period=20):
    typical = (df["h"] + df["l"] + df["c"]) / 3
    vwap = (typical * df["v"]).rolling(period).sum() / df["v"].rolling(period).sum()
    return (df["c"] - vwap) / vwap * 100

def calc_range_filter(df, period=200, qty=3.0):
    close = df["c"]
    rng = qty * calc_atr(df, period)
    filt = close.copy()
    for i in range(1, len(close)):
        c, r, pf = close.iloc[i], rng.iloc[i], filt.iloc[i-1]
        if c - r > pf:
            filt.iloc[i] = c - r
        elif c + r < pf:
            filt.iloc[i] = c + r
        else:
            filt.iloc[i] = pf
    up   = (filt > filt.shift(1)) & (close > filt)
    down = (filt < filt.shift(1)) & (close < filt)
    return filt, filt + rng, filt - rng, up, down

# ================== УРОВНИ ПОДДЕРЖКИ/СОПРОТИВЛЕНИЯ ==================
def _кластеризовать_уровни(levels: list, tolerance: float = SR_CLUSTER_TOL) -> list:
    """
    Группирует близкие уровни в кластеры.
    Возвращает список (средняя_цена, кол-во_касаний).
    Уровни в пределах tolerance (0.5%) считаются одним кластером.
    """
    if not levels:
        return []
    levels = sorted(levels)
    кластеры = []
    текущий = [levels[0]]
    for lvl in levels[1:]:
        # Если уровень близко к первому в текущем кластере — добавляем
        if (lvl - текущий[0]) / (текущий[0] + 1e-10) < tolerance:
            текущий.append(lvl)
        else:
            кластеры.append((float(np.mean(текущий)), len(текущий)))
            текущий = [lvl]
    кластеры.append((float(np.mean(текущий)), len(текущий)))
    return кластеры  # [(цена, касания), ...]


def calc_support_resistance(df: pd.DataFrame, period: int = SR_PERIOD) -> dict:
    """
    ИСПРАВЛЕННЫЙ алгоритм S/R:
    1. Берём только последние `period` свечей (не весь датафрейм)
    2. Находим локальные максимумы/минимумы (пики)
    3. Кластеризуем пики в пределах SR_CLUSTER_TOL (0.5%)
    4. near_support / near_resistance — только если кластер >= SR_MIN_TOUCHES (3)
       И расстояние < SR_PROXIMITY_PCT (0.3%)

    Это устраняет ложные срабатывания когда алгоритм находил
    случайный единичный пик рядом с ценой и считал его "поддержкой".
    """
    # Берём только последние period свечей
    df_sr  = df.tail(period).reset_index(drop=True)
    highs  = df_sr["h"].values
    lows   = df_sr["l"].values
    close  = float(df["c"].iloc[-1])

    raw_resistances = []
    raw_supports    = []

    # Ищем локальные пики (окно ±2 свечи)
    for i in range(2, len(highs) - 2):
        if (highs[i] > highs[i-1] and highs[i] > highs[i-2] and
                highs[i] > highs[i+1] and highs[i] > highs[i+2]):
            raw_resistances.append(highs[i])
        if (lows[i] < lows[i-1] and lows[i] < lows[i-2] and
                lows[i] < lows[i+1] and lows[i] < lows[i+2]):
            raw_supports.append(lows[i])

    # Кластеризуем
    res_clusters = _кластеризовать_уровни(raw_resistances, SR_CLUSTER_TOL)
    sup_clusters = _кластеризовать_уровни(raw_supports,    SR_CLUSTER_TOL)

    # Фильтруем: только уровни выше/ниже текущей цены
    res_above = [(p, n) for p, n in res_clusters if p > close]
    sup_below = [(p, n) for p, n in sup_clusters if p < close]

    # Ближайшие уровни (независимо от числа касаний — для TP/SL расчёта)
    res_above_sorted = sorted(res_above, key=lambda x: x[0])
    sup_below_sorted = sorted(sup_below, key=lambda x: x[0], reverse=True)

    nearest_resistance, res_cluster = res_above_sorted[0] if res_above_sorted else (close * 1.05, 0)
    nearest_support,    sup_cluster = sup_below_sorted[0] if sup_below_sorted else (close * 0.95, 0)

    dist_to_res = (nearest_resistance - close) / close * 100
    dist_to_sup = (close - nearest_support)    / close * 100

    # СТРОГИЕ условия near_*:
    # 1. Расстояние < SR_PROXIMITY_PCT (0.3%)
    # 2. Кластер >= SR_MIN_TOUCHES (3 касания) — исключаем случайные пики
    near_support    = dist_to_sup < SR_PROXIMITY_PCT and sup_cluster >= SR_MIN_TOUCHES
    near_resistance = dist_to_res < SR_PROXIMITY_PCT and res_cluster >= SR_MIN_TOUCHES

    log.debug(
        f"  S/R: sup={nearest_support:.6f}({sup_cluster}hits, {dist_to_sup:.3f}%) "
        f"res={nearest_resistance:.6f}({res_cluster}hits, {dist_to_res:.3f}%) "
        f"near_sup={near_support} near_res={near_resistance}"
    )

    return {
        "support":         round(nearest_support,    10),
        "resistance":      round(nearest_resistance, 10),
        "dist_to_sup_pct": round(dist_to_sup, 2),
        "dist_to_res_pct": round(dist_to_res, 2),
        "sup_cluster":     sup_cluster,
        "res_cluster":     res_cluster,
        "near_support":    near_support,
        "near_resistance": near_resistance,
    }

# ================== BYBIT AI СИГНАЛ ==================
def получить_bybit_ai(symbol: str) -> dict:
    result = {"signal": "neutral", "long_ratio": 0.5, "short_ratio": 0.5, "available": False}
    try:
        coin = symbol.split("/")[0]
        url = (
            f"https://api.bybit.com/v5/market/account-ratio"
            f"?category=linear&symbol={coin}USDT&period=1h&limit=1"
        )
        resp = requests.get(url, timeout=5)
        data = resp.json()
        if data.get("retCode") == 0:
            items = data.get("result", {}).get("list", [])
            if items:
                buy_ratio  = float(items[0].get("buyRatio", 0.5))
                sell_ratio = float(items[0].get("sellRatio", 0.5))
                result["long_ratio"]  = buy_ratio
                result["short_ratio"] = sell_ratio
                result["available"]   = True
                if buy_ratio > 0.6:
                    result["signal"] = "bullish"
                elif buy_ratio < 0.4:
                    result["signal"] = "bearish"
                else:
                    result["signal"] = "neutral"
    except Exception as e:
        log.debug(f"  Bybit AI/ratio недоступен для {symbol}: {e}")
    return result

# ================== ТЕХНИЧЕСКИЙ СКОР ==================
def получить_скор(symbol: str) -> dict:
    """
    Возвращает скор 0-100 и детали. Учитывает S/R уровни.
    Размер позиции будет масштабироваться пропорционально скору.
    """
    details = {}
    score   = 0
    price   = 0.0
    sr      = {}

    try:
        raw5  = exchange.fetch_ohlcv(symbol, TIMEFRAME_TA,    limit=300)
        raw1h = exchange.fetch_ohlcv(symbol, TIMEFRAME_TREND, limit=300)
        raw15 = exchange.fetch_ohlcv(symbol, TIMEFRAME_MID,   limit=100)

        if len(raw5) < 100 or len(raw1h) < 100:
            return {"score": 0, "details": {}, "price": 0, "sr": {}}

        cols = ["ts", "o", "h", "l", "c", "v"]
        df5  = pd.DataFrame(raw5,  columns=cols).reset_index(drop=True)
        df1h = pd.DataFrame(raw1h, columns=cols).reset_index(drop=True)
        df15 = pd.DataFrame(raw15, columns=cols).reset_index(drop=True)
        c5, c1h, c15 = df5["c"], df1h["c"], df15["c"]

        price = float(c5.iloc[-1])

        # --- RSI 5m [макс +20] ---
        rsi_val = calc_rsi(c5).iloc[-1]
        details["rsi"] = round(rsi_val, 1)
        if 25 <= rsi_val <= 42:
            score += 20
        elif 42 < rsi_val <= 52:
            score += 10
        elif rsi_val < 25:
            score += 12
        elif 52 < rsi_val <= 65:
            score += 5

        # --- RSI 1h [макс +8] ---
        rsi_1h = calc_rsi(c1h).iloc[-1]
        details["rsi_1h"] = round(rsi_1h, 1)
        if rsi_1h < 55:
            score += 8
        elif rsi_1h < 65:
            score += 4

        # --- MACD 5m [макс +18] ---
        ml, sl_macd, _ = calc_macd(c5)
        macd_bull  = ml.iloc[-1] > sl_macd.iloc[-1]
        macd_cross = macd_bull and ml.iloc[-2] <= sl_macd.iloc[-2]
        details["macd"] = "бычий" if macd_bull else "медвежий"
        if macd_cross:
            score += 18
        elif macd_bull:
            score += 8

        # --- Range Filter 5m [макс +15] ---
        _, _, _, rf_up, rf_down = calc_range_filter(df5)
        rf_up_now = rf_up.iloc[-1]
        details["range_filter"] = "вверх" if rf_up_now else ("вниз" if rf_down.iloc[-1] else "бок")
        if rf_up_now:
            score += 15

        # --- Supertrend 5m [макс +12] ---
        st_up, _ = calc_supertrend(df5)
        details["supertrend"] = "вверх" if st_up.iloc[-1] else "вниз"
        if st_up.iloc[-1]:
            score += 12

        # --- Supertrend 15m [макс +8] ---
        st_up_15, _ = calc_supertrend(df15)
        details["supertrend_15m"] = "вверх" if st_up_15.iloc[-1] else "вниз"
        if st_up_15.iloc[-1]:
            score += 8

        # --- Hull MA 5m [макс +8] ---
        hu_up, _ = calc_hull(c5)
        details["hull"] = "вверх" if hu_up.iloc[-1] else "вниз"
        if hu_up.iloc[-1]:
            score += 8

        # --- EMA 50/200 тренд 1h [макс +10] ---
        ema50  = _ema(c1h, 50).iloc[-1]
        ema200 = _ema(c1h, 200).iloc[-1]
        details["тренд_1h"] = "бычий" if ema50 > ema200 else "медвежий"
        if ema50 > ema200:
            score += 10

        # --- EMA 20/50 тренд 15m [макс +5] ---
        ema20_15 = _ema(c15, 20).iloc[-1]
        ema50_15 = _ema(c15, 50).iloc[-1]
        details["тренд_15m"] = "бычий" if ema20_15 > ema50_15 else "медвежий"
        if ema20_15 > ema50_15:
            score += 5

        # --- ADX [макс +8] ---
        adx, pdi, mdi = calc_adx(df5)
        adx_val = adx.iloc[-1]
        details["adx"] = round(adx_val, 1)
        if adx_val > 25 and pdi.iloc[-1] > mdi.iloc[-1]:
            score += 8
        elif adx_val > 20:
            score += 3

        # --- Stochastic [макс +8] ---
        k_ser, _ = calc_stochastic(df5)
        k_val = k_ser.iloc[-1]
        details["stoch_k"] = round(k_val, 1)
        if k_val < 25:
            score += 8
        elif k_val < 50:
            score += 4

        # --- Объём [макс +8] ---
        vol_avg   = df5["v"].rolling(20).mean().iloc[-1]
        vol_ratio = df5["v"].iloc[-1] / (vol_avg + 1e-10)
        details["объём_ratio"] = round(vol_ratio, 2)
        if vol_ratio > 1.5:
            score += 8
        elif vol_ratio > 1.2:
            score += 4

        # --- VWAP отклонение [макс +8] ---
        vwap_dev = calc_vwap_deviation(df5).iloc[-1]
        details["vwap_dev"] = round(vwap_dev, 2)
        if -3 <= vwap_dev <= -0.3:
            score += 8
        elif vwap_dev < -3:
            score += 4
        elif vwap_dev <= 1:
            score += 2

        # --- Поддержка/Сопротивление [макс +12, штраф -20] ---
        sr = calc_support_resistance(df5)
        details["support"]    = sr["support"]
        details["resistance"] = sr["resistance"]
        details["dist_sup"]   = sr["dist_to_sup_pct"]
        details["dist_res"]   = sr["dist_to_res_pct"]

        if sr["near_support"]:
            score += 12
            details["sr_signal"] = f"у поддержки ✅ ({sr['sup_cluster']} касаний)"
        elif sr["near_resistance"]:
            score -= 20
            details["sr_signal"] = f"у сопротивления ❌ ({sr['res_cluster']} касаний)"
        else:
            details["sr_signal"] = (
                f"нейтрально (sup={sr['dist_to_sup_pct']:.2f}% "
                f"res={sr['dist_to_res_pct']:.2f}%)"
            )

        # --- Штраф: 3 красных свечи подряд [-15] ---
        last3_bearish = all(df5["c"].iloc[-i] < df5["o"].iloc[-i] for i in range(1, 4))
        if last3_bearish:
            score -= 15
            details["свечи_3red"] = True

        score = max(0, min(100, score))

    except Exception as e:
        log.warning(f"Ошибка анализа {symbol}: {e}")

    return {"score": score, "details": details, "price": price, "sr": sr}

# ================== AI СКОР-КОРРЕКЦИЯ ==================
def применить_ai_корректировку(score: int, symbol: str) -> int:
    ai = получить_bybit_ai(symbol)
    if not ai["available"]:
        log.info(f"  🤖 Bybit ratio: недоступен")
        return score

    long_r  = ai["long_ratio"]
    signal  = ai["signal"]
    log.info(f"  🤖 Bybit ratio: long={long_r:.1%}  short={ai['short_ratio']:.1%}  сигнал={signal}")

    if signal == "bullish":
        return min(100, score + 8)
    elif signal == "bearish":
        return max(0, score - 8)
    return score

# ================== РАСЧЁТ РАЗМЕРА ПОЗИЦИИ ==================
def рассчитать_размер_позиции(score: int, баланс: float) -> float:
    if score <= MIN_SCORE:
        risk_pct = BASE_RISK_PCT
    else:
        factor   = (score - MIN_SCORE) / (100 - MIN_SCORE)
        risk_pct = BASE_RISK_PCT + (MAX_RISK_PCT - BASE_RISK_PCT) * factor

    max_loss_usdt = баланс * risk_pct / 100
    margin_usdt   = max_loss_usdt / (SL_PERCENT / 100)

    log.info(
        f"  📐 Скор={score} → риск={risk_pct:.1f}% "
        f"(макс.убыток={max_loss_usdt:.2f} USDT) → маржа={margin_usdt:.2f} USDT"
    )
    return round(max(1.0, margin_usdt), 2)

# ================== ОТКРЫТИЕ ПОЗИЦИИ ==================
def открыть_лонг(symbol: str, margin_usdt: float, tp_price: float, sl_price: float):
    try:
        установить_плечо(symbol, LEVERAGE)

        ticker    = exchange.fetch_ticker(symbol)
        price     = float(ticker["last"])
        pos_size_usdt = margin_usdt * LEVERAGE
        qty_raw   = pos_size_usdt / price

        qty = float(exchange.amount_to_precision(symbol, qty_raw))
        if qty <= 0:
            log.error(f"  Нулевое количество {symbol}")
            return None, None

        tp_str = exchange.price_to_precision(symbol, tp_price)
        sl_str = exchange.price_to_precision(symbol, sl_price)

        log.info(
            f"  Открываем лонг {symbol}: qty={qty}, маржа≈{margin_usdt:.2f}U, "
            f"плечо={LEVERAGE}x, TP={tp_str}, SL={sl_str}"
        )

        order = exchange.create_market_order(
            symbol, "buy", qty,
            params={
                "takeProfit": float(tp_str),
                "stopLoss":   float(sl_str),
            }
        )

        entry_price = price
        try:
            if order.get("average") and float(order["average"]) > 0:
                entry_price = float(order["average"])
        except:
            pass

        log.info(f"  📈 ЛОНГ открыт: {qty} {symbol} @ ~{entry_price:.8f}")
        return entry_price, qty

    except Exception as e:
        log.error(f"  ❌ Ошибка открытия лонга: {e}")
        return None, None

# ================== ОБНОВЛЕНИЕ SL НА БИРЖЕ ==================
def обновить_sl_на_бирже(symbol: str, new_sl: float) -> bool:
    try:
        sl_str = exchange.price_to_precision(symbol, new_sl)
        exchange.set_trading_stop(
            symbol,
            params={
                "category":    "linear",
                "stopLoss":    float(sl_str),
                "slTriggerBy": "MarkPrice",
                "positionIdx": 0,
            }
        )
        log.info(f"  🔧 SL обновлён → {sl_str}")
        return True
    except Exception as e:
        try:
            sl_str = exchange.price_to_precision(symbol, new_sl)
            coin_sym = symbol.replace("/", "").replace(":USDT", "")
            exchange.private_post_v5_position_trading_stop({
                "category":    "linear",
                "symbol":      coin_sym,
                "stopLoss":    sl_str,
                "slTriggerBy": "MarkPrice",
                "positionIdx": "0",
            })
            log.info(f"  🔧 SL обновлён (fallback) → {sl_str}")
            return True
        except Exception as e2:
            log.warning(f"  ⚠️ Не удалось обновить SL: {e} | {e2}")
            return False

# ================== МОНИТОРИНГ ПОЗИЦИИ ==================
def мониторить_позицию(symbol: str, entry_price: float, qty: float,
                        открыта_в: float, sl_цена: float) -> str:
    deadline = открыта_в + TRADE_MAX_LIFETIME
    coin     = symbol.split("/")[0]

    breakeven_price = entry_price * (1 + BYBIT_FEE * 2 + 0.0005)

    trailing_step    = TRAILING_STEP_PCT / 100
    trailing_offset  = TRAILING_OFFSET_PCT / 100

    фаза             = 1
    текущий_sl       = sl_цена
    пиковая_цена     = entry_price
    следующий_трейл  = entry_price * (1 + trailing_step)

    log.info(
        f"  🚦 Мониторинг запущен | вход={entry_price:.8f} "
        f"| безубыток @ {breakeven_price:.8f} (+{BYBIT_FEE*200+0.05:.2f}%)"
        f"  | трейлинг шаг={TRAILING_STEP_PCT}% отступ={TRAILING_OFFSET_PCT}%"
    )

    while True:
        сейчас = time.time()

        if сейчас >= deadline:
            log.warning("  ⏰ Дедлайн позиции — принудительное закрытие")
            try:
                exchange.create_market_order(symbol, "sell", qty, params={"reduceOnly": True})
            except Exception as e:
                log.warning(f"  Ошибка закрытия по дедлайну: {e}")
            return "таймаут"

        time.sleep(10)

        try:
            positions = exchange.fetch_positions([symbol])
            active = [p for p in positions
                      if float(p.get("contracts", 0) or 0) > 0 and p.get("side") == "long"]

            if not active:
                cur_price = float(exchange.fetch_ticker(symbol)["last"])
                if cur_price >= entry_price * (1 + TP_PERCENT / 100 * 0.7):
                    log.info("  ✅ Позиция закрыта по Тейк-профиту")
                    return "tp"
                elif фаза >= 2:
                    log.info("  🔒 Позиция закрыта по трейлинг/безубыток SL — без убытка")
                    return "tp"
                else:
                    log.info("  ❌ Позиция закрыта по Стоп-лоссу")
                    return "sl"

            pos       = active[0]
            cur_price = float(exchange.fetch_ticker(symbol)["last"])
            pnl       = float(pos.get("unrealizedPnl", 0) or 0)
            pnl_pct   = (cur_price - entry_price) / entry_price * 100
            до_дед    = int(deadline - сейчас)

            if фаза == 1 and cur_price >= breakeven_price:
                фаза = 2
                новый_sl = entry_price * (1 + BYBIT_FEE * 2 + 0.0003)
                if обновить_sl_на_бирже(symbol, новый_sl):
                    текущий_sl      = новый_sl
                    пиковая_цена    = cur_price
                    следующий_трейл = cur_price * (1 + trailing_step)
                    log.info(
                        f"  🔒 БЕЗУБЫТОК зафиксирован! "
                        f"SL перенесён → {новый_sl:.8f} "
                        f"(вход={entry_price:.8f}, покрыты комиссии)"
                    )

            elif фаза >= 2 and cur_price >= следующий_трейл:
                фаза = 3
                пиковая_цена    = max(пиковая_цена, cur_price)
                новый_sl        = пиковая_цена * (1 - trailing_offset)

                if новый_sl > текущий_sl:
                    if обновить_sl_на_бирже(symbol, новый_sl):
                        текущий_sl      = новый_sl
                        следующий_трейл = cur_price * (1 + trailing_step)
                        log.info(
                            f"  📈 ТРЕЙЛИНГ: пик={пиковая_цена:.8f} "
                            f"→ SL={новый_sl:.8f} "
                            f"(зафиксировано {(новый_sl-entry_price)/entry_price*100:+.2f}%)"
                        )

            if cur_price > пиковая_цена:
                пиковая_цена = cur_price

            фаза_лейбл = {1: "обычная", 2: "безубыток 🔒", 3: "трейлинг 📈"}.get(фаза, "?")
            log.info(
                f"  [{coin}] {cur_price:.8f}  P&L={pnl_pct:+.2f}% ({pnl:+.4f}U)"
                f"  SL={текущий_sl:.8f}  фаза={фаза_лейбл}  дед={до_дед}с"
            )

        except Exception as e:
            log.warning(f"  Ошибка мониторинга: {e}")

# ================== ИНВЕНТАРИЗАЦИЯ ==================
def полная_инвентаризация():
    log.info("🔄 Инвентаризация перед торговлей...")
    отменить_все_ордера()
    time.sleep(1)
    закрыть_все_позиции()
    log.info("✅ Инвентаризация завершена")
    time.sleep(2)

# ================== ТЕСТОВАЯ СДЕЛКА ==================
def тестовая_сделка():
    log.info("🧪 Тестовая сделка (фьючерсы) для проверки API...")
    try:
        sym = "DOGE/USDT:USDT"
        установить_плечо(sym, LEVERAGE)
        ticker  = exchange.fetch_ticker(sym)
        price   = float(ticker["last"])
        margin  = 1.0
        qty_raw = margin * LEVERAGE / price
        qty     = float(exchange.amount_to_precision(sym, qty_raw))

        log.info(f"  Покупка {qty} {sym} на маржу {margin} USDT...")
        order = exchange.create_market_order(sym, "buy", qty)
        time.sleep(2)
        exchange.create_market_order(sym, "sell", qty, params={"reduceOnly": True})
        log.info("  ✅ Тестовая сделка успешна!")
    except Exception as e:
        log.error(f"  ❌ Тестовая сделка не удалась: {e}")

# ================== ОТЧЁТ ==================
def печатать_отчёт():
    баланс  = баланс_usdt()
    старт   = stats["депозит_старт"]
    дельта  = баланс - старт
    чистый  = stats["прибыль_usdt"] - stats["убыток_usdt"]
    процент = (дельта / старт * 100) if старт > 0 else 0
    winrate = (stats["тейкпрофит"] / stats["сделок_всего"] * 100) if stats["сделок_всего"] > 0 else 0

    log.info("")
    log.info("=" * 60)
    log.info("  📊  ОТЧЁТ ЗА СЕССИЮ")
    log.info(f"  Время:               {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}")
    log.info(f"  Работает с:          {stats['старт_время']}")
    log.info("  ─" * 30)
    log.info(f"  Депозит при старте:  {старт:.2f} USDT")
    log.info(f"  Баланс сейчас:       {баланс:.2f} USDT  ({'+' if дельта >= 0 else ''}{дельта:.2f})")
    log.info(f"  Изменение:           {'+' if процент >= 0 else ''}{процент:.2f}%")
    log.info("  ─" * 30)
    log.info(f"  Сделок:              {stats['сделок_всего']}")
    log.info(f"  ✅ TP:               {stats['тейкпрофит']}  (winrate {winrate:.1f}%)")
    log.info(f"  ❌ SL:               {stats['стоплосс']}")
    log.info(f"  ⏰ Таймаут:          {stats['таймаут']}")
    log.info("  ─" * 30)
    log.info(f"  💰 Прибыль:         +{stats['прибыль_usdt']:.4f} USDT")
    log.info(f"  💸 Убыток:          -{stats['убыток_usdt']:.4f} USDT")
    log.info(f"  📈 Чистый P&L:       {'+' if чистый >= 0 else ''}{чистый:.4f} USDT")
    log.info("=" * 60)
    log.info("")

    stats["последний_отчёт"] = time.time()
    сохранить_состояние()
    аналитика_по_инструментам()

# ================== ГЛАВНЫЙ ЦИКЛ ==================
def main():
    полная_инвентаризация()
    тестовая_сделка()

    восстановлен  = загрузить_состояние()
    баланс_сейчас = баланс_usdt()
    stats["запусков"] += 1

    if not восстановлен or stats["депозит_старт"] == 0:
        stats["депозит_старт"] = баланс_сейчас
        stats["старт_время"]   = datetime.now().strftime("%d.%m.%Y %H:%M:%S")

    stats["последний_отчёт"] = time.time()

    история = загрузить_историю()
    if история:
        log.info(f"  📖 Найдено {len(история)} сделок в истории — запуск аналитики...")
        аналитика_по_инструментам()

    log.info("")
    log.info("=" * 60)
    log.info("  🤖  ФЬЮЧЕРСНЫЙ БОТ ЗАПУЩЕН  [v2 — исправленный S/R]")
    log.info("")
    log.info(f"  Запуск №:            {stats['запусков']}")
    log.info(f"  Плечо:               {LEVERAGE}x")
    log.info(f"  Дата/время:          {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}")
    log.info(f"  Работает с:          {stats['старт_время']}")
    log.info(f"  Риск на сделку:      {BASE_RISK_PCT}–{MAX_RISK_PCT}% от баланса")
    log.info(f"  Баланс:              {баланс_сейчас:.2f} USDT")
    log.info(f"  Пар для торговли:    {len(SYMBOLS)}")
    log.info(f"  MIN_SCORE:           {MIN_SCORE}")
    log.info(f"  TP / SL:             {TP_PERCENT}% / {SL_PERCENT}%")
    log.info(f"  S/R близость:        ±{SR_PROXIMITY_PCT}%  (мин. касаний: {SR_MIN_TOUCHES})")
    log.info(f"  S/R блок входа:      только dist < {SR_BLOCK_DIST_PCT}% к resistance")
    log.info(f"  AI Bybit:            включён (long/short ratio)")
    log.info("=" * 60)
    log.info("")

    while True:
        try:
            if time.time() - stats["последний_отчёт"] >= REPORT_INTERVAL:
                печатать_отчёт()

            баланс = баланс_usdt()

            if баланс < MIN_BALANCE:
                log.warning(f"  🛑 Баланс {баланс:.2f} USDT < минимума {MIN_BALANCE}. Пауза 10 мин.")
                time.sleep(600)
                continue

            if stats["депозит_старт"] > 0:
                просадка = (stats["депозит_старт"] - баланс) / stats["депозит_старт"] * 100
                if просадка > 30:
                    log.warning(f"  ⛔ Просадка {просадка:.1f}% > 30%. Пауза 1 час.")
                    time.sleep(3600)
                    continue

            активные = получить_позиции()
            if активные:
                log.info(f"  ⏳ Открыта позиция в {[p['symbol'] for p in активные]} — ждём закрытия")
                time.sleep(30)
                continue

            log.info(f"── Сканирование {len(SYMBOLS)} пар (баланс={баланс:.2f} USDT, порог={MIN_SCORE}) ──")
            scores = {}
            for sym in SYMBOLS:
                try:
                    res        = получить_скор(sym)
                    ai_score   = применить_ai_корректировку(res["score"], sym)
                    res["score_final"] = ai_score
                    scores[sym] = res
                    sr = res.get("sr", {})
                    log.info(
                        f"  {sym.split(':')[0]:12s}  скор={ai_score:3d}/100 "
                        f" rsi={res['details'].get('rsi', '?'):5}"
                        f"  rf={res['details'].get('range_filter', '?'):5}"
                        f"  st={res['details'].get('supertrend', '?'):5}"
                        f"  SR={res['details'].get('sr_signal', '?')}"
                    )
                except Exception as e:
                    log.warning(f"  Ошибка скора {sym}: {e}")
                    scores[sym] = {"score": 0, "score_final": 0, "details": {}, "price": 0, "sr": {}}

            if not scores:
                time.sleep(SCAN_INTERVAL)
                continue

            # Топ-5 кандидатов по скору
            кандидаты = sorted(
                [(s, d) for s, d in scores.items() if d["score_final"] >= MIN_SCORE],
                key=lambda x: x[1]["score_final"],
                reverse=True
            )[:5]

            if not кандидаты:
                log.info(f"  Нет пар с скором >= {MIN_SCORE} — ждём {SCAN_INTERVAL} сек")
                time.sleep(SCAN_INTERVAL)
                continue

            log.info(f"  🏆 Топ кандидаты: " +
                     ", ".join(f"{s.split(':')[0]}={d['score_final']}" for s, d in кандидаты))

            выбрана  = None
            фин_скор = 0
            цена     = 0.0
            sr_info  = {}
            sl_цена  = 0.0
            tp_цена  = 0.0
            margin   = 0.0

            for лучшая, данные in кандидаты:
                фин_скор = данные["score_final"]
                цена     = данные["price"]
                sr_info  = данные.get("sr", {})
                det      = данные.get("details", {})

                # ── ИСПРАВЛЕНО: блокируем только если resistance
                #    1) подтверждён кластером (near_resistance=True)
                #    2) И расстояние до него < SR_BLOCK_DIST_PCT (0.15%)
                #    Раньше блокировали при любом near_resistance,
                #    что отсекало сделки с resistance на 0.03–0.3%.
                dist_res_now = sr_info.get("dist_to_res_pct", 99)
                if sr_info.get("near_resistance") and dist_res_now < SR_BLOCK_DIST_PCT:
                    log.info(
                        f"  ⛔ {лучшая.split(':')[0]} (скор={фин_скор}): "
                        f"resistance {sr_info.get('resistance', '?'):.6f} "
                        f"в {dist_res_now:.3f}% ({sr_info.get('res_cluster', 0)} касаний) "
                        f"< {SR_BLOCK_DIST_PCT}% — пробуем следующего"
                    )
                    continue

                # Мягкий фильтр: RSI перекуплен + нет поддержки
                rsi_val = det.get("rsi", 50)
                try:
                    rsi_val = float(rsi_val)
                except:
                    rsi_val = 50
                if rsi_val > 68 and not sr_info.get("near_support"):
                    log.info(
                        f"  ⚠️ {лучшая.split(':')[0]}: RSI перекуплен ({rsi_val:.1f}) "
                        f"без поддержки — пробуем следующего"
                    )
                    continue

                выбрана = лучшая
                log.info(
                    f"  ► Выбрана {лучшая.split(':')[0]}  скор={фин_скор}  "
                    f"цена={цена:.8f}  dist_res={dist_res_now:.3f}%"
                )
                break

            if выбрана is None:
                log.info("  Все кандидаты отфильтрованы — ждём следующего цикла")
                time.sleep(SCAN_INTERVAL)
                continue

            лучшая = выбрана

            # ── Расчёт TP и SL с учётом S/R ──────────────────────
            support    = sr_info.get("support",    цена * (1 - SL_PERCENT / 100))
            resistance = sr_info.get("resistance", цена * (1 + TP_PERCENT / 100))

            sl_базовый = цена * (1 - SL_PERCENT / 100)
            sl_от_sup  = float(support) * 0.998
            sl_цена    = max(sl_базовый, sl_от_sup)

            tp_базовый = цена * (1 + TP_PERCENT / 100)
            dist_res   = sr_info.get("dist_to_res_pct", 99)
            if dist_res > TP_PERCENT * 1.2:
                tp_цена = цена + (float(resistance) - цена) * 0.90
            else:
                tp_цена = tp_базовый

            # ── Размер позиции ────────────────────────────────────
            margin = рассчитать_размер_позиции(фин_скор, баланс)
            if баланс < margin * 1.1:
                log.warning(f"  ⚠️ Баланс {баланс:.2f} < маржа {margin:.2f} — уменьшаем")
                margin = баланс * 0.8

            log.info(
                f"  ✅ ВХОД: скор={фин_скор} | "
                f"SL={sl_цена:.8f} | TP={tp_цена:.8f} | маржа={margin:.2f}U"
            )

            время_входа = time.time()
            вход_цена, кол_во = открыть_лонг(лучшая, margin, tp_цена, sl_цена)

            if вход_цена is None or кол_во is None:
                log.warning("  Не удалось открыть позицию — пауза 30 сек")
                time.sleep(30)
                continue

            stats["сделок_всего"] += 1
            сохранить_состояние()

            результат = мониторить_позицию(лучшая, вход_цена, кол_во, время_входа, sl_цена)

            объём      = margin * LEVERAGE
            комиссии   = объём * BYBIT_FEE * 2
            длит_мин   = (time.time() - время_входа) / 60
            pnl_сделки = 0.0

            if результат == "tp":
                pnl_сделки = объём * TP_PERCENT / 100 - комиссии
                stats["тейкпрофит"]   += 1
                stats["прибыль_usdt"] += max(0, pnl_сделки)
                log.info(f"  ✅ TP: прибыль ≈{pnl_сделки:.4f} USDT")

            elif результат == "sl":
                pnl_сделки = -(объём * SL_PERCENT / 100 + комиссии)
                stats["стоплосс"]    += 1
                stats["убыток_usdt"] += abs(pnl_сделки)
                log.warning(f"  ❌ SL: убыток ≈{pnl_сделки:.4f} USDT")

            elif результат == "таймаут":
                pnl_сделки = -комиссии
                stats["таймаут"]     += 1
                stats["убыток_usdt"] += комиссии
                log.warning(f"  ⏰ Таймаут: потери на комиссиях ≈{комиссии:.4f} USDT")

            запись_сделки = {
                "id":            stats["сделок_всего"],
                "время_входа":   datetime.fromtimestamp(время_входа).strftime("%d.%m.%Y %H:%M:%S"),
                "время_выхода":  datetime.now().strftime("%d.%m.%Y %H:%M:%S"),
                "duration_min":  round(длит_мин, 1),
                "symbol":        лучшая,
                "score":         фин_скор,
                "entry_price":   вход_цена,
                "sl_price":      sl_цена,
                "tp_price":      tp_цена,
                "margin_usdt":   margin,
                "leverage":      LEVERAGE,
                "результат":     результат,
                "pnl_usdt":      round(pnl_сделки, 4),
                "details":       scores[лучшая].get("details", {}),
                "sr":            {k: str(v) for k, v in (scores[лучшая].get("sr", {}) or {}).items()},
            }
            сохранить_сделку(запись_сделки)
            пост_трейд_анализ(запись_сделки)

            if stats["сделок_всего"] % 10 == 0:
                аналитика_по_инструментам()

            сохранить_состояние()
            log.info("  Сделка завершена — пауза 30 сек")
            time.sleep(30)

        except Exception as e:
            log.error(f"Глобальная ошибка: {e}", exc_info=True)
            time.sleep(60)

if __name__ == "__main__":
    main()
