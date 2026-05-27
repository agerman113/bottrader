#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
Bybit ГИБРИДНЫЙ ПРОФЕССИОНАЛЬНЫЙ БОТ — v11.1 Ultimate Pro Fixed
================================================================================
Версия: 11.1 Ultimate Pro Fixed
Дата: 28.05.2026

Исправления:
- Полная защита от крахов и зависаний
- Корректный расчёт P&L с учётом комиссий и проскальзывания
- Блокировка символов между перезапусками
- Очистка старых ML-записей
- Защита от дублирования ордеров
- Улучшенный мониторинг позиций
================================================================================
"""

import os
import sys
import time
import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any

# ============================================================
# ЛОГИРОВАНИЕ
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%d.%m.%Y %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("bot_v11_fixed.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# ============================================================
# ИМПОРТ МОДУЛЕЙ
# ============================================================
try:
    from config import *
    from engine import (
        exchange,
        ml_model as engine_ml_model,
        load_state,
        save_state,
        get_free_balance,
        get_total_balance,
        get_positions,
        update_day_start,
        is_daily_loss_exceeded,
        is_trading_time_allowed,
        trend_4h_bullish,
        trend_4h_bearish,
        get_score,
        get_score_short,
        apply_ai_correction,
        calc_position_size,
        open_position,
        close_position_with_confirm,
        confirm_entry,
        monitor_position,
        print_report,
        post_trade_analysis,
        load_trades_history,
        save_trade,
        log_ml_data,
        pending_ml_entries,
        calc_sl_tp,
        maybe_retrain_ml,
        get_order_flow_signals,
        get_quant_signals,
        calc_exact_pnl,
    )
except ImportError as e:
    log.critical(f"Ошибка импорта модулей: {e}")
    log.critical("Убедись, что config.py и engine.py в той же папке")
    sys.exit(1)

# ============================================================
# ИНИЦИАЦИЯ ML
# ============================================================
ml_model = engine_ml_model

def init_ml():
    """Инициализация ML модели."""
    global ml_model

    if not ML_ENABLED:
        log.info("ML отключён в конфигурации")
        return

    try:
        if os.path.exists(ML_MODEL_FILE):
            if ml_model.load_model(ML_MODEL_FILE):
                log.info(f"✅ ML модель загружена: Acc={ml_model.accuracy:.2f}")
                return
    except Exception as e:
        log.warning(f"Не удалось загрузить ML модель: {e}")

    # Пытаемся обучить на существующих данных
    try:
        if os.path.exists(ML_LOG_FILE):
            log.info("Пытаемся обучить ML модель на существующих данных...")
            if ml_model.train():
                ml_model.save_model()
                log.info("✅ ML модель обучена и сохранена")
    except Exception as e:
        log.warning(f"Не удалось обучить ML: {e}")

# ============================================================
# ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ
# ============================================================
stats = {}
blocked_symbols = {}

# ============================================================
# ПРЕДСТАРТОВЫЕ ПРОВЕРКИ
# ============================================================
def preflight_check() -> bool:
    """Выполняет предстартовые проверки."""
    log.info("")
    log.info("=" * 65)
    log.info(f"🔍 ПРЕДСТАРТОВАЯ ПРОВЕРКА | {BOT_VERSION}")
    log.info("=" * 65)

    all_ok = True

    # 1. API ключи
    log.info("\n▶ Проверка API ключей...")
    try:
        balance = exchange.fetch_balance({"type": "linear"})
        usdt = float(balance.get("USDT", {}).get("free", 0))
        if usdt < MIN_BALANCE:
            log.warning(f"⚠️ Баланс низкий: {usdt:.2f} USDT")
        log.info(f"✅ API OK | Баланс: {usdt:.2f} USDT")
    except Exception as e:
        log.error(f"❌ Ошибка подключения: {e}")
        all_ok = False

    # 2. Конфигурация
    log.info("\n▶ Проверка конфигурации...")
    rr = TP_PERCENT / SL_PERCENT
    if rr < MIN_RR_RATIO:
        log.error(f"❌ RR {rr:.1f}:1 < {MIN_RR_RATIO}:1")
        all_ok = False
    else:
        log.info(f"✅ Конфигурация OK | RR={rr:.1f}:1")

    # 3. Рынок
    log.info("\n▶ Проверка рынка...")
    test_count = 0
    for sym in SYMBOLS[:3]:
        try:
            ticker = exchange.fetch_ticker(sym)
            if float(ticker["last"]) > 0:
                test_count += 1
        except Exception as e:
            log.warning(f"⚠️ {sym}: {e}")

    if test_count == 0:
        log.error("❌ Рынок недоступен")
        all_ok = False
    else:
        log.info(f"✅ Рынок OK ({test_count}/3 пар доступны)")

    # 4. Существующие позиции
    log.info("\n▶ Проверка позиций...")
    positions = get_positions()
    if positions:
        for p in positions:
            log.warning(f"⚠️ Уже открыта: {p.get('symbol')} {p.get('side')}")
    else:
        log.info("✅ Открытых позиций нет")

    log.info("")
    log.info("=" * 65)
    if all_ok:
        log.info("✅ ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ")
    else:
        log.error("❌ ЕСТЬ ОШИБКИ")
    log.info("=" * 65)
    log.info("")

    return all_ok

# ============================================================
# ЗАГРУЗКА БЛОКИРОВОК
# ============================================================
def load_blocked_symbols(state: dict):
    """Загружает заблокированные символы из состояния."""
    global blocked_symbols
    blocked_symbols = {}

    saved_blocked = state.get("blocked_symbols", {})
    now = time.time()

    for sym, unblock_time in saved_blocked.items():
        if unblock_time > now:
            blocked_symbols[sym] = unblock_time

    if blocked_symbols:
        log.info(f"Заблокированы: {', '.join([s.split(':')[0] for s in blocked_symbols])}")

def save_blocked_symbols(state: dict):
    """Сохраняет заблокированные символы в состояние."""
    state["blocked_symbols"] = blocked_symbols

def is_symbol_blocked(symbol: str) -> bool:
    """Проверяет, заблокирован ли символ."""
    if symbol in blocked_symbols:
        if time.time() < blocked_symbols[symbol]:
            return True
        else:
            del blocked_symbols[symbol]
    return False

def block_symbol(symbol: str, minutes: int):
    """Блокирует символ на указанное время."""
    blocked_symbols[symbol] = time.time() + minutes * 60
    log.info(f"🔒 {symbol.split(':')[0]} заблокирован на {minutes} мин")

# ============================================================
# СКАНИРОВАНИЕ
# ============================================================
def scan_for_trade() -> Tuple[Optional[str], int, float, Dict, str]:
    """Сканирует рынок в поисках сделок."""
    global stats

    log.info(f"── Сканирование {len(SYMBOLS)} пар (баланс={get_free_balance():.2f}U) ──")

    scores = {}

    for sym in SYMBOLS:
        # Пропускаем заблокированные
        if is_symbol_blocked(sym):
            continue

        # Фильтр 4H тренда
        if not trend_4h_bullish(sym):
            continue

        # Получаем скор
        result = get_score(sym)
        if result["score"] == 0:
            continue

        # AI корректировка
        ai_score = apply_ai_correction(result["score"], sym)
        result["score_final"] = ai_score
        scores[sym] = result

        log.debug(f"{sym.split(':')[0]:12s} скор={ai_score:3.0f}/100")

    if not scores:
        log.info("Нет кандидатов на лонг")
        return None, 0, 0.0, {}, "long"

    # Сортируем по скору
    candidates = sorted(
        [(s, d) for s, d in scores.items() if d.get("score_final", 0) >= MIN_SCORE],
        key=lambda x: x[1]["score_final"],
        reverse=True
    )[:5]  # Топ 5 кандидатов

    # Выбираем лучший кандидат
    for best, data in candidates:
        final_score = data["score_final"]
        price = data["price"]
        sr_info = data.get("sr", {})
        det = data.get("details", {})

        # Фильтр S/R
        if sr_info.get("near_resistance") and sr_info.get("dist_to_res_pct", 99) < SR_BLOCK_DIST_PCT:
            log.info(f"⛔ {best.split(':')[0]}: сопротивление {sr_info.get('dist_to_res_pct', 0):.2f}%")
            continue

        # Фильтр RSI
        rsi_val = float(det.get("rsi", 50) or 50)
        if rsi_val > 65 and not sr_info.get("near_support"):
            log.info(f"⚠️ {best.split(':')[0]}: RSI={rsi_val:.1f}")
            continue

        # MA кроссовер
        if MA_CROSSOVER_ENABLED and not det.get("ma_cross", True):
            continue

        # Volume spike
        if not det.get("vol_spike_ok", True):
            continue

        log.info(f"► Выбрана {best.split(':')[0]} (лонг) скор={final_score}")
        return best, final_score, price, sr_info, "long"

    # Если нет лонгов — ищем шорты
    log.info("Лонгов нет, ищем шорты...")
    for sym in SYMBOLS:
        if is_symbol_blocked(sym):
            continue
        if trend_4h_bearish(sym):
            short_res = get_score_short(sym)
            if short_res["score"] >= MIN_SCORE:
                det_sh = short_res.get("details", {})
                if MA_CROSSOVER_ENABLED and not det_sh.get("ma_cross", True):
                    continue
                if not det_sh.get("vol_spike_ok", True):
                    continue
                log.info(f"🐻 Шорт: {sym.split(':')[0]} скор={short_res['score']}")
                return sym, short_res["score"], short_res["price"], short_res.get("sr", {}), "short"

    return None, 0, 0.0, {}, "long"

# ============================================================
# ГЛАВНЫЙ ЦИКЛ
# ============================================================
def main():
    global stats, ml_model

    log.info("=" * 65)
    log.info(f"🤖 ГИБРИДНЫЙ БОТ {BOT_VERSION}")
    log.info("=" * 65)

    # Предстартовая проверка
    if not preflight_check():
        log.error("🛑 Ошибки предстартовой проверки")
        return

    # Инициализация ML
    init_ml()

    # Загрузка состояния
    stats = load_state()
    stats["starts"] = stats.get("starts", 0) + 1
    stats["bot_version"] = BOT_VERSION

    balance_now = get_total_balance()
    if stats.get("deposit_start", 0) <= 0:
        stats["deposit_start"] = balance_now
    if not stats.get("start_time"):
        stats["start_time"] = datetime.now().strftime("%d.%m.%Y %H:%M:%S")

    update_day_start(stats, balance_now)
    load_blocked_symbols(stats)
    save_state(stats)

    log.info(f"Плечо: {LEVERAGE}x | RR: {TP_PERCENT}/{SL_PERCENT} ({TP_PERCENT / SL_PERCENT:.1f}:1)")
    log.info(f"Баланс: {balance_now:.2f} USDT")
    log.info(f"Пар: {len(SYMBOLS)}")
    log.info(f"Мин. скор: {MIN_SCORE}")
    log.info(f"Quant: {'ВКЛ' if QUANT_ENABLED else 'ВЫКЛ'} | "
             f"Order Flow: {'ВКЛ' if ORDER_FLOW_ENABLED else 'ВЫКЛ'} | "
             f"ML: {'ВКЛ' if ML_ENABLED and ml_model.trained else 'ВЫКЛ'}")
    log.info("=" * 65)
    log.info("")

    # Очистка старых ML записей (если бот был перезапущен)
    pending_ml_entries.clear()

    # Главный цикл
    while True:
        try:
            # Отчёт
            if time.time() - stats.get("last_report", 0) >= REPORT_INTERVAL:
                print_report(stats)

            # Баланс
            balance = get_total_balance()
            free_balance = get_free_balance()
            update_day_start(stats, balance)

            # Проверка свободного баланса
            if free_balance < MIN_BALANCE:
                log.warning(f"🛑 Баланс {free_balance:.2f} < {MIN_BALANCE}. Пауза 10 мин.")
                time.sleep(600)
                continue

            # Максимальная просадка
            if stats["deposit_start"] > 0:
                drawdown = (stats["deposit_start"] - balance) / stats["deposit_start"] * 100
                if drawdown > MAX_DRAWDOWN_PCT:
                    log.warning(f"⛔ Просадка {drawdown:.1f}% > {MAX_DRAWDOWN_PCT}%. Пауза 2ч.")
                    time.sleep(7200)
                    continue

            # Дневной лимит
            if is_daily_loss_exceeded(stats):
                log.warning(f"⛔ Дневной лимит убытков. Пауза {DAILY_LOSS_PAUSE_SEC // 60} мин.")
                time.sleep(DAILY_LOSS_PAUSE_SEC)
                continue

            # Фильтр времени
            if not is_trading_time_allowed():
                log.info("🕐 Заблокировано по времени. Пауза 5 мин.")
                time.sleep(300)
                continue

            # SL стрик
            if stats.get("sl_streak", 0) >= SL_STREAK_LIMIT:
                log.warning(f"🧊 {SL_STREAK_LIMIT} SL подряд — cooldown")
                stats["sl_streak"] = 0
                save_state(stats)
                save_blocked_symbols(stats)
                time.sleep(SL_STREAK_PAUSE + SL_STREAK_EXTRA_PAUSE)
                continue

            # Активные позиции
            active_positions = get_positions()
            if active_positions:
                log.info(f"⏳ Открытые позиции: {[p['symbol'] for p in active_positions]}")
                time.sleep(60)
                continue

            # Сканирование
            selected, score, price, sr_info, side = scan_for_trade()

            if selected is None:
                log.info(f"Нет кандидатов — ждём {SCAN_INTERVAL} сек")
                time.sleep(SCAN_INTERVAL)
                continue

            # Расчёт SL/TP
            sl_price, tp_price, sl_dist_pct, real_rr = calc_sl_tp(selected, price, side, sr_info)

            log.info(f"📐 SL={sl_dist_pct:.2f}% RR={real_rr:.1f}:1")

            # Проверка RR
            if real_rr < MIN_RR_RATIO:
                log.warning(f"⛔ RR={real_rr:.1f}:1 < {MIN_RR_RATIO}:1")
                time.sleep(SCAN_INTERVAL)
                continue

            # Размер позиции
            history = load_trades_history()
            margin = calc_position_size(score, free_balance, sl_dist_pct, history)

            if free_balance < margin * 1.1:
                log.warning(f"⚠️ Баланс меньше маржи")
                margin = free_balance * 0.8

            # ML предсказание
            ml_prediction = None
            ml_features = None

            if ML_ENABLED and ml_model.trained:
                try:
                    result = get_score(selected) if side == "long" else get_score_short(selected)
                    df_ta = result.get("df_ta")
                    df_1h = result.get("df_1h")

                    if df_ta is not None and df_1h is not None:
                        order_flow_data = get_order_flow_signals(selected)
                        quant_data = get_quant_signals(selected)
                        risk_data = calc_exact_pnl(price, tp_price, sl_price, margin, LEVERAGE, selected, side)

                        ml_features = ml_model.create_features(
                            selected, df_ta, df_1h, order_flow_data, quant_data, risk_data
                        )
                        ml_prediction = ml_model.predict(ml_features)

                        if ml_prediction["valid"]:
                            log.info(f"🤖 ML: {ml_prediction['signal']} (prob={ml_prediction['probability']:.2f})")

                            # Если ML против — пропускаем
                            if (side == "long" and ml_prediction["signal"] == "sell" and ml_prediction["probability"] > 0.6):
                                log.info("⛔ ML против лонга — пропуск")
                                time.sleep(60)
                                continue
                            elif (side == "short" and ml_prediction["signal"] == "buy" and ml_prediction["probability"] > 0.6):
                                log.info("⛔ ML против шорта — пропуск")
                                time.sleep(60)
                                continue
                except Exception as e:
                    log.warning(f"Ошибка ML предсказания: {e}")

            log.info(f"✅ ВХОД {side.upper()}: скор={score} | SL={sl_price:.8f} | TP={tp_price:.8f} | маржа={margin:.2f}U")

            # Подтверждение входа
            if ENTRY_CONFIRM_BARS > 0:
                if not confirm_entry(selected, score, side):
                    log.info(f"⛔ Вход в {selected} отменён")
                    time.sleep(30)
                    continue

            # Открытие позиции
            balance_before = get_total_balance()
            entry_time = time.time()
            entry_price, qty = open_position(selected, margin, tp_price, sl_price, side)

            if entry_price is None or qty is None:
                log.warning("Не удалось открыть позицию — пауза 30 сек")
                time.sleep(30)
                continue

            # Сохраняем ML фичи для последующего обучения
            if ml_features and ml_prediction:
                trade_id = stats["trades_total"] + 1
                pending_ml_entries[trade_id] = {
                    "symbol": selected,
                    "features": ml_features,
                    "prediction": ml_prediction,
                    "entry_time": entry_time,
                }

            # Обновляем статистику
            stats["trades_total"] += 1
            save_state(stats)
            save_blocked_symbols(stats)

            # Мониторинг
            result = "sl"
            try:
                result = monitor_position(
                    selected, entry_price, qty, entry_time,
                    sl_price, tp_price, side
                )
            except Exception as e:
                log.error(f"💥 Краш мониторинга: {e}")
                try:
                    close_position_with_confirm(selected, qty, side)
                except Exception:
                    pass
                result = "sl"

            # Закрытие — считаем P&L
            time.sleep(3)
            balance_after = get_total_balance()
            real_pnl = balance_after - balance_before
            duration_min = (time.time() - entry_time) / 60

            # Обновляем статистику
            if result == "tp":
                stats["take_profit"] += 1
                stats["profit_usdt"] += max(0, real_pnl)
                stats["sl_streak"] = 0
                log.info(f"✅ TP: +{real_pnl:+.4f} USDT")
                block_symbol(selected, SYMBOL_BLOCK_AFTER_TP)

            elif result == "sl":
                stats["stop_loss"] += 1
                stats["loss_usdt"] += abs(min(0, real_pnl))
                stats["sl_streak"] = stats.get("sl_streak", 0) + 1
                block_symbol(selected, SYMBOL_BLOCK_AFTER_SL)
                log.warning(f"❌ SL: {real_pnl:+.4f} USDT (streak={stats['sl_streak']})")

            else:  # timeout
                stats["timeout"] += 1
                stats["loss_usdt"] += abs(min(0, real_pnl))
                stats["sl_streak"] = 0
                block_symbol(selected, SYMBOL_BLOCK_AFTER_TP)
                log.warning(f"⏰ Таймаут: {real_pnl:+.4f} USDT")

            # Сохраняем сделку
            trade_record = {
                "id": stats["trades_total"],
                "entry_time": datetime.fromtimestamp(entry_time).strftime("%d.%m.%Y %H:%M:%S"),
                "exit_time": datetime.now().strftime("%d.%m.%Y %H:%M:%S"),
                "duration_min": round(duration_min, 1),
                "symbol": selected,
                "side": side,
                "score": score,
                "entry_price": entry_price,
                "sl_price": sl_price,
                "tp_price": tp_price,
                "sl_dist_pct": round(sl_dist_pct, 3),
                "margin_usdt": margin,
                "leverage": LEVERAGE,
                "result": result,
                "pnl_usdt": round(real_pnl, 4),
                "rr_ratio": round(real_rr, 2),
                "ml_probability": ml_prediction.get("probability") if ml_prediction else None,
                "details": {},
                "bot_version": BOT_VERSION,
            }

            save_trade(trade_record)

            # Пост-трейд анализ
            post_trade_analysis(trade_record, ml_model)

            # Сохраняем состояние
            save_state(stats)
            save_blocked_symbols(stats)

            # Переобучение ML
            maybe_retrain_ml()

            log.info("Сделка завершена — пауза 60 сек")
            time.sleep(60)

        except KeyboardInterrupt:
            log.info("\n👋 Остановка по Ctrl+C")
            save_state(stats)
            save_blocked_symbols(stats)
            break

        except Exception as e:
            log.error(f"Глобальная ошибка: {e}", exc_info=True)
            time.sleep(60)

# ============================================================
# КОМАНДНАЯ СТРОКА
# ============================================================
def print_help():
    """Выводит справку."""
    print(f"""
{'=' * 65}
🤖 Bybit Гибридный Бот {BOT_VERSION}
{'=' * 65}

Использование:
    python bot.py           — Запустить бота
    python bot.py backtest  — Запустить бэктест
    python bot.py check     — Проверить конфигурацию
    python bot.py help      — Показать эту справку

Файлы конфигурации:
    .env        — API ключи (BYBIT_API_KEY, BYBIT_API_SECRET)
    config.py   — Все настройки бота
    engine.py   — Логика (индикаторы, ML, анализ)
    bot.py      — Главный цикл

{'=' * 65}
""")

def run_backtest():
    """Запускает простой бэктест."""
    from engine import backtest_simple
    import pandas as pd

    print(f"\n{'=' * 65}")
    print(f"📊 БЭКТЕСТ {BOT_VERSION}")
    print(f"{'=' * 65}\n")

    for symbol in SYMBOLS[:5]:  # Тест на первых 5 парах
        try:
            print(f"Тестируем {symbol}...")
            raw = exchange.fetch_ohlcv(symbol, "1h", limit=1000)
            if len(raw) < 200:
                print(f"  ⚠️ Мало данных: {len(raw)}")
                continue

            df = pd.DataFrame(raw, columns=["ts", "o", "h", "l", "c", "v"])
            result = backtest_simple(df)

            if "error" in result:
                print(f"  ❌ {result['error']}")
            else:
                print(f"  ✅ Сделок: {result['total_trades']}")
                print(f"     WinRate: {result['winrate']}%")
                print(f"     Ср. P&L: {result['avg_pnl']}%")
                print(f"     Итого: {result['total_pnl']}%")
                print(f"     Лучшая: {result['max_pnl']}% | Худшая: {result['min_pnl']}%")
            print()
        except Exception as e:
            print(f"  ❌ Ошибка: {e}\n")

    print(f"{'=' * 65}\n")

def run_check():
    """Проверяет конфигурацию без запуска торговли."""
    print(f"\n{'=' * 65}")
    print(f"🔍 ПРОВЕРКА КОНФИГУРАЦИИ {BOT_VERSION}")
    print(f"{'=' * 65}\n")

    # Конфигурация
    print("КОНФИГУРАЦИЯ:")
    print(f"  Плечо: {LEVERAGE}x")
    print(f"  TP: {TP_PERCENT}% | SL: {SL_PERCENT}% | RR: {TP_PERCENT / SL_PERCENT:.1f}:1")
    print(f"  Мин. скор: {MIN_SCORE}")
    print(f"  Риск: {BASE_RISK_PCT}-{MAX_RISK_PCT}%")
    print(f"  Пар: {len(SYMBOLS)}")
    print(f"  Quant: {'ВКЛ' if QUANT_ENABLED else 'ВЫКЛ'}")
    print(f"  Order Flow: {'ВКЛ' if ORDER_FLOW_ENABLED else 'ВЫКЛ'}")
    print(f"  ML: {'ВКЛ' if ML_ENABLED else 'ВЫКЛ'}")
    print()

    # API
    print("API:")
    try:
        balance = exchange.fetch_balance({"type": "linear"})
        usdt = float(balance.get("USDT", {}).get("free", 0))
        print(f"  ✅ Подключение OK")
        print(f"  Баланс: {usdt:.2f} USDT")
    except Exception as e:
        print(f"  ❌ Ошибка: {e}")
    print()

    # Рынок
    print("РЫНОК:")
    for sym in SYMBOLS[:5]:
        try:
            ticker = exchange.fetch_ticker(sym)
            price = float(ticker["last"])
            vol = float(ticker.get("quoteVolume", 0))
            print(f"  {sym.split('/')[0]:8s}: {price:12.4f} | Объём: {vol / 1e6:.1f}M")
        except Exception as e:
            print(f"  {sym.split('/')[0]:8s}: ❌ {e}")
    print()

    # Позиции
    print("ПОЗИЦИИ:")
    positions = get_positions()
    if positions:
        for p in positions:
            print(f"  ⚠️ {p.get('symbol')} {p.get('side')} qty={p.get('contracts')}")
    else:
        print(f"  Нет открытых позиций")
    print()

    # История
    history = load_trades_history()
    print(f"ИСТОРИЯ: {len(history)} сделок")
    if history:
        wins = sum(1 for t in history if t.get("result") == "tp")
        losses = sum(1 for t in history if t.get("result") == "sl")
        pnl = sum(t.get("pnl_usdt", 0) for t in history)
        print(f"  TP: {wins} | SL: {losses}")
        print(f"  WinRate: {wins / len(history) * 100:.1f}%")
        print(f"  Чистый P&L: {pnl:+.4f} USDT")
    print()

    # ML модель
    if ML_ENABLED:
        print("ML МОДЕЛЬ:")
        if os.path.exists(ML_MODEL_FILE):
            try:
                ml_model.load_model()
                print(f"  ✅ Модель загружена")
                print(f"  Accuracy: {ml_model.accuracy:.2f}")
                print(f"  Precision: {ml_model.precision:.2f}")
            except Exception as e:
                print(f"  ⚠️ Не загружена: {e}")
        else:
            print(f"  ⚠️ Файл модели не найден")

        if os.path.exists(ML_LOG_FILE):
            log_size = os.path.getsize(ML_LOG_FILE) / 1024
            print(f"  Данные для обучения: {log_size:.1f} KB")
        print()

    print(f"{'=' * 65}\n")

# ============================================================
# ТОЧКА ВХОДА
# ============================================================
if __name__ == "__main__":
    args = sys.argv[1:]

    if not args:
        # Запуск бота
        main()
    elif args[0] == "backtest":
        run_backtest()
    elif args[0] == "check":
        run_check()
    elif args[0] == "help":
        print_help()
    else:
        print(f"Неизвестная команда: {args[0]}")
        print("Используй: python bot.py [backtest|check|help]")
