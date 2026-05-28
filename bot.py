#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
Bybit БОТ ДЛЯ МЕМ-КОИНОВ — v12.3 MemeCoin Turbo Fixed
================================================================================
Версия: 12.3 MemeCoin Turbo Fixed
Дата: 28.05.2026

Особенности:
- Торговля только мем-коинами (PEPE, WIF, BONK и др.)
- Частые сделки (сканирование каждые 30 секунд)
- Агрессивная стратегия (высокий RR, увеличенный риск)
- Уведомления в Telegram
- Понятная сводка работы
================================================================================
"""

import os
import sys
import time
import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np

# ============================================================
# ЛОГИРОВАНИЕ
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%d.%m.%Y %H:%M:%S",
    handlers=[
        logging.FileHandler("bot_meme.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# ============================================================
# ИМПОРТ МОДУЛЕЙ
# ============================================================
try:
    from config import *
    from engine import (
        exchange,
        ml_model,
        load_state,
        save_state,
        get_free_balance,
        get_total_balance,
        get_positions,
        update_day_start,
        is_daily_loss_exceeded,
        trend_4h_bullish,
        trend_4h_bearish,
        get_score,
        get_score_short,
        calc_position_size,
        open_position_with_retries,
        close_position_with_confirm,
        emergency_close_position,
        monitor_position,
        print_report,
        post_trade_analysis,
        load_trades_history,
        save_trade,
        send_telegram_message,
        check_liquidity,
        check_slippage,
        calc_sl_tp,
    )
except ImportError as e:
    log.critical(f"Ошибка импорта модулей: {e}")
    log.critical("Убедись, что config.py и engine.py в той же папке")
    sys.exit(1)

# ============================================================
# ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ
# ============================================================
stats = {}
blocked_symbols = {}
last_action_time = time.time()
current_status = "Инициализация..."

# ============================================================
# ПРЕДСТАРТОВАЯ ПРОВЕРКА
# ============================================================
def preflight_check() -> bool:
    """Выполняет предстартовую проверку."""
    global current_status
    current_status = "Проверка API ключей..."
    log.info("")
    log.info("=" * 65)
    log.info(f"🔍 ПРЕДСТАРТОВАЯ ПРОВЕРКА | {BOT_VERSION}")
    log.info("=" * 65)

    all_ok = True

    # 1. API ключи
    current_status = "Проверка API ключей..."
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
    current_status = "Проверка конфигурации..."
    log.info("\n▶ Проверка конфигурации...")
    rr = TP_PERCENT / SL_PERCENT
    if rr < MIN_RR_RATIO:
        log.error(f"❌ RR {rr:.1f}:1 < {MIN_RR_RATIO}:1")
        all_ok = False
    else:
        log.info(f"✅ Конфигурация OK | RR={rr:.1f}:1")

    # 3. Рынок мем-коинов
    current_status = "Проверка рынка мем-коинов..."
    log.info("\n▶ Проверка рынка мем-коинов...")
    test_count = 0
    for sym in SYMBOLS[:3]:
        try:
            ticker = exchange.fetch_ticker(sym)
            if float(ticker["last"]) > 0:
                test_count += 1
                log.info(f"  ✅ {sym}: {float(ticker['last']):.8f} USDT")
        except Exception as e:
            log.warning(f"⚠️ {sym}: {e}")

    if test_count == 0:
        log.error("❌ Рынок мем-коинов недоступен")
        all_ok = False
    else:
        log.info(f"✅ Рынок OK ({test_count}/3 мем-коина доступны)")

    # 4. Позиции
    current_status = "Проверка позиций..."
    log.info("\n▶ Проверка позиций...")
    positions = get_positions()
    if positions:
        for p in positions:
            log.warning(f"⚠️ Уже открыта: {p.get('symbol')} {p.get('side')}")
    else:
        log.info("✅ Открытых позиций нет")

    # 5. Telegram
    current_status = "Проверка Telegram..."
    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        try:
            send_telegram_message(f"🤖 Бот {BOT_VERSION} проходит предстартовую проверку")
            log.info("✅ Telegram уведомления включены")
        except Exception as e:
            log.warning(f"⚠️ Ошибка отправки тестового уведомления: {e}")
    else:
        log.info("ℹ️ Telegram уведомления отключены")

    log.info("")
    log.info("=" * 65)
    if all_ok:
        log.info("✅ ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ")
        current_status = "Готов к торговле"
    else:
        log.error("❌ ЕСТЬ ОШИБКИ")
        current_status = "Ошибка предстартовой проверки"
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
        log.info(f"Заблокированы: {', '.join(blocked_symbols)}")

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
    log.info(f"🔒 {symbol} заблокирован на {minutes} мин")

# ============================================================
# ПРОВЕРКА СОЕДИНЕНИЯ
# ============================================================
def check_connection() -> bool:
    """Проверяет соединение с биржей."""
    try:
        exchange.fetch_ticker("BTC/USDT")  # ← Исправлено: используем fetch_ticker
        return True
    except Exception as e:
        log.error(f"Нет соединения с биржей: {e}")
        return False

# ============================================================
# СКАНИРОВАНИЕ
# ============================================================
def scan_for_trade() -> Tuple[Optional[str], int, float, Dict, str]:
    """Сканирует мем-коины в поисках сделок."""
    global stats, current_status
    current_status = f"Сканирование {len(SYMBOLS)} мем-коинов..."

    log.info(f"── {current_status} (баланс={get_free_balance():.2f}U) ──")

    candidates = []
    open_positions = get_positions()

    for sym in SYMBOLS:
        if is_symbol_blocked(sym):
            continue
        if not check_liquidity(sym):
            continue
        if not trend_4h_bullish(sym):
            continue

        result = get_score(sym)
        if result["score"] == 0:
            continue

        candidates.append((sym, result))
        log.debug(f"{sym}: скор={result['score']:3.0f}/100")

    if not candidates:
        current_status = "Нет кандидатов на вход"
        log.info("Нет кандидатов на лонг")
        return None, 0, 0.0, {}, "long"

    candidates.sort(key=lambda x: x[1]["score"], reverse=True)
    candidates = candidates[:10]

    for best, data in candidates:
        final_score = data["score"]
        price = data["price"]
        sr_info = data.get("sr", {})
        det = data.get("details", {})

        if sr_info.get("near_resistance") and sr_info.get("dist_to_res_pct", 99) < SR_BLOCK_DIST_PCT:
            log.info(f"⛔ {best}: сопротивление {sr_info.get('dist_to_res_pct', 0):.2f}%")
            continue
        if det.get("rsi", 50) > 65 and not sr_info.get("near_support"):
            log.info(f"⚠️ {best}: RSI={det.get('rsi', 50):.1f}")
            continue
        if not det.get("ema_cross", True):
            continue

        current_status = f"Найден кандидат: {best} (скор={final_score})"
        log.info(f"► Выбрана {best} (лонг) скор={final_score}")
        return best, final_score, price, sr_info, "long"

    current_status = "Поиск шортов..."
    log.info("Лонгов нет, ищем шорты...")
    for sym in SYMBOLS:
        if is_symbol_blocked(sym):
            continue
        if not check_liquidity(sym):
            continue
        if trend_4h_bearish(sym):
            short_res = get_score_short(sym)
            if short_res["score"] >= MIN_SCORE:
                current_status = f"Найден шорт: {sym} (скор={short_res['score']})"
                log.info(f"🐻 Шорт: {sym} скор={short_res['score']}")
                return sym, short_res["score"], short_res["price"], short_res.get("sr", {}), "short"

    current_status = "Нет подходящих сделок"
    return None, 0, 0.0, {}, "long"

# ============================================================
# ВЫВОД СТАТУСА
# ============================================================
def print_status():
    """Выводит текущий статус бота."""
    global current_status, last_action_time
    uptime = time.time() - last_action_time
    log.info(f"📌 Статус: {current_status} | Время с последнего действия: {uptime:.0f}с")

# ============================================================
# ГЛАВНЫЙ ЦИКЛ
# ============================================================
def main():
    global stats, current_status, last_action_time

    log.info("=" * 65)
    log.info(f"🤖 БОТ ДЛЯ МЕМ-КОИНОВ {BOT_VERSION}")
    log.info("=" * 65)

    if not preflight_check():
        log.error("🛑 Ошибки предстартовой проверки")
        if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
            send_telegram_message("🚨 Ошибки предстартовой проверки! Бот остановлен.")
        return

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
    log.info(f"Мем-коинов: {len(SYMBOLS)}")
    log.info(f"Мин. скор: {MIN_SCORE}")
    log.info(f"Quant: {'ВКЛ' if QUANT_ENABLED else 'ВЫКЛ'} | "
             f"Order Flow: {'ВКЛ' if ORDER_FLOW_ENABLED else 'ВЫКЛ'} | "
             f"ML: {'ВКЛ' if ML_ENABLED else 'ВЫКЛ'}")
    log.info("=" * 65)
    log.info("")

    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        send_telegram_message(
            f"🤖 Бот для мем-коинов {BOT_VERSION} запущен!\n"
            f"Баланс: {balance_now:.2f} USDT\n"
            f"Торгуемые пары: {', '.join(SYMBOLS)}"
        )

    current_status = "Ожидание возможностей для входа..."
    last_action_time = time.time()

    while True:
        try:
            if time.time() - last_action_time > 60:
                print_status()
                last_action_time = time.time()

            if not check_connection():
                current_status = "Нет соединения с биржей!"
                log.error("🚨 Нет соединения с биржей! Пауза 30 сек...")
                if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
                    send_telegram_message("🚨 Нет соединения с биржей! Бот на паузе.")
                time.sleep(30)
                continue

            if time.time() - stats.get("last_report", 0) >= REPORT_INTERVAL:
                print_report(stats)

            balance = get_total_balance()
            free_balance = get_free_balance()
            update_day_start(stats, balance)

            if free_balance < MIN_BALANCE:
                current_status = f"Низкий баланс: {free_balance:.2f} USDT"
                log.warning(f"🛑 Баланс {free_balance:.2f} < {MIN_BALANCE}. Пауза 5 мин.")
                if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
                    send_telegram_message(f"🛑 Низкий баланс: {free_balance:.2f} USDT")
                time.sleep(300)
                continue

            if stats["deposit_start"] > 0:
                drawdown = (stats["deposit_start"] - balance) / stats["deposit_start"] * 100
                if drawdown > MAX_DRAWDOWN_PCT:
                    current_status = f"Просадка {drawdown:.1f}%"
                    log.warning(f"⛔ Просадка {drawdown:.1f}% > {MAX_DRAWDOWN_PCT}%. Пауза 1ч.")
                    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
                        send_telegram_message(f"⛔ Просадка {drawdown:.1f}% > {MAX_DRAWDOWN_PCT}%!")
                    time.sleep(3600)
                    continue

            if is_daily_loss_exceeded(stats):
                current_status = "Дневной лимит убытков"
                log.warning(f"⛔ Дневной лимит убытков. Пауза {DAILY_LOSS_PAUSE_SEC // 60} мин.")
                if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
                    send_telegram_message("⛔ Дневной лимит убытков достигнут!")
                time.sleep(DAILY_LOSS_PAUSE_SEC)
                continue

            if not is_trading_time_allowed():
                current_status = "Заблокировано по времени"
                log.info("🕐 Заблокировано по времени. Пауза 1 мин.")
                time.sleep(60)
                continue

            if stats.get("sl_streak", 0) >= SL_STREAK_LIMIT:
                current_status = f"{SL_STREAK_LIMIT} SL подряд — cooldown"
                log.warning(f"🧊 {SL_STREAK_LIMIT} SL подряд — cooldown")
                stats["sl_streak"] = 0
                save_state(stats)
                save_blocked_symbols(stats)
                if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
                    send_telegram_message(f"🧊 {SL_STREAK_LIMIT} SL подряд — cooldown")
                time.sleep(SL_STREAK_PAUSE + SL_STREAK_EXTRA_PAUSE)
                continue

            active_positions = get_positions()
            if len(active_positions) >= MAX_OPEN_POSITIONS:
                current_status = f"Лимит позиций ({MAX_OPEN_POSITIONS})"
                log.info(f"⏳ Достигнут лимит открытых позиций ({MAX_OPEN_POSITIONS})")
                time.sleep(30)
                continue
            elif active_positions:
                current_status = f"Открытые позиции: {[p['symbol'] for p in active_positions]}"
                log.info(f"⏳ Открытые позиции: {[p['symbol'] for p in active_positions]}")
                time.sleep(30)
                continue

            current_status = "Сканирование рынка..."
            selected, score, price, sr_info, side = scan_for_trade()

            if selected is None:
                current_status = "Нет подходящих сделок"
                log.info(f"Нет кандидатов — ждём {SCAN_INTERVAL} сек")
                time.sleep(SCAN_INTERVAL)
                continue

            sl_price, tp_price, sl_dist_pct, real_rr = calc_sl_tp(selected, price, side, sr_info)
            current_status = f"Анализ {selected} (RR={real_rr:.1f}:1)"
            log.info(f"📐 SL={sl_dist_pct:.2f}% RR={real_rr:.1f}:1")

            if real_rr < MIN_RR_RATIO:
                current_status = f"Низкое RR для {selected}"
                log.warning(f"⛔ RR={real_rr:.1f}:1 < {MIN_RR_RATIO}:1")
                time.sleep(10)
                continue

            history = load_trades_history()
            margin = calc_position_size(score, free_balance, sl_dist_pct, history)
            if free_balance < margin * 1.1:
                log.warning(f"⚠️ Баланс меньше маржи")
                margin = free_balance * 0.8

            current_status = f"Готов к открытию {selected} (маржа={margin:.2f}U)"

            balance_before = get_total_balance()
            entry_time = time.time()
            current_status = f"Открытие позиции {selected}..."
            entry_price, qty = open_position_with_retries(selected, margin, tp_price, sl_price, side)

            if entry_price is None or qty is None:
                current_status = f"Не удалось открыть {selected}"
                log.warning("Не удалось открыть позицию — пауза 10 сек")
                time.sleep(10)
                continue

            if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
                send_telegram_message(
                    f"🆕 Открыта позиция: {selected} {side.upper()}\n"
                    f"Скор: {score}/100 | Маржа: {margin:.2f} USDT\n"
                    f"SL: {sl_price:.8f} | TP: {tp_price:.8f}"
                )

            current_status = f"Мониторинг {selected} (вход={entry_price:.8f})"
            stats["trades_total"] += 1
            save_state(stats)
            save_blocked_symbols(stats)

            result = "sl"
            try:
                result = monitor_position(selected, entry_price, qty, entry_time, sl_price, tp_price, side)
            except Exception as e:
                log.error(f"💥 Краш мониторинга: {e}")
                emergency_close_position(selected, side)
                result = "sl"

            time.sleep(3)
            balance_after = get_total_balance()
            real_pnl = balance_after - balance_before
            duration_min = (time.time() - entry_time) / 60

            if result == "tp":
                stats["take_profit"] += 1
                stats["profit_usdt"] += max(0, real_pnl)
                stats["sl_streak"] = 0
                current_status = f"TP: {selected} (+{real_pnl:+.4f}U)"
                log.info(f"✅ TP: +{real_pnl:+.4f} USDT")
                block_symbol(selected, SYMBOL_BLOCK_AFTER_TP)
                if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
                    send_telegram_message(f"✅ TP: {selected} | P&L: {real_pnl:+.4f} USDT")
            elif result == "sl":
                stats["stop_loss"] += 1
                stats["loss_usdt"] += abs(min(0, real_pnl))
                stats["sl_streak"] = stats.get("sl_streak", 0) + 1
                current_status = f"SL: {selected} ({real_pnl:+.4f}U)"
                log.warning(f"❌ SL: {real_pnl:+.4f} USDT (streak={stats['sl_streak']})")
                block_symbol(selected, SYMBOL_BLOCK_AFTER_SL)
                if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
                    send_telegram_message(f"❌ SL: {selected} | P&L: {real_pnl:+.4f} USDT (streak={stats['sl_streak']})")
            else:
                stats["timeout"] += 1
                stats["loss_usdt"] += abs(min(0, real_pnl))
                stats["sl_streak"] = 0
                current_status = f"Таймаут: {selected} ({real_pnl:+.4f}U)"
                log.warning(f"⏰ Таймаут: {real_pnl:+.4f} USDT")
                block_symbol(selected, SYMBOL_BLOCK_AFTER_TP)
                if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
                    send_telegram_message(f"⏰ Таймаут: {selected} | P&L: {real_pnl:+.4f} USDT")

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
                "bot_version": BOT_VERSION,
            }
            save_trade(trade_record)
            post_trade_analysis(trade_record)

            save_state(stats)
            save_blocked_symbols(stats)

            current_status = f"Сделка завершена ({result.upper()})"
            log.info(f"Сделка завершена ({result.upper()}) — пауза 10 сек")
            time.sleep(10)

        except KeyboardInterrupt:
            current_status = "Остановлен пользователем"
            log.info("\n👋 Остановка по Ctrl+C")
            save_state(stats)
            save_blocked_symbols(stats)
            if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
                send_telegram_message("👋 Бот остановлен пользователем")
            break

        except Exception as e:
            current_status = f"Ошибка: {str(e)[:50]}..."
            log.error(f"Глобальная ошибка: {e}", exc_info=True)
            if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
                send_telegram_message(f"🚨 Глобальная ошибка: {e}")
            time.sleep(30)

# ============================================================
# КОМАНДНАЯ СТРОКА
# ============================================================
def print_help():
    """Выводит справку."""
    print(f"""
{'=' * 65}
🤖 Бот для мем-коинов {BOT_VERSION}
{'=' * 65}

Использование:
    python bot.py           — Запустить бота
    python bot.py check     — Проверить конфигурацию
    python bot.py help      — Показать эту справку

Файлы конфигурации:
    .env        — API ключи (BYBIT_API_KEY, BYBIT_API_SECRET)
    config.py   — Настройки бота (мем-коины)
    engine.py   — Логика (индикаторы, анализ)
    bot.py      — Главный цикл

Дополнительные настройки:
    TELEGRAM_BOT_TOKEN — Токен бота для уведомлений
    TELEGRAM_CHAT_ID   — ID чата для уведомлений

Особенности:
    - Торговля только мем-коинами (PEPE, WIF, BONK и др.)
    - Частые сделки (сканирование каждые 30 секунд)
    - Агрессивная стратегия (высокий RR, увеличенный риск)
    - Уведомления в Telegram о всех действиях

{'=' * 65}
""")

def run_check():
    """Проверяет конфигурацию без запуска торговли."""
    print(f"\n{'=' * 65}")
    print(f"🔍 ПРОВЕРКА КОНФИГУРАЦИИ {BOT_VERSION}")
    print(f"{'=' * 65}\n")

    print("КОНФИГУРАЦИЯ:")
    print(f"  Плечо: {LEVERAGE}x")
    print(f"  TP: {TP_PERCENT}% | SL: {SL_PERCENT}% | RR: {TP_PERCENT / SL_PERCENT:.1f}:1")
    print(f"  Мин. скор: {MIN_SCORE}")
    print(f"  Риск: {BASE_RISK_PCT}-{MAX_RISK_PCT}%")
    print(f"  Макс. открытых позиций: {MAX_OPEN_POSITIONS}")
    print(f"  Мем-коинов: {len(SYMBOLS)}")
    print(f"  Интервал сканирования: {SCAN_INTERVAL} сек")
    print()

    print("API:")
    try:
        balance = exchange.fetch_balance({"type": "linear"})
        usdt = float(balance.get("USDT", {}).get("free", 0))
        print(f"  ✅ Подключение OK")
        print(f"  Баланс: {usdt:.2f} USDT")
    except Exception as e:
        print(f"  ❌ Ошибка: {e}")
    print()

    print("РЫНОК МЕМ-КОИНОВ:")
    for sym in SYMBOLS:
        try:
            ticker = exchange.fetch_ticker(sym)
            price = float(ticker["last"])
            print(f"  {sym}: {price:.8f} USDT")
        except Exception as e:
            print(f"  {sym}: ❌ {e}")
    print()

    print("ПОЗИЦИИ:")
    positions = get_positions()
    if positions:
        for p in positions:
            print(f"  ⚠️ {p.get('symbol')} {p.get('side')} qty={p.get('contracts')}")
    else:
        print(f"  Нет открытых позиций")
    print()

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

    print(f"{'=' * 65}\n")

# ============================================================
# ТОЧКА ВХОДА
# ============================================================
if __name__ == "__main__":
    args = sys.argv[1:]

    if not args:
        main()
    elif args[0] == "check":
        run_check()
    elif args[0] == "help":
        print_help()
    else:
        print(f"Неизвестная команда: {args[0]}")
        print("Используй: python bot.py [check|help]")
