"""
Tkinter GUI for the MT5 exhaustion pattern scanner with integrated backtesting.
Double-click any pattern to view M5 candles chart with FVG zones.
Click 'Run Backtest' to test trading strategy on detected patterns.
"""
import threading
import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
from typing import List, Dict, Tuple
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from matplotlib.patches import Rectangle

import mt5_core as mt5_core
import backtest_engine
import virtual_account


DEFAULT_MIN_CANDLE_PIPS = 50
DEFAULT_LOOKBACK_DAYS = 5


def _fetch_next_trading_day_m5(symbol: str, start_day, lookahead_days: int = 10):
    """Fetch M5 candles for the first day on/after start_day that actually has data."""
    # Do not force midnight: MT5 daily candles often start at a broker-specific hour.
    start_dt = pd.to_datetime(start_day).to_pydatetime().replace(microsecond=0)

    print(f"\n[DEBUG] Looking for next trading day starting from: {start_dt}")
    for i in range(max(0, int(lookahead_days)) + 1):
        day_start = start_dt + timedelta(days=i)
        day_end = day_start + timedelta(days=1)
        print(f"[DEBUG] Checking day {i}: {day_start} to {day_end}")
        m5 = mt5_core.fetch_m5_candles(symbol, day_start, day_end)
        if m5 is not None and len(m5) > 0:
            # Filter out any candles that are before our search start time (avoid C2 overlap)
            m5_filtered = m5[m5['time'] > start_dt]
            if len(m5_filtered) > 0:
                print(f"[DEBUG] Found {len(m5_filtered)} NEW candles! First: {m5_filtered.iloc[0]['time']}, Last: {m5_filtered.iloc[-1]['time']}")
                return m5_filtered.reset_index(drop=True), day_start, day_end
            print(f"[DEBUG] Found {len(m5)} candles but all overlap with previous day, skipping...")
        else:
            print(f"[DEBUG] No data for this day.")

    print(f"[DEBUG] No trading day found in {lookahead_days} day lookahead!")
    return None, None, None


def find_extreme_candle_index(m5_df: pd.DataFrame, pattern_type: str) -> int:
    """Find index of M5 candle with lowest low (TB Bullish) or highest high (TB Bearish)."""
    if pattern_type == 'TB Bullish':
        return m5_df['low'].idxmin()
    else:  # TB Bearish
        return m5_df['high'].idxmax()


def detect_bullish_fvg(c1, c2, c3) -> Tuple[bool, float, float]:
    """
    Detect bullish FVG: candle1.high < candle3.low
    Returns (is_fvg, gap_bottom, gap_top)
    """
    if c1['high'] < c3['low']:
        return True, c1['high'], c3['low']
    return False, 0, 0


def detect_bearish_fvg(c1, c2, c3) -> Tuple[bool, float, float]:
    """
    Detect bearish FVG: candle1.low > candle3.high
    Returns (is_fvg, gap_top, gap_bottom)
    """
    if c1['low'] > c3['high']:
        return True, c1['low'], c3['high']
    return False, 0, 0


def is_fvg_filled(fvg_top: float, fvg_bottom: float, subsequent_candles: pd.DataFrame) -> bool:
    """Check if FVG has been filled by subsequent price action."""
    for _, candle in subsequent_candles.iterrows():
        if candle['low'] <= fvg_top and candle['high'] >= fvg_bottom:
            return True
    return False


def find_unfilled_fvgs_structural(
    m5_df: pd.DataFrame,
    start_idx: int,
    pattern_type: str
) -> List[Dict]:
    """
    PASS 1: Detect unfilled FVGs only (no validation).
    """
    fvgs = []

    range_df = m5_df.loc[start_idx:].reset_index(drop=True)
    if len(range_df) < 3:
        return fvgs

    for i in range(len(range_df) - 2):
        c1 = range_df.iloc[i]
        c3 = range_df.iloc[i + 2]

        if pattern_type == "TB Bullish":
            if c1.high >= c3.low:
                continue
            bottom, top = c1.high, c3.low
            fvg_type = "Bullish"
        else:
            if c1.low <= c3.high:
                continue
            top, bottom = c1.low, c3.high
            fvg_type = "Bearish"

        subsequent = range_df.iloc[i + 3:]
        if is_fvg_filled(top, bottom, subsequent):
            continue

        fvgs.append({
            "start_time": c1.time,
            "end_time": range_df.iloc[-1].time,
            "top": top,
            "bottom": bottom,
            "type": fvg_type,
            "validation_levels": [],
            "is_validated": False
        })

    return fvgs


def validate_fvgs_by_price_projection(
    all_candles: pd.DataFrame,
    fvgs: List[Dict],
    lookahead: int = 12
) -> None:
    """
    PASS 2: Validate FVGs using price-only projection logic.
    Modifies fvgs in-place.
    """
    for fvg in fvgs:
        candles = all_candles[all_candles["time"] <= fvg["start_time"]]
        bottom = fvg["bottom"]
        top = fvg["top"]
        fvg_type = fvg["type"]

        validations = []

        for i in range(len(all_candles) - 2):
            c1 = all_candles.iloc[i]
            c2 = all_candles.iloc[i + 1]

            if fvg_type == "Bullish":
                reaction_ok = (
                    c1.close > c1.open and
                    bottom <= c1.close <= top and
                    c2.close < c2.open
                )
                displacement_ok = lambda c: c.close < bottom
            else:
                reaction_ok = (
                    c1.close < c1.open and
                    bottom <= c1.close <= top and
                    c2.close > c2.open
                )
                displacement_ok = lambda c: c.close > top

            if not reaction_ok:
                continue

            for j in range(i + 2, min(i + 2 + lookahead, len(candles))):
                disp_candle = candles.iloc[j]

                if displacement_ok(disp_candle):
                    reaction_level = c2.open
                    fvg_time = fvg["start_time"]
                    violated = False

                    for k in range(j + 1, len(all_candles)):
                        future_candle = all_candles.iloc[k]

                        if future_candle.time >= fvg_time:
                            break

                        if fvg_type == "Bullish" and future_candle.high > reaction_level:
                            violated = True
                            break

                        if fvg_type == "Bearish" and future_candle.low < reaction_level:
                            violated = True
                            break

                    if violated:
                        break

                    validations.append({
                        "level": reaction_level,
                        "time": c2.time
                    })
                    break

        if validations:
            fvg["validation_levels"] = validations[-1:]
            fvg["is_validated"] = True
            break


def get_validated_fvgs_for_pattern(symbol: str, c2_time, pattern_type: str) -> List[Dict]:
    """Get all validated FVGs for a pattern. Used for backtesting."""
    if not mt5_core.initialize_connection():
        return []
    
    try:
        if isinstance(c2_time, pd.Timestamp):
            c2_date = c2_time.to_pydatetime()
        else:
            c2_date = pd.to_datetime(c2_time).to_pydatetime()
        
        # Use the D1 candle timestamp as the 24h anchor (do not force midnight).
        start_time = pd.to_datetime(c2_time).to_pydatetime().replace(microsecond=0)
        end_time = start_time + timedelta(days=1)
        
        m5_df = mt5_core.fetch_m5_candles(symbol, start_time, end_time)
        
        if m5_df is None or len(m5_df) == 0:
            return []
        
        extreme_idx = find_extreme_candle_index(m5_df, pattern_type)
        unfilled_fvgs = find_unfilled_fvgs_structural(m5_df, extreme_idx, pattern_type)
        validate_fvgs_by_price_projection(m5_df, unfilled_fvgs)
        
        return [fvg for fvg in unfilled_fvgs if fvg['is_validated']]
    
    finally:
        mt5_core.terminate_connection()


def show_m5_chart(symbol: str, c2_time, pattern_type: str):
    """Display M5 candles for C2+C3 with FVG zones and trade levels."""
    if not mt5_core.initialize_connection():
        messagebox.showerror('Error', 'Failed to connect to MT5')
        return
    
    try:
        if isinstance(c2_time, pd.Timestamp):
            c2_date = c2_time.to_pydatetime()
        else:
            c2_date = pd.to_datetime(c2_time).to_pydatetime()
        
        # Use the D1 candle timestamp as the 24h anchor (do not force midnight).
        c2_start = pd.to_datetime(c2_time).to_pydatetime().replace(microsecond=0)
        c2_end = c2_start + timedelta(days=1)

        print(f"\n[INFO] Fetching C2 M5 data for {symbol}")
        print(f"[INFO] C2 window: {c2_start} to {c2_end}")
        m5_c2 = mt5_core.fetch_m5_candles(symbol, c2_start, c2_end)

        if m5_c2 is None or len(m5_c2) == 0:
            messagebox.showinfo('No Data', f'No M5 data available for {symbol} on {c2_date.strftime("%Y-%m-%d")}')
            return
        
        print(f"[INFO] C2 M5 data: {len(m5_c2)} candles, First: {m5_c2.iloc[0]['time']}, Last: {m5_c2.iloc[-1]['time']}")

        # Fetch C3 day (next trading day after C2) to show entry/SL/TP context
        print(f"\n[INFO] C2 ends at: {c2_end}, now looking for C3...")
        m5_c3, c3_start, c3_end = _fetch_next_trading_day_m5(symbol, c2_end, lookahead_days=10)
        
        # If no C3 found, use empty DataFrame
        if m5_c3 is None:
            m5_c3 = pd.DataFrame()
            c3_start = None
        
        extreme_idx = find_extreme_candle_index(m5_c2, pattern_type)
        unfilled_fvgs = find_unfilled_fvgs_structural(m5_c2, extreme_idx, pattern_type)
        validate_fvgs_by_price_projection(m5_c2, unfilled_fvgs)

        validated_count = sum(1 for fvg in unfilled_fvgs if fvg['is_validated'])

        # Pick the validated FVG farthest from C2 close (matches backtest selection)
        c2_close = float(m5_c2.iloc[-1]['close'])
        validated_fvgs = [f for f in unfilled_fvgs if f.get('is_validated') and f.get('validation_levels')]
        chosen_fvg = None
        if validated_fvgs:
            def _score(f):
                return abs(float(f['validation_levels'][0]['level']) - c2_close)
            chosen_fvg = max(validated_fvgs, key=_score)

        # Combine C2 + C3 for charting (ensure sorted by time)
        m5_df = pd.concat([m5_c2, m5_c3], ignore_index=True) if len(m5_c3) else m5_c2.copy()
        m5_df = m5_df.sort_values('time').reset_index(drop=True)
        
        chart_window = tk.Toplevel()
        chart_window.title(
            f"{symbol} - {pattern_type} - C2+C3 from {c2_date.strftime('%Y-%m-%d')} "
            f"| C2:{len(m5_c2)} candles C3:{len(m5_c3)} candles "
            f"| {len(unfilled_fvgs)} FVGs ({validated_count} validated)"
        )
        chart_window.geometry('1200x700')
        
        fig, ax = plt.subplots(figsize=(14, 7))
        
        for idx, row in m5_df.iterrows():
            color = 'green' if row['close'] > row['open'] else 'red'
            
            ax.plot([row['time'], row['time']], [row['open'], row['close']], 
                   color=color, linewidth=2, solid_capstyle='round')
            
            ax.plot([row['time'], row['time']], [row['low'], row['high']], 
                   color=color, linewidth=0.5)
        
        extreme_candle = m5_c2.iloc[extreme_idx]
        extreme_label = 'Lowest Low' if pattern_type == 'TB Bullish' else 'Highest High'
        ax.scatter(extreme_candle['time'], 
                  extreme_candle['low'] if pattern_type == 'TB Bullish' else extreme_candle['high'],
                  color='blue', s=100, zorder=5, marker='v' if pattern_type == 'TB Bullish' else '^',
                  label=extreme_label)
        
        last_time = m5_df.iloc[-1]['time']

        # Visually separate C2 and C3 (only if C3 exists)
        if c3_start is not None and len(m5_c3) > 0:
            ax.axvline(c3_start, color='gray', linestyle='--', linewidth=1.0, alpha=0.6, label='C3 Start')
        else:
            # If C3 is missing, make it explicit on chart
            ax.text(0.01, 0.98, 'No C3 M5 data available (holiday/weekend)', transform=ax.transAxes,
                fontsize=10, ha='left', va='top', color='red')
        
        for fvg in unfilled_fvgs:
            fvg_color = 'lightgreen' if fvg['type'] == 'Bullish' else 'lightcoral'
            edge_color = 'darkgreen' if fvg['type'] == 'Bullish' else 'darkred'
            
            alpha = 0.4 if fvg['is_validated'] else 0.2
            edge_width = 2 if fvg['is_validated'] else 1
            
            start_time_mpl = mdates.date2num(fvg['start_time'])
            end_time_mpl = mdates.date2num(fvg['end_time'])
            width = end_time_mpl - start_time_mpl
            height = fvg['top'] - fvg['bottom']
            
            rect = Rectangle((start_time_mpl, fvg['bottom']), width, height,
                           facecolor=fvg_color, alpha=alpha, edgecolor=edge_color, 
                           linewidth=edge_width, label=f"{fvg['type']} FVG")
            ax.add_patch(rect)
            
            if fvg['validation_levels']:
                line_color = 'darkblue' if fvg['type'] == 'Bullish' else 'purple'
                for val in fvg['validation_levels']:
                    ax.plot([val['time'], last_time], [val['level'], val['level']], 
                           color=line_color, linestyle='--', linewidth=1.5, 
                           alpha=0.7, label='Validation Level')

        # Overlay trade levels for the chosen validated FVG
        if chosen_fvg is not None:
            validation_level = float(chosen_fvg['validation_levels'][0]['level'])
            validation_time = chosen_fvg['validation_levels'][0]['time']
            fvg_formation_time = chosen_fvg['start_time']
            fvg_type = chosen_fvg['type']

            validation_ts = pd.to_datetime(validation_time)
            formation_ts = pd.to_datetime(fvg_formation_time)
            invalid_validation_time = validation_ts >= formation_ts

            # SL/TP for visualization:
            # - If timing is valid, use backtest rule.
            # - If timing is invalid, still compute a "viz SL" from the min/max time window so the user can see levels,
            #   but clearly label as INVALID (not a valid setup).
            sl_level = backtest_engine.calculate_sl_level(
                m5_c2,
                validation_ts,
                formation_ts,
                fvg_type
            )

            if sl_level is None and invalid_validation_time:
                t0 = min(validation_ts, formation_ts)
                t1 = max(validation_ts, formation_ts)
                range_candles = m5_c2[(m5_c2['time'] >= t0) & (m5_c2['time'] <= t1)]
                if len(range_candles):
                    sl_level = float(range_candles['low'].min()) if fvg_type == 'Bullish' else float(range_candles['high'].max())

            # Calculate TP using C1 midpoint
            print(f"\n[INFO] Calculating TP for {symbol} with C2 date: {c2_date}")
            tp_level = backtest_engine.get_c1_midpoint(symbol, c2_date)
            print(f"[INFO] TP level (C1 midpoint): {tp_level}")
            
            # Validate entry vs TP
            invalid_tp = False
            if tp_level is not None:
                if fvg_type == 'Bullish' and validation_level >= tp_level:
                    invalid_tp = True
                elif fvg_type == 'Bearish' and validation_level <= tp_level:
                    invalid_tp = True

            # Check if entry touched on C3
            entry_text = 'Not Activated'
            entry_time = None
            entry_candle_idx = None
            if len(m5_c3):
                entry_result = backtest_engine.check_entry_triggered(m5_c3, validation_level, fvg_type)
                if entry_result is not None:
                    entry_time, entry_price, entry_candle_idx = entry_result
                    entry_text = f"Activated @ {pd.to_datetime(entry_time).strftime('%H:%M')}"

            if invalid_validation_time:
                entry_text = f"{entry_text} | INVALID validation time"
            
            if invalid_tp:
                entry_text = f"{entry_text} | INVALID TP (entry/TP mismatch)"

            # Draw horizontal lines from C3 start (or entry time if activated) to end of chart
            line_start = m5_c3.iloc[0]['time'] if len(m5_c3) else m5_c2.iloc[-1]['time']
            if entry_time is not None:
                line_start = entry_time

            ax.plot([line_start, last_time], [validation_level, validation_level],
                    color='black', linestyle='--', linewidth=2.0, label=f'Entry ({entry_text})')

            if sl_level is not None:
                ax.plot([line_start, last_time], [float(sl_level), float(sl_level)],
                        color='black', linestyle=':', linewidth=1.8, label='Stop Loss')
            else:
                ax.text(0.01, 0.02, 'SL/TP unavailable (insufficient data)', transform=ax.transAxes,
                        fontsize=9, ha='left', va='bottom')

            if tp_level is not None:
                ax.plot([line_start, last_time], [float(tp_level), float(tp_level)],
                        color='black', linestyle='-', linewidth=1.8, label='Take Profit')

            if entry_time is not None:
                ax.scatter(entry_time, validation_level, color='black', s=60, zorder=6, marker='o', label='Entry Fill')

                # Mark the first SL/TP hit on C3 (if any), so it's visually obvious
                if len(m5_c3) and sl_level is not None and tp_level is not None and entry_candle_idx is not None:
                    exit_marker_added = False
                    for i in range(entry_candle_idx, len(m5_c3)):
                        candle = m5_c3.iloc[i]
                        outcome = backtest_engine.check_sl_tp_hit_on_candle(
                            candle, validation_level, float(sl_level), float(tp_level), fvg_type
                        )
                        if outcome == 'discard':
                            ax.scatter(candle['time'], validation_level, color='orange', s=70, zorder=7, marker='x',
                                       label='Discarded (SL+TP same candle)')
                            exit_marker_added = True
                            break
                        if outcome == 'loss':
                            ax.scatter(candle['time'], float(sl_level), color='red', s=70, zorder=7, marker='x',
                                       label='SL Hit (C3)')
                            exit_marker_added = True
                            break
                        if outcome == 'win':
                            ax.scatter(candle['time'], float(tp_level), color='green', s=70, zorder=7, marker='x',
                                       label='TP Hit (C3)')
                            exit_marker_added = True
                            break

                    if not exit_marker_added:
                        ax.text(0.01, 0.94, 'No SL/TP hit on C3 (may hit later)', transform=ax.transAxes,
                                fontsize=9, ha='left', va='top')
        
        ax.set_xlabel('Time', fontsize=10)
        ax.set_ylabel('Price', fontsize=10)
        
        title = f'{symbol} - M5 Candles - C2+C3 from {c2_date.strftime("%Y-%m-%d")} - {pattern_type}'
        if unfilled_fvgs:
            title += f' - {len(unfilled_fvgs)} FVG(s) ({validated_count} validated)'
        ax.set_title(title, fontsize=12, fontweight='bold')
        
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
        plt.xticks(rotation=45)
        
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper left')
        
        fig.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, master=chart_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
        
        if unfilled_fvgs:
            print(f"\n{'='*80}")
            print(f"UNFILLED FVGs for {symbol} on {c2_date.strftime('%Y-%m-%d')} ({pattern_type})")
            print(f"{'='*80}")
            for i, fvg in enumerate(unfilled_fvgs, 1):
                validation_status = "✓ VALIDATED" if fvg['is_validated'] else "✗ Not Validated"
                print(f"\nFVG #{i} ({fvg['type']}) - {validation_status}:")
                print(f"  Start Time: {fvg['start_time'].strftime('%Y-%m-%d %H:%M')}")
                print(f"  Top: {fvg['top']:.5f}")
                print(f"  Bottom: {fvg['bottom']:.5f}")
                print(f"  Size: {(fvg['top'] - fvg['bottom']):.5f}")
                if fvg['validation_levels']:
                    print(f"  Validation Levels:")
                    for j, val in enumerate(fvg['validation_levels'], 1):
                        print(f"    #{j}: {val['level']:.5f} at {val['time'].strftime('%H:%M')}")
            print(f"\nTotal: {len(unfilled_fvgs)} FVGs ({validated_count} validated)")
            print(f"{'='*80}\n")
        
    except Exception as e:
        messagebox.showerror('Error', f'Failed to fetch M5 data: {str(e)}')
    finally:
        mt5_core.terminate_connection()


def show_backtest_results(results: List[Dict], stats: Dict):
    """Display backtest results in a new window."""
    result_window = tk.Toplevel()
    result_window.title('Backtest Results')
    result_window.geometry('900x750')
    
    # Calculate R-multiple distribution for wins and losses
    winning_trades = [r for r in results if r.get('outcome') == 'win' and r.get('r_multiple') is not None]
    losing_trades = [r for r in results if r.get('outcome') == 'loss' and r.get('r_multiple') is not None]
    
    # Classify winning trades by R achieved
    wins_0_1 = sum(1 for t in winning_trades if 0 <= t['r_multiple'] < 1)
    wins_1_3 = sum(1 for t in winning_trades if 1 <= t['r_multiple'] < 3)
    wins_3_6 = sum(1 for t in winning_trades if 3 <= t['r_multiple'] < 6)
    wins_6_plus = sum(1 for t in winning_trades if t['r_multiple'] >= 6)
    
    # For losing trades, show what their potential R was (i.e., if they had won)
    # The potential R is stored as 'potential_r' in backtest results
    losses_0_1 = sum(1 for t in losing_trades if t.get('potential_r', 0) < 1)
    losses_1_3 = sum(1 for t in losing_trades if 1 <= t.get('potential_r', 0) < 3)
    losses_3_6 = sum(1 for t in losing_trades if 3 <= t.get('potential_r', 0) < 6)
    losses_6_plus = sum(1 for t in losing_trades if t.get('potential_r', 0) >= 6)
    
    # Statistics Frame
    stats_frame = ttk.LabelFrame(result_window, text='Statistics', padding=10)
    stats_frame.pack(fill='x', padx=10, pady=10)
    
    stats_text = f"""
Total Patterns Analyzed: {stats.get('total_patterns', 0)}
Entries Triggered: {stats.get('entries_triggered', 0)} ({stats.get('entry_rate', 0):.1f}%)

Total Trades Completed: {stats.get('total_trades', 0)}
Wins: {stats.get('wins', 0)}
Losses: {stats.get('losses', 0)}
Discarded: {stats.get('discarded', 0)}
Open Trades: {stats.get('open_trades', 0)}
Pending Trades: {stats.get('pending_trades', 0)}

Win Rate: {stats.get('win_rate', 0):.1f}%
Average R-Multiple: {stats.get('average_r', 0):.2f}R
Total R: {stats.get('total_r', 0):.2f}R

Total Pips: {stats.get('total_pips', 0):.1f}
Average Pips per Trade: {stats.get('average_pips', 0):.1f}
Average Hours Held: {stats.get('avg_hours_held', 0):.1f}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WINNING TRADES R-MULTIPLE DISTRIBUTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
0-1R:    {wins_0_1:3d} trades ({(wins_0_1/len(winning_trades)*100) if winning_trades else 0:.1f}%)
1-3R:    {wins_1_3:3d} trades ({(wins_1_3/len(winning_trades)*100) if winning_trades else 0:.1f}%)
3-6R:    {wins_3_6:3d} trades ({(wins_3_6/len(winning_trades)*100) if winning_trades else 0:.1f}%)
6+R:     {wins_6_plus:3d} trades ({(wins_6_plus/len(winning_trades)*100) if winning_trades else 0:.1f}%)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LOSING TRADES - POTENTIAL R IF WON
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
0-1R:    {losses_0_1:3d} trades ({(losses_0_1/len(losing_trades)*100) if losing_trades else 0:.1f}%)
1-3R:    {losses_1_3:3d} trades ({(losses_1_3/len(losing_trades)*100) if losing_trades else 0:.1f}%)
3-6R:    {losses_3_6:3d} trades ({(losses_3_6/len(losing_trades)*100) if losing_trades else 0:.1f}%)
6+R:     {losses_6_plus:3d} trades ({(losses_6_plus/len(losing_trades)*100) if losing_trades else 0:.1f}%)
    """
    
    stats_label = ttk.Label(stats_frame, text=stats_text, justify='left', font=('Courier', 10))
    stats_label.pack()
    
    # Buttons Frame (at the top)
    buttons_frame = ttk.Frame(result_window, padding=10)
    buttons_frame.pack(fill='x', padx=10)
    
    # Virtual Account Simulation Button
    def simulate_account():
        """Launch virtual account simulation with equity curve visualization."""
        # Filter only win and loss trades (note: backtest uses 'win' and 'loss', not 'won'/'lost')
        completed_trades = [r for r in results if r.get('outcome') in ['win', 'loss']]
        
        if not completed_trades:
            messagebox.showinfo('No Trades', 'No completed trades (win/loss) to simulate.')
            return
        
        # Prepare trade data for simulation
        trade_data = []
        for trade in completed_trades:
            if trade.get('entry_time') and trade.get('exit_time'):
                # Map 'win'/'loss' to 'won'/'lost' for virtual_account module
                result_status = 'won' if trade['outcome'] == 'win' else 'lost'
                trade_data.append({
                    'entry_time': trade['entry_time'],
                    'exit_time': trade['exit_time'],
                    'r_multiple': trade.get('r_multiple', 0),
                    'result': result_status,
                    'symbol': trade.get('symbol', 'N/A')
                })
        
        if not trade_data:
            messagebox.showinfo('No Valid Trades', 'No trades have both entry and exit times.')
            return
        
        print(f"\n[DEBUG] Starting virtual account simulation with {len(trade_data)} trades")
        
        # Run simulation
        initial_balance = 5000
        risk_pct = 1.0
        equity_curve, trade_results = virtual_account.simulate_virtual_account(
            trade_data, initial_balance, risk_pct
        )
        
        print(f"[DEBUG] Simulation complete. Equity curve points: {len(equity_curve)}")
        
        # Calculate statistics
        account_stats = virtual_account.calculate_account_statistics(
            equity_curve, trade_results, initial_balance
        )
        
        print(f"[DEBUG] Creating NEW INDEPENDENT window (not parented to result_window)")
        
        # Create independent visualization window
        sim_window = tk.Toplevel()
        print(f"[DEBUG] Toplevel created: {sim_window}")
        print(f"[DEBUG] Parent window: {sim_window.master}")
        
        sim_window.title('Virtual Account Simulation')
        
        # Larger window by default (chart is tall)
        sim_window.geometry('1200x950+100+80')
        
        print(f"[DEBUG] Window configured at position +100+100, starting to add widgets")
        
        # Scrollable container (so large charts + stats remain usable on smaller screens)
        container = ttk.Frame(sim_window)
        container.pack(fill='both', expand=True)

        scroll_canvas = tk.Canvas(container, highlightthickness=0)
        vscroll = ttk.Scrollbar(container, orient='vertical', command=scroll_canvas.yview)
        scroll_canvas.configure(yscrollcommand=vscroll.set)

        vscroll.pack(side='right', fill='y')
        scroll_canvas.pack(side='left', fill='both', expand=True)

        content = ttk.Frame(scroll_canvas)
        content_window = scroll_canvas.create_window((0, 0), window=content, anchor='nw')

        def _on_content_configure(_event=None):
            scroll_canvas.configure(scrollregion=scroll_canvas.bbox('all'))

        def _on_canvas_configure(event):
            # Keep the content frame the same width as the visible area
            scroll_canvas.itemconfigure(content_window, width=event.width)

        content.bind('<Configure>', _on_content_configure)
        scroll_canvas.bind('<Configure>', _on_canvas_configure)

        def _on_mousewheel(event):
            # Windows: event.delta is typically +/-120 per notch
            if event.delta:
                scroll_canvas.yview_scroll(int(-1 * (event.delta / 120)), 'units')

        # Bind mousewheel when cursor is over the window
        sim_window.bind_all('<MouseWheel>', _on_mousewheel)

        # Statistics Frame
        sim_stats_frame = ttk.LabelFrame(content, text='Account Performance', padding=10)
        sim_stats_frame.pack(fill='x', padx=10, pady=(10, 6))
        
        sim_stats_text = f"""
Initial Balance: ${account_stats['initial_balance']:.2f}
Final Balance: ${account_stats['final_balance']:.2f}
Total Return: {account_stats['total_return']:.2f}% (${account_stats['total_return_dollars']:.2f})
Max Drawdown: ${account_stats['max_drawdown']:.2f} ({account_stats['max_drawdown_pct']:.2f}%)

Total Trades: {account_stats['total_trades']}
Winning Trades: {account_stats['winning_trades']}
Losing Trades: {account_stats['losing_trades']}
Win Rate: {account_stats['win_rate']:.1f}%

Average Win: ${account_stats['avg_win']:.2f}
Average Loss: ${account_stats['avg_loss']:.2f}
Profit Factor: {account_stats['profit_factor']:.2f}

Risk Per Trade: {risk_pct}% of current balance
        """
        
        sim_stats_label = ttk.Label(sim_stats_frame, text=sim_stats_text,
                                     justify='left', font=('Courier', 10))
        sim_stats_label.pack()
        
        # Equity Curve Chart
        chart_frame = ttk.LabelFrame(content, text='Equity Curve', padding=10)
        chart_frame.pack(fill='both', expand=True, padx=10, pady=(6, 10))

        # A cleaner looking style without changing global matplotlib settings
        with plt.style.context('seaborn-v0_8-darkgrid'):
            fig, (ax, ax_dd) = plt.subplots(
                2,
                1,
                figsize=(12, 9),
                sharex=True,
                gridspec_kw={'height_ratios': [3, 1]},
            )
        
        # Build plot series based on ENTRIES (not exits)
        # Note: the balance changes are still realized after each trade, but we anchor them on entry timestamps
        # to match the requested visualization.
        plot_points = []
        if trade_results:
            first_entry = pd.to_datetime(trade_results[0].get('entry_time') or trade_results[0].get('exit_time'))
            plot_points.append((first_entry, float(initial_balance)))
            for t in trade_results:
                ts = pd.to_datetime(t.get('entry_time') or t.get('exit_time'))
                plot_points.append((ts, float(t.get('balance_after', initial_balance))))
        else:
            # Fallback: keep something valid for plotting
            plot_points = [(pd.to_datetime(equity_curve[0][0]), float(equity_curve[0][1]))]

        timestamps = [p[0] for p in plot_points]
        balances = [p[1] for p in plot_points]

        # Compute drawdown series (useful context)
        peak = float(balances[0]) if balances else float(initial_balance)
        dd_pct = []
        dd_ts = []
        for ts, bal in zip(timestamps, balances):
            b = float(bal)
            if b > peak:
                peak = b
            dd_ts.append(ts)
            dd_pct.append(((b - peak) / peak) * 100 if peak > 0 else 0.0)

        # Plot equity curve (anchored on entry timestamps)
        ax.plot(timestamps, balances, linewidth=2.5, label='Account Balance')
        
        # Add horizontal line for initial balance
        ax.axhline(y=initial_balance, linestyle='--', linewidth=1.5, label=f'Initial Balance (${initial_balance})')
        
        # Mark winning and losing trades (at entry timestamps)
        for trade in trade_results:
            marker_time = pd.to_datetime(trade.get('entry_time') or trade.get('exit_time'))
            if trade['result'] == 'won':
                ax.scatter(marker_time, trade['balance_after'],
                           color='green', s=45, alpha=0.65, marker='^', zorder=5)
            else:
                ax.scatter(marker_time, trade['balance_after'],
                           color='red', s=45, alpha=0.65, marker='v', zorder=5)
        
        # Formatting
        ax.set_ylabel('Account Balance ($)', fontsize=12)
        ax.set_title('Virtual Account Equity Curve', fontsize=14, fontweight='bold')
        ax.yaxis.set_major_formatter(mticker.StrMethodFormatter('${x:,.0f}'))

        # Drawdown subplot
        ax_dd.fill_between(dd_ts, dd_pct, 0, alpha=0.25)
        ax_dd.plot(dd_ts, dd_pct, linewidth=1.25)
        ax_dd.set_ylabel('Drawdown %', fontsize=11)
        ax_dd.yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:.0f}%'))
        ax_dd.axhline(0, linewidth=1)

        locator = mdates.AutoDateLocator()
        ax_dd.xaxis.set_major_locator(locator)
        ax_dd.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        fig.autofmt_xdate()
        
        # Add custom legend
        from matplotlib.lines import Line2D
        custom_lines = [
            Line2D([0], [0], linewidth=2.5),
            Line2D([0], [0], linestyle='--', linewidth=1.5),
            Line2D([0], [0], marker='^', color='w', markerfacecolor='green', markersize=8),
            Line2D([0], [0], marker='v', color='w', markerfacecolor='red', markersize=8)
        ]
        ax.legend(custom_lines, ['Equity Curve', f'Initial (${initial_balance})', 
                                 'Win', 'Loss'], loc='upper left')

        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=chart_frame)
        toolbar = NavigationToolbar2Tk(canvas, chart_frame)
        toolbar.update()

        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
        
        print(f"[DEBUG] Chart canvas created and packed")
        print(f"[DEBUG] Simulation window should now be visible as SEPARATE window")
        print(f"[DEBUG] If it appears inside backtest window, this is a Tkinter parent issue")
    
    simulate_btn = ttk.Button(buttons_frame, text='Simulate Virtual Account', command=simulate_account)
    simulate_btn.pack(side='left', padx=5)
    
    # Export Button
    def export_results():
        fp = filedialog.asksaveasfilename(defaultextension='.csv', filetypes=[('CSV', '*.csv')])
        if not fp:
            return
        
        df = pd.DataFrame(results)
        df.to_csv(fp, index=False)
        messagebox.showinfo('Saved', f'Exported {len(results)} results to {fp}')
    
    export_btn = ttk.Button(buttons_frame, text='Export to CSV', command=export_results)
    export_btn.pack(side='left', padx=5)
    
    # Results Table
    table_frame = ttk.LabelFrame(result_window, text='Trade Details', padding=10)
    table_frame.pack(fill='both', expand=True, padx=10, pady=10)
    
    columns = ('symbol', 'date', 'type', 'entry', 'outcome', 'pips', 'r_mult', 'hours')
    tree = ttk.Treeview(table_frame, columns=columns, show='headings', height=15)
    
    tree.heading('symbol', text='Symbol')
    tree.heading('date', text='FVG Date')
    tree.heading('type', text='Type')
    tree.heading('entry', text='Entry')
    tree.heading('outcome', text='Outcome')
    tree.heading('pips', text='Pips')
    tree.heading('r_mult', text='R')
    tree.heading('hours', text='Hours')
    
    tree.column('symbol', width=80)
    tree.column('date', width=100)
    tree.column('type', width=80)
    tree.column('entry', width=80)
    tree.column('outcome', width=80)
    tree.column('pips', width=80)
    tree.column('r_mult', width=60)
    tree.column('hours', width=60)
    
    vsb = ttk.Scrollbar(table_frame, orient='vertical', command=tree.yview)
    tree.configure(yscroll=vsb.set)
    tree.pack(side='left', fill='both', expand=True)
    vsb.pack(side='right', fill='y')
    
    # Populate results
    for result in results:
        if result.get('entry_triggered', False):
            date_str = result['fvg_date'].strftime('%Y-%m-%d')
            entry_str = 'Yes' if result.get('entry_time') else 'No'
            outcome = result.get('outcome', 'N/A')
            pips = result.get('pips', 0)
            r_mult = result.get('r_multiple', 0)
            hours = result.get('hours_held', 0)
            
            tree.insert('', 'end', values=(
                result['symbol'],
                date_str,
                result['fvg_type'],
                entry_str,
                outcome,
                f"{pips:.1f}",
                f"{r_mult:.1f}",
                f"{hours:.1f}"
            ))


def create_gui() -> tk.Tk:
    root = tk.Tk()
    root.title('MT5 Exhaustion Pattern Scanner with Backtest')
    root.geometry('800x550')

    frm = ttk.Frame(root, padding=10)
    frm.pack(fill='x')

    ttk.Label(frm, text='Min candle size (pips):').grid(row=0, column=0, sticky='w')
    min_pips_var = tk.StringVar(value=str(DEFAULT_MIN_CANDLE_PIPS))
    min_entry = ttk.Entry(frm, textvariable=min_pips_var, width=10)
    min_entry.grid(row=0, column=1, padx=6)

    # Date range inputs (dates only)
    today = datetime.now().date()
    default_start = (today - timedelta(days=DEFAULT_LOOKBACK_DAYS)).strftime('%Y-%m-%d')
    default_end = today.strftime('%Y-%m-%d')

    ttk.Label(frm, text='Start date (YYYY-MM-DD):').grid(row=1, column=0, sticky='w')
    start_date_var = tk.StringVar(value=default_start)
    start_entry = ttk.Entry(frm, textvariable=start_date_var, width=14)
    start_entry.grid(row=1, column=1, padx=6, sticky='w')

    ttk.Label(frm, text='End date (YYYY-MM-DD):').grid(row=2, column=0, sticky='w')
    end_date_var = tk.StringVar(value=default_end)
    end_entry = ttk.Entry(frm, textvariable=end_date_var, width=14)
    end_entry.grid(row=2, column=1, padx=6, sticky='w')

    status_var = tk.StringVar(value='Idle')
    status_lbl = ttk.Label(frm, textvariable=status_var)
    status_lbl.grid(row=0, column=2, padx=10, rowspan=3)

    btn_frame = ttk.Frame(frm)
    btn_frame.grid(row=0, column=3, padx=6, rowspan=3)

    start_btn = ttk.Button(btn_frame, text='Start Scan')
    stop_btn = ttk.Button(btn_frame, text='Stop', state='disabled')
    backtest_btn = ttk.Button(btn_frame, text='Run Backtest', state='disabled')
    save_btn = ttk.Button(btn_frame, text='Save CSV', state='disabled')

    start_btn.grid(row=0, column=0, padx=2)
    stop_btn.grid(row=0, column=1, padx=2)
    backtest_btn.grid(row=0, column=2, padx=2)
    save_btn.grid(row=0, column=3, padx=2)

    list_frame = ttk.Frame(root, padding=(10, 6))
    list_frame.pack(fill='both', expand=True)

    columns = ('symbol', 'date', 'type')
    tree = ttk.Treeview(list_frame, columns=columns, show='headings')
    tree.heading('symbol', text='Symbol')
    tree.heading('date', text='Date')
    tree.heading('type', text='Pattern')
    
    tree.column('symbol', width=100)
    tree.column('date', width=100)
    tree.column('type', width=80)

    vsb = ttk.Scrollbar(list_frame, orient='vertical', command=tree.yview)
    tree.configure(yscroll=vsb.set)
    tree.pack(side='left', fill='both', expand=True)
    vsb.pack(side='right', fill='y')

    stop_event = threading.Event()
    scan_thread = None
    found_patterns: List[Dict] = []

    def on_progress(text: str):
        root.after(0, status_var.set, text)

    def on_found(patt: Dict):
        def _insert():
            date_str = patt['date'].strftime('%Y-%m-%d')
            
            tree.insert('', 'end', values=(
                patt['symbol'], 
                date_str, 
                patt['pattern_type'],
            ))
            found_patterns.append(patt)
            save_btn.config(state='normal')
            backtest_btn.config(state='normal')

        root.after(0, _insert)

    def on_tree_double_click(event):
        selection = tree.selection()
        if not selection:
            return
        
        item = tree.item(selection[0])
        values = item['values']
        
        if len(values) >= 3:
            symbol = values[0]
            date_str = values[1]
            pattern_type = values[2]
            
            for patt in found_patterns:
                if (patt['symbol'] == symbol and 
                    patt['date'].strftime('%Y-%m-%d') == date_str and
                    patt['pattern_type'] == pattern_type):
                    show_m5_chart(symbol, patt['date'], pattern_type)
                    break

    tree.bind('<Double-Button-1>', on_tree_double_click)

    def start_scan():
        nonlocal scan_thread, stop_event, found_patterns
        try:
            min_pips = int(min_pips_var.get())
        except ValueError:
            messagebox.showerror('Input error', 'Min pips must be an integer')
            return

        try:
            start_dt = datetime.strptime(start_date_var.get().strip(), '%Y-%m-%d')
            end_dt = datetime.strptime(end_date_var.get().strip(), '%Y-%m-%d')
        except ValueError:
            messagebox.showerror('Input error', 'Dates must be in YYYY-MM-DD format')
            return

        if start_dt > end_dt:
            messagebox.showerror('Input error', 'Start date must be <= end date')
            return

        for i in tree.get_children():
            tree.delete(i)
        found_patterns = []
        stop_event.clear()

        start_btn.config(state='disabled')
        stop_btn.config(state='normal')
        save_btn.config(state='disabled')
        backtest_btn.config(state='disabled')
        status_var.set('Initializing...')

        def worker():
            mt5_core.scan_all(min_pips, on_progress, on_found, stop_event, start_date=start_dt, end_date=end_dt)
            root.after(0, lambda: start_btn.config(state='normal'))
            root.after(0, lambda: stop_btn.config(state='disabled'))

        scan_thread = threading.Thread(target=worker, daemon=True)
        scan_thread.start()

    def stop_scan():
        stop_event.set()
        status_var.set('Stopping...')

    def run_backtest():
        if not found_patterns:
            messagebox.showinfo('No Data', 'No patterns to backtest')
            return
        
        backtest_btn.config(state='disabled')
        status_var.set('Preparing backtest...')
        
        def backtest_worker():
            # Add validated FVGs to patterns
            patterns_with_fvgs = []
            for pattern in found_patterns:
                validated_fvgs = get_validated_fvgs_for_pattern(
                    pattern['symbol'],
                    pattern['date'],
                    pattern['pattern_type']
                )
                pattern['validated_fvgs'] = validated_fvgs
                patterns_with_fvgs.append(pattern)
            
            results, stats = backtest_engine.run_backtest_on_patterns(
                patterns_with_fvgs,
                progress_callback=on_progress
            )
            
            root.after(0, lambda: show_backtest_results(results, stats))
            root.after(0, lambda: backtest_btn.config(state='normal'))
            root.after(0, lambda: status_var.set('Backtest complete'))
        
        thread = threading.Thread(target=backtest_worker, daemon=True)
        thread.start()

    def save_csv():
        if not found_patterns:
            messagebox.showinfo('No data', 'No patterns to save')
            return
        fp = filedialog.asksaveasfilename(defaultextension='.csv', filetypes=[('CSV', '*.csv')])
        if not fp:
            return
        
        df = pd.DataFrame(found_patterns)
        df.to_csv(fp, index=False)
        messagebox.showinfo('Saved', f'Saved {len(found_patterns)} patterns to {fp}')

    start_btn.config(command=start_scan)
    stop_btn.config(command=stop_scan)
    backtest_btn.config(command=run_backtest)
    save_btn.config(command=save_csv)

    root.protocol('WM_DELETE_WINDOW', root.destroy)
    return root


if __name__ == '__main__':
    app = create_gui()
    app.mainloop()