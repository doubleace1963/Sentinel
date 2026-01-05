import MetaTrader5 as mt5
import pandas as pd
from typing import List, Dict, Callable
import threading
from datetime import datetime, timedelta



def initialize_connection() -> bool:
    if not mt5.initialize():
        return False
    return True


def terminate_connection() -> None:
    """Shutdown MT5 gracefully."""
    try:
        mt5.shutdown()
    except Exception:
        pass


def get_available_symbols() -> List[str]:
    """Return list of forex majors/minors, metals (XAU/XAG), and select crypto (BTC/ETH).
    
    Filters for:
    - Major/minor forex pairs (e.g., EURUSD, GBPJPY)
    - XAGUSD (silver), XAUUSD (gold)
    - BTCUSD, ETHUSD
    
    Works with both legacy brokers (FX suffix/path) and modern ones (Zero\Forex\ path).
    """
    symbols = mt5.symbols_get()
    if not symbols:
        return []
    
    # Define forex majors and minors base pairs (without suffixes)
    forex_base = [
        'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD', 'NZDUSD',  # Majors
        'EURGBP', 'EURJPY', 'EURCHF', 'EURAUD', 'EURCAD', 'EURNZD',  # EUR crosses
        'GBPJPY', 'GBPCHF', 'GBPAUD', 'GBPCAD', 'GBPNZD',  # GBP crosses
        'AUDJPY', 'AUDCHF', 'AUDCAD', 'AUDNZD',  # AUD crosses
        'NZDJPY', 'NZDCHF', 'NZDCAD',  # NZD crosses
        'CADJPY', 'CADCHF', 'CHFJPY',  # Other crosses
    ]
    
    # Metals and crypto
    special_instruments = ['XAGUSD', 'XAUUSD', 'BTCUSD', 'ETHUSD']
    
    # Combined whitelist (case-insensitive base matching)
    whitelist = set(s.upper() for s in forex_base + special_instruments)
    
    filtered = []
    for sym in symbols:
        name = sym.name
        path = getattr(sym, 'path', '')
        
        # Legacy broker pattern: contains 'FX' in path or ends with 'FX'
        if 'FX' in path or name.endswith('FX'):
            filtered.append(name)
            continue
        
        # Modern broker pattern: check if base symbol (without suffix) is in our whitelist
        # Remove common suffixes: 'z', 'm', '.a', etc.
        base_name = name.rstrip('zm').rstrip('.a').rstrip('.m').upper()
        
        if base_name in whitelist:
            filtered.append(name)
    
    return filtered


def fetch_daily_candles(symbol: str, days: int = 5):
    """Return pandas.DataFrame of last `days` daily candles for `symbol`, or None."""
    try:
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_D1, 0, days)
        if rates is None:
            return None
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        return df.sort_values('time').reset_index(drop=True)
    except Exception:
        return None


def fetch_daily_candles_range(symbol: str, start: datetime, end: datetime):
    """Return pandas.DataFrame of D1 candles in [start, end], best-effort.

    Notes:
    - MT5 daily candle timestamps are broker-time anchored; this fetch is a practical filter.
    - For date-only inputs, callers should pass end as the date's midnight and we will
      include the full end day by extending end by 1 day.
    """
    try:
        start_dt = pd.to_datetime(start).to_pydatetime().replace(microsecond=0)
        end_dt = pd.to_datetime(end).to_pydatetime().replace(microsecond=0) + timedelta(days=1)
        rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_D1, start_dt, end_dt)
        if rates is None or len(rates) == 0:
            return None
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        return df.sort_values('time').reset_index(drop=True)
    except Exception:
        return None


def is_bullish(candle) -> bool:
    return candle['close'] > candle['open']


def is_large_candle(candle, symbol: str, min_candle_size_pips: int) -> bool:
    info = mt5.symbol_info(symbol)
    if info is None:
        return False
    body_size = abs(candle['close'] - candle['open'])
    point_value = info.point
    if not point_value:
        return False
    pips = body_size / point_value
    return pips >= min_candle_size_pips


def detect_exhaustion_pattern(c1, c2, symbol: str, min_candle_size_pips: int) -> bool:
    """Return True if two-candle exhaustion pattern is present."""
    if not is_large_candle(c1, symbol, min_candle_size_pips):
        return False

    if is_bullish(c1):
        goes_above = c2['high'] > c1['high']
        closes_in_range = (c1['open'] < c2['close'] < c1['high'])
        return goes_above and closes_in_range
    else:
        goes_below = c2['low'] < c1['low']
        closes_in_range = (c1['low'] < c2['close'] < c1['open'])
        return goes_below and closes_in_range
    
def get_directional_m5_slice(m5_df: pd.DataFrame, pattern_type: str) -> pd.DataFrame:

    if m5_df is None or len(m5_df) < 3:
        return None

    if pattern_type == 'TB Bullish':
        idx = m5_df['low'].idxmin()
    elif pattern_type == 'TB Bearish':
        idx = m5_df['high'].idxmax()
    else:
        return None

    return m5_df.loc[idx:].reset_index(drop=True)

def scan_all(
    min_candle_size_pips: int,
    *args,
    lookback_days: int | None = None,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    fetch_m5_data: bool = False,
) -> List[Dict]:
    """Scan all symbols and call callbacks on progress and when patterns found."""
    patterns: List[Dict] = []

    # Backward compatible argument parsing.
    # Supported call styles:
    # - scan_all(min_pips, lookback_days, on_progress, on_found, stop_event)
    # - scan_all(min_pips, on_progress, on_found, stop_event)  (older GUIs)
    if len(args) == 3:
        on_progress, on_found, stop_event = args
        lookback_days = int(lookback_days or 5)
    elif len(args) >= 4 and isinstance(args[0], int):
        lookback_days = int(args[0])
        on_progress, on_found, stop_event = args[1], args[2], args[3]
    else:
        raise TypeError(
            "scan_all expects either (min_pips, lookback_days, on_progress, on_found, stop_event) "
            "or (min_pips, on_progress, on_found, stop_event)"
        )

    if not initialize_connection():
        on_progress('MT5 init failed')
        return patterns

    try:
        symbols = get_available_symbols()
        if not symbols:
            on_progress('No symbols')
            return patterns

        for idx, symbol in enumerate(symbols, 1):
            if stop_event.is_set():
                on_progress('Scan stopped')
                break
            on_progress(f"Scanning {idx}/{len(symbols)}: {symbol}")

            if start_date is not None and end_date is not None:
                df = fetch_daily_candles_range(symbol, start=start_date, end=end_date)
            else:
                df = fetch_daily_candles(symbol, days=int(lookback_days or 5))
            if df is None or len(df) < 2:
                continue

            for i in range(len(df) - 1):
                if stop_event.is_set():
                    break
                c1 = df.iloc[i]
                c2 = df.iloc[i + 1]
                if detect_exhaustion_pattern(c1, c2, symbol, min_candle_size_pips):
                    pattern_type = 'TB Bearish' if is_bullish(c1) else 'TB Bullish'
                    
                    patt = {
                        'symbol': symbol,
                        'date': c2['time'],
                        'pattern_type': pattern_type,
                    }
                    
                    patterns.append(patt)
                    on_found(patt)

                    if fetch_m5_data:
                        print_m5_data_for_c2(symbol, c2['time'])

        on_progress('Scan complete')
    except Exception as e:
        on_progress(f'Error: {e}')
    finally:
        terminate_connection()

    return patterns

def fetch_m5_candles(symbol: str, start_time, end_time):
    """Fetch M5 candles between start_time and end_time."""
    try:
        rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M5, start_time, end_time)
        if rates is None or len(rates) == 0:
            return None
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        return df.sort_values('time').reset_index(drop=True)
    except Exception:
        return None


def print_m5_data_for_c2(symbol: str, c2_time):
    if isinstance(c2_time, pd.Timestamp):
        c2_date = c2_time.to_pydatetime()
    else:
        c2_date = c2_time
    
    # Get start and end of the C2 day
    start_time = c2_date.replace(hour=0, minute=0, second=0, microsecond=0)
    end_time = start_time + timedelta(days=1)
    
    print(f"\n{'='*80}")
    print(f"M5 CANDLES FOR: {symbol} on {c2_date.strftime('%Y-%m-%d')}")
    print(f"{'='*80}")
    
    # Fetch M5 data
    m5_df = fetch_m5_candles(symbol, start_time, end_time)
    
    if m5_df is None or len(m5_df) == 0:
        print(f"No M5 data available for {symbol} on {c2_date.strftime('%Y-%m-%d')}")
        return
    
    # Print column headers
    print(f"\n{'Time':<20} {'Open':<12} {'High':<12} {'Low':<12} {'Close':<12}")
    print(f"{'-'*70}")
    
    # Print each M5 candle
    for _, row in m5_df.iterrows():
        print(f"{row['time'].strftime('%Y-%m-%d %H:%M'):<20} "
              f"{row['open']:<12.5f} "
              f"{row['high']:<12.5f} "
              f"{row['low']:<12.5f} "
              f"{row['close']:<12.5f}")
    
    print(f"\nTotal M5 candles: {len(m5_df)}")
    print(f"{'='*80}\n")

