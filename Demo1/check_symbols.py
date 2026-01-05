"""
Quick diagnostic: Show all available symbols from your current MT5 broker connection.
Run this to see what symbol names your new broker uses.
"""
import MetaTrader5 as mt5

if not mt5.initialize():
    print("❌ Failed to connect to MT5. Make sure MT5 is running and logged in.")
    exit(1)

print("✅ Connected to MT5")
print(f"Account: {mt5.account_info().login if mt5.account_info() else 'N/A'}")
print(f"Server: {mt5.account_info().server if mt5.account_info() else 'N/A'}")
print("\n" + "="*80)

symbols = mt5.symbols_get()
if not symbols:
    print("❌ No symbols available from broker.")
    mt5.shutdown()
    exit(1)

print(f"Total symbols available: {len(symbols)}\n")

# Show all symbols with their paths (helps identify forex vs others)
print("Symbol Name".ljust(20), "Path".ljust(40), "Description")
print("-" * 80)

for s in symbols:
    path = getattr(s, 'path', 'N/A')
    desc = getattr(s, 'description', 'N/A')
    print(f"{s.name.ljust(20)} {path.ljust(40)} {desc}")

print("\n" + "="*80)

# Show just the likely forex pairs (contains "USD", "EUR", "GBP", etc.)
forex_keywords = ['USD', 'EUR', 'GBP', 'JPY', 'AUD', 'NZD', 'CAD', 'CHF']
likely_forex = [s for s in symbols if any(kw in s.name.upper() for kw in forex_keywords)]

print(f"\nLikely forex pairs ({len(likely_forex)} found):")
for s in likely_forex:
    print(f"  - {s.name}")

print("\n" + "="*80)
print("\n📌 Copy one of the symbol names above and try it in the GUI.")
print("📌 If symbols have suffixes (like '.a' or 'm'), you'll need to use the exact name.")

mt5.shutdown()
