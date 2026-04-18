"""
Spot Market Making Bot
======================
Strategy: Adaptive market-making with inventory skew and trend awareness.

- Quotes a two-sided market (bid + ask) around the mid-price.
- Skews quotes based on current inventory to avoid runaway exposure.
- Widens spread during high volatility (measured from recent tape).
- Tracks PnL and position size, pausing if exposure gets too large.
- Cancels and refreshes quotes on every cycle (throttled by REFRESH_SECS).
- Handles all competition states cleanly (pre_open / paused / post_close).
"""

import time
import collections
from knight_trader import ExchangeClient

# ─────────────────────────────────────────────
# CONFIG  –  edit these before uploading
# ─────────────────────────────────────────────
SYMBOL = "RUX"          # Replace with the live symbol on competition day

ORDER_SIZE      = 1.0        # Base order size in units
MIN_SPREAD      = 0.04       # Minimum acceptable market spread to quote into
BASE_EDGE       = 0.01       # How far inside the best bid/ask we start our quotes
REFRESH_SECS    = 0.5        # Minimum seconds between quote refreshes
MAX_POSITION    = 10.0       # Max net inventory (units) before skew clamps quotes
SKEW_FACTOR     = 0.005      # Price skew per unit of inventory (pushes quotes away from risk)
VOL_WINDOW      = 20         # Number of recent mid-prices used for volatility estimate
VOL_MULTIPLIER  = 2.0        # Extra spread added per 1% of rolling volatility
MAX_SPREAD_MULT = 3.0        # Cap on spread multiplier from volatility
# ─────────────────────────────────────────────


def compute_volatility(mid_prices: collections.deque) -> float:
    """Return simple rolling std-dev of mid prices (as a fraction of mean)."""
    if len(mid_prices) < 2:
        return 0.0
    prices = list(mid_prices)
    mean = sum(prices) / len(prices)
    if mean == 0:
        return 0.0
    variance = sum((p - mean) ** 2 for p in prices) / len(prices)
    return (variance ** 0.5) / mean   # coefficient of variation


def run():
    client = ExchangeClient()

    resting_bid  = None
    resting_ask  = None
    next_refresh = 0.0
    position     = 0.0    # net inventory: positive = long, negative = short
    realized_pnl = 0.0
    last_buy_px  = None
    last_sell_px = None
    mid_history  = collections.deque(maxlen=VOL_WINDOW)
    tick_count   = 0

    print(f"[bot] Starting spot market-maker on {SYMBOL}")

    try:
        for state in client.stream_state():
            try:
                comp_state = state.get("competition_state", "pre_open")

                # ── Wait for live market ──────────────────────────────────
                if comp_state != "live":
                    if resting_bid or resting_ask:
                        print(f"[bot] Market not live ({comp_state}), cancelling all orders")
                        client.cancel_all()
                        resting_bid = resting_ask = None
                    continue

                # ── Throttle refresh rate ─────────────────────────────────
                if time.monotonic() < next_refresh:
                    continue

                tick_count += 1

                # ── Pull order book ───────────────────────────────────────
                book  = state.get("book", {}).get(SYMBOL, {})
                bids  = sorted((float(p) for p in book.get("bids", {}).keys()), reverse=True)
                asks  = sorted(float(p) for p in book.get("asks", {}).keys())

                if not bids or not asks:
                    continue

                best_bid = bids[0]
                best_ask = asks[0]
                spread   = best_ask - best_bid
                mid      = (best_bid + best_ask) / 2.0

                # Track mid-price history for vol estimate
                mid_history.append(mid)

                # ── Spread filter ─────────────────────────────────────────
                if spread < MIN_SPREAD:
                    continue   # Market is too tight; skip this tick

                # ── Volatility-adjusted edge ──────────────────────────────
                vol    = compute_volatility(mid_history)
                vol_adj = min(MAX_SPREAD_MULT, 1.0 + VOL_MULTIPLIER * vol * 100)
                edge   = BASE_EDGE * vol_adj

                # ── Inventory skew ────────────────────────────────────────
                # If we are long, push bid down and ask down → sell more.
                # If we are short, push bid up and ask up → buy more.
                skew     = -SKEW_FACTOR * position          # oppose inventory
                clamped  = max(-MAX_POSITION, min(MAX_POSITION, position))
                skew     = -SKEW_FACTOR * clamped

                bid_px = round(best_bid + edge + skew, 4)
                ask_px = round(best_ask - edge + skew, 4)

                # Sanity check: quotes must not cross
                if bid_px >= ask_px:
                    half = round((bid_px + ask_px) / 2, 4)
                    bid_px = round(half - 0.005, 4)
                    ask_px = round(half + 0.005, 4)

                # ── Size: reduce when inventory is large ──────────────────
                inv_ratio  = min(1.0, abs(position) / MAX_POSITION)
                size_scale = max(0.2, 1.0 - inv_ratio * 0.8)

                # When very long, only place ask; when very short, only place bid.
                want_bid = position <= MAX_POSITION * 0.9
                want_ask = position >= -MAX_POSITION * 0.9

                # ── Cancel stale quotes ───────────────────────────────────
                if resting_bid:
                    client.cancel(resting_bid)
                    resting_bid = None
                if resting_ask:
                    client.cancel(resting_ask)
                    resting_ask = None

                # ── Place fresh quotes ────────────────────────────────────
                qty = round(ORDER_SIZE * size_scale, 3)
                qty = max(0.001, qty)

                if want_bid:
                    new_bid = client.buy(SYMBOL, bid_px, qty)
                    if new_bid:
                        resting_bid  = new_bid
                        last_buy_px  = bid_px

                if want_ask:
                    new_ask = client.sell(SYMBOL, ask_px, qty)
                    if new_ask:
                        resting_ask  = new_ask
                        last_sell_px = ask_px

                # ── Estimate fills from tape ──────────────────────────────
                # Parse recent tape to update position approximation.
                recent_trades = state.get("trades", [])
                for trade in recent_trades:
                    if trade.get("symbol", "").upper() != SYMBOL:
                        continue
                    buyer  = trade.get("buyer_id", "")
                    seller = trade.get("seller_id", "")
                    my_id  = client.bot_id
                    qty_t  = float(trade.get("quantity", 0))
                    px_t   = float(trade.get("price", 0))
                    if buyer == my_id:
                        position     += qty_t
                        realized_pnl -= qty_t * px_t
                    elif seller == my_id:
                        position     -= qty_t
                        realized_pnl += qty_t * px_t

                # ── Periodic diagnostics ──────────────────────────────────
                if tick_count % 20 == 0:
                    unreal = position * mid
                    total  = realized_pnl + unreal
                    diag   = client.diagnostics_snapshot()
                    print(
                        f"[bot] tick={tick_count} | mid={mid:.4f} | "
                        f"pos={position:+.3f} | rpnl={realized_pnl:+.2f} | "
                        f"upnl={unreal:+.2f} | total={total:+.2f} | "
                        f"vol={vol*100:.2f}% | edge={edge:.4f} | "
                        f"rejects={diag.get('order_rejects', 0)}"
                    )

                next_refresh = time.monotonic() + REFRESH_SECS

            except Exception as exc:
                print(f"[bot] loop error: {exc}")
                time.sleep(1.0)

    except KeyboardInterrupt:
        print("[bot] Shutting down...")
    finally:
        print("[bot] Cancelling all open orders before exit...")
        client.cancel_all()
        client.close()
        print(f"[bot] Final position: {position:+.3f} | Realized PnL: {realized_pnl:+.4f}")


if __name__ == "__main__":
    run()
