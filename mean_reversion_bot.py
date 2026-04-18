"""
Mean Reversion Bot for SPOT markets — Trade the Knight competition.

Strategy:
  - Maintains a rolling price history using tick data from stream_state().
  - Computes a rolling mean and standard deviation (z-score).
  - When price is significantly BELOW the mean (z < -threshold), places a
    buy limit order inside the spread to capture reversion upward.
  - When price is significantly ABOVE the mean (z > +threshold), places a
    sell/short limit order inside the spread to capture reversion downward.
  - Cancels stale resting orders on each refresh cycle.
  - Respects PRD constraints: quantity snaps to 0.001, naked short allowed
    but exposure tracked, self-match prevention is handled by the exchange,
    and orders are only placed while competition_state == "live".
"""

import time
import collections
import math
from knight_trader import ExchangeClient

# ── Configuration ─────────────────────────────────────────────────────────────

SYMBOL = "SPOTDEMO"          # Replace with the real symbol on competition day.

# Mean-reversion parameters
LOOKBACK = 30                # Number of price observations for rolling stats.
Z_ENTRY = 1.5                # Z-score magnitude required to enter a position.
Z_EXIT = 0.3                 # Z-score magnitude at which we consider mean-reached.

# Order sizing
ORDER_SIZE = 1.0             # Units per order (snaps to 0.001 per PRD).
MAX_POSITION = 5.0           # Maximum absolute position (long or short) in units.

# Timing
REFRESH_SECS = 0.5           # Minimum seconds between order refresh cycles.

# Price edge inside the spread for limit orders
EDGE = 0.01                  # How far inside the spread to post.

# Minimum spread guard — don't trade in an illiquid book.
MIN_SPREAD = 0.02


# ── Helpers ───────────────────────────────────────────────────────────────────

def rolling_stats(prices):
    """Return (mean, stdev) for the price deque. Returns (None, None) if not enough data."""
    n = len(prices)
    if n < 2:
        return None, None
    mu = sum(prices) / n
    variance = sum((p - mu) ** 2 for p in prices) / (n - 1)
    sigma = math.sqrt(variance) if variance > 0 else 0.0
    return mu, sigma


def zscore(price, mu, sigma):
    if sigma == 0:
        return 0.0
    return (price - mu) / sigma


def snap_qty(qty):
    """Snap quantity to 0.001 as required by the PRD."""
    return round(round(qty / 0.001) * 0.001, 3)


# ── Bot logic ─────────────────────────────────────────────────────────────────

def run():
    client = ExchangeClient()

    price_history = collections.deque(maxlen=LOOKBACK)

    # Resting order IDs — at most one bid and one ask at a time.
    resting_bid = None
    resting_ask = None

    # Simple position tracker (net units held by this bot).
    # The exchange tracks real inventory; this mirrors it locally for sizing.
    net_position = 0.0

    next_refresh = 0.0

    try:
        for state in client.stream_state():
            try:
                # ── Gate: only trade when the exchange is live ────────────────
                if state.get("competition_state") != "live":
                    continue

                # ── Throttle refresh rate ─────────────────────────────────────
                now = time.monotonic()
                if now < next_refresh:
                    continue

                # ── Pull book for our symbol ──────────────────────────────────
                book = state.get("book", {}).get(SYMBOL, {})
                bids_raw = book.get("bids", {})
                asks_raw = book.get("asks", {})

                bids = sorted((float(p) for p in bids_raw.keys()), reverse=True)
                asks = sorted(float(p) for p in asks_raw.keys())

                if not bids or not asks:
                    continue

                best_bid = bids[0]
                best_ask = asks[0]
                spread = best_ask - best_bid
                mid = (best_bid + best_ask) / 2.0

                # ── Update price history ──────────────────────────────────────
                price_history.append(mid)

                # ── Compute rolling stats ─────────────────────────────────────
                mu, sigma = rolling_stats(price_history)
                if mu is None:
                    # Not enough history yet — keep accumulating.
                    continue

                z = zscore(mid, mu, sigma)

                # ── Cancel stale resting orders on every cycle ────────────────
                if resting_bid:
                    try:
                        client.cancel(resting_bid)
                    except Exception:
                        pass
                    resting_bid = None

                if resting_ask:
                    try:
                        client.cancel(resting_ask)
                    except Exception:
                        pass
                    resting_ask = None

                # ── Skip if spread is too thin ────────────────────────────────
                if spread < MIN_SPREAD:
                    next_refresh = now + REFRESH_SECS
                    continue

                qty = snap_qty(ORDER_SIZE)

                # ── Entry signals ─────────────────────────────────────────────
                #
                # BUY signal: price is well below the mean → expect reversion up.
                # Place a limit buy just inside the best ask to improve fill odds
                # while still capturing spread.
                #
                # SELL/SHORT signal: price is well above the mean → expect reversion down.
                # Naked shorts are allowed per PRD; uncovered exposure costs 1.5× market
                # value, so we keep MAX_POSITION as a hard cap.

                if z < -Z_ENTRY and net_position < MAX_POSITION:
                    # Buy — post just above best bid (inside spread).
                    bid_px = round(min(best_bid + EDGE, best_ask - EDGE), 4)
                    if bid_px < best_ask:  # Sanity check: don't cross the spread.
                        try:
                            order_id = client.buy(SYMBOL, bid_px, qty)
                            resting_bid = order_id
                            net_position += qty
                            print(
                                f"[BUY ] z={z:.2f}  mid={mid:.4f}  mu={mu:.4f}  "
                                f"px={bid_px}  qty={qty}  pos={net_position:.3f}"
                            )
                        except Exception as exc:
                            print(f"buy order error: {exc}")

                elif z > Z_ENTRY and net_position > -MAX_POSITION:
                    # Sell/short — post just below best ask (inside spread).
                    ask_px = round(max(best_ask - EDGE, best_bid + EDGE), 4)
                    if ask_px > best_bid:  # Sanity check: don't cross the spread.
                        try:
                            order_id = client.sell(SYMBOL, ask_px, qty)
                            resting_ask = order_id
                            net_position -= qty
                            print(
                                f"[SELL] z={z:.2f}  mid={mid:.4f}  mu={mu:.4f}  "
                                f"px={ask_px}  qty={qty}  pos={net_position:.3f}"
                            )
                        except Exception as exc:
                            print(f"sell order error: {exc}")

                # ── Exit / flatten signal ─────────────────────────────────────
                #
                # When z returns near zero, our position has served its purpose.
                # Flatten by posting on the opposite side at mid or better.

                elif abs(z) < Z_EXIT and abs(net_position) >= qty:
                    if net_position > 0:
                        # Long → sell to flatten.
                        ask_px = round(max(best_ask - EDGE, best_bid + EDGE), 4)
                        close_qty = snap_qty(min(abs(net_position), ORDER_SIZE))
                        if ask_px > best_bid and close_qty >= 0.001:
                            try:
                                order_id = client.sell(SYMBOL, ask_px, close_qty)
                                resting_ask = order_id
                                net_position -= close_qty
                                print(
                                    f"[FLAT] z={z:.2f}  mid={mid:.4f}  "
                                    f"sell to flatten  qty={close_qty}  pos={net_position:.3f}"
                                )
                            except Exception as exc:
                                print(f"flatten sell error: {exc}")

                    elif net_position < 0:
                        # Short → buy to flatten.
                        bid_px = round(min(best_bid + EDGE, best_ask - EDGE), 4)
                        close_qty = snap_qty(min(abs(net_position), ORDER_SIZE))
                        if bid_px < best_ask and close_qty >= 0.001:
                            try:
                                order_id = client.buy(SYMBOL, bid_px, close_qty)
                                resting_bid = order_id
                                net_position += close_qty
                                print(
                                    f"[FLAT] z={z:.2f}  mid={mid:.4f}  "
                                    f"buy to flatten  qty={close_qty}  pos={net_position:.3f}"
                                )
                            except Exception as exc:
                                print(f"flatten buy error: {exc}")

                next_refresh = now + REFRESH_SECS

            except Exception as exc:
                print(f"bot loop error: {exc}")
                time.sleep(1.0)

    finally:
        # Clean up any open orders before the bot exits.
        try:
            client.cancel_all()
        except Exception:
            pass
        client.close()


if __name__ == "__main__":
    run()
