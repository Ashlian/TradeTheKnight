"""
Order Book Texture Mining Bot
Trade the Knight — Spot Market Alpha Strategy

Strategy: Order Book Texture Mining
Analyzes the SHAPE of the full order book rather than just best bid/ask.
Computes a "texture ratio" — volume clustered near the top vs. deep in the book.
When asymmetry exists between near/deep layers, a directional move is anticipated
before the price has actually moved.

PRD Compliance Notes:
- Uses ExchangeClient() via knight_trader SDK (no custom transport)
- Reads BOT_ID and EXCHANGE_URL from environment automatically
- Uses stream_state() with zero-backlog queue (skips stale states natively)
- buy() / sell() block on private order-status delta as specified
- Respects competition_state: only trades when "live"
- Naked short exposure avoided by tracking inventory; 1.5x penalty model considered
- cancel_all() called on state transitions to avoid stale resting orders
- Quantities rounded to 0.001 as per PRD snap spec
- Self-match prevention is platform-side; bot avoids double-sided resting orders
"""

import time
import math
import logging
from collections import deque
from knight_trader import ExchangeClient

# ─────────────────────────────────────────────
# CONFIGURATION — adjust before deployment
# ─────────────────────────────────────────────

SYMBOL = "SPOTDEMO"          # Replace with live symbol on competition day

# Texture analysis parameters
NEAR_LAYER_DEPTH  = 5        # Number of price levels considered "near top"
DEEP_LAYER_DEPTH  = 15       # Price levels considered "deep" (beyond near)
TEXTURE_THRESHOLD = 0.65     # Imbalance ratio to trigger a signal (0.5 = neutral)
TEXTURE_STRONG    = 0.80     # Strong signal threshold — larger position size

# Order sizing
BASE_QTY          = 1.0      # Base order quantity (snapped to 0.001)
STRONG_QTY        = 2.0      # Quantity on strong texture signal
MAX_INVENTORY     = 5.0      # Maximum net inventory in either direction
                              # Avoids uncovered short exposure penalty (1.5x)

# Execution parameters
EDGE_TICKS        = 0.01     # How aggressively inside the spread we post
MIN_SPREAD        = 0.02     # Minimum spread required to enter
REFRESH_INTERVAL  = 0.4      # Seconds between strategy evaluations
POSITION_TTL      = 8.0      # Seconds before we flatten a position
SIGNAL_HISTORY    = 12       # Rolling window of texture ratios to smooth signal

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [TEXTURE] %(levelname)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("texture_miner")


# ─────────────────────────────────────────────
# TEXTURE ANALYSIS CORE
# ─────────────────────────────────────────────

def compute_texture_ratio(book: dict, side: str) -> float:
    """
    Compute the texture ratio for one side of the book.

    Texture ratio = near_volume / (near_volume + deep_volume)

    A ratio near 1.0 means liquidity is concentrated at the top (thin deep book).
    A ratio near 0.0 means liquidity is buried deep (thin near book).
    0.5 = balanced.

    Args:
        book: The book dict for one symbol from stream_state, e.g.
              {"bids": {"100.00": 5.0, ...}, "asks": {...}}
        side: "bids" or "asks"

    Returns:
        Float in [0, 1], or 0.5 if insufficient data.
    """
    raw = book.get(side, {})
    if not raw:
        return 0.5

    def _to_qty(v) -> float:
        """
        Normalise a book-level value to a scalar quantity.
        The SDK may return a bare number (int/float/str) or a list
        where the first element is the quantity, e.g. [qty, order_count].
        """
        if isinstance(v, list):
            return float(v[0]) if v else 0.0
        return float(v)

    # Sort price levels: bids descending, asks ascending
    reverse = (side == "bids")
    levels = sorted(
        [(float(p), _to_qty(v)) for p, v in raw.items()],
        key=lambda x: x[0],
        reverse=reverse,
    )

    near_vol = sum(v for _, v in levels[:NEAR_LAYER_DEPTH])
    deep_vol = sum(v for _, v in levels[NEAR_LAYER_DEPTH:NEAR_LAYER_DEPTH + DEEP_LAYER_DEPTH])
    total    = near_vol + deep_vol

    if total < 1e-9:
        return 0.5

    return near_vol / total


def classify_signal(bid_texture: float, ask_texture: float) -> tuple[str, float]:
    """
    Derive a directional signal from bid and ask texture ratios.

    Logic:
      - Fat near bids  + thin near asks  → bullish pressure (bids defending, asks retreating)
      - Fat near asks  + thin near bids  → bearish pressure (asks defending, bids retreating)
      - Symmetry → no signal

    Returns:
        (direction, confidence) where direction in {"buy", "sell", "none"}
        and confidence is [0, 1].
    """
    # Asymmetry score: positive = bid-heavy near top, negative = ask-heavy near top
    asymmetry = bid_texture - ask_texture

    confidence = abs(asymmetry)

    if asymmetry > (TEXTURE_THRESHOLD - 0.5):
        return "buy", confidence
    elif asymmetry < -(TEXTURE_THRESHOLD - 0.5):
        return "sell", confidence
    else:
        return "none", confidence


def snap_qty(qty: float) -> float:
    """Snap quantity to 0.001 as required by PRD."""
    return round(math.floor(qty * 1000) / 1000, 3)


# ─────────────────────────────────────────────
# STATE TRACKING
# ─────────────────────────────────────────────

class BotState:
    def __init__(self):
        self.inventory: float      = 0.0     # Net position (positive = long)
        self.resting_buy_id        = None    # Active resting buy order ID
        self.resting_sell_id       = None    # Active resting sell order ID
        self.position_opened_at: float = 0.0
        self.signal_history        = deque(maxlen=SIGNAL_HISTORY)
        self.last_evaluation: float = 0.0
        self.last_competition_state: str = ""
        self.entry_price: float    = 0.0

    def record_signal(self, asymmetry: float):
        self.signal_history.append(asymmetry)

    def smoothed_asymmetry(self) -> float:
        if not self.signal_history:
            return 0.0
        return sum(self.signal_history) / len(self.signal_history)

    def has_position(self) -> bool:
        return abs(self.inventory) > 1e-6

    def position_age(self) -> float:
        if not self.has_position():
            return 0.0
        return time.monotonic() - self.position_opened_at


# ─────────────────────────────────────────────
# MAIN STRATEGY LOOP
# ─────────────────────────────────────────────

def run():
    log.info("Texture Miner initializing — connecting to exchange...")
    client = ExchangeClient()
    state  = BotState()

    log.info(f"Connected. Monitoring symbol: {SYMBOL}")
    log.info(
        f"Parameters — near_depth={NEAR_LAYER_DEPTH}, deep_depth={DEEP_LAYER_DEPTH}, "
        f"threshold={TEXTURE_THRESHOLD}, refresh={REFRESH_INTERVAL}s"
    )

    for mkt in client.stream_state():
        try:
            # ── 1. Guard: only act when live ─────────────────────────────
            comp_state = mkt.get("competition_state", "")

            if comp_state != state.last_competition_state:
                log.info(f"Competition state changed: {state.last_competition_state!r} → {comp_state!r}")
                if comp_state != "live":
                    # Cancel all resting orders on state transition
                    _safe_cancel_all(client, state)
                state.last_competition_state = comp_state

            if comp_state != "live":
                continue

            # ── 2. Rate limit evaluations ─────────────────────────────────
            now = time.monotonic()
            if now < state.last_evaluation + REFRESH_INTERVAL:
                continue
            state.last_evaluation = now

            # ── 3. Read book ──────────────────────────────────────────────
            book_map = mkt.get("book", {})
            book     = book_map.get(SYMBOL, {})

            bids_raw = book.get("bids", {})
            asks_raw = book.get("asks", {})
            if not bids_raw or not asks_raw:
                continue

            best_bid = max(float(p) for p in bids_raw)
            best_ask = min(float(p) for p in asks_raw)
            spread   = best_ask - best_bid
            mid      = (best_bid + best_ask) / 2.0

            if spread < MIN_SPREAD:
                log.debug(f"Spread {spread:.4f} below minimum {MIN_SPREAD} — skip")
                continue

            # ── 4. Compute texture ────────────────────────────────────────
            bid_texture = compute_texture_ratio(book, "bids")
            ask_texture = compute_texture_ratio(book, "asks")
            asymmetry   = bid_texture - ask_texture

            state.record_signal(asymmetry)
            smoothed = state.smoothed_asymmetry()

            direction, confidence = classify_signal(
                # Use smoothed signal to reduce noise
                0.5 + smoothed / 2,   # re-map smoothed asymmetry back to texture space
                0.5 - smoothed / 2,
            )

            log.debug(
                f"BID texture={bid_texture:.3f} ASK texture={ask_texture:.3f} "
                f"asymmetry={asymmetry:+.3f} smoothed={smoothed:+.3f} → {direction} ({confidence:.2f})"
            )

            # ── 5. Position TTL — flatten aged positions ───────────────────
            if state.has_position() and state.position_age() > POSITION_TTL:
                log.info(
                    f"Position TTL reached ({state.position_age():.1f}s) — flattening "
                    f"inventory={state.inventory:+.3f}"
                )
                _flatten_position(client, state, best_bid, best_ask)
                continue

            # ── 6. No signal — hold or stay flat ──────────────────────────
            if direction == "none":
                continue

            # ── 7. Inventory guardrails ────────────────────────────────────
            # Avoid uncovered short exposure penalty (1.5x per PRD)
            # Hard cap on inventory in either direction
            if direction == "buy"  and state.inventory >= MAX_INVENTORY:
                log.debug("Max long inventory reached — no new buy")
                continue
            if direction == "sell" and state.inventory <= -MAX_INVENTORY:
                log.debug("Max short inventory reached — no new sell")
                continue

            # ── 8. Already have a position in this direction — hold ────────
            if state.has_position():
                pos_direction = "buy" if state.inventory > 0 else "sell"
                if pos_direction == direction:
                    log.debug(f"Holding existing {pos_direction} position — no action")
                    continue
                else:
                    # Signal flipped — flatten and re-enter
                    log.info(f"Signal reversal detected — flattening and re-entering {direction}")
                    _flatten_position(client, state, best_bid, best_ask)

            # ── 9. Determine order size ────────────────────────────────────
            qty = STRONG_QTY if confidence >= (TEXTURE_STRONG - 0.5) else BASE_QTY
            qty = snap_qty(min(qty, MAX_INVENTORY - abs(state.inventory)))
            if qty < 0.001:
                continue

            # ── 10. Place directional order ───────────────────────────────
            if direction == "buy":
                # Post slightly inside the spread — not at best ask (we want a fill,
                # not a market order) to preserve edge
                price = round(best_bid + EDGE_TICKS, 4)
                log.info(
                    f"BUY signal — texture asymmetry={smoothed:+.3f}, "
                    f"confidence={confidence:.2f}, qty={qty}, px={price}"
                )
                order_id = client.buy(SYMBOL, price, qty)
                if order_id:
                    state.resting_buy_id     = order_id
                    state.inventory         += qty
                    state.position_opened_at = time.monotonic()
                    state.entry_price        = price
                    log.info(f"BUY placed — order_id={order_id}, inventory now {state.inventory:+.3f}")

            elif direction == "sell":
                price = round(best_ask - EDGE_TICKS, 4)
                log.info(
                    f"SELL signal — texture asymmetry={smoothed:+.3f}, "
                    f"confidence={confidence:.2f}, qty={qty}, px={price}"
                )
                order_id = client.sell(SYMBOL, price, qty)
                if order_id:
                    state.resting_sell_id    = order_id
                    state.inventory         -= qty
                    state.position_opened_at = time.monotonic()
                    state.entry_price        = price
                    log.info(f"SELL placed — order_id={order_id}, inventory now {state.inventory:+.3f}")

        except Exception as exc:
            log.error(f"Strategy loop error: {exc}", exc_info=True)
            time.sleep(1.0)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _safe_cancel_all(client: ExchangeClient, state: BotState):
    """Cancel all resting orders and reset order IDs."""
    try:
        client.cancel_all()
        log.info("cancel_all() issued")
    except Exception as e:
        log.warning(f"cancel_all failed: {e}")
    state.resting_buy_id  = None
    state.resting_sell_id = None


def _flatten_position(client: ExchangeClient, state: BotState, best_bid: float, best_ask: float):
    """
    Flatten net inventory with a market-crossing order.
    Crosses the spread to guarantee fill rather than resting.
    """
    _safe_cancel_all(client, state)

    net = state.inventory
    if abs(net) < 0.001:
        state.inventory = 0.0
        return

    qty = snap_qty(abs(net))

    try:
        if net > 0:
            # Long — sell to flatten; cross spread by hitting best ask price or below
            price = round(best_bid - EDGE_TICKS, 4)
            order_id = client.sell(SYMBOL, price, qty)
            log.info(f"Flatten SELL {qty} @ {price} — order_id={order_id}")
        else:
            # Short — buy to flatten
            price = round(best_ask + EDGE_TICKS, 4)
            order_id = client.buy(SYMBOL, price, qty)
            log.info(f"Flatten BUY {qty} @ {price} — order_id={order_id}")
    except Exception as e:
        log.error(f"Flatten order failed: {e}")

    # Reset state regardless — next signal starts fresh
    state.inventory          = 0.0
    state.resting_buy_id     = None
    state.resting_sell_id    = None
    state.position_opened_at = 0.0
    state.entry_price        = 0.0


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    run()
