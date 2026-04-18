import math
import os
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, Iterable, List, Optional, Set, Tuple

from knight_trader import ExchangeClient


PRICE_TICK = 0.0001
SIZE_STEP = 0.001


def env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)


def env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return int(default)


def env_csv(name: str) -> List[str]:
    raw = os.environ.get(name, "")
    return [item.strip().upper() for item in raw.split(",") if item.strip()]


def floor_to_step(value: float, step: float) -> float:
    if value <= 0:
        return 0.0
    units = math.floor((value + 1e-12) / step)
    return round(units * step, 6)


def round_price(value: float) -> float:
    return round(max(value, PRICE_TICK), 4)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass
class ManagedOrder:
    order_id: str
    symbol: str
    side: str
    price: float
    quantity: float
    created_at: float


@dataclass
class SymbolState:
    mids: Deque[float]
    last_action_ts: float = 0.0


class MeanReversionSpotBot:
    def __init__(self):
        self.lookback = env_int("MR_LOOKBACK", 60)
        self.min_history = env_int("MR_MIN_HISTORY", max(24, self.lookback // 2))
        self.entry_z = env_float("MR_ENTRY_Z", 1.8)
        self.exit_z = env_float("MR_EXIT_Z", 0.35)
        self.refresh_secs = env_float("MR_REFRESH_SECS", 0.75)
        self.order_stale_secs = env_float("MR_ORDER_STALE_SECS", 6.0)
        self.team_state_refresh_secs = env_float("MR_TEAM_STATE_REFRESH_SECS", 2.5)
        self.asset_refresh_secs = env_float("MR_ASSET_REFRESH_SECS", 30.0)
        self.notional_per_trade = env_float("MR_NOTIONAL_PER_TRADE", 100.0)
        self.max_order_qty = env_float("MR_MAX_ORDER_QTY", 5.0)
        self.max_inventory_per_symbol = env_float("MR_MAX_INVENTORY_PER_SYMBOL", 10.0)
        self.max_symbols = env_int("MR_MAX_SYMBOLS", 3)
        self.max_spread_bps = env_float("MR_MAX_SPREAD_BPS", 60.0)
        self.min_std_bps = env_float("MR_MIN_STD_BPS", 3.0)
        self.forced_symbols = env_csv("MR_SYMBOLS")

        self.client = ExchangeClient()
        self.bot_id = str(getattr(self.client, "bot_id", "") or os.environ.get("BOT_ID", ""))
        self.symbol_states: Dict[str, SymbolState] = {}
        self.orders: Dict[str, ManagedOrder] = {}
        self.inferred_inventory: Dict[str, float] = defaultdict(float)
        self.last_asset_refresh = 0.0
        self.last_team_state_refresh = 0.0
        self.next_refresh = 0.0
        self.eligible_symbols: Set[str] = set(self.forced_symbols)
        self.logged_universe = False

    def run(self):
        print("Starting mean-reversion spot bot")
        if self.forced_symbols:
            print(f"Using explicit symbol override: {', '.join(sorted(self.forced_symbols))}")

        try:
            for state in self.client.stream_state():
                try:
                    self._reconcile_filled_orders()
                    competition_state = str(state.get("competition_state", "pre_open"))
                    book_by_symbol = state.get("book", {}) or {}
                    self._refresh_universe(book_by_symbol)
                    self._update_price_history(book_by_symbol)
                    self._refresh_inventory_from_team_state()

                    if competition_state != "live":
                        continue

                    now = time.monotonic()
                    if now < self.next_refresh:
                        continue
                    self.next_refresh = now + self.refresh_secs

                    symbols = self._select_symbols(book_by_symbol)
                    for symbol in symbols:
                        self._trade_symbol(symbol, book_by_symbol.get(symbol, {}), now)
                    self._cancel_irrelevant_orders(symbols, now)
                except Exception as exc:
                    print(f"bot loop error: {exc}")
                    time.sleep(1.0)
        finally:
            self.client.close()

    def _state_for_symbol(self, symbol: str) -> SymbolState:
        symbol = symbol.upper()
        if symbol not in self.symbol_states:
            self.symbol_states[symbol] = SymbolState(mids=deque(maxlen=self.lookback))
        return self.symbol_states[symbol]

    def _refresh_universe(self, book_by_symbol: Dict[str, Any]):
        now = time.monotonic()
        if self.forced_symbols:
            self.eligible_symbols = set(self.forced_symbols)
            return
        if self.eligible_symbols and now - self.last_asset_refresh < self.asset_refresh_secs:
            return

        asset_symbols = set()
        for asset in self.client.get_assets():
            symbol = self._asset_symbol(asset)
            if symbol and self._asset_is_spot(asset):
                asset_symbols.add(symbol)

        if asset_symbols:
            self.eligible_symbols = asset_symbols
        else:
            self.eligible_symbols = {
                symbol.upper()
                for symbol in book_by_symbol.keys()
                if self._looks_like_spot_symbol(symbol)
            }

        if self.eligible_symbols and not self.logged_universe:
            print(f"Eligible symbols: {', '.join(sorted(self.eligible_symbols))}")
            self.logged_universe = True

        self.last_asset_refresh = now

    def _update_price_history(self, book_by_symbol: Dict[str, Any]):
        symbols: Iterable[str]
        if self.eligible_symbols:
            symbols = self.eligible_symbols
        else:
            symbols = [symbol.upper() for symbol in book_by_symbol.keys()]

        for symbol in symbols:
            book = book_by_symbol.get(symbol, {})
            best_bid, best_ask = self._best_prices(book)
            if best_bid <= 0 or best_ask <= 0 or best_ask <= best_bid:
                continue
            mid = (best_bid + best_ask) / 2.0
            self._state_for_symbol(symbol).mids.append(mid)

    def _refresh_inventory_from_team_state(self):
        now = time.monotonic()
        if now - self.last_team_state_refresh < self.team_state_refresh_secs:
            return
        self.last_team_state_refresh = now

        team_state = self.client.get_team_state()
        positions = self._extract_positions_for_bot(team_state, self.bot_id)
        if not positions:
            return

        for symbol, quantity in positions.items():
            self.inferred_inventory[symbol] = quantity

    def _extract_positions_for_bot(self, payload: Any, bot_id: str) -> Dict[str, float]:
        matches: List[Dict[str, float]] = []

        def walk(node: Any, matched_bot: bool = False):
            if isinstance(node, dict):
                local_match = matched_bot or self._dict_matches_bot(node, bot_id)
                parsed = self._extract_symbol_quantity_map(node)
                if local_match and parsed:
                    matches.append(parsed)
                for key, value in node.items():
                    child_match = local_match or str(key) == bot_id
                    walk(value, child_match)
            elif isinstance(node, list):
                for item in node:
                    walk(item, matched_bot)

        walk(payload, False)

        positions: Dict[str, float] = {}
        for match in matches:
            for symbol, quantity in match.items():
                positions[symbol] = quantity
        return positions

    def _dict_matches_bot(self, node: Dict[str, Any], bot_id: str) -> bool:
        if not bot_id:
            return False
        for key in ("bot_id", "owner_id", "id", "botId"):
            if str(node.get(key, "")) == bot_id:
                return True
        return False

    def _extract_symbol_quantity_map(self, node: Dict[str, Any]) -> Dict[str, float]:
        if "symbol" in node:
            symbol = str(node.get("symbol", "")).upper()
            quantity = self._read_quantity(node)
            if symbol and quantity is not None:
                return {symbol: quantity}

        for container_key in ("positions", "inventory", "holdings", "balances", "assets"):
            container = node.get(container_key)
            if not isinstance(container, dict):
                continue
            mapped: Dict[str, float] = {}
            for symbol, raw in container.items():
                quantity = self._coerce_quantity(raw)
                symbol_text = str(symbol).upper()
                if quantity is None or not symbol_text:
                    continue
                mapped[symbol_text] = quantity
            if mapped:
                return mapped

        return {}

    def _read_quantity(self, node: Dict[str, Any]) -> Optional[float]:
        for key in (
            "quantity",
            "qty",
            "position",
            "net_position",
            "balance",
            "amount",
            "free",
            "total",
        ):
            if key in node:
                return self._coerce_quantity(node.get(key))
        return None

    def _coerce_quantity(self, raw: Any) -> Optional[float]:
        if isinstance(raw, dict):
            for key in ("quantity", "qty", "position", "balance", "amount", "free", "total"):
                if key in raw:
                    return self._coerce_quantity(raw.get(key))
            return None
        if isinstance(raw, (int, float)):
            return round(float(raw), 3)
        if isinstance(raw, str):
            try:
                return round(float(raw), 3)
            except ValueError:
                return None
        return None

    def _select_symbols(self, book_by_symbol: Dict[str, Any]) -> List[str]:
        candidates: List[Tuple[float, float, str]] = []
        symbols = self.eligible_symbols or {symbol.upper() for symbol in book_by_symbol.keys()}
        for symbol in symbols:
            book = book_by_symbol.get(symbol, {})
            best_bid, best_ask = self._best_prices(book)
            if best_bid <= 0 or best_ask <= 0 or best_ask <= best_bid:
                continue
            state = self._state_for_symbol(symbol)
            if len(state.mids) < self.min_history:
                continue
            spread_bps = ((best_ask - best_bid) / max((best_ask + best_bid) / 2.0, PRICE_TICK)) * 10_000.0
            candidates.append((spread_bps, -len(state.mids), symbol))

        candidates.sort()
        return [symbol for _, _, symbol in candidates[: self.max_symbols]]

    def _trade_symbol(self, symbol: str, book: Dict[str, Any], now: float):
        best_bid, best_ask = self._best_prices(book)
        if best_bid <= 0 or best_ask <= 0 or best_ask <= best_bid:
            self._cancel_symbol_orders(symbol)
            return

        mid = (best_bid + best_ask) / 2.0
        spread = best_ask - best_bid
        spread_bps = (spread / max(mid, PRICE_TICK)) * 10_000.0
        if spread_bps > self.max_spread_bps:
            self._cancel_symbol_orders(symbol)
            return

        fair_value, zscore = self._signal(symbol)
        if fair_value is None or zscore is None:
            return

        inventory = round(self.inferred_inventory.get(symbol, 0.0), 3)
        outstanding_buys = self._outstanding_quantity(symbol, "buy")
        outstanding_sells = self._outstanding_quantity(symbol, "sell")
        remaining_capacity = max(0.0, self.max_inventory_per_symbol - inventory - outstanding_buys)

        want_buy = zscore <= -self.entry_z and remaining_capacity >= SIZE_STEP
        want_sell = inventory - outstanding_sells >= SIZE_STEP and zscore >= -self.exit_z

        desired_buy = None
        desired_sell = None

        if want_buy:
            buy_qty = min(self._base_order_quantity(mid), remaining_capacity)
            if buy_qty >= SIZE_STEP:
                desired_buy = (
                    self._passive_buy_price(best_bid, best_ask, fair_value),
                    floor_to_step(buy_qty, SIZE_STEP),
                )

        if want_sell:
            sell_qty = min(self._base_order_quantity(mid), inventory - outstanding_sells)
            if sell_qty >= SIZE_STEP:
                desired_sell = (
                    self._passive_sell_price(best_bid, best_ask, fair_value),
                    floor_to_step(sell_qty, SIZE_STEP),
                )

        self._sync_order(symbol, "buy", desired_buy, now)
        self._sync_order(symbol, "sell", desired_sell, now)

    def _signal(self, symbol: str) -> Tuple[Optional[float], Optional[float]]:
        prices = list(self._state_for_symbol(symbol).mids)
        if len(prices) < self.min_history:
            return None, None

        mean = sum(prices) / len(prices)
        variance = sum((price - mean) ** 2 for price in prices) / len(prices)
        std = math.sqrt(variance)
        if mean <= 0:
            return None, None
        if std <= mean * (self.min_std_bps / 10_000.0):
            return None, None

        zscore = (prices[-1] - mean) / std
        return mean, zscore

    def _sync_order(self, symbol: str, side: str, desired: Optional[Tuple[float, float]], now: float):
        existing = self._find_order(symbol, side)
        if desired is None:
            if existing and now - existing.created_at >= self.refresh_secs:
                self._cancel_order(existing.order_id)
            return

        desired_price, desired_qty = desired
        if desired_qty < SIZE_STEP:
            if existing:
                self._cancel_order(existing.order_id)
            return

        if existing:
            price_changed = abs(existing.price - desired_price) >= PRICE_TICK / 2.0
            qty_changed = abs(existing.quantity - desired_qty) >= SIZE_STEP / 2.0
            too_old = now - existing.created_at >= self.order_stale_secs
            if price_changed or qty_changed or too_old:
                self._cancel_order(existing.order_id)
                existing = None

        if existing is None:
            order_id = self._place_order(symbol, side, desired_price, desired_qty)
            if order_id:
                self.orders[order_id] = ManagedOrder(
                    order_id=order_id,
                    symbol=symbol,
                    side=side,
                    price=desired_price,
                    quantity=desired_qty,
                    created_at=now,
                )

    def _place_order(self, symbol: str, side: str, price: float, quantity: float) -> Optional[str]:
        price = round_price(price)
        quantity = floor_to_step(quantity, SIZE_STEP)
        if quantity < SIZE_STEP:
            return None
        if side == "buy":
            return self.client.buy(symbol, price, quantity)
        return self.client.sell(symbol, price, quantity)

    def _cancel_irrelevant_orders(self, active_symbols: Iterable[str], now: float):
        active_set = set(active_symbols)
        for order in list(self.orders.values()):
            stale = now - order.created_at >= self.order_stale_secs
            if stale or order.symbol not in active_set:
                self._cancel_order(order.order_id)

    def _cancel_symbol_orders(self, symbol: str):
        for order in list(self.orders.values()):
            if order.symbol == symbol:
                self._cancel_order(order.order_id)

    def _cancel_order(self, order_id: str):
        if order_id not in self.orders:
            return
        if self.client.cancel(order_id):
            self.orders.pop(order_id, None)

    def _reconcile_filled_orders(self):
        active_order_ids = self._active_order_ids()
        for order_id, order in list(self.orders.items()):
            if order_id in active_order_ids:
                continue
            self.orders.pop(order_id, None)
            if order.side == "buy":
                self.inferred_inventory[order.symbol] = round(
                    self.inferred_inventory.get(order.symbol, 0.0) + order.quantity,
                    3,
                )
            else:
                self.inferred_inventory[order.symbol] = round(
                    self.inferred_inventory.get(order.symbol, 0.0) - order.quantity,
                    3,
                )

    def _active_order_ids(self) -> Set[str]:
        state_lock = getattr(self.client, "_state_lock", None)
        if state_lock is not None:
            try:
                with state_lock:
                    return set(getattr(self.client, "_active_orders", set()) or set())
            except Exception:
                pass
        return set(getattr(self.client, "_active_orders", set()) or set())

    def _find_order(self, symbol: str, side: str) -> Optional[ManagedOrder]:
        for order in self.orders.values():
            if order.symbol == symbol and order.side == side:
                return order
        return None

    def _outstanding_quantity(self, symbol: str, side: str) -> float:
        total = 0.0
        for order in self.orders.values():
            if order.symbol == symbol and order.side == side:
                total += order.quantity
        return round(total, 3)

    def _base_order_quantity(self, mid: float) -> float:
        if mid <= 0:
            return 0.0
        raw_qty = self.notional_per_trade / mid
        capped = min(raw_qty, self.max_order_qty)
        return floor_to_step(capped, SIZE_STEP)

    def _passive_buy_price(self, best_bid: float, best_ask: float, fair_value: float) -> float:
        inside_price = best_bid + PRICE_TICK if best_ask - best_bid > 2 * PRICE_TICK else best_bid
        target_price = min(inside_price, fair_value)
        return round_price(max(best_bid, min(target_price, best_ask - PRICE_TICK)))

    def _passive_sell_price(self, best_bid: float, best_ask: float, fair_value: float) -> float:
        inside_price = best_ask - PRICE_TICK if best_ask - best_bid > 2 * PRICE_TICK else best_ask
        target_price = max(inside_price, fair_value)
        return round_price(min(best_ask, max(target_price, best_bid + PRICE_TICK)))

    def _best_prices(self, book: Dict[str, Any]) -> Tuple[float, float]:
        bids = [safe_float(price) for price in (book.get("bids", {}) or {}).keys()]
        asks = [safe_float(price) for price in (book.get("asks", {}) or {}).keys()]
        best_bid = max((price for price in bids if price > 0), default=0.0)
        best_ask = min((price for price in asks if price > 0), default=0.0)
        return best_bid, best_ask

    def _asset_symbol(self, asset: Dict[str, Any]) -> str:
        for key in ("symbol", "ticker", "code", "asset_symbol"):
            value = asset.get(key)
            if value:
                return str(value).upper()
        return ""

    def _asset_is_spot(self, asset: Dict[str, Any]) -> bool:
        symbol = self._asset_symbol(asset)
        if not symbol or symbol == "RUD":
            return False
        if asset.get("tradable") is False:
            return False

        text = " ".join(str(value).lower() for value in asset.values() if value is not None)
        if any(token in text for token in ("bond", "option", "prediction", "call", "put", "forex", "fx")):
            return False
        if any(token in text for token in ("spot", "equity", "stock")):
            return True
        return self._looks_like_spot_symbol(symbol)

    def _looks_like_spot_symbol(self, symbol: str) -> bool:
        symbol = str(symbol).upper()
        if not symbol or symbol == "RUD":
            return False
        banned_fragments = ("BOND", "OPT", "CALL", "PUT", "YES", "NO", "FX", "/")
        if any(fragment in symbol for fragment in banned_fragments):
            return False
        return True


def run():
    bot = MeanReversionSpotBot()
    bot.run()


if __name__ == "__main__":
    run()
