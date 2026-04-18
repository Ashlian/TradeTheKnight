import copy
import json
import os
import queue
import threading
import time
import uuid
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional

import requests
import websocket

from exchange_wire import decode_state_message


class ExchangeException(Exception):
    """Base class for exchange errors."""


class AuthenticationError(ExchangeException):
    """Raised when BOT_ID is missing."""


class OrderError(ExchangeException):
    """Raised when an order is rejected."""


class ExchangeClient:
    @staticmethod
    def _env_flag(name: str, default: bool = False) -> bool:
        raw = os.environ.get(name)
        if raw is None:
            return default
        return raw.strip().lower() in {"1", "true", "yes", "on"}

    def __init__(self, api_url: Optional[str] = None, bot_id: Optional[str] = None):
        self.api_url = api_url or os.environ.get("EXCHANGE_URL", "http://127.0.0.1:3000")
        self.bot_id = str(bot_id or os.environ.get("BOT_ID") or "")

        if not self.bot_id:
            print(" Warning: BOT_ID not found in environment variables or constructor.")
            print(" Off-platform scripts require a valid BOT_ID to place orders.")
            print(" Public market data will still stream without authenticated order access.\n")
        else:
            print(f"Connected to Exchange at {self.api_url} as Agent {self.bot_id[:8]}...")

        ws_url = self.api_url.replace("http://", "ws://").replace("https://", "wss://")
        self.state_ws_url = f"{ws_url}/ws/state"

        self._latest_state: Dict[str, Any] = {
            "tick": 0,
            "competition_state": "pre_open",
            "book": {},
            "timeseries": {},
            "trades": [],
        }
        self._state_lock = threading.Lock()
        self._pending_lock = threading.Lock()
        self._active_orders = set()
        self._state_seq = 0
        self._pending_orders: Dict[str, Dict[str, Any]] = {}
        self._order_client_map: Dict[str, str] = {}
        self._tape_by_symbol: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self._tick_queue: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=1)
        self._stop_stream = False
        self._state_ws_app = None
        self._state_stream_healthy = False
        self._ack_timeout_secs = float(os.environ.get("LOAD_ACK_TIMEOUT_SECS", "3.0"))
        self._max_pending_orders = int(os.environ.get("LOAD_MAX_PENDING_ORDERS", "96"))
        self._load_backpressure_enabled = self._env_flag(
            "SDK_ENABLE_LOAD_BACKPRESSURE",
            default=bool(os.environ.get("LOAD_PROFILE") or os.environ.get("LOAD_MODE")),
        )
        self._cooldown_until = 0.0
        self._cooldown_secs = float(os.environ.get("LOAD_BACKOFF_SECS", "0.35"))
        self._max_cooldown_secs = float(os.environ.get("LOAD_MAX_BACKOFF_SECS", "2.0"))
        self._ping_interval_secs = float(os.environ.get("LOAD_WS_PING_INTERVAL_SECS", "25.0"))
        self._ping_timeout_secs = float(os.environ.get("LOAD_WS_PING_TIMEOUT_SECS", "15.0"))
        self._reconnect_sleep_secs = float(os.environ.get("LOAD_WS_RECONNECT_SLEEP_SECS", "0.75"))
        self._max_reconnect_sleep_secs = float(os.environ.get("LOAD_WS_MAX_RECONNECT_SLEEP_SECS", "5.0"))
        self._log_lock = threading.Lock()
        self._log_squelch: Dict[str, float] = {}
        self._symbol_cooldowns: Dict[str, float] = {}
        self._diagnostics_lock = threading.Lock()
        self._diagnostics: Dict[str, Any] = {
            "ws_reconnects": 0,
            "ws_errors": 0,
            "ack_timeouts": 0,
            "order_rejects": 0,
            "cooldown_skips": 0,
            "reject_reasons": defaultdict(int),
        }

        self._recover_snapshot()
        self._recover_active_orders()
        self._state_stream_thread = threading.Thread(target=self._state_websocket_stream, daemon=True)
        self._state_stream_thread.start()

    def _log_limited(self, key: str, message: str, min_interval: float = 5.0):
        now = time.time()
        with self._log_lock:
            last = self._log_squelch.get(key, 0.0)
            if now - last < min_interval:
                return
            self._log_squelch[key] = now
        print(message)

    def _set_cooldown(self, multiplier: float = 1.0):
        duration = min(self._max_cooldown_secs, self._cooldown_secs * max(1.0, multiplier))
        self._cooldown_until = max(self._cooldown_until, time.time() + duration)

    def _pending_depth(self) -> int:
        with self._pending_lock:
            return len(self._pending_orders)

    def _bump_diagnostic(self, key: str, amount: int = 1):
        with self._diagnostics_lock:
            self._diagnostics[key] = int(self._diagnostics.get(key, 0)) + amount

    def _bump_reject_reason(self, reason: str):
        with self._diagnostics_lock:
            self._diagnostics["reject_reasons"][reason] += 1

    def diagnostics_snapshot(self) -> Dict[str, Any]:
        with self._diagnostics_lock:
            return {
                "ws_reconnects": int(self._diagnostics["ws_reconnects"]),
                "ws_errors": int(self._diagnostics["ws_errors"]),
                "ack_timeouts": int(self._diagnostics["ack_timeouts"]),
                "order_rejects": int(self._diagnostics["order_rejects"]),
                "cooldown_skips": int(self._diagnostics["cooldown_skips"]),
                "reject_reasons": dict(self._diagnostics["reject_reasons"]),
                "pending_depth": self._pending_depth(),
            }

    def _symbol_is_cooled_down(self, symbol: str) -> bool:
        return self._symbol_cooldowns.get(symbol.upper(), 0.0) > time.time()

    def _cooldown_symbol(self, symbol: str, seconds: float):
        symbol = symbol.upper()
        self._symbol_cooldowns[symbol] = max(
            self._symbol_cooldowns.get(symbol, 0.0),
            time.time() + seconds,
        )

    @staticmethod
    def _is_invalid_market_state(reason: str) -> bool:
        lowered = reason.lower()
        return (
            "not tradable" in lowered
            or "paused" in lowered
            or "pre-open" in lowered
            or "closed" in lowered
        )

    def _fail_all_pending(self, reason: str):
        with self._pending_lock:
            pending_items = list(self._pending_orders.items())
            self._pending_orders.clear()
        for _, pending in pending_items:
            pending["result"] = {"status": "rejected", "reason": reason}
            pending["event"].set()

    def _state_websocket_stream(self):
        def on_message(ws, message):
            if not message:
                return
            try:
                self._handle_state_message(message)
            except Exception as exc:
                print(f"WebSocket state decode error: {exc}")

        def on_error(ws, error):
            self._state_stream_healthy = False
            self._set_cooldown(2.0)
            self._fail_all_pending("state_stream_error")
            self._bump_diagnostic("ws_errors")
            self._log_limited("state_ws_error", f"State WebSocket Error: {error}", min_interval=10.0)

        def on_close(ws, close_status_code, close_msg):
            self._state_stream_healthy = False
            self._set_cooldown(2.0)
            self._fail_all_pending("state_stream_closed")
            self._log_limited(
                "state_ws_close",
                f"State WebSocket Closed: {close_status_code} - {close_msg}",
                min_interval=10.0,
            )

        def on_open(ws):
            self._state_stream_healthy = True
            self._cooldown_until = 0.0
            self._reconnect_sleep_secs = float(os.environ.get("LOAD_WS_RECONNECT_SLEEP_SECS", "0.75"))
            self._log_limited("state_ws_open", f"State stream connected to {self.state_ws_url}", min_interval=2.0)

        headers = [f"X-API-Key: {self.bot_id}"] if self.bot_id else []
        while not self._stop_stream:
            try:
                self._state_ws_app = websocket.WebSocketApp(
                    self.state_ws_url,
                    header=headers,
                    on_message=on_message,
                    on_error=on_error,
                    on_close=on_close,
                    on_open=on_open,
                )
                self._state_ws_app.run_forever(
                    ping_interval=self._ping_interval_secs,
                    ping_timeout=self._ping_timeout_secs,
                )
            except Exception as exc:
                self._state_stream_healthy = False
                self._set_cooldown(2.0)
                self._fail_all_pending("state_ws_crash")
                self._bump_diagnostic("ws_errors")
                self._log_limited("state_ws_crash", f"Critical state WS crash: {exc}", min_interval=10.0)

            if not self._stop_stream:
                self._bump_diagnostic("ws_reconnects")
                time.sleep(self._reconnect_sleep_secs)
                self._reconnect_sleep_secs = min(self._max_reconnect_sleep_secs, self._reconnect_sleep_secs * 1.5)

    def _handle_state_message(self, message: Any):
        payload = self._decode_market_message(message)
        message_type = payload.get("type")

        if message_type == "snapshot":
            self._replace_state_v2_snapshot(payload)
            return

        if message_type == "heartbeat":
            return

        if message_type == "resync_required":
            self._state_seq = 0
            self._state_stream_healthy = False
            self._set_cooldown(2.0)
            self._fail_all_pending("state_resync_required")
            self._recover_snapshot()
            self._recover_active_orders()
            if self._state_ws_app:
                try:
                    self._state_ws_app.close()
                except Exception:
                    pass
            return

        if message_type != "delta":
            return

        delta = payload.get("payload") or {}
        seq = int(payload.get("seq", 0) or 0)
        private_only = (
            seq == 0
            and bool(delta.get("order_updates"))
            and not delta.get("book_updates")
            and not delta.get("timeseries_updates")
            and not delta.get("tape")
        )
        if private_only:
            self._apply_state_v2_delta(payload, advance_seq=False)
            return

        with self._state_lock:
            last_seq = self._state_seq
        if last_seq and seq != last_seq + 1:
            self._state_seq = 0
            self._state_stream_healthy = False
            self._set_cooldown(2.0)
            self._fail_all_pending("state_seq_gap")
            self._recover_snapshot()
            self._recover_active_orders()
            if self._state_ws_app:
                try:
                    self._state_ws_app.close()
                except Exception:
                    pass
            return

        self._apply_state_v2_delta(payload)

    def _decode_market_message(self, message: Any) -> Dict[str, Any]:
        return decode_state_message(message)

    def _replace_state_v2_snapshot(self, payload: Dict[str, Any]):
        envelope_payload = payload.get("payload") or {}
        book = self._convert_v2_books(envelope_payload.get("books") or {})
        trades = self._convert_v2_tape(envelope_payload.get("tape") or [])
        with self._state_lock:
            self._latest_state = {
                "tick": int(payload.get("tick", 0) or 0),
                "competition_state": payload.get("competition_state", "live"),
                "book": book,
                "timeseries": copy.deepcopy(envelope_payload.get("timeseries") or {}),
                "trades": trades,
            }
            self._state_seq = int(payload.get("seq", 0) or 0)
            self._tape_by_symbol = defaultdict(lambda: deque(maxlen=1000))
            for trade in envelope_payload.get("tape") or []:
                self._push_tape_record_locked(trade)
            state_snapshot = copy.deepcopy(self._latest_state)
        self._publish_state_snapshot(state_snapshot)

    def _apply_state_v2_delta(self, payload: Dict[str, Any], advance_seq: bool = True):
        delta = payload.get("payload") or {}
        with self._state_lock:
            state = self._latest_state
            self._apply_v2_book_updates_locked(state["book"], delta.get("book_updates") or [])
            for update in delta.get("timeseries_updates") or []:
                key = str(update.get("key", ""))
                if not key:
                    continue
                state["timeseries"][key] = float(update.get("value", 0.0))
            for trade in delta.get("tape") or []:
                self._push_tape_record_locked(trade)
                state["trades"] = [self._convert_v2_trade(trade), *state.get("trades", [])][:200]
            self._apply_v2_order_updates_locked(delta.get("order_updates") or [])
            state["tick"] = int(payload.get("tick", state.get("tick", 0)) or 0)
            state["competition_state"] = payload.get("competition_state", state.get("competition_state", "pre_open"))
            if advance_seq:
                self._state_seq = int(payload.get("seq", 0) or 0)
            snapshot = copy.deepcopy(state)

        self._publish_state_snapshot(snapshot)

    def _apply_v2_order_updates_locked(self, updates: List[Dict[str, Any]]):
        for update in updates:
            order_id = str(update.get("order_id", ""))
            client_order_id = str(update.get("client_order_id", "") or "")
            status = str(update.get("status", "accepted"))
            reason = update.get("reason")
            if status == "accepted":
                if order_id:
                    self._active_orders.add(order_id)
                if client_order_id:
                    self._order_client_map[order_id] = client_order_id
                    self._resolve_pending_locked(
                        client_order_id,
                        {"status": "accepted", "order_id": order_id},
                    )
            elif status in ("canceled", "filled"):
                if order_id:
                    self._active_orders.discard(order_id)
                    self._order_client_map.pop(order_id, None)
                if client_order_id:
                    self._resolve_pending_locked(
                        client_order_id,
                        {"status": status, "order_id": order_id, "reason": reason},
                    )
            elif status == "rejected" and client_order_id:
                self._resolve_pending_locked(
                    client_order_id,
                    {"status": "rejected", "order_id": order_id, "reason": reason},
                )

    def _resolve_pending_from_snapshot_locked(self):
        book = self._latest_state.get("book", {})
        for symbol_book in book.values():
            for side_key in ("bids", "asks"):
                for orders in symbol_book.get(side_key, {}).values():
                    for order in orders:
                        if str(order.get("owner_id", "")) == self.bot_id:
                            self._active_orders.add(str(order.get("id", "")))
                        client_order_id = order.get("client_order_id")
                        if client_order_id:
                            self._order_client_map[str(order.get("id", ""))] = str(client_order_id)
                            self._resolve_pending_locked(
                                str(client_order_id),
                                {
                                    "status": "accepted",
                                    "order_id": str(order.get("id", "")),
                                    "order": copy.deepcopy(order),
                                },
                            )

    def _resolve_pending_locked(self, client_order_id: str, result: Dict[str, Any]):
        with self._pending_lock:
            pending = self._pending_orders.get(client_order_id)
            if not pending:
                return
            pending["result"] = result
            pending["event"].set()

    def _publish_state_snapshot(self, state_snapshot: Dict[str, Any]):
        if self._tick_queue.full():
            try:
                self._tick_queue.get_nowait()
            except queue.Empty:
                pass
        try:
            self._tick_queue.put_nowait(state_snapshot)
        except queue.Full:
            pass

    def _recover_snapshot(self):
        book = self._get("/api/exchange/book")
        if not isinstance(book, dict):
            return
        trades = self._get("/api/exchange/trades")
        trades_list = trades if isinstance(trades, list) else []
        with self._state_lock:
            current_tick = int(self._latest_state.get("tick", 0))
            current_timeseries = copy.deepcopy(self._latest_state.get("timeseries", {}))
        with self._state_lock:
            self._latest_state = {
                "tick": current_tick,
                "competition_state": self._latest_state.get("competition_state", "pre_open"),
                "book": copy.deepcopy(book),
                "timeseries": current_timeseries,
                "trades": copy.deepcopy(trades_list[:200]),
            }
            self._resolve_pending_from_snapshot_locked()
            snapshot = copy.deepcopy(self._latest_state)
        self._publish_state_snapshot(snapshot)

    def _recover_active_orders(self):
        book = self._get("/api/exchange/book")
        if not isinstance(book, dict):
            return
        active_orders = set()
        order_client_map: Dict[str, str] = {}
        for symbol_book in book.values():
            for side_key in ("bids", "asks"):
                for orders in symbol_book.get(side_key, {}).values():
                    for order in orders:
                        order_id = str(order.get("id", ""))
                        if str(order.get("owner_id", "")) == self.bot_id:
                            active_orders.add(order_id)
                        client_order_id = order.get("client_order_id")
                        if client_order_id:
                            order_client_map[order_id] = str(client_order_id)
        with self._pending_lock:
            self._active_orders = active_orders
            self._order_client_map = order_client_map

    def _post(self, endpoint: str, data: Dict[str, Any]) -> Any:
        if not self.bot_id:
            raise AuthenticationError("CRITICAL: Cannot place orders or cancel without a BOT_ID.")

        headers = {"X-API-Key": self.bot_id}

        try:
            resp = requests.post(f"{self.api_url}{endpoint}", json=data, headers=headers, timeout=1.0)
            if resp.status_code in (200, 202):
                if not resp.content:
                    return {}
                try:
                    return resp.json()
                except ValueError:
                    return resp.text
            if resp.status_code == 400:
                raise OrderError(f"Rejected: {resp.text}")
            if resp.status_code == 403:
                raise AuthenticationError(f"Forbidden: {resp.text}")
            raise ExchangeException(f"Server Error {resp.status_code}: {resp.text}")
        except requests.exceptions.ConnectionError:
            raise ExchangeException("Could not connect to Exchange (Is it running?)")
        except requests.exceptions.Timeout:
            raise ExchangeException("Exchange timed out")

    def _get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Any:
        headers = {"X-API-Key": self.bot_id} if self.bot_id else {}
        try:
            resp = requests.get(f"{self.api_url}{endpoint}", params=params, headers=headers, timeout=1.0)
            if resp.status_code == 200:
                return resp.json()
            return None
        except Exception:
            return None

    def get_assets(self) -> List[Dict[str, Any]]:
        result = self._get("/api/exchange/assets")
        return result if isinstance(result, list) else []

    def _convert_v2_books(self, books: Dict[str, Any]) -> Dict[str, Any]:
        converted: Dict[str, Any] = {}
        for symbol, book in books.items():
            bids = self._convert_v2_side(book.get("bids") or {})
            asks = self._convert_v2_side(book.get("asks") or {})
            converted[str(symbol).upper()] = {
                "bids": bids,
                "asks": asks,
                "last_trade_ticks": book.get("last_trade_ticks"),
            }
        return converted

    def _convert_v2_side(self, side: Dict[Any, Any]) -> Dict[str, List[Dict[str, Any]]]:
        levels: Dict[str, List[Dict[str, Any]]] = {}
        for price_ticks, quantity_lots in side.items():
            price_str = self._format_ticks(int(price_ticks), 4)
            qty_str = self._format_ticks(int(quantity_lots), 3)
            levels[price_str] = [
                {
                    "id": "",
                    "owner_id": "",
                    "symbol": "",
                    "price": price_str,
                    "quantity": qty_str,
                    "side": "",
                    "timestamp": 0,
                }
            ]
        return levels

    def _apply_v2_book_updates_locked(self, book_state: Dict[str, Any], updates: List[Dict[str, Any]]):
        for update in updates:
            symbol = str(update.get("symbol", "")).upper()
            if not symbol:
                continue
            book = book_state.setdefault(symbol, {"bids": {}, "asks": {}, "last_trade_ticks": None})
            for price_ticks, quantity_lots in update.get("bids") or []:
                self._apply_v2_level_update(book["bids"], int(price_ticks), int(quantity_lots))
            for price_ticks, quantity_lots in update.get("asks") or []:
                self._apply_v2_level_update(book["asks"], int(price_ticks), int(quantity_lots))
            if update.get("last_trade_ticks") is not None:
                book["last_trade_ticks"] = int(update["last_trade_ticks"])

    def _apply_v2_level_update(self, side_levels: Dict[str, List[Dict[str, Any]]], price_ticks: int, quantity_lots: int):
        price_str = self._format_ticks(price_ticks, 4)
        if quantity_lots <= 0:
            side_levels.pop(price_str, None)
            return
        qty_str = self._format_ticks(quantity_lots, 3)
        side_levels[price_str] = [
            {
                "id": "",
                "owner_id": "",
                "symbol": "",
                "price": price_str,
                "quantity": qty_str,
                "side": "",
                "timestamp": 0,
            }
        ]

    def _convert_v2_tape(self, tape: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [self._convert_v2_trade(record) for record in tape][:200]

    def _convert_v2_trade(self, record: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "bid_order_id": "",
            "ask_order_id": "",
            "buyer_id": str(record.get("buyer_bot_id", "")),
            "seller_id": str(record.get("seller_bot_id", "")),
            "symbol": str(record.get("symbol", "")),
            "price": self._format_ticks(int(record.get("price_ticks", 0) or 0), 4),
            "quantity": self._format_ticks(int(record.get("quantity_lots", 0) or 0), 3),
            "timestamp": int(record.get("ts_ms", 0) or 0),
            "tick": int(record.get("tick", 0) or 0),
        }

    def _push_tape_record_locked(self, record: Dict[str, Any]):
        symbol = str(record.get("symbol", "")).upper()
        if symbol:
            self._tape_by_symbol[symbol].append(copy.deepcopy(record))

    @staticmethod
    def _format_ticks(value: int, scale_digits: int) -> str:
        sign = "-" if value < 0 else ""
        value = abs(value)
        whole = value // (10 ** scale_digits)
        frac = value % (10 ** scale_digits)
        if scale_digits == 0:
            return f"{sign}{whole}"
        return f"{sign}{whole}.{frac:0{scale_digits}d}"

    def _book_contains_order_locked(self, order_id: str) -> bool:
        if not order_id:
            return False
        for symbol_book in self._latest_state.get("book", {}).values():
            for side_key in ("bids", "asks"):
                for orders in symbol_book.get(side_key, {}).values():
                    if any(str(order.get("id", "")) == order_id for order in orders):
                        return True
        return False

    def _sync_active_orders_locked(self):
        active_orders = set()
        order_client_map: Dict[str, str] = {}
        for symbol_book in self._latest_state.get("book", {}).values():
            for side_key in ("bids", "asks"):
                for orders in symbol_book.get(side_key, {}).values():
                    for order in orders:
                        order_id = str(order.get("id", ""))
                        if str(order.get("owner_id", "")) == self.bot_id:
                            active_orders.add(order_id)
                        client_order_id = order.get("client_order_id")
                        if client_order_id:
                            order_client_map[order_id] = str(client_order_id)
        self._active_orders = active_orders
        self._order_client_map = order_client_map

    @staticmethod
    def _apply_trade_to_levels(levels: Dict[str, List[Dict[str, Any]]], order_id: str, quantity: Any):
        if not order_id:
            return
        trade_qty = float(quantity or 0)
        for price in list(levels.keys()):
            orders = levels[price]
            for index, order in enumerate(list(orders)):
                if str(order.get("id", "")) != order_id:
                    continue
                remaining = float(order.get("quantity", "0")) - trade_qty
                if remaining <= 0:
                    del orders[index]
                else:
                    order["quantity"] = f"{remaining:.3f}"
                    orders[index] = order
                if not orders:
                    del levels[price]
                return

    @staticmethod
    def _remove_order_from_levels(levels: Dict[str, List[Dict[str, Any]]], order_id: str):
        if not order_id:
            return
        for price in list(levels.keys()):
            orders = [order for order in levels[price] if str(order.get("id", "")) != order_id]
            if len(orders) != len(levels[price]):
                if orders:
                    levels[price] = orders
                else:
                    del levels[price]
                return

    def stream_state(self):
        while not self._stop_stream:
            try:
                yield self._tick_queue.get(timeout=1.0)
            except queue.Empty:
                continue

    def get_book(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        with self._state_lock:
            full_book = copy.deepcopy(self._latest_state.get("book", {}))
        if symbol:
            return full_book.get(symbol.upper(), {"bids": {}, "asks": {}})
        if "bids" not in full_book and len(full_book) == 1:
            return list(full_book.values())[0]
        return full_book

    def get_team_state(self) -> Dict[str, Any]:
        result = self._get("/api/exchange/team/state")
        return result if isinstance(result, dict) else {}

    def get_best_bid(self, symbol: str) -> float:
        book = self.get_book(symbol)
        bids = [float(price) for price in book.get("bids", {}).keys()]
        return max(bids) if bids else 0.0

    def get_best_ask(self, symbol: str) -> float:
        book = self.get_book(symbol)
        asks = [float(price) for price in book.get("asks", {}).keys()]
        return min(asks) if asks else 0.0

    def get_price(self, symbol: str) -> float:
        best_bid = self.get_best_bid(symbol)
        best_ask = self.get_best_ask(symbol)
        if best_bid <= 0 or best_ask <= 0:
            return 0.0
        return (best_bid + best_ask) / 2.0

    def cancel_all(self, symbol: Optional[str] = None) -> bool:
        with self._state_lock:
            to_cancel = list(self._active_orders)
        for order_id in to_cancel:
            self.cancel(order_id)
        return True

    def place_order(self, side: str, price: float, quantity: float, symbol: str) -> Optional[str]:
        symbol = symbol.upper()
        if quantity <= 0 or price <= 0:
            self._log_limited("invalid_order", f"Warning: Invalid order parameters (Price: {price}, Qty: {quantity})", min_interval=2.0)
            return None
        if time.time() < self._cooldown_until:
            return None
        if self._symbol_is_cooled_down(symbol):
            self._bump_diagnostic("cooldown_skips")
            return None
        if not self._state_stream_healthy:
            self._set_cooldown(1.0)
            return None
        if self._load_backpressure_enabled and self._pending_depth() >= self._max_pending_orders:
            self._set_cooldown(2.0)
            self._log_limited(
                "pending_cap",
                f"ExchangeClient pending order cap reached ({self._max_pending_orders}); backing off",
                min_interval=5.0,
            )
            return None

        order_id = str(uuid.uuid4())
        client_order_id = str(uuid.uuid4())
        event = threading.Event()
        with self._pending_lock:
            self._pending_orders[client_order_id] = {"event": event, "result": None}

        order = {
            "id": order_id,
            "owner_id": self.bot_id,
            "client_order_id": client_order_id,
            "symbol": symbol,
            "price": f"{price:.4f}",
            "quantity": f"{quantity:.3f}",
            "side": side.capitalize(),
            "order_type": "Limit",
            "timestamp": int(time.time() * 1000),
        }

        try:
            self._post("/api/exchange/order", order)
            if not event.wait(timeout=self._ack_timeout_secs):
                with self._pending_lock:
                    self._pending_orders.pop(client_order_id, None)
                self._bump_diagnostic("ack_timeouts")
                self._set_cooldown(2.0)
                raise ExchangeException("Timed out waiting for order acknowledgement")

            with self._pending_lock:
                result = self._pending_orders.pop(client_order_id, {}).get("result") or {}

            status = result.get("status")
            if status == "accepted":
                accepted_order_id = str(result.get("order_id") or order_id)
                with self._state_lock:
                    self._active_orders.add(accepted_order_id)
                    self._order_client_map[accepted_order_id] = client_order_id
                return accepted_order_id

            reason = result.get("reason", "Order rejected")
            raise OrderError(str(reason))
        except OrderError as exc:
            with self._pending_lock:
                self._pending_orders.pop(client_order_id, None)
            reason = str(exc)
            self._bump_diagnostic("order_rejects")
            self._bump_reject_reason(reason)
            if self._is_invalid_market_state(reason):
                self._cooldown_symbol(symbol, 10.0)
            elif self._load_backpressure_enabled:
                lowered = reason.lower()
                if "insufficient" in lowered or "balance" in lowered or "capital" in lowered:
                    self._cooldown_symbol(symbol, 2.0)
                self._set_cooldown(1.0)
            self._log_limited("order_rejected", f"Order Rejected: {exc}", min_interval=2.0)
            return None
        except ExchangeException as exc:
            with self._pending_lock:
                self._pending_orders.pop(client_order_id, None)
            self._set_cooldown(2.0)
            self._log_limited("exchange_error", f"Exchange Error: {exc}", min_interval=2.0)
            return None

    def cancel(self, order_id: str) -> bool:
        try:
            self._post("/api/exchange/order/cancel", {"order_id": order_id, "bot_id": self.bot_id})
            with self._state_lock:
                self._active_orders.discard(order_id)
                self._order_client_map.pop(order_id, None)
            return True
        except OrderError as exc:
            self._log_limited("cancel_rejected", f"Cancel Rejected: {exc}", min_interval=2.0)
            return False
        except ExchangeException as exc:
            self._set_cooldown(1.0)
            self._log_limited("cancel_exchange_error", f"Exchange Error: {exc}", min_interval=2.0)
            return False

    def buy(self, symbol: str, price: float, quantity: float) -> Optional[str]:
        return self.place_order("Bid", price, quantity, symbol)

    def sell(self, symbol: str, price: float, quantity: float) -> Optional[str]:
        return self.place_order("Ask", price, quantity, symbol)

    def _agent_url(self) -> str:
        return self.api_url.replace(":3000", ":8000")

    def list_timeseries(self) -> List[Dict[str, Any]]:
        headers = {"X-API-Key": self.bot_id} if self.bot_id else {}
        try:
            resp = requests.get(f"{self._agent_url()}/timeseries", headers=headers, timeout=1.0)
            if resp.status_code == 200:
                return resp.json()
            return []
        except Exception:
            return []

    def get_timeseries(self, name: str, limit: int = 100) -> List[Dict[str, Any]]:
        headers = {"X-API-Key": self.bot_id} if self.bot_id else {}
        try:
            resp = requests.get(
                f"{self._agent_url()}/timeseries/{name}/data",
                params={"limit": limit},
                headers=headers,
                timeout=1.0,
            )
            if resp.status_code == 200:
                return resp.json().get("data", [])
            return []
        except Exception:
            return []

    def place_auction_bid(self, symbol: str, yield_rate: float, quantity: float) -> bool:
        symbol = symbol.upper()
        if quantity <= 0 or yield_rate <= 0:
            self._log_limited("invalid_auction_bid", f"Warning: Invalid bid parameters (Yield: {yield_rate}, Qty: {quantity})", min_interval=2.0)
            return False
        if time.time() < self._cooldown_until or self._symbol_is_cooled_down(symbol):
            self._bump_diagnostic("cooldown_skips")
            return False

        payload = {
            "symbol": symbol,
            "yield_rate": yield_rate,
            "quantity": round(quantity, 3),
            "bot_id": self.bot_id,
        }
        try:
            self._post("/api/exchange/auction/bid", payload)
            return True
        except OrderError as exc:
            reason = str(exc)
            self._bump_diagnostic("order_rejects")
            self._bump_reject_reason(reason)
            if self._is_invalid_market_state(reason) or "auction" in reason.lower():
                self._cooldown_symbol(symbol, 10.0)
            elif self._load_backpressure_enabled:
                lowered = reason.lower()
                if "insufficient" in lowered or "balance" in lowered or "capital" in lowered:
                    self._cooldown_symbol(symbol, 3.0)
            self._log_limited("auction_rejected", f"Auction Bid Rejected: {exc}", min_interval=2.0)
            return False
        except ExchangeException as exc:
            self._set_cooldown(1.0)
            self._log_limited("auction_exchange_error", f"Auction Bid Failed: {exc}", min_interval=2.0)
            return False

    def close(self):
        self._stop_stream = True
        if self._state_ws_app:
            try:
                self._state_ws_app.close()
            except Exception:
                pass