import socket
import select
import json
import websocket
import base64
import time

class WebSocketManager:
    def __init__(self, ip: str, port: int, local_ip: str):
        self.ip = ip
        self.port = port
        self.local_ip = local_ip
        self.ws = None

    def connect(self):
        if self.ws is None or not self.check_health():
            # Close stale socket before reconnecting
            if self.ws is not None:
                try:
                    self.ws.close()
                except Exception:
                    pass
                self.ws = None
            sock = socket.create_connection((self.ip, self.port), source_address=(self.local_ip, 0))
            ws = websocket.WebSocket()
            ws.sock = sock
            ws.connect(f"ws://{self.ip}:{self.port}")
            self.ws = ws
            print("[WebSocket] Connected")

    def send(self, message: dict):
        self.connect()
        if self.ws:
            try:
                # Ensure message is JSON serializable
                json_msg = json.dumps(message)
                self.ws.send(json_msg)
            except TypeError as e:
                print(f"[WebSocket] JSON serialization error: {e}")
                self.close()
            except Exception as e:
                print(f"[WebSocket] Send error: {e}")
                self.close()


    def receive_binary(self, timeout: float = None) -> bytes:
        self.connect()
        if self.ws:
            try:
                if timeout:
                    self.ws.settimeout(timeout)
                raw = self.ws.recv()  # raw is JSON string (type: str)
                if timeout:
                    self.ws.settimeout(None)  # Reset timeout
                return raw
            except (socket.timeout, TimeoutError, OSError) as e:
                # Handle various timeout exceptions
                if "timeout" in str(e).lower() or "timed out" in str(e).lower() or isinstance(e, (socket.timeout, TimeoutError)):
                    print(f"[WebSocket] Receive timeout after {timeout}s")
                    if timeout:
                        self.ws.settimeout(None)  # Reset timeout on error
                    raise TimeoutError(f"WebSocket receive timed out after {timeout}s")
                raise  # Re-raise if it's a different OSError
            except Exception as e:
                print(f"Receive error: {e}")
                if timeout:
                    self.ws.settimeout(None)  # Reset timeout on error
                # Don't close on timeout - let caller handle it
                if "timeout" not in str(e).lower() and "timed out" not in str(e).lower():
                    self.close()
                # Re-raise timeout-related exceptions
                if "timeout" in str(e).lower() or "timed out" in str(e).lower():
                    raise TimeoutError(f"WebSocket receive timed out: {e}")
                raise  # Re-raise other exceptions
        return b""
    
    def flush_pending_messages(self, timeout: float = 0.1):
        """
        Flush/drain any pending messages from the WebSocket buffer.
        This helps ensure we get fresh messages on the next subscription.
        """
        if not self.ws or not self.ws.connected:
            return
        
        try:
            # Try to receive messages with a short timeout until no more messages arrive
            while True:
                self.ws.settimeout(timeout)
                try:
                    _ = self.ws.recv()
                    # If we got a message, continue draining
                except:
                    # No more messages or timeout - we're done
                    break
            self.ws.settimeout(None)  # Reset timeout
        except Exception as e:
            # If flushing fails, that's okay - just continue
            try:
                self.ws.settimeout(None)
            except:
                pass
    
    def get_topics(self) -> list[tuple[str, str]]:
        self.connect()
        if self.ws:
            try:
                self.send({
                    "op": "call_service",
                    "service": "/rosapi/topics",
                    "id": "get_topics_request_1"
                })
                response = self.receive_binary()
                print(f"[WebSocket] Received response: {response}")
                if response:
                    data = json.loads(response)
                    if "values" in data:
                        topics = data["values"].get("topics", [])
                        types = data["values"].get("types", [])
                        if topics and types and len(topics) == len(types):
                            return list(zip(topics, types))
                        else:
                            print("[WebSocket] Mismatch in topics and types length")
            except json.JSONDecodeError as e:
                print(f"[WebSocket] JSON decode error: {e}")
            except Exception as e:
                print(f"[WebSocket] Error: {e}")
        return []

    def close(self):
        if self.ws and self.ws.connected:
            try:
                self.ws.close()
                print("[WebSocket] Closed")
            except Exception as e:
                print(f"[WebSocket] Close error: {e}")
            finally:
                self.ws = None

    def check_health(self) -> bool:
        """Check if the WebSocket connection is truly alive (not half-open).

        Uses a 3-method approach adapted from Isaac Sim's MCP extension:
        1. getpeername() — detects disconnected sockets
        2. select() — detects socket error state
        3. recv(MSG_PEEK) non-blocking — detects remote closure
        """
        if self.ws is None or not self.ws.connected:
            return False

        sock = self.ws.sock
        if sock is None:
            return False

        # Method 1: getpeername() — fails if socket is disconnected
        try:
            sock.getpeername()
        except (OSError, socket.error):
            print("[WebSocket] Health check failed: getpeername()")
            return False

        # Method 2: select() — check for error condition on socket
        try:
            _, _, err = select.select([], [], [sock], 0)
            if err:
                print("[WebSocket] Health check failed: select() error state")
                return False
        except (OSError, socket.error, ValueError):
            print("[WebSocket] Health check failed: select() exception")
            return False

        # Method 3: non-blocking recv with MSG_PEEK — detects remote closure
        try:
            sock.setblocking(False)
            try:
                data = sock.recv(1, socket.MSG_PEEK)
                # If recv returns empty bytes, the remote side closed
                if data == b'':
                    print("[WebSocket] Health check failed: remote closed (empty recv)")
                    return False
            except BlockingIOError:
                # No data available — socket is alive, just nothing to read
                pass
            except (OSError, socket.error):
                print("[WebSocket] Health check failed: recv(MSG_PEEK) error")
                return False
            finally:
                sock.setblocking(True)
        except Exception:
            # If we can't even set blocking mode, socket is broken
            return False

        return True
