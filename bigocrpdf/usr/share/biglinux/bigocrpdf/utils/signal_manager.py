"""
BigOcrPdf - Signal Manager

This module provides centralized signal/callback management for GTK widgets.
It ensures proper cleanup of signal handlers to prevent memory leaks.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple

from gi.repository import GObject
from utils.logger import logger


class SignalManager:
    """Centralized manager for GTK signal handlers.
    
    This class tracks all signal connections and provides methods to
    cleanly disconnect them when no longer needed. This prevents memory
    leaks and dangling handler issues.
    
    Usage:
        signals = SignalManager()
        signals.connect(button, "clicked", on_button_clicked)
        signals.connect(entry, "changed", on_entry_changed)
        
        # Later, when cleaning up:
        signals.disconnect_all()
    """

    def __init__(self):
        """Initialize the signal manager."""
        self._handlers: List[Tuple[GObject.Object, int, str]] = []
        self._named_handlers: Dict[str, Tuple[GObject.Object, int]] = {}

    def connect(
        self,
        widget: GObject.Object,
        signal: str,
        callback: Callable,
        *args,
        name: Optional[str] = None
    ) -> int:
        """Connect a signal handler to a widget.
        
        Args:
            widget: The GTK widget to connect the signal to
            signal: The signal name (e.g., "clicked", "changed")
            callback: The callback function to invoke
            *args: Additional arguments to pass to the callback
            name: Optional unique name for this handler (for named disconnect)
            
        Returns:
            The handler ID from GTK
        """
        try:
            if args:
                handler_id = widget.connect(signal, callback, *args)
            else:
                handler_id = widget.connect(signal, callback)
            
            self._handlers.append((widget, handler_id, signal))
            
            if name:
                self._named_handlers[name] = (widget, handler_id)
            
            return handler_id
            
        except Exception as e:
            logger.error(f"Error connecting signal {signal}: {e}")
            return 0

    def connect_after(
        self,
        widget: GObject.Object,
        signal: str,
        callback: Callable,
        *args,
        name: Optional[str] = None
    ) -> int:
        """Connect a signal handler to run after the default handler.
        
        Args:
            widget: The GTK widget to connect the signal to
            signal: The signal name
            callback: The callback function to invoke
            *args: Additional arguments to pass to the callback
            name: Optional unique name for this handler
            
        Returns:
            The handler ID from GTK
        """
        try:
            if args:
                handler_id = widget.connect_after(signal, callback, *args)
            else:
                handler_id = widget.connect_after(signal, callback)
            
            self._handlers.append((widget, handler_id, signal))
            
            if name:
                self._named_handlers[name] = (widget, handler_id)
            
            return handler_id
            
        except Exception as e:
            logger.error(f"Error connecting signal after {signal}: {e}")
            return 0

    def disconnect(self, name: str) -> bool:
        """Disconnect a named signal handler.
        
        Args:
            name: The unique name given when connecting
            
        Returns:
            True if handler was disconnected, False otherwise
        """
        if name not in self._named_handlers:
            return False
        
        widget, handler_id = self._named_handlers.pop(name)
        
        try:
            if widget.handler_is_connected(handler_id):
                widget.disconnect(handler_id)
                
                # Also remove from main handler list
                self._handlers = [
                    (w, h, s) for w, h, s in self._handlers
                    if not (w == widget and h == handler_id)
                ]
                return True
                
        except Exception as e:
            logger.error(f"Error disconnecting handler {name}: {e}")
        
        return False

    def disconnect_widget(self, widget: GObject.Object) -> int:
        """Disconnect all signal handlers for a specific widget.
        
        Args:
            widget: The widget to disconnect all handlers for
            
        Returns:
            Number of handlers disconnected
        """
        count = 0
        remaining_handlers = []
        
        for w, handler_id, signal in self._handlers:
            if w == widget:
                try:
                    if w.handler_is_connected(handler_id):
                        w.disconnect(handler_id)
                        count += 1
                except Exception as e:
                    logger.error(f"Error disconnecting widget handler: {e}")
            else:
                remaining_handlers.append((w, handler_id, signal))
        
        self._handlers = remaining_handlers
        
        # Also clean named handlers for this widget
        self._named_handlers = {
            name: (w, h) for name, (w, h) in self._named_handlers.items()
            if w != widget
        }
        
        return count

    def disconnect_all(self) -> int:
        """Disconnect all tracked signal handlers.
        
        Returns:
            Number of handlers disconnected
        """
        count = 0
        
        for widget, handler_id, signal in self._handlers:
            try:
                if widget.handler_is_connected(handler_id):
                    widget.disconnect(handler_id)
                    count += 1
            except Exception as e:
                logger.debug(f"Error disconnecting handler: {e}")
        
        self._handlers.clear()
        self._named_handlers.clear()
        
        if count > 0:
            logger.debug(f"Disconnected {count} signal handlers")
        
        return count

    def block(self, name: str) -> bool:
        """Temporarily block a named signal handler.
        
        Args:
            name: The unique name given when connecting
            
        Returns:
            True if handler was blocked, False otherwise
        """
        if name not in self._named_handlers:
            return False
        
        widget, handler_id = self._named_handlers[name]
        
        try:
            widget.handler_block(handler_id)
            return True
        except Exception as e:
            logger.error(f"Error blocking handler {name}: {e}")
            return False

    def unblock(self, name: str) -> bool:
        """Unblock a previously blocked named signal handler.
        
        Args:
            name: The unique name given when connecting
            
        Returns:
            True if handler was unblocked, False otherwise
        """
        if name not in self._named_handlers:
            return False
        
        widget, handler_id = self._named_handlers[name]
        
        try:
            widget.handler_unblock(handler_id)
            return True
        except Exception as e:
            logger.error(f"Error unblocking handler {name}: {e}")
            return False

    def is_connected(self, name: str) -> bool:
        """Check if a named handler is still connected.
        
        Args:
            name: The unique name to check
            
        Returns:
            True if handler exists and is connected
        """
        if name not in self._named_handlers:
            return False
        
        widget, handler_id = self._named_handlers[name]
        
        try:
            return widget.handler_is_connected(handler_id)
        except Exception:
            return False

    def get_handler_count(self) -> int:
        """Get the number of tracked handlers.
        
        Returns:
            Number of handlers being tracked
        """
        return len(self._handlers)

    def get_handler_info(self) -> List[Tuple[str, str, int]]:
        """Get information about all tracked handlers.
        
        Returns:
            List of tuples (widget_type, signal_name, handler_id)
        """
        return [
            (type(widget).__name__, signal, handler_id)
            for widget, handler_id, signal in self._handlers
        ]

    def __del__(self):
        """Cleanup on destruction."""
        self.disconnect_all()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        """Context manager exit - disconnect all handlers."""
        self.disconnect_all()
        return False
