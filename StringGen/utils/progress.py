#!/usr/bin/env python3
"""
Progress tracking utilities for SCA generation.

Provides real-time progress bars, ETAs, and performance metrics
during generation processes.
"""

import sys
import time
import threading
from typing import Iterator, Optional, Any, TypeVar
from datetime import datetime, timedelta


T = TypeVar('T')


class ProgressTracker:
    """
    Thread-safe progress tracker with ETA calculation.
    
    Features:
    - Real-time progress bar in terminal
    - Accurate ETA based on rolling average
    - Performance metrics (items/sec)
    - Memory-efficient streaming updates
    """
    
    def __init__(self, 
                 total: int,
                 desc: str = "Processing",
                 bar_width: int = 40,
                 update_interval: float = 0.1):
        """
        Initialize progress tracker.
        
        Args:
            total: Total number of items to process
            desc: Description to show before progress bar
            bar_width: Width of progress bar in characters
            update_interval: Minimum time between display updates (seconds)
        """
        self.total = total
        self.desc = desc
        self.bar_width = bar_width
        self.update_interval = update_interval
        
        # Progress state
        self.current = 0
        self.start_time = time.time()
        self.last_update = 0
        
        # Thread safety
        self._lock = threading.Lock()
        self._closed = False
        
        # Performance tracking
        self._rate_window = []  # Rolling window for rate calculation
        self._window_size = 20
        
        # Initial display
        self._display()
    
    def update(self, n: int = 1) -> None:
        """
        Update progress by n items.
        
        Args:
            n: Number of items completed
        """
        with self._lock:
            if self._closed:
                return
                
            self.current += n
            
            # Track rate
            current_time = time.time()
            self._rate_window.append((current_time, n))
            
            # Maintain window size
            if len(self._rate_window) > self._window_size:
                self._rate_window.pop(0)
            
            # Update display if enough time has passed
            if current_time - self.last_update >= self.update_interval:
                self._display()
                self.last_update = current_time
    
    def _display(self) -> None:
        """Display current progress bar."""
        if self.total <= 0:
            return
        
        # Calculate percentage
        percentage = min(100, self.current * 100 / self.total)
        
        # Build progress bar
        filled = int(self.bar_width * percentage / 100)
        bar = '█' * filled + '░' * (self.bar_width - filled)
        
        # Calculate rate and ETA
        elapsed = time.time() - self.start_time
        rate = self._calculate_rate()
        eta = self._calculate_eta(rate)
        
        # Format output
        output = f"\r{self.desc}: [{bar}] {percentage:6.1f}% "
        output += f"({self.current:,}/{self.total:,}) "
        
        if rate > 0:
            output += f"[{rate:,.0f} items/s] "
        
        if eta:
            output += f"ETA: {eta}"
        
        # Write to stderr to avoid mixing with stdout
        sys.stderr.write(output)
        sys.stderr.flush()
    
    def _calculate_rate(self) -> float:
        """Calculate current processing rate using rolling window."""
        if len(self._rate_window) < 2:
            return 0.0
        
        # Use window to calculate rate
        time_span = self._rate_window[-1][0] - self._rate_window[0][0]
        if time_span <= 0:
            return 0.0
        
        total_items = sum(items for _, items in self._rate_window)
        return total_items / time_span
    
    def _calculate_eta(self, rate: float) -> str:
        """Calculate estimated time to completion."""
        if rate <= 0 or self.current >= self.total:
            return ""
        
        remaining = self.total - self.current
        seconds_left = remaining / rate
        
        # Format time remaining
        if seconds_left < 60:
            return f"{int(seconds_left)}s"
        elif seconds_left < 3600:
            minutes = int(seconds_left / 60)
            seconds = int(seconds_left % 60)
            return f"{minutes}m {seconds}s"
        else:
            hours = int(seconds_left / 3600)
            minutes = int((seconds_left % 3600) / 60)
            return f"{hours}h {minutes}m"
    
    def track(self, iterator: Iterator[T]) -> Iterator[T]:
        """
        Wrap an iterator to automatically track progress.
        
        Args:
            iterator: Iterator to track
            
        Yields:
            Items from the iterator
        """
        try:
            for item in iterator:
                yield item
                self.update(1)
        finally:
            self.close()
    
    def close(self) -> None:
        """Close progress tracker and show final statistics."""
        with self._lock:
            if self._closed:
                return
            
            self._closed = True
            
            # Final update
            self._display()
            
            # Print newline to move to next line
            sys.stderr.write('\n')
            sys.stderr.flush()
            
            # Show final statistics
            elapsed = time.time() - self.start_time
            if elapsed > 0 and self.current > 0:
                final_rate = self.current / elapsed
                sys.stderr.write(
                    f"Completed {self.current:,} items in "
                    f"{timedelta(seconds=int(elapsed))} "
                    f"({final_rate:,.0f} items/s average)\n"
                )
                sys.stderr.flush()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


class SpinnerProgress:
    """
    Simple spinner for indeterminate progress.
    
    Used when total count is unknown or for long-running operations.
    """
    
    def __init__(self, desc: str = "Processing"):
        """Initialize spinner."""
        self.desc = desc
        self.spinner_chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        self.current = 0
        self._stop = False
        self._thread = None
    
    def start(self):
        """Start spinner in background thread."""
        self._stop = False
        self._thread = threading.Thread(target=self._spin)
        self._thread.daemon = True
        self._thread.start()
    
    def _spin(self):
        """Spin animation loop."""
        while not self._stop:
            char = self.spinner_chars[self.current % len(self.spinner_chars)]
            sys.stderr.write(f"\r{self.desc}: {char}")
            sys.stderr.flush()
            self.current += 1
            time.sleep(0.1)
    
    def stop(self):
        """Stop spinner."""
        self._stop = True
        if self._thread:
            self._thread.join()
        sys.stderr.write("\r" + " " * (len(self.desc) + 4) + "\r")
        sys.stderr.flush()