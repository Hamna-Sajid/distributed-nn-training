"""
Network Simulator for Heterogeneous Communication

Injects realistic network effects (latency, jitter, packet loss)
to simulate real-world network heterogeneity without requiring
different physical network configurations.
"""

import time
import random
from typing import Optional, Dict, Tuple
from enum import Enum


class NetworkCondition(Enum):
    """Predefined network conditions."""
    IDEAL = ("ideal", 0.0, 0.0)           # No delay
    FAST_LAN = ("fast_lan", 1.0, 0.1)      # 1ms ± 0.1ms
    STANDARD_LAN = ("standard_lan", 5.0, 1.0)   # 5ms ± 1ms
    SLOW_WAN = ("slow_wan", 50.0, 10.0)    # 50ms ± 10ms
    VERY_SLOW = ("very_slow", 200.0, 50.0) # 200ms ± 50ms
    
    def __init__(self, name: str, latency_ms: float, jitter_ms: float):
        self.name = name
        self.latency_ms = latency_ms
        self.jitter_ms = jitter_ms


class NetworkSimulator:
    """
    Simulates network effects for communication between master and workers.
    
    Can simulate:
    - Variable latency (one-way trip time)
    - Jitter (variance in latency)
    - Packet loss (percentage of messages lost)
    - Bandwidth throttling (if needed later)
    """
    
    def __init__(self, enabled: bool = False, loss_rate: float = 0.0):
        """
        Args:
            enabled: Whether network simulation is active
            loss_rate: Probability of packet loss (0.0 to 1.0)
        """
        self.enabled = enabled
        self.loss_rate = max(0.0, min(1.0, loss_rate))
        self.worker_conditions: Dict[int, NetworkCondition] = {}
        self.stats = {
            "packets_sent": 0,
            "packets_lost": 0,
            "total_delay_s": 0.0,
            "num_delays": 0
        }
    
    def set_worker_condition(self, worker_id: int, condition: NetworkCondition) -> None:
        """
        Set network condition for a specific worker.
        
        Args:
            worker_id: ID of the worker
            condition: NetworkCondition enum value
        """
        self.worker_conditions[worker_id] = condition
    
    def set_worker_custom_latency(self, worker_id: int, 
                                 latency_ms: float, jitter_ms: float = 0.0) -> None:
        """
        Set custom latency parameters for a worker.
        
        Args:
            worker_id: ID of the worker
            latency_ms: Base latency in milliseconds
            jitter_ms: Jitter (standard deviation) in milliseconds
        """
        # Create a custom condition
        custom_cond = NetworkCondition.__new__(NetworkCondition)
        custom_cond.name = f"custom_w{worker_id}"
        custom_cond.latency_ms = latency_ms
        custom_cond.jitter_ms = jitter_ms
        self.worker_conditions[worker_id] = custom_cond
    
    def simulate_send_delay(self, worker_id: int) -> Tuple[float, bool]:
        """
        Simulate the delay for sending a message to a worker.
        
        Returns:
            Tuple of (delay_seconds, did_packet_loss)
            If packet loss occurs, delay = 0 and flag = True
        """
        if not self.enabled:
            return 0.0, False
        
        self.stats["packets_sent"] += 1
        
        # Check for packet loss
        if random.random() < self.loss_rate:
            self.stats["packets_lost"] += 1
            return 0.0, True
        
        # Get network condition for this worker
        condition = self.worker_conditions.get(
            worker_id, 
            NetworkCondition.IDEAL
        )
        
        # Add jitter to base latency
        if condition.jitter_ms > 0:
            jitter = random.gauss(0, condition.jitter_ms)
            latency_ms = max(0, condition.latency_ms + jitter)
        else:
            latency_ms = condition.latency_ms
        
        delay_s = latency_ms / 1000.0
        self.stats["total_delay_s"] += delay_s
        self.stats["num_delays"] += 1
        
        return delay_s, False
    
    def apply_send_delay(self, worker_id: int) -> bool:
        """
        Apply network delay by sleeping.
        
        Returns:
            False if packet loss occurred (caller should retry or skip),
            True if message was successfully delayed
        """
        delay, lost = self.simulate_send_delay(worker_id)
        
        if lost:
            return False  # Packet lost, caller handles retry logic
        
        if delay > 0:
            time.sleep(delay)
        
        return True
    
    def get_statistics(self) -> Dict:
        """Get cumulative network simulation statistics."""
        stats = self.stats.copy()
        
        if stats["num_delays"] > 0:
            stats["avg_delay_ms"] = (stats["total_delay_s"] / stats["num_delays"]) * 1000.0
        else:
            stats["avg_delay_ms"] = 0.0
        
        if stats["packets_sent"] > 0:
            stats["loss_rate_actual"] = stats["packets_lost"] / stats["packets_sent"]
        else:
            stats["loss_rate_actual"] = 0.0
        
        return stats
    
    def reset_statistics(self) -> None:
        """Reset all statistics counters."""
        self.stats = {
            "packets_sent": 0,
            "packets_lost": 0,
            "total_delay_s": 0.0,
            "num_delays": 0
        }
    
    def log_configuration(self) -> str:
        """Generate log string describing network configuration."""
        if not self.enabled:
            return "[Network Simulator] Disabled"
        
        lines = ["[Network Simulator] Configuration:"]
        lines.append(f"  Global packet loss rate: {self.loss_rate * 100:.1f}%")
        
        if self.worker_conditions:
            lines.append("  Worker conditions:")
            for w_id, cond in sorted(self.worker_conditions.items()):
                lines.append(
                    f"    Worker {w_id}: {cond.name} "
                    f"({cond.latency_ms:.1f}ms ± {cond.jitter_ms:.1f}ms)"
                )
        else:
            lines.append("  No worker-specific conditions set")
        
        return "\n".join(lines)
    
    def log_statistics(self) -> str:
        """Generate log string with current statistics."""
        stats = self.get_statistics()
        lines = ["[Network Simulator] Statistics:"]
        lines.append(f"  Packets sent: {stats['packets_sent']}")
        lines.append(f"  Packets lost: {stats['packets_lost']} ({stats['loss_rate_actual']*100:.2f}%)")
        lines.append(f"  Avg one-way delay: {stats['avg_delay_ms']:.2f}ms")
        lines.append(f"  Total simulated delay: {stats['total_delay_s']:.2f}s")
        return "\n".join(lines)
