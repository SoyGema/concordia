#!/usr/bin/env python3
"""
Hierarchical timeout management for evolutionary simulations.
Provides intelligent timeout handling rather than crude timeout flag increases.
"""

import time
import logging
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass, field
from threading import Event, Thread
import signal

logger = logging.getLogger(__name__)

@dataclass
class TimeoutConfig:
    """Configuration for hierarchical timeouts."""
    inference_timeout: float = 120.0  # 2 minutes per LLM call
    round_timeout: float = 300.0      # 5 minutes per round
    generation_timeout: float = 900.0  # 15 minutes per generation
    simulation_timeout: float = 3600.0 # 1 hour total simulation
    
    # Progress monitoring
    progress_callback: Optional[Callable[[str, float], None]] = None
    user_interrupt_check: Optional[Callable[[], bool]] = None

@dataclass
class TimeoutContext:
    """Context for tracking timeout state."""
    start_time: float = field(default_factory=time.time)
    current_phase: str = "initialization"
    current_generation: int = 0
    total_generations: int = 0
    current_round: int = 0
    total_rounds: int = 0
    
    # Progress tracking
    completed_generations: int = 0
    completed_rounds: int = 0
    completed_inferences: int = 0
    
    # State management
    should_stop: bool = False
    timeout_occurred: bool = False
    interrupted_by_user: bool = False

class TimeoutManager:
    """Hierarchical timeout manager for evolutionary simulations."""
    
    def __init__(self, config: TimeoutConfig):
        self.config = config
        self.context = TimeoutContext()
        self._stop_event = Event()
        
    def start_simulation(self, total_generations: int, total_rounds: int):
        """Initialize simulation-level timeout tracking."""
        self.context.start_time = time.time()
        self.context.total_generations = total_generations
        self.context.total_rounds = total_rounds
        self.context.current_phase = "simulation_started"
        
        logger.info(f"🕒 Timeout Manager initialized:")
        logger.info(f"   Simulation timeout: {self.config.simulation_timeout:.0f}s")
        logger.info(f"   Generation timeout: {self.config.generation_timeout:.0f}s")
        logger.info(f"   Round timeout: {self.config.round_timeout:.0f}s")
        logger.info(f"   Inference timeout: {self.config.inference_timeout:.0f}s")
        
    def start_generation(self, generation: int) -> bool:
        """Start a new generation. Returns False if should stop."""
        if self._should_stop():
            return False
            
        self.context.current_generation = generation
        self.context.current_phase = f"generation_{generation}"
        self.context.current_round = 0
        
        elapsed = time.time() - self.context.start_time
        remaining = self.config.simulation_timeout - elapsed
        
        if remaining <= 0:
            logger.warning(f"⏰ Simulation timeout reached before generation {generation}")
            self.context.timeout_occurred = True
            return False
            
        self._report_progress(f"Starting generation {generation}", 
                            self.context.current_generation / self.context.total_generations)
        return True
        
    def start_round(self, round_num: int) -> bool:
        """Start a new round. Returns False if should stop."""
        if self._should_stop():
            return False
            
        self.context.current_round = round_num
        self.context.current_phase = f"generation_{self.context.current_generation}_round_{round_num}"
        
        # Check generation timeout
        generation_start = time.time() - (self.context.completed_rounds * self.config.round_timeout)
        if time.time() - generation_start > self.config.generation_timeout:
            logger.warning(f"⏰ Generation timeout reached at round {round_num}")
            self.context.timeout_occurred = True
            return False
            
        return True
        
    def start_inference(self, agent_name: str, inference_type: str = "decision") -> bool:
        """Start an LLM inference. Returns False if should stop."""
        if self._should_stop():
            return False
            
        self.context.current_phase = f"inference_{agent_name}_{inference_type}"
        return True
        
    def complete_inference(self, success: bool = True):
        """Mark inference as completed."""
        if success:
            self.context.completed_inferences += 1
            
    def complete_round(self):
        """Mark round as completed."""
        self.context.completed_rounds += 1
        self.context.current_round += 1
        
    def complete_generation(self):
        """Mark generation as completed."""
        self.context.completed_generations += 1
        self._report_progress(f"Completed generation {self.context.current_generation}", 
                            self.context.completed_generations / self.context.total_generations)
        
    def should_save_checkpoint(self) -> bool:
        """Check if we should save a checkpoint due to time constraints."""
        elapsed = time.time() - self.context.start_time
        remaining = self.config.simulation_timeout - elapsed
        
        # Save checkpoint if less than one generation timeout remaining
        return remaining < self.config.generation_timeout
        
    def get_remaining_time(self, level: str = "simulation") -> float:
        """Get remaining time for specified level."""
        elapsed = time.time() - self.context.start_time
        
        if level == "simulation":
            return max(0, self.config.simulation_timeout - elapsed)
        elif level == "generation":
            gen_elapsed = elapsed % self.config.generation_timeout
            return max(0, self.config.generation_timeout - gen_elapsed)
        elif level == "round":
            round_elapsed = elapsed % self.config.round_timeout
            return max(0, self.config.round_timeout - round_elapsed)
        else:
            return self.config.inference_timeout
            
    def get_progress_info(self) -> Dict[str, Any]:
        """Get current progress information."""
        elapsed = time.time() - self.context.start_time
        return {
            'elapsed_time': elapsed,
            'remaining_time': self.get_remaining_time(),
            'current_phase': self.context.current_phase,
            'current_generation': self.context.current_generation,
            'total_generations': self.context.total_generations,
            'completed_generations': self.context.completed_generations,
            'current_round': self.context.current_round,
            'completed_rounds': self.context.completed_rounds,
            'completed_inferences': self.context.completed_inferences,
            'progress_percentage': (self.context.completed_generations / max(1, self.context.total_generations)) * 100,
            'should_stop': self._should_stop(),
            'timeout_occurred': self.context.timeout_occurred,
            'interrupted_by_user': self.context.interrupted_by_user
        }
        
    def request_stop(self, reason: str = "user_request"):
        """Request graceful stop."""
        logger.info(f"🛑 Stop requested: {reason}")
        if reason == "user_interrupt":
            self.context.interrupted_by_user = True
        self.context.should_stop = True
        self._stop_event.set()
        
    def _should_stop(self) -> bool:
        """Check if simulation should stop."""
        if self.context.should_stop:
            return True
            
        # Check user interrupt
        if self.config.user_interrupt_check and self.config.user_interrupt_check():
            self.request_stop("user_interrupt")
            return True
            
        # Check simulation timeout
        elapsed = time.time() - self.context.start_time
        if elapsed > self.config.simulation_timeout:
            logger.warning("⏰ Simulation timeout reached")
            self.context.timeout_occurred = True
            return True
            
        return False
        
    def _report_progress(self, message: str, progress: float):
        """Report progress if callback is configured."""
        if self.config.progress_callback:
            self.config.progress_callback(message, progress)
        else:
            logger.info(f"📊 Progress: {message} ({progress:.1%})")

class ProgressMonitor:
    """Simple progress monitor for console output."""
    
    def __init__(self):
        self.last_update = 0
        self.update_interval = 30  # Update every 30 seconds
        
    def __call__(self, message: str, progress: float):
        """Progress callback function."""
        current_time = time.time()
        if current_time - self.last_update >= self.update_interval:
            print(f"\n📊 {message} ({progress:.1%})")
            print(f"⏰ {time.strftime('%H:%M:%S', time.localtime())}")
            self.last_update = current_time

def create_gemma_timeout_config() -> TimeoutConfig:
    """Create timeout configuration optimized for Gemma models."""
    return TimeoutConfig(
        inference_timeout=180.0,    # 3 minutes per inference (Gemma can be slow)
        round_timeout=600.0,        # 10 minutes per round
        generation_timeout=1800.0,  # 30 minutes per generation  
        simulation_timeout=7200.0,  # 2 hours total simulation
        progress_callback=ProgressMonitor()
    )