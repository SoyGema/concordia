"""Optimized engines for Mac-specific evolutionary simulations."""

from collections.abc import Mapping, Sequence
from typing import Any, Callable
import os

from concordia.environment.engines import simultaneous
from concordia.utils import concurrency


class OptimizedSimultaneous(simultaneous.Simultaneous):
    """Simultaneous engine optimized for Mac with configurable max_workers."""
    
    def __init__(
        self,
        max_workers: int | None = None,
        **kwargs
    ):
        """Initialize with Mac-optimized defaults.
        
        Args:
            max_workers: Maximum number of worker threads. If None, uses
                min(8, os.cpu_count()) for Mac optimization.
            **kwargs: Additional arguments passed to parent class.
        """
        super().__init__(**kwargs)
        
        if max_workers is None:
            # Mac-optimized default: balance between parallelism and resource usage
            self._max_workers = min(8, os.cpu_count() or 4)
        else:
            self._max_workers = max_workers
    
    def run_loop(
        self,
        game_masters: Sequence,
        entities: Sequence,
        premise: str = '',
        max_steps: int = 100,
        verbose: bool = False,
        log: list[Mapping[str, Any]] | None = None,
        checkpoint_callback: Callable[[int], None] | None = None,
    ):
        """Run the game loop with optimized concurrency for entity actions."""
        # Store original run_tasks function
        original_run_tasks = concurrency.run_tasks
        
        # Define optimized version that uses our max_workers
        def optimized_run_tasks(tasks, **kwargs):
            # Only set max_workers if not already specified
            kwargs.setdefault('max_workers', self._max_workers)
            return original_run_tasks(tasks, **kwargs)
        
        try:
            # Temporarily patch concurrency for this run
            concurrency.run_tasks = optimized_run_tasks
            
            # Call parent's run_loop method
            return super().run_loop(
                game_masters=game_masters,
                entities=entities,
                premise=premise,
                max_steps=max_steps,
                verbose=verbose,
                log=log,
                checkpoint_callback=checkpoint_callback,
            )
        finally:
            # Always restore original function
            concurrency.run_tasks = original_run_tasks


def create_optimized_engine(pop_size: int, engine_type: str = 'simultaneous'):
    """Create an optimized engine for evolutionary simulations.
    
    Args:
        pop_size: Population size to optimize for
        engine_type: Type of engine ('simultaneous' or 'parallel')
        
    Returns:
        Optimized engine instance
    """
    if engine_type == 'simultaneous':
        # Optimize max_workers based on population size and Mac capabilities
        max_workers = min(8, pop_size, os.cpu_count() or 4)
        return OptimizedSimultaneous(max_workers=max_workers)
    else:
        raise ValueError(f"Unsupported engine type: {engine_type}")