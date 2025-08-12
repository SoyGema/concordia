"""Safe engines that avoid threading conflicts entirely."""

from collections.abc import Mapping, Sequence
from typing import Any, Callable

from concordia.environment.engines import sequential
from concordia.environment.engines import simultaneous
from concordia.utils import concurrency


class SafeSequential(sequential.Sequential):
    """Sequential engine that processes agents one by one - no threading conflicts."""
    pass


class SafeSimultaneous(simultaneous.Simultaneous):
    """Simultaneous engine that disables nested concurrency to prevent conflicts."""
    
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
        """Run the game loop with concurrency disabled to prevent threading conflicts."""
        # Store original functions
        original_run_tasks = concurrency.run_tasks
        original_map_parallel = concurrency.map_parallel
        
        # Define safe versions that use sequential processing
        def safe_run_tasks(tasks, **kwargs):
            """Run tasks sequentially instead of in parallel to avoid threading conflicts."""
            results = {}
            for key, task in tasks.items():
                try:
                    results[key] = task()
                except Exception as e:
                    print(f"Error in task {key}: {e}")
                    raise
            return results
        
        def safe_map_parallel(fn, *args, **kwargs):
            """Map function sequentially instead of in parallel."""
            return [fn(*arg) for arg in zip(*args, strict=True)]
        
        try:
            # Replace concurrency functions with safe versions
            concurrency.run_tasks = safe_run_tasks
            concurrency.map_parallel = safe_map_parallel
            
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
            # Always restore original functions
            concurrency.run_tasks = original_run_tasks
            concurrency.map_parallel = original_map_parallel


def create_safe_engine(engine_type: str = 'sequential'):
    """Create a safe engine that avoids threading conflicts.
    
    Args:
        engine_type: 'sequential' or 'safe_simultaneous'
        
    Returns:
        Safe engine instance
    """
    if engine_type == 'sequential':
        return SafeSequential()
    elif engine_type == 'safe_simultaneous':
        return SafeSimultaneous()
    else:
        raise ValueError(f"Unsupported engine type: {engine_type}")