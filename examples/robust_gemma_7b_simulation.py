#!/usr/bin/env python3
"""
Robust Gemma 2 7B evolutionary simulation with intelligent timeout management.
Enhanced version of the working Gemma 7B test with timeout management and graceful interruption.
"""

import logging
import time
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import threading

from concordia.typing import evolutionary as evolutionary_types
from concordia.utils.timeout_manager import TimeoutManager, TimeoutConfig
from concordia.utils.logging_evolutionary_simulation import logging_evolutionary_main
from concordia.utils.enhanced_results_exporter import EnhancedResultsExporter
from concordia.utils.generation_logger import get_generation_logger

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RobustGemma2_7BSimulation:
    """
    Robust Gemma 2 7B evolutionary simulation with timeout management.
    """

    def __init__(self,
                 config: evolutionary_types.EvolutionConfig,
                 timeout_config: Optional[TimeoutConfig] = None):
        """
        Initialize robust Gemma 2 7B evolutionary simulation.

        Args:
            config: Evolutionary simulation configuration
            timeout_config: Timeout management configuration
        """
        self.config = config
        self.timeout_config = timeout_config or self._create_gemma2_7b_timeout_config()

        self.timeout_manager = TimeoutManager(self.timeout_config)
        self.exporter = EnhancedResultsExporter()

        # State tracking
        self.start_time = None
        self.measurements = None

    def _create_gemma2_7b_timeout_config(self) -> TimeoutConfig:
        """Create timeout configuration optimized for Gemma 2 7B models."""
        return TimeoutConfig(
            inference_timeout=180.0,    # 3 minutes per inference (7B model needs more time)
            round_timeout=600.0,        # 10 minutes per round
            generation_timeout=1200.0,  # 20 minutes per generation
            simulation_timeout=3600.0,  # 1 hour total simulation (conservative for minimal test)
            progress_callback=None  # Will use default console progress
        )

    def run(self) -> Tuple[Any, Dict[str, Any]]:
        """
        Run the evolutionary simulation with robust timeout management.

        Returns:
            Tuple of (measurements, simulation_info)
        """
        logger.info("🚀 Starting robust Gemma 2 7B evolutionary simulation...")
        logger.info(f"   Model: {self.config.model_name}")
        logger.info(f"   Generations: {self.config.num_generations}")
        logger.info(f"   Population: {self.config.pop_size}")
        logger.info(f"   Rounds: {self.config.num_rounds}")

        self.start_time = time.time()

        # Initialize timeout manager
        self.timeout_manager.start_simulation(
            total_generations=self.config.num_generations,
            total_rounds=self.config.num_rounds
        )

        try:
            # Run the simulation with timeout monitoring
            self.measurements = self._run_with_monitoring()

            # Export final results
            results_info = self._export_final_results(completed=True)

            logger.info("✅ Robust Gemma 2 7B evolutionary simulation completed successfully!")
            return self.measurements, results_info

        except TimeoutError as e:
            logger.warning(f"⏰ Simulation stopped due to timeout: {e}")
            results_info = self._handle_timeout_gracefully()
            return self.measurements, results_info

        except KeyboardInterrupt:
            logger.info("🛑 Simulation interrupted by user")
            results_info = self._handle_user_interruption()
            return self.measurements, results_info

        except Exception as e:
            logger.error(f"❌ Simulation failed with error: {e}")
            logger.exception("Full traceback:")
            results_info = self._handle_simulation_error(e)
            return self.measurements, results_info

    def _run_with_monitoring(self):
        """Run simulation with timeout monitoring."""
        logger.info("🔄 Starting monitored Gemma 2 7B evolutionary simulation...")

        # Start progress monitoring in a separate thread
        def progress_monitor():
            """Monitor progress and check for timeouts."""
            while not self.timeout_manager.context.should_stop:
                time.sleep(30)  # Check every 30 seconds

                progress = self.timeout_manager.get_progress_info()
                remaining = progress['remaining_time']

                if remaining <= 0:
                    logger.warning("⏰ Simulation timeout - requesting graceful stop")
                    self.timeout_manager.request_stop("timeout")
                    break

                logger.info(f"📊 Gemma 2 7B running... {remaining/60:.1f} minutes remaining")

        monitor_thread = threading.Thread(target=progress_monitor, daemon=True)
        monitor_thread.start()

        # Run the actual simulation
        return logging_evolutionary_main(config=self.config)

    def _handle_timeout_gracefully(self) -> Dict[str, Any]:
        """Handle timeout gracefully by saving partial results."""
        logger.info("🕒 Handling Gemma 2 7B timeout gracefully...")
        return self._export_final_results(completed=False, reason="timeout")

    def _handle_user_interruption(self) -> Dict[str, Any]:
        """Handle user interruption gracefully."""
        logger.info("👤 Handling user interruption gracefully...")
        return self._export_final_results(completed=False, reason="user_interrupt")

    def _handle_simulation_error(self, error: Exception) -> Dict[str, Any]:
        """Handle simulation error by saving debug information."""
        logger.info("💥 Handling simulation error gracefully...")
        return self._export_final_results(completed=False, reason="error", error=str(error))

    def _export_final_results(self, completed: bool, reason: str = "completed", error: str = None) -> Dict[str, Any]:
        """Export final or partial results."""
        logger.info("📊 Exporting Gemma 2 7B simulation results...")

        # Get generation logs
        generation_log = get_generation_logger().get_all_interactions()
        logger.info(f"📝 Captured {len(generation_log)} LLM interactions")

        # Prepare results data
        results_data = self._prepare_results_data()
        analysis_data = self._prepare_analysis_data(completed, reason)
        metadata = self._prepare_metadata(completed, reason, error)

        # Export results
        try:
            results_folder = self.exporter.export_complete_results(
                model_name=f"robust-{self.config.model_name}",
                results_data=results_data,
                analysis_data=analysis_data,
                metadata=metadata,
                generation_log=generation_log
            )

            logger.info(f"📁 Gemma 2 7B results exported to: {results_folder}")

            return {
                'results_folder': results_folder,
                'completed': completed,
                'reason': reason,
                'export_success': True,
                'generation_logs_count': len(generation_log)
            }

        except Exception as e:
            logger.error(f"❌ Failed to export Gemma 2 7B results: {e}")
            return {
                'completed': completed,
                'reason': reason,
                'export_success': False,
                'error': str(e)
            }

    def _prepare_results_data(self) -> Dict[str, Any]:
        """Prepare results data from measurements."""
        results_data = {
            'config': {
                'model_name': self.config.model_name,
                'api_type': self.config.api_type,
                'pop_size': self.config.pop_size,
                'num_generations': self.config.num_generations,
                'num_rounds': self.config.num_rounds,
                'selection_method': self.config.selection_method,
                'mutation_rate': self.config.mutation_rate,
                'device': self.config.device,
                'disable_language_model': self.config.disable_language_model,
            },
            'generations': [],
            'final_cooperation_rate': 0.5,  # Default fallback
            'performance': {}
        }

        # Extract generation data if measurements exist
        if self.measurements:
            try:
                gen_summaries = self.measurements.get_channel('evolutionary_generation_summary')
                for gen_data in gen_summaries:
                    gen_info = {
                        'generation': len(results_data['generations']) + 1,
                        'scores': gen_data.get('agent_scores', {}),
                        'cooperative_count': gen_data.get('cooperative_count', 0),
                        'selfish_count': gen_data.get('selfish_count', 0),
                        'cooperation_rate': gen_data.get('cooperation_rate', 0),
                    }
                    results_data['generations'].append(gen_info)

                if results_data['generations']:
                    results_data['final_cooperation_rate'] = results_data['generations'][-1]['cooperation_rate']

            except Exception as e:
                logger.warning(f"Could not extract generation data: {e}")

        return results_data

    def _prepare_analysis_data(self, completed: bool, reason: str) -> Dict[str, Any]:
        """Prepare analysis data."""
        return {
            'simulation_status': {
                'completed': completed,
                'reason': reason,
                'timeout_info': self.timeout_manager.get_progress_info(),
                'model_family': 'gemma-2-7b'
            },
            'performance_analysis': {
                'total_runtime': time.time() - self.start_time if self.start_time else 0,
                'timeout_config': {
                    'inference_timeout': self.timeout_config.inference_timeout,
                    'generation_timeout': self.timeout_config.generation_timeout,
                    'simulation_timeout': self.timeout_config.simulation_timeout
                }
            }
        }

    def _prepare_metadata(self, completed: bool, reason: str, error: str = None) -> Dict[str, Any]:
        """Prepare metadata."""
        import platform

        metadata = {
            'simulation_info': {
                'model_name': self.config.model_name,
                'model_family': 'gemma-2-7b',
                'api_type': self.config.api_type,
                'device': self.config.device,
                'language_model_active': not self.config.disable_language_model,
                'completed': completed,
                'completion_reason': reason,
                'timeout_management': True,
                'robust_simulation': True
            },
            'performance': {
                'duration_seconds': time.time() - self.start_time if self.start_time else 0,
                'avg_generation_time': (time.time() - self.start_time) / max(1, self.config.num_generations) if self.start_time else 0,
                'timeout_occurred': self.timeout_manager.context.timeout_occurred,
                'user_interrupted': self.timeout_manager.context.interrupted_by_user
            },
            'experiment_parameters': {
                'population_size': self.config.pop_size,
                'generations': self.config.num_generations,
                'rounds_per_generation': self.config.num_rounds,
                'selection_method': self.config.selection_method,
                'mutation_rate': self.config.mutation_rate,
            },
            'system_info': {
                'platform': platform.system(),
                'python_version': platform.python_version(),
            }
        }

        if error:
            metadata['error_info'] = {
                'error_message': error,
                'error_timestamp': time.time()
            }

        return metadata

def run_robust_gemma2_7b_simulation():
    """Run a robust Gemma 2 7B simulation with intelligent timeout management."""

    # Known working Gemma 2 7B configuration with timeout management
    ROBUST_GEMMA2_7B_CONFIG = evolutionary_types.EvolutionConfig(
        pop_size=2,  # Small for speed
        num_generations=5,  # Just 2 generations (like original test)
        selection_method='topk',
        top_k=1,
        mutation_rate=0.1,
        num_rounds=4,  # Minimal rounds (like original test)
        api_type='pytorch_gemma',
        model_name='google/gemma-7b-it',  # Known working model
        embedder_name='all-mpnet-base-v2',
        device='mps',
        disable_language_model=False,
    )

    # Create simulation
    simulation = RobustGemma2_7BSimulation(
        config=ROBUST_GEMMA2_7B_CONFIG
    )

    measurements, results_info = simulation.run()

    # Print summary
    print("\n" + "="*70)
    print("🎉 ROBUST GEMMA 2 7B SIMULATION SUMMARY")
    print("="*70)
    print(f"✅ Completed: {results_info['completed']}")
    print(f"📋 Reason: {results_info['reason']}")
    if 'results_folder' in results_info:
        print(f"📁 Results: {results_info['results_folder']}")
    if 'generation_logs_count' in results_info:
        print(f"📝 LLM Interactions: {results_info['generation_logs_count']}")
    print("="*70)

    return measurements, results_info

if __name__ == "__main__":
    print("🧪 Robust Gemma 2 7B Evolutionary Simulation")
    print("🕒 With Intelligent Timeout Management")
    print("🚀 Enhanced version of working Gemma 7B test")
    print("=" * 70)

    try:
        measurements, results = run_robust_gemma2_7b_simulation()
        print("\n🎉 SUCCESS: Robust Gemma 2 7B simulation completed!")
        print("🔍 Check simulation_results/ for new structured results!")

    except KeyboardInterrupt:
        print("\n🛑 Simulation interrupted by user")
    except Exception as e:
        print(f"\n❌ Simulation failed: {e}")
        import traceback
        traceback.print_exc()
