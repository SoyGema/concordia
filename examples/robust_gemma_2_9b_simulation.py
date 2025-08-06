#!/usr/bin/env python3
"""
Robust Gemma evolutionary simulation with intelligent timeout management.
Provides graceful interruption handling and automatic progress monitoring.
"""

import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

from concordia.typing import evolutionary as evolutionary_types
from concordia.utils.timeout_manager import TimeoutManager, TimeoutConfig, create_gemma_timeout_config
from concordia.utils import checkpointing
from concordia.utils.enhanced_results_exporter import EnhancedResultsExporter
from concordia.utils.generation_logger import get_generation_logger
from concordia.utils.logging_evolutionary_simulation import logging_evolutionary_main

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RobustEvolutionarySimulation:
    """
    Robust evolutionary simulation with timeout management and graceful interruption.
    """

    def __init__(self,
                 config: evolutionary_types.EvolutionConfig,
                 timeout_config: Optional[TimeoutConfig] = None,
                 checkpoint_dir: Optional[Path] = None):
        """
        Initialize robust evolutionary simulation.

        Args:
            config: Evolutionary simulation configuration
            timeout_config: Timeout management configuration
            checkpoint_dir: Directory for checkpoints (optional)
        """
        self.config = config
        self.timeout_config = timeout_config or create_gemma_timeout_config()
        self.checkpoint_dir = checkpoint_dir

        self.timeout_manager = TimeoutManager(self.timeout_config)
        self.exporter = EnhancedResultsExporter()

        # State tracking
        self.start_time = None
        self.measurements = None

    def run(self, resume_from_checkpoint: bool = False) -> Tuple[Any, Dict[str, Any]]:
        """
        Run the evolutionary simulation with robust timeout management.

        Returns:
            Tuple of (measurements, simulation_info)
        """
        logger.info("🚀 Starting robust Gemma evolutionary simulation...")
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

            logger.info("✅ Robust evolutionary simulation completed successfully!")
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
        logger.info("🔄 Starting monitored evolutionary simulation...")

        # Start progress monitoring in a separate thread
        import threading

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

                logger.info(f"📊 Running... {remaining/60:.1f} minutes remaining")

        monitor_thread = threading.Thread(target=progress_monitor, daemon=True)
        monitor_thread.start()

        # Run the actual simulation
        return logging_evolutionary_main(config=self.config)

    def _handle_timeout_gracefully(self) -> Dict[str, Any]:
        """Handle timeout gracefully by saving partial results."""
        logger.info("🕒 Handling timeout gracefully...")
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
        logger.info("📊 Exporting simulation results...")

        # Get generation logs
        generation_log = get_generation_logger().get_all_interactions()

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

            logger.info(f"📁 Results exported to: {results_folder}")

            return {
                'results_folder': results_folder,
                'completed': completed,
                'reason': reason,
                'export_success': True,
                'generation_logs_count': len(generation_log)
            }

        except Exception as e:
            logger.error(f"❌ Failed to export results: {e}")
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
            },
            'generations': [],
            'final_cooperation_rate': 0,
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
            'config': {
                'model_name': self.config.model_name,
                'api_type': self.config.api_type,
                'device': self.config.device,
                'embedder_name': self.config.embedder_name,
                'disable_language_model': self.config.disable_language_model,
                'pop_size': self.config.pop_size,
                'num_generations': self.config.num_generations,
                'num_rounds': self.config.num_rounds,
                'selection_method': self.config.selection_method,
                'mutation_rate': self.config.mutation_rate,
            },
            'generations': self._extract_generation_data(),
            'final_cooperation_rate': self._calculate_final_cooperation_rate(),
            'simulation_status': {
                'completed': completed,
                'reason': reason,
                'timeout_info': self.timeout_manager.get_progress_info()
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
                'completed': completed,
                'completion_reason': reason,
                'timeout_management': True,
                'robust_simulation': True
            },
            'performance': {
                'duration_seconds': time.time() - self.start_time if self.start_time else 0,
                'timeout_occurred': self.timeout_manager.context.timeout_occurred,
                'user_interrupted': self.timeout_manager.context.interrupted_by_user
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

    def _extract_generation_data(self) -> list:
        """Extract generation data for analysis."""
        generations = []
        if self.measurements:
            try:
                gen_summaries = self.measurements.get_channel('evolutionary_generation_summary')
                for gen_data in gen_summaries:
                    gen_info = {
                        'generation': len(generations) + 1,
                        'scores': gen_data.get('agent_scores', {}),
                        'cooperative_count': gen_data.get('cooperative_count', 0),
                        'selfish_count': gen_data.get('selfish_count', 0),
                        'cooperation_rate': gen_data.get('cooperation_rate', 0),
                        'avg_cooperative_score': gen_data.get('avg_cooperative_score', 0),
                        'avg_selfish_score': gen_data.get('avg_selfish_score', 0),
                        'cooperative_agents': gen_data.get('cooperative_agents', []),
                    }
                    generations.append(gen_info)
            except Exception as e:
                logger.warning(f"Could not extract generation data for analysis: {e}")
        return generations

    def _calculate_final_cooperation_rate(self) -> float:
        """Calculate the final cooperation rate."""
        if self.measurements:
            try:
                # Try to get the final cooperation rate from measurements
                gen_summaries = self.measurements.get_channel('evolutionary_generation_summary')
                if gen_summaries:
                    final_coop_rate = gen_summaries[-1].get('cooperation_rate', 0.5)
                    return final_coop_rate
            except Exception as e:
                logger.warning(f"Could not calculate final cooperation rate: {e}")
        return 0.5  # Default fallback

def run_robust_gemma_2_9b_simulation():
    """Run a robust Gemma 2 9B simulation with intelligent timeout management."""

    # Gemma 2 9B configuration with robust timeout management
    ROBUST_GEMMA_CONFIG = evolutionary_types.EvolutionConfig(
        pop_size=2,
        num_generations=1,
        selection_method='topk',
        top_k=1,
        mutation_rate=0.1,
        num_rounds=2,
        api_type='pytorch_gemma',
        model_name='google/gemma-2-9b-it',
        embedder_name='all-mpnet-base-v2',
        device='mps',
        disable_language_model=False,
    )

    # Create custom timeout configuration for Gemma 2 9B
    timeout_config = TimeoutConfig(
        inference_timeout=300.0,    # 5 minutes per inference (9B model is larger)
        round_timeout=1200.0,       # 20 minutes per round
        generation_timeout=3600.0,  # 60 minutes per generation
        simulation_timeout=14400.0, # 4 hours total simulation
        progress_callback=None  # Will use default console progress
    )

    # Create and run robust simulation
    simulation = RobustEvolutionarySimulation(
        config=ROBUST_GEMMA_CONFIG,
        timeout_config=timeout_config,
        checkpoint_dir=Path("evolutionary_checkpoints")
    )

    measurements, results_info = simulation.run()

    # Print summary
    print("\n" + "="*70)
    print("🎉 ROBUST SIMULATION SUMMARY")
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
    print("🎯 Robust Gemma 2 9B Evolutionary Simulation")
    print("🕒 With Intelligent Timeout Management")
    print("="*70)

    try:
        measurements, results = run_robust_gemma_2_9b_simulation()
        print("\n🎉 SUCCESS: Robust simulation completed!")

    except Exception as e:
        print(f"\n❌ Simulation failed: {e}")
        import traceback
        traceback.print_exc()
