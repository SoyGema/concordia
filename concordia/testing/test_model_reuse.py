#!/usr/bin/env python3
"""
Test script to verify model reuse fix for evolutionary simulation.
This should show "Reusing cached language model" after the first generation.
"""

import logging
import sys
import os

# Configure logging to see cache messages
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from concordia.typing import evolutionary as evolutionary_types
from concordia.utils.logging_evolutionary_simulation import logging_evolutionary_main

# Small test configuration - only 3 generations, small population
TEST_CONFIG = evolutionary_types.EvolutionConfig(
    pop_size=2,                    # Minimal population
    num_generations=3,             # Just enough to test caching (Gen 1: create, Gen 2-3: reuse)
    selection_method='topk',
    top_k=1,
    mutation_rate=0.1,
    num_rounds=2,                  # Minimal rounds
    api_type='pytorch_gemma',
    model_name='google/gemma-7b-it',
    embedder_name='all-mpnet-base-v2',
    device='mps',
    disable_language_model=False,
)

def test_model_reuse():
    """Test the model reuse functionality."""
    print("🧪 TESTING MODEL REUSE")
    print("=" * 50)
    print(f"Configuration: {TEST_CONFIG.pop_size} agents, {TEST_CONFIG.num_generations} generations")
    print("Expected behavior:")
    print("  Generation 1: 'Creating cached language model'")
    print("  Generation 2: 'Reusing cached language model' ← This is the fix!")
    print("  Generation 3: 'Reusing cached language model'")
    print("=" * 50)

    try:
        measurements = logging_evolutionary_main(config=TEST_CONFIG)
        print("\n✅ SUCCESS: Model reuse test completed!")
        print("🎯 Check the logs above - you should see model reuse messages")
        return True

    except Exception as e:
        print(f"\n❌ FAILED: Model reuse test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔧 Testing model reuse fix...")
    success = test_model_reuse()
    sys.exit(0 if success else 1)
