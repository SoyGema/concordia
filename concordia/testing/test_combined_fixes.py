#!/usr/bin/env python3
"""
Test script to verify both model reuse and memory cleanup fixes.
This should prevent the Generation 3 hanging issue.
"""

import logging
import sys

# Configure logging to see both model reuse and memory cleanup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from concordia.typing import evolutionary as evolutionary_types
from concordia.utils.logging_evolutionary_simulation import logging_evolutionary_main

# Test configuration - same as the problematic one but shorter
TEST_CONFIG = evolutionary_types.EvolutionConfig(
    pop_size=2,                    # Same as problematic config
    num_generations=4,             # Go to Gen 4 to test the fix (original failed at Gen 3)
    selection_method='topk',       # Same parameters
    top_k=1,
    mutation_rate=0.1,
    num_rounds=3,                  # Slightly fewer rounds for speed
    api_type='pytorch_gemma',
    model_name='google/gemma-7b-it',
    embedder_name='all-mpnet-base-v2',
    device='mps',
    disable_language_model=False,
)

def test_combined_fixes() -> bool:
    """Test both model reuse and memory cleanup fixes."""
    print("🧪 TESTING COMBINED FIXES (Model Reuse + Memory Cleanup)")
    print("=" * 70)
    print(f"Configuration: {TEST_CONFIG.pop_size} agents, {TEST_CONFIG.num_generations} generations")
    print("Expected behavior:")
    print("  Generation 1: 'Creating cached language model' + normal execution")
    print("  Generation 2: 'Reusing cached language model' + 'Memory cleanup complete'")  
    print("  Generation 3: 'Reusing cached language model' + 'Memory cleanup complete'")
    print("  Generation 4: 'Reusing cached language model' + 'Memory cleanup complete'")
    print("🎯 This should NOT hang at Generation 3!")
    print("=" * 70)

    try:
        measurements = logging_evolutionary_main(config=TEST_CONFIG)
        print("\n🎉 SUCCESS: Combined fixes test completed!")
        print("✅ No hanging at Generation 3 - problem solved!")
        return True
        
    except Exception as e:
        print(f"\n❌ FAILED: Combined fixes test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔧 Testing both model reuse and memory cleanup fixes...")
    success = test_combined_fixes()
    sys.exit(0 if success else 1)