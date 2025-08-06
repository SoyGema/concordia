# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**🚨 CRITICAL**: All Python commands MUST use the `evolutionary_env` virtual environment. Always run `source evolutionary_env/bin/activate` before any Python operations.

## Project Overview

Concordia is a library for generative social simulation with agent-based models. This fork specializes in evolutionary simulation research with language model integration.

## Essential Commands

### Environment Setup
```bash
# Create and activate virtual environment (required)
python -m venv evolutionary_env
source evolutionary_env/bin/activate  # Linux/Mac
pip install sentence-transformers torch
```

### Testing & Validation
```bash
# ALWAYS activate environment first
source evolutionary_env/bin/activate

# Run all tests
PYTHONPATH=. pytest --pyargs concordia

# Run specific evolutionary tests
PYTHONPATH=. pytest concordia/testing/test_evolutionary_simulation.py

# Type checking
pytype concordia/

# Linting
pylint --errors-only concordia

# Import validation
PYTHONPATH=. python -c "import concordia; print('Import successful')"
```

### Core Simulation Commands
```bash
# Basic evolutionary simulation (dummy model)
PYTHONPATH=. python examples/evolutionary_simulation.py

# With Gemma model
PYTHONPATH=. python -c "
from examples.evolutionary_simulation import evolutionary_main, GEMMA_CONFIG
measurements = evolutionary_main(config=GEMMA_CONFIG)
"

# With OpenAI (requires OPENAI_API_KEY env var)
PYTHONPATH=. python -c "
import os
from examples.evolutionary_simulation import evolutionary_main, OPENAI_CONFIG
openai_config = OPENAI_CONFIG
openai_config.api_key = os.getenv('OPENAI_API_KEY')
measurements = evolutionary_main(config=openai_config)
"
```

## Architecture Overview

**Key Directories:**
- `concordia/agents/` - Agent implementations
- `concordia/components/` - Modular components for agents and game masters
- `concordia/environment/` - Simulation engines (sequential, simultaneous, parallel)
- `concordia/language_model/` - LLM integrations (OpenAI, Gemma, Mistral, etc.)
- `concordia/prefabs/` - Pre-built templates for entities and simulations
- `concordia/typing/` - Type definitions, especially `evolutionary.py`
- `concordia/utils/` - Utilities including checkpointing and measurement systems
- `examples/evolutionary_simulation.py` - Main evolutionary simulation framework

**Core Simulation Pattern:**
1. **Agents** make decisions via language models answering: "What kind of situation is this?", "What kind of person am I?", "What should I do?"
2. **Game Master** manages environment, resolves actions, tracks measurements
3. **Engine** coordinates multi-agent interactions (sequential vs simultaneous)

## Language Model Configuration

**Configuration Type:** `evolutionary_types.EvolutionConfig`

**Supported Models:**
- `pytorch_gemma` - Local Gemma models (google/gemma-2b-it, google/gemma-7b-it)
- `openai` - GPT models (gpt-4o, gpt-4o-mini, gpt-3.5-turbo)
- `mistral` - Mistral API models
- `amazon_bedrock` - AWS Bedrock models
- `google_aistudio_model` - Google AI Studio models

**Device Options:**
- `cpu` - CPU processing
- `cuda:0` - NVIDIA GPU
- `mps` - Mac GPU (Metal Performance Shaders)

## Evolutionary Simulation Framework

**Location:** `examples/evolutionary_simulation.py`

**Purpose:** Studies cooperation vs selfishness evolution in public goods games through:
- Population-based evolution with selection and mutation
- Multi-round public goods games
- Comprehensive measurement tracking
- Checkpointing system for long simulations

**Key Type Definitions:** `concordia/typing/evolutionary.py`
- `Strategy.COOPERATIVE` vs `Strategy.SELFISH`
- `EvolutionConfig` for simulation parameters
- Selection methods: 'topk', 'probabilistic'

## Common Issues & Automated Fixes

### Import Resolution Issues
The codebase includes automated fixing utilities:

**Upstream Fix Utility:** `concordia/utils/upstream_fixes.py`
```bash
# Apply all upstream import fixes
PYTHONPATH=. python -c "from concordia.utils.upstream_fixes import main; main()"
```

**CI Validation Utility:** `concordia/utils/ci_validation.py`
```bash
# Full validation before push
PYTHONPATH=. python -c "from concordia.utils.ci_validation import main; main()"
```

### Virtual Environment Issues
- **Problem:** `ModuleNotFoundError` or dependency issues
- **Solution:** ALWAYS activate `evolutionary_env` first
- **Check:** `which python` should show evolutionary_env path

## Git Workflow

**Important:** This is a fork. PRs should target `SoyGema/concordia`, not upstream.

```bash
# Create feature branch
git checkout -b feature_name

# Standard development cycle
source evolutionary_env/bin/activate
# Make changes, run tests
PYTHONPATH=. python -m pytest

# Push to fork
git push origin feature_name

# Create PR targeting SoyGema/concordia
gh pr create --repo SoyGema/concordia --base main --head feature_name
```

## Project-Specific Conventions

**File Naming:**
- Test files: `test_*.py` in `concordia/testing/`
- Components: `snake_case.py`
- Classes: `PascalCase`
- Functions: `snake_case`

**Testing Structure:**
- Core tests: `concordia/testing/test_evolutionary_simulation.py`
- Enhanced exports: `concordia/testing/test_enhanced_results_exporter.py`
- Model validation: `concordia/testing/test_gemma_robust.py`

**Build System:**
- Uses setuptools with `pyproject.toml`
- Python 3.11+ required
- Dependencies managed via `setup.py`
- Development tools: pytest, pytype, pylint, pyink

## Performance Optimization

**Model Selection Guidance:**
- **Development/Testing:** Dummy model (`disable_language_model=True`)
- **Research:** Gemma 2B (balanced performance/quality)
- **Production:** GPT-4o or Gemma 7B (highest quality)

**Mac-Specific:** Use `device='mps'` for Metal Performance Shaders GPU acceleration

## CI/CD Integration

**GitHub Actions Workflows:**
- `test-concordia.yml` - Core library testing
- `test-examples.yml` - Examples and notebook testing
- `upstream-sync.yml` - Automated upstream synchronization

**Pre-Push Validation:** Always run local tests before pushing to catch upstream issues early.