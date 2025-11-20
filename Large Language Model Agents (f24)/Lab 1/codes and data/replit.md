# Agentic AI - LLM Agent Labs

## Overview
This repository contains a collection of educational labs focused on Large Language Model (LLM) agents, multi-agent systems, and LLM security. The project is designed for learning and experimentation with agentic AI workflows, inspired by the Berkeley RDI LLM Agents Hackathon and AgentX/AgentBeats competitions.

## Project Structure

### Lab 1: Restaurant Review Analysis with AutoGen
**Location:** `Large Language Model Agents (f24)/Lab 1/`

A complete implementation of a multi-agent restaurant review analysis system using the **AutoGen framework** with Groq API for LLM inference. This implementation follows the lab assignment specifications using a sequential multi-agent architecture.

**Multi-Agent Architecture:**
1. **Entrypoint Agent** - Extracts restaurant name from natural language queries using LLM
2. **Data Fetch** - Retrieves reviews from local dataset based on extracted restaurant name
3. **Review Analysis Agent** - Uses LLM to analyze reviews and extract food/service scores (1-5) based on keyword mapping
4. **Scoring Calculation** - Computes overall score using formula: Σ(sqrt(food² × service)) / (N × sqrt(125)) × 10

The system:
- Uses AutoGen's `ConversableAgent` for multi-agent orchestration
- Leverages Groq's Llama 3.3 70B model for fast, cost-effective LLM inference
- Implements sequential chat pattern for agent communication
- Includes fallback to deterministic keyword matching if LLM analysis fails
- Analyzes unstructured reviews to extract food and service scores (1-5)

**Key Files:**
- `main.py` - AutoGen multi-agent implementation
- `test.py` - Public test suite (validates AutoGen pipeline)
- `restaurant-data.txt` - Restaurant reviews dataset
- `requirements.txt` - Python dependencies (AutoGen, OpenAI client, etc.)
- `Instructions.md` - Original lab assignment instructions

**How to Run:**
```bash
cd "Large Language Model Agents (f24)/Lab 1"
python test.py  # Run test suite (4/4 tests passing)
python main.py "How good is In N Out?"  # Query a restaurant
```

**Test Results:** All 4 public tests passing ✓

### Lab 2: LLM Security - Attack Prompts
**Location:** `Large Language Model Agents (f24)/Lab 2/lab02_release/`

Focus on writing attack prompts to extract secret keys from LLM system messages. Explores prompt injection, jailbreaking, and other security vulnerabilities.

**Files:**
- `attack-1.txt` - First attack prompt (to be implemented)
- `attack-2.txt` - Second attack prompt (to be implemented)
- `Instructions.md` - Full lab instructions

### Lab 3: LLM Security - Defense Prompts
**Location:** `Large Language Model Agents (f24)/Lab 3/lab03_release/`

Focus on creating robust defense prompts to prevent secret key leakage and resist adversarial attacks.

**Files:**
- `defense.txt` - Defense prompt (to be implemented)
- `Instructions.md` - Full lab instructions

## Recent Changes
- **2025-11-20**: Rebuilt Lab 1 with AutoGen multi-agent system
  - Installed Python 3.11 and all required dependencies (AutoGen, etc.)
  - Rebuilt main.py from scratch using AutoGen framework with ConversableAgent
  - Configured Groq API (Llama 3.3 70B) as LLM provider for cost-effective inference
  - Implemented sequential multi-agent architecture as specified in lab instructions:
    - Entrypoint Agent for restaurant name extraction
    - Data fetch with fuzzy matching
    - Review Analysis Agent with LLM-powered keyword detection
    - Scoring calculation with mathematical formula
  - Fixed file paths to work with Replit directory structure
  - Configured workflow to run Lab 1 test suite
  - Created Python .gitignore
  - All Lab 1 tests passing (4/4) ✓

## Environment Setup

### Python Version
Python 3.11

### Dependencies
All dependencies are managed via `requirements.txt` in Lab 1. Key packages used:
- `autogen` (v0.3.0) - Multi-agent orchestration framework
- `openai` (v1.44.1) - OpenAI-compatible API client for Groq
- `httpx` - HTTP client for API requests
- Standard library packages for data processing

### Environment Variables
**Required:**
- `GROQ_API_KEY` - API key for Groq LLM inference (configured in Replit Secrets)

**Optional (for Labs 2 & 3):**
- `OPENAI_API_KEY` - OpenAI API key for security labs

## Architecture

### Lab 1 Implementation (AutoGen Multi-Agent System)

The implementation follows the lab specifications using AutoGen's sequential chat pattern:

**Agent Flow:**
1. **Entrypoint Agent** (`ConversableAgent`)
   - Receives user query and extracts restaurant name using LLM (Groq Llama 3.3 70B)
   - Coordinates workflow by initiating sequential chats with other agents
   
2. **Data Fetch** (Programmatic with Agent Coordination)
   - Uses entrypoint agent's extracted restaurant name
   - Performs fuzzy matching against restaurant database
   - Returns canonical name and all reviews for the restaurant

3. **Review Analysis Agent** (`ConversableAgent`)
   - Receives reviews from previous step (via AutoGen's summary mechanism)
   - Uses LLM to identify keyword adjectives in each review
   - Maps keywords to scores using the lab's specified scoring rules:
     - Score 1: awful, horrible, disgusting
     - Score 2: bad, unpleasant, offensive
     - Score 3: average, uninspiring, forgettable
     - Score 4: good, enjoyable, satisfying
     - Score 5: awesome, incredible, amazing
   - Extracts first keyword as food_score, second keyword as service_score
   - Falls back to deterministic keyword matching if LLM parsing fails

4. **Scoring Calculation** (Programmatic)
   - Takes extracted scores from review analysis
   - Calculates overall score: Σ(sqrt(food² × service)) / (N × sqrt(125)) × 10
   - Returns final restaurant score (0-10 scale)

**Key Design Decisions:**
- Uses Groq API (OpenAI-compatible) instead of OpenAI for cost efficiency
- Implements `ConversableAgent` from AutoGen for agent orchestration
- Uses `initiate_chat` for sequential agent communication
- Includes robust fallback mechanisms for reliability
- Leverages AutoGen's summary passing between agents for context transfer

## User Preferences
None specified yet.

## Testing
Lab 1 includes a comprehensive test suite that validates:
- Restaurant name extraction from queries
- Review analysis accuracy
- Overall score calculations
- Case-insensitive restaurant matching

Run tests: `python test.py` from Lab 1 directory

## Notes
- This is an educational project focused on learning LLM agent architectures
- Labs 2 and 3 contain starter code but require implementation
- The agentic pipeline in Lab 1 is fully functional but optional
- All paths have been updated to work with Replit's directory structure
