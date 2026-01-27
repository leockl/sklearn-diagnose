# AGENTS.md

> Documentation for AI agents used in sklearn-diagnose

## Overview

sklearn-diagnose uses a **multi-agent architecture** powered by [LangChain](https://langchain.com/) to diagnose machine learning model failures. The system employs four specialized AI agents: three for batch diagnosis and one for interactive conversation.

```
┌─────────────────────────────────────────────────────────────────────┐
│                     sklearn-diagnose Agent System                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   Signal Extraction (Deterministic)                                  │
│   ─────────────────────────────────                                  │
│   • Train/validation scores                                          │
│   • Cross-validation metrics                                         │
│   • Feature correlations                                             │
│   • Class distributions                                              │
│                         │                                            │
│                         ▼                                            │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │                   HYPOTHESIS AGENT                           │   │
│   │   Analyzes signals → Detects failure modes                   │   │
│   │   Output: List of hypotheses with confidence & evidence      │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                         │                                            │
│                         ▼                                            │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │                 RECOMMENDATION AGENT                         │   │
│   │   Analyzes hypotheses → Generates actionable fixes           │   │
│   │   Output: Prioritized list of recommendations                │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                         │                                            │
│                         ▼                                            │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │                    SUMMARY AGENT                             │   │
│   │   Synthesizes findings → Creates human-readable report       │   │
│   │   Output: Markdown-formatted diagnostic summary              │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                         │                                            │
│                         ▼                                            │
│                  DiagnosisReport                                     │
│                         │                                            │
│                         ▼                                            │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │                     CHAT AGENT (Interactive)                 │   │
│   │   User conversations → Context-aware Q&A                     │   │
│   │   Output: Interactive responses with conversation history    │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Agent Specifications

### 1. Hypothesis Agent

**Purpose**: Analyze model performance signals and identify potential failure modes.

**Role**: Expert ML diagnostician that detects issues like overfitting, underfitting, data leakage, class imbalance, and more.

| Property | Value |
|----------|-------|
| **Location** | `sklearn_diagnose/llm/client.py` |
| **Method** | `LangChainClient.generate_hypotheses()` |
| **System Prompt** | `HYPOTHESIS_SYSTEM_PROMPT` |
| **Input** | Extracted signals (dict), Task type (str) |
| **Output** | List of `Hypothesis` objects |

#### System Prompt

```
You are an expert ML diagnostician agent. Your task is to analyze model 
performance signals and identify potential failure modes.

You will be given:
1. Performance metrics (train score, validation score, CV scores, etc.)
2. A list of possible failure modes to consider

For each failure mode you detect, you must provide:
- failure_mode: The name of the failure mode (must be one of the provided options)
- confidence: A score between 0.0 and 1.0 indicating how confident you are (0.95 max)
- severity: "low", "medium", or "high"
- evidence: A list of specific observations that support this hypothesis

Guidelines:
- Only report failure modes with confidence >= 0.25
- Be conservative - don't over-diagnose
- Base your assessment solely on the provided signals
- Provide specific, quantitative evidence when possible
- A model can have multiple failure modes simultaneously
```

#### Output Schema

```json
{
  "hypotheses": [
    {
      "failure_mode": "overfitting",
      "confidence": 0.85,
      "severity": "high",
      "evidence": [
        "Train-val gap of 25% indicates memorization",
        "Perfect training score (100%) with 75% validation"
      ]
    }
  ]
}
```

#### Detectable Failure Modes

| Failure Mode | Description |
|--------------|-------------|
| `overfitting` | Model memorizes training data, poor generalization |
| `underfitting` | Model too simple, poor performance on both train/val |
| `high_variance` | Unstable performance across data splits |
| `class_imbalance` | Skewed class distribution affecting predictions |
| `feature_redundancy` | Highly correlated or duplicate features |
| `label_noise` | Incorrect or noisy target labels |
| `data_leakage` | Information from validation leaking into training |

---

### 2. Recommendation Agent

**Purpose**: Generate actionable recommendations to fix detected issues.

**Role**: Expert ML engineer that provides specific, prioritized solutions based on detected failure modes.

| Property | Value |
|----------|-------|
| **Location** | `sklearn_diagnose/llm/client.py` |
| **Method** | `LangChainClient.generate_recommendations()` |
| **System Prompt** | `RECOMMENDATION_SYSTEM_PROMPT` |
| **Input** | List of hypotheses, Example recommendations (dict) |
| **Output** | List of `Recommendation` objects |

#### System Prompt

```
You are an expert ML engineer agent helping users fix model issues.

You will be given:
1. Detected failure modes (hypotheses) with their confidence and severity
2. Example recommendations for each failure mode (these are suggestions, not exhaustive)

Your task is to generate the most impactful recommendations to address the 
detected issues.

Guidelines:
- Focus on the highest confidence and severity issues first
- Consider root causes vs symptoms (fixing root cause may resolve multiple issues)
- Recommendations should be specific and actionable
- You can use the example recommendations as guidance, but feel free to suggest others
- Avoid redundant recommendations
- Order recommendations from most to least impactful
```

#### Output Schema

```json
{
  "recommendations": [
    {
      "action": "Increase regularization strength (e.g., increase C parameter)",
      "rationale": "Stronger regularization penalizes model complexity and reduces overfitting",
      "related_failure_mode": "overfitting"
    }
  ]
}
```

---

### 3. Summary Agent

**Purpose**: Create a human-readable diagnostic report synthesizing all findings.

**Role**: Expert ML diagnostician that communicates complex technical findings in clear, actionable language.

| Property | Value |
|----------|-------|
| **Location** | `sklearn_diagnose/llm/client.py` |
| **Method** | `LangChainClient.generate_summary()` |
| **System Prompt** | `SUMMARY_SYSTEM_PROMPT` |
| **Input** | Hypotheses, Recommendations, Signals, Task type |
| **Output** | Markdown-formatted string |

#### System Prompt

```
You are an expert ML diagnostician agent helping users understand model issues.

Your role is to provide a comprehensive diagnostic summary that includes:
1. A summary of the detected issues
2. The recommended actions to fix them

Guidelines:
- Be concise and direct
- Focus on the most important issues first
- Present recommendations in order of importance
- Use markdown formatting for clarity
- Include specific numbers and metrics from the evidence
- For feature_redundancy, include the specific correlated feature pairs
- For class_imbalance, include class distribution and recall disparities
- For data_leakage, include suspicious feature correlations and CV-holdout gaps

Structure your response as:
## Diagnosis
[Brief summary of detected issues with evidence]

## Recommendations
[Numbered list of recommendations with rationale]
```

#### Output Format

```markdown
## Diagnosis

Based on the analysis, here are the key findings:

- **Overfitting** (85% confidence, high severity)
  - Train-validation gap of 25% indicates memorization
  - Perfect training score (100%) with 75% validation

## Recommendations

**1. Increase regularization strength**
   Stronger regularization penalizes model complexity and reduces overfitting.
   *(Addresses: overfitting)*

**2. Reduce model complexity**
   Use a simpler model or reduce the number of features.
   *(Addresses: overfitting)*
```

---

### 4. Chat Agent

**Purpose**: Enable interactive conversations about diagnosis results through a web-based chatbot interface.

**Role**: Conversational ML assistant that answers user questions about their model's diagnosis, provides explanations, and offers code examples.

| Property | Value |
|----------|-------|
| **Location** | `sklearn_diagnose/server/chat_agent.py` |
| **Class** | `ChatAgent` |
| **Input** | DiagnosisReport, User message (str) |
| **Output** | Conversational response (str) |

#### Architecture

Unlike the batch-processing agents (Hypothesis, Recommendation, Summary), the Chat Agent is **interactive** and **stateful**:

- Maintains conversation history across multiple turns
- Uses the DiagnosisReport as context for all responses
- Generates a welcome message based on detected issues
- Supports clearing/resetting conversation history

#### System Prompt

```
You are an expert ML diagnostician assistant helping users understand and fix
issues with their machine learning models.

You have access to a diagnosis report containing:
- Detected issues (hypotheses) with confidence scores and evidence
- Actionable recommendations to fix the issues
- Performance metrics and signals

Your role is to:
1. Answer questions about the diagnosis in a clear, conversational manner
2. Explain technical concepts when needed
3. Provide code examples when asked
4. Help users understand how to implement the recommendations
5. Be encouraging and supportive while being technically accurate

Guidelines:
- Be concise but complete
- Use specific numbers and evidence from the diagnosis
- Suggest concrete next steps when appropriate
- If asked for code, provide working Python examples
- Always ground your responses in the actual diagnosis data
```

#### Key Methods

```python
class ChatAgent:
    def __init__(self, report: DiagnosisReport):
        """Initialize agent with diagnosis report."""

    def chat(self, message: str) -> str:
        """Process user message and return response."""

    def get_welcome_message(self) -> str:
        """Generate welcome message based on detected issues."""

    def get_history(self) -> List[Dict[str, str]]:
        """Retrieve conversation history."""

    def clear_history(self) -> None:
        """Reset conversation to start fresh."""
```

#### Usage Example

```python
from sklearn_diagnose import diagnose, launch_chatbot

# Run diagnosis
report = diagnose(model, datasets, task="classification")

# Launch interactive chatbot (opens browser automatically)
launch_chatbot(report)

# Or use ChatAgent directly in Python
from sklearn_diagnose.server.chat_agent import ChatAgent

agent = ChatAgent(report)
response = agent.chat("What are the main issues with my model?")
print(response)
```

#### Web Interface

The Chat Agent is exposed through a FastAPI web server with:

- **Frontend**: React + Vite SPA with bundled static files
- **Backend**: FastAPI REST API with endpoints:
  - `POST /api/chat` - Send message, receive response
  - `GET /api/report` - Retrieve diagnosis report
  - `GET /api/welcome` - Get welcome message
  - `POST /api/clear` - Clear conversation history
- **Single-port architecture**: Both frontend and API served on port 8000

---

## Implementation Details

### LangChain Integration

All agents are implemented using LangChain's `create_agent` abstraction (v1.2.0+):

```python
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

# Initialize model
model = ChatOpenAI(model="gpt-4o", api_key=api_key)

# Create agent (no tools needed for generation tasks)
agent = create_agent(model=model, tools=[])

# Invoke agent
result = agent.invoke({"messages": [system_message, user_message]})
```

### Supported LLM Providers

| Provider | Model Examples | Environment Variable |
|----------|----------------|---------------------|
| **OpenAI** | `gpt-4o`, `gpt-4o-mini`, `gpt-4.1-mini` | `OPENAI_API_KEY` |
| **Anthropic** | `claude-3-5-sonnet-latest`, `claude-3-haiku-20240307` | `ANTHROPIC_API_KEY` |
| **OpenRouter** | `deepseek/deepseek-r1-0528`, `openai/gpt-4o` | `OPENROUTER_API_KEY` |

### Agent Invocation Flow

```python
# 1. Setup LLM provider
from sklearn_diagnose import setup_llm
setup_llm(provider="openai", model="gpt-4o")

# 2. Run diagnosis (invokes 3 batch agents sequentially)
from sklearn_diagnose import diagnose
report = diagnose(model, datasets, task="classification")

# Internal flow:
# ├── Extract signals (deterministic)
# ├── Hypothesis Agent → report.hypotheses
# ├── Recommendation Agent → report.recommendations
# └── Summary Agent → report.summary()

# 3. (Optional) Launch interactive chatbot (4th agent)
from sklearn_diagnose import launch_chatbot
launch_chatbot(report)  # Opens browser, starts Chat Agent
```

---

## Agent Communication

The agents communicate through a **sequential pipeline** where each agent's output becomes input for the next:

```
Signals (dict)
    │
    ▼
┌─────────────────┐
│ Hypothesis Agent │──────► hypotheses: List[Hypothesis]
└─────────────────┘                │
                                   ▼
                    ┌──────────────────────────┐
                    │   Recommendation Agent    │──► recommendations: List[Recommendation]
                    └──────────────────────────┘           │
                                                           ▼
                                            ┌─────────────────────┐
                                            │    Summary Agent     │──► summary: str
                                            └─────────────────────┘
                                                           │
                                                           ▼
                                                   DiagnosisReport (all findings)
                                                           │
                                                           ▼
                                            ┌─────────────────────┐
                                            │    Chat Agent        │──► Interactive Q&A
                                            │    (Interactive)     │    (stateful)
                                            └─────────────────────┘
```

### Data Flow

| Stage | Input | Output |
|-------|-------|--------|
| Signal Extraction | Model, Data | `signals: Dict[str, Any]` |
| Hypothesis Agent | `signals`, `task` | `hypotheses: List[Hypothesis]` |
| Recommendation Agent | `hypotheses`, `examples` | `recommendations: List[Recommendation]` |
| Summary Agent | `hypotheses`, `recommendations`, `signals`, `task` | `summary: str` |

---

## Testing Agents

For testing, a `MockLLMClient` is provided that simulates agent behavior without making API calls:

```python
from tests.conftest import MockLLMClient
from sklearn_diagnose.llm.client import _set_global_client

# Use mock client for testing
_set_global_client(MockLLMClient())

# Now diagnose() uses deterministic mock responses
report = diagnose(model, datasets, task="classification")
```

The mock client implements the same interface as `LangChainClient`:
- `generate_hypotheses()` - Rule-based hypothesis detection
- `generate_recommendations()` - Template-based recommendations
- `generate_summary()` - Formatted string output

---

## Configuration

### Basic Setup

```python
from sklearn_diagnose import setup_llm

# OpenAI
setup_llm(provider="openai", model="gpt-4o", api_key="sk-...")

# Anthropic
setup_llm(provider="anthropic", model="claude-3-5-sonnet-latest", api_key="sk-ant-...")

# OpenRouter
setup_llm(provider="openrouter", model="deepseek/deepseek-r1-0528", api_key="sk-or-...")
```

### Using Environment Variables

```bash
# .env file
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
OPENROUTER_API_KEY=sk-or-...
```

```python
from sklearn_diagnose import setup_llm

# API key loaded automatically from environment
setup_llm(provider="openai", model="gpt-4o")
```

---

## Extending Agents

### Adding Custom Failure Modes

To add new failure modes, update:

1. `sklearn_diagnose/core/schemas.py` - Add to `FailureMode` enum
2. `sklearn_diagnose/core/recommendations.py` - Add example recommendations
3. `sklearn_diagnose/llm/client.py` - Update `HYPOTHESIS_SYSTEM_PROMPT` with the new failure mode

### Custom Agent Behavior

For advanced customization, subclass `LangChainClient`:

```python
from sklearn_diagnose.llm.client import LangChainClient

class CustomClient(LangChainClient):
    def generate_hypotheses(self, signals, task):
        # Custom hypothesis generation logic
        ...
```

---

## Error Handling

Each agent has fallback behavior if the LLM call fails:

| Agent | Fallback Behavior |
|-------|-------------------|
| Hypothesis Agent | Returns empty list `[]` |
| Recommendation Agent | Returns empty list `[]` |
| Summary Agent | Returns `_generate_fallback_summary()` with basic formatting |

---

## Performance Considerations

| Aspect | Details |
|--------|---------|
| **API Calls** | 3 sequential calls per diagnosis (one per agent) |
| **Latency** | ~2-5 seconds total (depends on model and network) |
| **Token Usage** | ~500-2000 tokens per diagnosis |
| **Cost** | ~$0.01-0.10 per diagnosis (depends on model) |

### Optimization Tips

1. Use smaller models (`gpt-4o-mini`) for faster, cheaper diagnostics
2. Cache results for repeated diagnoses on same model/data
3. Use `MockLLMClient` for development and testing

---

## Maintenance

> **Important**: This document must be kept in sync with code changes.

### When to Update AGENTS.md

Update this document whenever changes are made to:

| Change Type | Files Affected | Sections to Update |
|-------------|----------------|-------------------|
| New feature added | Any | Relevant sections (Overview, Agent Specifications, Configuration, etc.) |
| System prompts modified | `llm/client.py` | Agent Specifications → System Prompt |
| New failure mode added | `core/schemas.py`, `llm/client.py` | Detectable Failure Modes table |
| Agent method signature changed | `llm/client.py` | Agent Specifications → Input/Output |
| New LLM provider added | `llm/client.py` | Supported LLM Providers table |
| Output schema changed | `llm/client.py` | Output Schema sections |
| New agent added | `llm/client.py` | Add new Agent Specification section |
| Agent communication flow changed | `llm/client.py`, `api/diagnose.py` | Agent Communication diagram |
| Performance characteristics changed | Any | Performance Considerations |
| Dependencies updated | `pyproject.toml` | Implementation Details, Configuration |
| API interface changed | `api/diagnose.py`, `__init__.py` | Configuration, Agent Invocation Flow |

### Checklist for Code Changes

When modifying agent-related code, ensure:

- [ ] AGENTS.md reflects current system prompts
- [ ] Input/output schemas match actual implementations
- [ ] Diagrams accurately represent data flow
- [ ] All supported providers are documented
- [ ] Code examples are tested and working
- [ ] Version-specific features are noted

### How to Verify AGENTS.md is Current

```bash
# Check that documented prompts match code
grep -A 20 "HYPOTHESIS_SYSTEM_PROMPT" sklearn_diagnose/llm/client.py
grep -A 20 "RECOMMENDATION_SYSTEM_PROMPT" sklearn_diagnose/llm/client.py
grep -A 20 "SUMMARY_SYSTEM_PROMPT" sklearn_diagnose/llm/client.py

# Check failure modes enum matches documentation
grep -A 10 "class FailureMode" sklearn_diagnose/core/schemas.py

# Check agent methods exist
grep "def generate_" sklearn_diagnose/llm/client.py
```

---

## References

- [LangChain Agents Documentation](https://docs.langchain.com/oss/python/langchain/agents)
- [LangChain create_agent API](https://reference.langchain.com/python/langchain/agents/#langchain.agents.create_agent)
- [sklearn-diagnose README](README.md)
- [sklearn-diagnose Architecture](README.md#architecture)
