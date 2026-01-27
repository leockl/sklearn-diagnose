"""
Chat agent for interactive diagnosis discussions.

The ChatAgent maintains conversation history and provides contextual
responses about a diagnosis report using the configured LLM.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from sklearn_diagnose.core.schemas import DiagnosisReport, Hypothesis, Recommendation
from sklearn_diagnose.llm.client import _get_global_client


@dataclass
class ChatMessage:
    """Represents a single message in the conversation."""
    role: str  # "user" or "assistant"
    content: str


class ChatAgent:
    """
    Chat agent for discussing diagnosis reports.

    This agent maintains conversation context and uses the configured LLM
    to answer questions about a specific diagnosis report.
    """

    def __init__(self, report: DiagnosisReport):
        """
        Initialize the chat agent with a diagnosis report.

        Args:
            report: The DiagnosisReport to provide context for conversations
        """
        self.report = report
        self.conversation_history: List[ChatMessage] = []
        self.llm_client = _get_global_client()

        if self.llm_client is None:
            raise RuntimeError(
                "LLM client not configured. Please call setup_llm() before launching the chatbot."
            )

    def _format_hypotheses(self) -> str:
        """Format hypotheses for the system prompt."""
        if not self.report.hypotheses:
            return "No significant issues detected."

        lines = []
        for i, hyp in enumerate(self.report.hypotheses, 1):
            lines.append(f"{i}. **{hyp.name.value}** (Confidence: {hyp.confidence:.2%}, Severity: {hyp.severity})")
            lines.append(f"   Evidence:")
            for evidence in hyp.evidence:
                lines.append(f"   - {evidence}")
            lines.append("")

        return "\n".join(lines)

    def _format_recommendations(self) -> str:
        """Format recommendations for the system prompt."""
        if not self.report.recommendations:
            return "No specific recommendations available."

        lines = []
        for i, rec in enumerate(self.report.recommendations, 1):
            lines.append(f"{i}. **{rec.action}**")
            lines.append(f"   Rationale: {rec.rationale}")
            if rec.related_hypothesis:
                lines.append(f"   Addresses: {rec.related_hypothesis.value}")
            lines.append("")

        return "\n".join(lines)

    def _format_key_signals(self) -> str:
        """Format key signals for the system prompt."""
        signals = self.report.signals
        lines = [
            f"- Task: {self.report.task.value}",
            f"- Estimator: {self.report.estimator_type}",
            f"- Training samples: {signals.n_samples_train}",
            f"- Validation samples: {signals.n_samples_val}",
            f"- Features: {signals.n_features}",
            f"- Training score: {signals.train_score:.4f}",
            f"- Validation score: {signals.val_score:.4f}",
            f"- Train-val gap: {signals.train_val_gap:.4f}",
        ]

        if signals.cv_mean is not None:
            lines.extend([
                f"- CV mean score: {signals.cv_mean:.4f}",
                f"- CV std: {signals.cv_std:.4f}",
            ])

        return "\n".join(lines)

    def _build_system_prompt(self) -> str:
        """Build the system prompt with diagnosis context."""
        return f"""You are an expert machine learning diagnostician assistant helping a user understand their model's diagnosis results.

You have access to the following diagnosis report:

## Detected Issues (Hypotheses)
{self._format_hypotheses()}

## Recommendations
{self._format_recommendations()}

## Key Model Signals
{self._format_key_signals()}

## Your Role
Help the user understand:
- What each issue means and why it was detected
- How to implement the recommendations with concrete code examples
- The relationship between different issues (e.g., how feature redundancy can contribute to overfitting)
- Best practices for fixing their specific problems
- Trade-offs between different solutions

## Guidelines
- Be conversational, friendly, and helpful
- IMPORTANT: Do NOT repeat the full issue summary in every response. Only provide a complete overview when the user explicitly asks "what are the issues" or similar overview questions
- For specific "how to fix X" questions, jump straight to the solution without restating all issues
- For follow-up questions, assume the user already knows the context from previous messages
- Provide code examples when relevant (use Python and scikit-learn)
- Reference specific evidence from the diagnosis only when directly relevant to the current question
- If asked about something not in the diagnosis, provide general ML advice but clarify it's not specific to their model
- Use markdown formatting for better readability
- Keep responses concise and focused on answering the specific question asked
- When discussing confidence levels: HIGH (≥75%), MEDIUM (≥50%), LOW (≥25%)
"""

    def _build_conversation_context(self) -> List[Dict[str, str]]:
        """Build the conversation context for the LLM."""
        messages = [{"role": "system", "content": self._build_system_prompt()}]

        for msg in self.conversation_history:
            messages.append({"role": msg.role, "content": msg.content})

        return messages

    def get_welcome_message(self) -> str:
        """Generate a welcome message summarizing key findings."""
        issues = self.report.hypotheses

        if not issues:
            return (
                "👋 Hello! Your model looks healthy - no significant issues were detected. "
                "Feel free to ask me any questions about your model's performance or machine learning in general!"
            )

        # Get top issues
        top_issues = sorted(issues, key=lambda h: h.confidence, reverse=True)[:3]

        issue_summary = []
        for hyp in top_issues:
            issue_summary.append(
                f"- **{hyp.name.value.replace('_', ' ').title()}** "
                f"({hyp.confidence:.0%} confidence, {hyp.severity} severity)"
            )

        message = (
            f"👋 Hello! I've analyzed your {self.report.task.value} model "
            f"and detected {len(issues)} potential issue(s):\n\n"
            + "\n".join(issue_summary) +
            "\n\nI'm here to help you understand these issues and how to fix them. "
            "Ask me anything!"
        )

        return message

    def chat(self, user_message: str) -> str:
        """
        Process a user message and generate a response.

        Args:
            user_message: The user's message

        Returns:
            The assistant's response
        """
        # Add user message to history
        self.conversation_history.append(ChatMessage(role="user", content=user_message))

        # Build conversation context
        messages = self._build_conversation_context()
        messages.append({"role": "user", "content": user_message})

        # Generate response using LLM
        try:
            # Use the LLM client to generate a response
            # We'll use a simple approach: format messages as a single prompt
            response = self._generate_response(messages)

            # Add assistant response to history
            self.conversation_history.append(ChatMessage(role="assistant", content=response))

            return response

        except Exception as e:
            error_msg = (
                f"I encountered an error while processing your message: {str(e)}\n\n"
                "Please try rephrasing your question or ask something else."
            )
            self.conversation_history.append(ChatMessage(role="assistant", content=error_msg))
            return error_msg

    def _generate_response(self, messages: List[Dict[str, str]]) -> str:
        """
        Generate a response using the LLM client.

        Args:
            messages: List of conversation messages

        Returns:
            The generated response
        """
        # Extract system message and conversation messages
        system_message = messages[0]["content"] if messages[0]["role"] == "system" else ""
        conversation_messages = [msg for msg in messages if msg["role"] != "system"]

        # Format as a single prompt for the LLM
        prompt_parts = [system_message, "\n\n---\n\n"]

        for msg in conversation_messages:
            role = "User" if msg["role"] == "user" else "Assistant"
            prompt_parts.append(f"{role}: {msg['content']}\n\n")

        prompt_parts.append("Assistant: ")
        prompt = "".join(prompt_parts)

        # Use the LLM client's underlying model
        try:
            # Access the LangChain model directly
            from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

            lc_messages = []
            if system_message:
                lc_messages.append(SystemMessage(content=system_message))

            for msg in conversation_messages:
                if msg["role"] == "user":
                    lc_messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    lc_messages.append(AIMessage(content=msg["content"]))

            response = self.llm_client._chat_model.invoke(lc_messages)
            return response.content

        except Exception as e:
            # Fallback to simple string concatenation
            raise RuntimeError(f"Failed to generate response: {str(e)}")

    def clear_history(self) -> None:
        """Clear the conversation history."""
        self.conversation_history = []

    def get_history(self) -> List[Dict[str, str]]:
        """
        Get the conversation history.

        Returns:
            List of messages as dicts with 'role' and 'content' keys
        """
        return [
            {"role": msg.role, "content": msg.content}
            for msg in self.conversation_history
        ]
