"""
Chatbot launcher for sklearn-diagnose.

This module provides functionality to launch an interactive chatbot
in the browser for discussing diagnosis results.
"""

import webbrowser
import threading
import time
import sys

import uvicorn

from sklearn_diagnose.core.schemas import DiagnosisReport


def launch_chatbot(
    report: DiagnosisReport,
    host: str = "127.0.0.1",
    port: int = 8000,
    auto_open_browser: bool = True,
) -> None:
    """
    Launch an interactive chatbot for discussing the diagnosis report.

    This function starts a FastAPI server with a bundled React frontend
    where users can have conversations with an LLM about their model's diagnosis.

    The server runs on a single port and serves both the API and the web interface.
    No separate frontend setup is required - everything is bundled in the package!

    Args:
        report: The DiagnosisReport to discuss
        host: Server host address (default: "127.0.0.1")
        port: Server port (default: 8000)
        auto_open_browser: Whether to automatically open the browser (default: True)

    Example:
        >>> from sklearn_diagnose import setup_llm, diagnose, launch_chatbot
        >>> setup_llm(provider="openai", model="gpt-4o", api_key="sk-...")
        >>> report = diagnose(model, datasets, task="classification")
        >>> launch_chatbot(report)

    Note:
        Make sure you've called setup_llm() before using this function.
        The chatbot uses the configured LLM to answer questions.
    """
    from sklearn_diagnose.server.app import set_diagnosis_report

    # Set the diagnosis report for the server
    set_diagnosis_report(report)

    # The URL where the chatbot will be accessible
    chatbot_url = f"http://{host}:{port}"

    print("\n" + "=" * 70)
    print("[*] Starting sklearn-diagnose chatbot")
    print("=" * 70)
    print(f"\n[SERVER] Running at: {chatbot_url}")
    print(f"\n[i] The chatbot will open in your browser automatically.")
    print(f"    If it doesn't, navigate to: {chatbot_url}")
    print(f"\n[!] Press Ctrl+C to stop the server")
    print("=" * 70 + "\n")

    # Open browser after a short delay
    if auto_open_browser:
        def open_browser():
            time.sleep(2)  # Wait for server to start
            try:
                webbrowser.open(chatbot_url)
            except Exception as e:
                print(f"Could not open browser automatically: {e}")
                print(f"Please navigate to {chatbot_url} manually.")

        browser_thread = threading.Thread(target=open_browser, daemon=True)
        browser_thread.start()

    # Run the server
    try:
        uvicorn.run(
            "sklearn_diagnose.server.app:app",
            host=host,
            port=port,
            log_level="info",
        )
    except KeyboardInterrupt:
        print("\n\n[*] Shutting down chatbot server...")
        sys.exit(0)


def main():
    """
    CLI entry point for the chatbot.

    This is used when running: python -m sklearn_diagnose.chatbot
    """
    print("[X] Error: No diagnosis report provided.")
    print("\nUsage:")
    print("  from sklearn_diagnose import launch_chatbot")
    print("  launch_chatbot(report)")
    print("\nYou must first run diagnose() to generate a report, then pass it to launch_chatbot().")
    sys.exit(1)


if __name__ == "__main__":
    main()
