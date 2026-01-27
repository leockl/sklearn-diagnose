"""
Example script showing how to use sklearn-diagnose with the interactive chatbot.

This script demonstrates the complete workflow:
1. Set up an LLM provider
2. Train a model (with deliberate issues for diagnosis)
3. Run diagnosis
4. Launch the interactive chatbot

Run this script with:
    python my_diagnosis.py
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_validate
from dotenv import load_dotenv

# Load API keys from .env file
load_dotenv()

from sklearn_diagnose import setup_llm, diagnose, launch_chatbot


def main():
    print("=" * 80)
    print("  SKLEARN-DIAGNOSE CHATBOT DEMO")
    print("=" * 80)

    # Step 1: Setup LLM
    print("\n[1/4] Setting up LLM...")
    print("   Using OpenAI gpt-4o (with fallback to gpt-4o-mini)")

    # Try OpenAI gpt-4o first (API key loaded from .env)
    try:
        setup_llm(provider="openai", model="gpt-4o")
        print("   [OK] LLM configured successfully")
    except Exception as e:
        print(f"   [ERROR] Error setting up LLM: {e}")
        print("\n   Make sure you have OPENAI_API_KEY in your .env file")
        return

    # Step 2: Generate test data with deliberate issues
    print("\n[2/4] Generating synthetic test data...")
    X, y = make_classification(
        n_samples=500,
        n_features=10,
        n_informative=5,
        n_redundant=3,
        n_classes=2,
        weights=[0.9, 0.1],  # Class imbalance - deliberate!
        random_state=42,
    )

    # Add highly correlated features - deliberate redundancy!
    X = np.column_stack([X, X[:, 0] + np.random.normal(0, 0.01, len(X))])
    X = np.column_stack([X, X[:, 1] + np.random.normal(0, 0.01, len(X))])

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"   Training set: {X_train.shape[0]} samples")
    print(f"   Validation set: {X_val.shape[0]} samples")
    print(f"   Features: {X_train.shape[1]}")

    # Step 3: Train model (with weak regularization to induce issues)
    print("\n[3/4] Training model...")
    model = LogisticRegression(C=100.0, max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    train_score = model.score(X_train, y_train)
    val_score = model.score(X_val, y_val)

    print(f"   Train score: {train_score:.4f}")
    print(f"   Val score: {val_score:.4f}")
    print(f"   Gap: {train_score - val_score:.4f}")

    # Run cross-validation
    cv_results = cross_validate(
        model, X_train, y_train, cv=5, return_train_score=True, scoring="accuracy"
    )

    # Step 4: Run diagnosis
    print("\n[4/4] Running diagnosis (this may take 10-30 seconds)...")

    report = diagnose(
        estimator=model,
        datasets={"train": (X_train, y_train), "val": (X_val, y_val)},
        task="classification",
        cv_results=cv_results,
    )

    print(f"   [OK] Detected {len(report.hypotheses)} issues")
    print(f"   [OK] Generated {len(report.recommendations)} recommendations")

    # Display full static report
    print("\n" + "=" * 80)
    print("  FULL DIAGNOSIS REPORT")
    print("=" * 80 + "\n")
    print(report.summary())

    print("\n" + "=" * 80)
    print("  LAUNCHING CHATBOT")
    print("=" * 80)
    print("\nThe chatbot will open in your browser automatically.")
    print("You can ask questions like:")
    print("  - 'What are the main issues with my model?'")
    print("  - 'How do I fix the class imbalance?'")
    print("  - 'Show me code to implement your first recommendation'")
    print("\nPress Ctrl+C to stop the server when done.")
    print("=" * 80 + "\n")

    # Launch the chatbot!
    launch_chatbot(report)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[OK] Stopped by user.")
    except Exception as e:
        print(f"\n\n[ERROR] Error: {str(e)}")
        import traceback
        traceback.print_exc()
