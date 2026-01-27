"""
Tests for the main diagnose() function.

These tests cover:
- Basic functionality with various estimators
- Detection of known failure modes
- Edge cases and error handling
- Read-only behavior validation
- LLM requirement validation
"""

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from sklearn_diagnose import diagnose
from sklearn_diagnose.core import FailureMode, TaskType


# Note: MockLLMClient fixture is automatically applied from conftest.py


# Fixtures for test data
@pytest.fixture
def classification_data():
    """Generate balanced classification data."""
    X, y = make_classification(
        n_samples=500,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        n_classes=2,
        random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_val, y_train, y_val


@pytest.fixture
def regression_data():
    """Generate regression data."""
    X, y = make_regression(
        n_samples=500,
        n_features=20,
        n_informative=10,
        noise=10,
        random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_val, y_train, y_val


@pytest.fixture
def imbalanced_data():
    """Generate imbalanced classification data."""
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_classes=2,
        weights=[0.95, 0.05],  # 95% class 0, 5% class 1
        random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    return X_train, X_val, y_train, y_val


class TestLLMRequirement:
    """Test that LLM is properly required."""
    
    def test_diagnose_fails_without_llm(self, classification_data):
        """Test that diagnose fails when no LLM is configured."""
        from sklearn_diagnose.llm.client import _set_global_client
        
        X_train, X_val, y_train, y_val = classification_data
        
        # Clear the mock LLM
        _set_global_client(None)
        
        model = LogisticRegression(random_state=42)
        model.fit(X_train, y_train)
        
        with pytest.raises(RuntimeError, match="No LLM provider configured"):
            diagnose(
                estimator=model,
                datasets={
                    "train": (X_train, y_train),
                    "val": (X_val, y_val)
                },
                task="classification"
            )
    
    def test_diagnose_works_with_mock_llm(self, classification_data):
        """Test that diagnose works with mock LLM configured."""
        X_train, X_val, y_train, y_val = classification_data
        
        model = LogisticRegression(random_state=42)
        model.fit(X_train, y_train)
        
        # Mock LLM is already configured via autouse fixture
        report = diagnose(
            estimator=model,
            datasets={
                "train": (X_train, y_train),
                "val": (X_val, y_val)
            },
            task="classification"
        )
        
        assert report is not None
        assert isinstance(report.summary(), str)
        assert isinstance(report.recommendations, list)


class TestBasicFunctionality:
    """Test basic diagnose() functionality."""
    
    def test_diagnose_logistic_regression(self, classification_data):
        """Test diagnose with a simple LogisticRegression."""
        X_train, X_val, y_train, y_val = classification_data
        
        model = LogisticRegression(random_state=42)
        model.fit(X_train, y_train)
        
        report = diagnose(
            estimator=model,
            datasets={
                "train": (X_train, y_train),
                "val": (X_val, y_val)
            },
            task="classification"
        )
        
        assert report is not None
        assert report.task == TaskType.CLASSIFICATION
        assert report.signals is not None
        assert isinstance(report.hypotheses, list)
        assert isinstance(report.recommendations, list)
    
    def test_diagnose_pipeline(self, classification_data):
        """Test diagnose with a Pipeline."""
        X_train, X_val, y_train, y_val = classification_data
        
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(random_state=42))
        ])
        pipeline.fit(X_train, y_train)
        
        report = diagnose(
            estimator=pipeline,
            datasets={
                "train": (X_train, y_train),
                "val": (X_val, y_val)
            },
            task="classification"
        )
        
        assert report is not None
        assert report.has_pipeline is True
        assert "Pipeline" in report.estimator_type
    
    def test_diagnose_regression(self, regression_data):
        """Test diagnose with regression task."""
        X_train, X_val, y_train, y_val = regression_data
        
        model = Ridge(random_state=42)
        model.fit(X_train, y_train)
        
        report = diagnose(
            estimator=model,
            datasets={
                "train": (X_train, y_train),
                "val": (X_val, y_val)
            },
            task="regression"
        )
        
        assert report is not None
        assert report.task == TaskType.REGRESSION
    
    def test_diagnose_with_cv_results(self, classification_data):
        """Test diagnose with cross-validation results."""
        X_train, X_val, y_train, y_val = classification_data
        
        model = LogisticRegression(random_state=42)
        model.fit(X_train, y_train)
        
        cv_results = cross_validate(
            model, X_train, y_train,
            cv=5, return_train_score=True
        )
        
        report = diagnose(
            estimator=model,
            datasets={"train": (X_train, y_train)},
            task="classification",
            cv_results=cv_results
        )
        
        assert report is not None
        assert report.signals.cv_mean is not None
        assert report.signals.cv_std is not None
    
    def test_diagnose_without_val_set(self, classification_data):
        """Test diagnose with only training data."""
        X_train, _, y_train, _ = classification_data
        
        model = LogisticRegression(random_state=42)
        model.fit(X_train, y_train)
        
        # Should work with warning
        report = diagnose(
            estimator=model,
            datasets={"train": (X_train, y_train)},
            task="classification"
        )
        
        assert report is not None
        assert report.signals.val_score is None


class TestFailureModeDetection:
    """Test detection of specific failure modes."""
    
    def test_detect_overfitting(self, classification_data):
        """Test detection of overfitting with deep decision tree."""
        X_train, X_val, y_train, y_val = classification_data
        
        # Create an overfit model (very deep tree)
        model = DecisionTreeClassifier(max_depth=None, min_samples_leaf=1, random_state=42)
        model.fit(X_train, y_train)
        
        report = diagnose(
            estimator=model,
            datasets={
                "train": (X_train, y_train),
                "val": (X_val, y_val)
            },
            task="classification"
        )
        
        # Should detect overfitting
        overfitting_hypotheses = [
            h for h in report.hypotheses 
            if h.name == FailureMode.OVERFITTING
        ]
        
        # Check that we detected the train-val gap
        assert report.signals.train_val_gap is not None
        if report.signals.train_val_gap > 0.15:
            assert len(overfitting_hypotheses) > 0
    
    def test_detect_class_imbalance(self, imbalanced_data):
        """Test detection of class imbalance."""
        X_train, X_val, y_train, y_val = imbalanced_data
        
        model = LogisticRegression(random_state=42)
        model.fit(X_train, y_train)
        
        report = diagnose(
            estimator=model,
            datasets={
                "train": (X_train, y_train),
                "val": (X_val, y_val)
            },
            task="classification"
        )
        
        # Should detect class imbalance
        imbalance_hypotheses = [
            h for h in report.hypotheses 
            if h.name == FailureMode.CLASS_IMBALANCE
        ]
        
        assert report.signals.minority_class_ratio is not None
        assert report.signals.minority_class_ratio < 0.2
        assert len(imbalance_hypotheses) > 0
    
    def test_detect_high_variance(self, classification_data):
        """Test detection of high variance from CV results."""
        X_train, X_val, y_train, y_val = classification_data
        
        # Small tree - might show variance
        model = DecisionTreeClassifier(max_depth=3, random_state=42)
        model.fit(X_train, y_train)
        
        # Create CV results with artificial high variance
        cv_results = {
            "test_score": np.array([0.6, 0.9, 0.65, 0.85, 0.7]),
            "train_score": np.array([0.8, 0.82, 0.78, 0.81, 0.79])
        }
        
        report = diagnose(
            estimator=model,
            datasets={
                "train": (X_train, y_train),
                "val": (X_val, y_val)
            },
            task="classification",
            cv_results=cv_results
        )
        
        # Should have high CV std
        assert report.signals.cv_std is not None
        assert report.signals.cv_std > 0.05


class TestInputValidation:
    """Test input validation."""
    
    def test_reject_unfitted_model(self, classification_data):
        """Test that unfitted models are rejected."""
        X_train, X_val, y_train, y_val = classification_data
        
        model = LogisticRegression()  # Not fitted
        
        with pytest.raises(ValueError, match="not fitted"):
            diagnose(
                estimator=model,
                datasets={
                    "train": (X_train, y_train),
                    "val": (X_val, y_val)
                },
                task="classification"
            )
    
    def test_reject_missing_train(self, classification_data):
        """Test that missing training data is rejected."""
        X_train, X_val, y_train, y_val = classification_data
        
        model = LogisticRegression(random_state=42)
        model.fit(X_train, y_train)
        
        with pytest.raises(ValueError, match="train"):
            diagnose(
                estimator=model,
                datasets={"val": (X_val, y_val)},  # Missing train
                task="classification"
            )
    
    def test_reject_invalid_task(self, classification_data):
        """Test that invalid task types are rejected."""
        X_train, X_val, y_train, y_val = classification_data
        
        model = LogisticRegression(random_state=42)
        model.fit(X_train, y_train)
        
        with pytest.raises(ValueError):
            diagnose(
                estimator=model,
                datasets={
                    "train": (X_train, y_train),
                    "val": (X_val, y_val)
                },
                task="invalid_task"
            )


class TestReadOnlyBehavior:
    """Test that diagnose() is truly read-only."""
    
    def test_estimator_not_modified(self, classification_data):
        """Test that the estimator is not modified."""
        X_train, X_val, y_train, y_val = classification_data
        
        model = LogisticRegression(random_state=42, C=1.0)
        model.fit(X_train, y_train)
        
        # Store original state
        original_coef = model.coef_.copy()
        original_intercept = model.intercept_.copy()
        
        # Run diagnosis
        _ = diagnose(
            estimator=model,
            datasets={
                "train": (X_train, y_train),
                "val": (X_val, y_val)
            },
            task="classification"
        )
        
        # Verify no changes
        np.testing.assert_array_equal(model.coef_, original_coef)
        np.testing.assert_array_equal(model.intercept_, original_intercept)
    
    def test_data_not_modified(self, classification_data):
        """Test that input data is not modified."""
        X_train, X_val, y_train, y_val = classification_data
        
        # Store original data
        X_train_orig = X_train.copy()
        y_train_orig = y_train.copy()
        X_val_orig = X_val.copy()
        y_val_orig = y_val.copy()
        
        model = LogisticRegression(random_state=42)
        model.fit(X_train, y_train)
        
        _ = diagnose(
            estimator=model,
            datasets={
                "train": (X_train, y_train),
                "val": (X_val, y_val)
            },
            task="classification"
        )
        
        # Verify no changes
        np.testing.assert_array_equal(X_train, X_train_orig)
        np.testing.assert_array_equal(y_train, y_train_orig)
        np.testing.assert_array_equal(X_val, X_val_orig)
        np.testing.assert_array_equal(y_val, y_val_orig)


class TestReportOutput:
    """Test the DiagnosisReport output."""
    
    def test_summary_generation(self, classification_data):
        """Test that summary is generated properly."""
        X_train, X_val, y_train, y_val = classification_data
        
        model = LogisticRegression(random_state=42)
        model.fit(X_train, y_train)
        
        report = diagnose(
            estimator=model,
            datasets={
                "train": (X_train, y_train),
                "val": (X_val, y_val)
            },
            task="classification"
        )
        
        summary = report.summary()
        assert isinstance(summary, str)
        assert len(summary) > 0
    
    def test_to_dict_serialization(self, classification_data):
        """Test that report can be serialized to dict."""
        X_train, X_val, y_train, y_val = classification_data
        
        model = LogisticRegression(random_state=42)
        model.fit(X_train, y_train)
        
        report = diagnose(
            estimator=model,
            datasets={
                "train": (X_train, y_train),
                "val": (X_val, y_val)
            },
            task="classification"
        )
        
        result = report.to_dict()
        assert isinstance(result, dict)
        assert "hypotheses" in result
        assert "recommendations" in result
        assert "signals" in result
    
    def test_hypothesis_confidence_bounds(self, classification_data):
        """Test that all confidence scores are valid."""
        X_train, X_val, y_train, y_val = classification_data
        
        model = DecisionTreeClassifier(max_depth=None, random_state=42)
        model.fit(X_train, y_train)
        
        report = diagnose(
            estimator=model,
            datasets={
                "train": (X_train, y_train),
                "val": (X_val, y_val)
            },
            task="classification"
        )
        
        for hypothesis in report.hypotheses:
            assert 0.0 <= hypothesis.confidence <= 1.0
    
    def test_recommendations_have_required_fields(self, classification_data):
        """Test that recommendations have all required fields."""
        X_train, X_val, y_train, y_val = classification_data
        
        # Create overfit model to get recommendations
        model = DecisionTreeClassifier(max_depth=None, random_state=42)
        model.fit(X_train, y_train)
        
        report = diagnose(
            estimator=model,
            datasets={
                "train": (X_train, y_train),
                "val": (X_val, y_val)
            },
            task="classification"
        )
        
        for rec in report.recommendations:
            assert hasattr(rec, 'action')
            assert hasattr(rec, 'rationale')
            assert isinstance(rec.action, str)
            assert isinstance(rec.rationale, str)
            assert len(rec.action) > 0
            assert len(rec.rationale) > 0


class TestEdgeCases:
    """Test edge cases and special scenarios."""
    
    def test_small_dataset(self):
        """Test with very small dataset."""
        X = np.random.randn(20, 5)
        y = np.random.randint(0, 2, 20)
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        model = LogisticRegression(random_state=42)
        model.fit(X_train, y_train)
        
        report = diagnose(
            estimator=model,
            datasets={
                "train": (X_train, y_train),
                "val": (X_val, y_val)
            },
            task="classification"
        )
        
        assert report is not None
    
    def test_single_feature(self):
        """Test with single feature."""
        X = np.random.randn(100, 1)
        y = (X[:, 0] > 0).astype(int)
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        model = LogisticRegression(random_state=42)
        model.fit(X_train, y_train)
        
        report = diagnose(
            estimator=model,
            datasets={
                "train": (X_train, y_train),
                "val": (X_val, y_val)
            },
            task="classification"
        )
        
        assert report is not None
    
    def test_multiclass(self):
        """Test with multiclass classification."""
        X, y = make_classification(
            n_samples=300,
            n_features=20,
            n_classes=3,
            n_clusters_per_class=1,
            random_state=42
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        model = LogisticRegression(random_state=42, max_iter=500)
        model.fit(X_train, y_train)
        
        report = diagnose(
            estimator=model,
            datasets={
                "train": (X_train, y_train),
                "val": (X_val, y_val)
            },
            task="classification"
        )
        
        assert report is not None
        assert report.signals.class_distribution is not None
        assert len(report.signals.class_distribution) == 3
    
    def test_max_recommendations_parameter(self, classification_data):
        """Test that max_recommendations parameter works."""
        X_train, X_val, y_train, y_val = classification_data
        
        model = DecisionTreeClassifier(max_depth=None, random_state=42)
        model.fit(X_train, y_train)
        
        report = diagnose(
            estimator=model,
            datasets={
                "train": (X_train, y_train),
                "val": (X_val, y_val)
            },
            task="classification",
            max_recommendations=3
        )
        
        assert len(report.recommendations) <= 3


# ============================================================================
# Chatbot Tests
# ============================================================================

class TestChatAgent:
    """Test the ChatAgent class for the interactive chatbot."""

    def test_chat_agent_initialization(self, sample_diagnosis_report):
        """Test that ChatAgent initializes correctly with a diagnosis report."""
        from sklearn_diagnose.server.chat_agent import ChatAgent

        agent = ChatAgent(sample_diagnosis_report)

        assert agent.report == sample_diagnosis_report
        assert agent.conversation_history == []
        assert agent.llm_client is not None

    def test_chat_agent_requires_llm(self, sample_diagnosis_report):
        """Test that ChatAgent fails if no LLM is configured."""
        from sklearn_diagnose.server.chat_agent import ChatAgent
        from sklearn_diagnose.llm.client import _set_global_client

        # Temporarily remove the LLM client
        _set_global_client(None)

        with pytest.raises(RuntimeError, match="LLM client not configured"):
            ChatAgent(sample_diagnosis_report)

        # Restore for other tests (will be restored by autouse fixture anyway)
        from tests.conftest import MockLLMClient
        _set_global_client(MockLLMClient())

    def test_chat_response(self, sample_diagnosis_report):
        """Test that chat method produces a response."""
        from sklearn_diagnose.server.chat_agent import ChatAgent

        agent = ChatAgent(sample_diagnosis_report)
        response = agent.chat("What issues were detected?")

        assert isinstance(response, str)
        assert len(response) > 0
        assert len(agent.conversation_history) == 2  # user + assistant

    def test_conversation_history_tracking(self, sample_diagnosis_report):
        """Test that conversation history is properly maintained."""
        from sklearn_diagnose.server.chat_agent import ChatAgent

        agent = ChatAgent(sample_diagnosis_report)
        agent.chat("First message")
        agent.chat("Second message")

        history = agent.get_history()
        assert len(history) == 4  # 2 user + 2 assistant
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"
        assert history[2]["role"] == "user"
        assert history[3]["role"] == "assistant"

    def test_clear_history(self, sample_diagnosis_report):
        """Test clearing conversation history."""
        from sklearn_diagnose.server.chat_agent import ChatAgent

        agent = ChatAgent(sample_diagnosis_report)
        agent.chat("First message")
        agent.chat("Second message")

        assert len(agent.conversation_history) == 4

        agent.clear_history()
        assert len(agent.conversation_history) == 0

    def test_welcome_message_generation(self, sample_diagnosis_report):
        """Test that welcome message is generated correctly."""
        from sklearn_diagnose.server.chat_agent import ChatAgent

        agent = ChatAgent(sample_diagnosis_report)
        welcome = agent.get_welcome_message()

        assert isinstance(welcome, str)
        assert len(welcome) > 0
        # Should mention the issues detected
        assert "overfitting" in welcome.lower() or "issue" in welcome.lower()

    def test_welcome_message_no_issues(self, sample_diagnosis_report):
        """Test welcome message when no issues are detected."""
        from sklearn_diagnose.server.chat_agent import ChatAgent

        # Create a report with no issues
        report_no_issues = sample_diagnosis_report
        report_no_issues.hypotheses = []

        agent = ChatAgent(report_no_issues)
        welcome = agent.get_welcome_message()

        assert isinstance(welcome, str)
        assert "healthy" in welcome.lower() or "no" in welcome.lower()

    def test_system_prompt_includes_context(self, sample_diagnosis_report):
        """Test that system prompt includes diagnosis context."""
        from sklearn_diagnose.server.chat_agent import ChatAgent

        agent = ChatAgent(sample_diagnosis_report)
        system_prompt = agent._build_system_prompt()

        assert isinstance(system_prompt, str)
        assert "overfitting" in system_prompt.lower()
        assert "recommendation" in system_prompt.lower()
        # Should include confidence scores (85% or 85.00% or 0.85)
        assert "85" in system_prompt and ("%" in system_prompt or "0.85" in system_prompt)


class TestChatbotLauncher:
    """Test the chatbot launcher functionality."""

    def test_set_diagnosis_report_does_not_raise(self, sample_diagnosis_report):
        """Test that set_diagnosis_report accepts a valid report without errors."""
        from sklearn_diagnose.server.app import set_diagnosis_report

        # This should not raise an error
        try:
            set_diagnosis_report(sample_diagnosis_report)
            success = True
        except Exception:
            success = False

        assert success, "set_diagnosis_report should not raise an exception"

    def test_set_diagnosis_report_returns_none(self, sample_diagnosis_report):
        """Test that set_diagnosis_report returns None."""
        from sklearn_diagnose.server.app import set_diagnosis_report

        result = set_diagnosis_report(sample_diagnosis_report)

        assert result is None


class TestFastAPIEndpoints:
    """Test FastAPI endpoints (without running the server)."""

    def test_get_chat_agent_after_set_report(self, sample_diagnosis_report):
        """Test that get_chat_agent works after setting a report."""
        from sklearn_diagnose.server.app import set_diagnosis_report, get_chat_agent

        set_diagnosis_report(sample_diagnosis_report)
        agent = get_chat_agent()

        # Should return a ChatAgent instance
        from sklearn_diagnose.server.chat_agent import ChatAgent
        assert isinstance(agent, ChatAgent)
        assert agent.report == sample_diagnosis_report


class TestChatbotIntegration:
    """Integration tests for the complete chatbot flow."""

    def test_complete_chat_flow(self, sample_diagnosis_report):
        """Test a complete chat interaction flow."""
        from sklearn_diagnose.server.chat_agent import ChatAgent

        # Create agent
        agent = ChatAgent(sample_diagnosis_report)

        # Send first message
        response1 = agent.chat("What's the main issue with my model?")
        assert isinstance(response1, str)
        assert len(response1) > 0

        # Send follow-up
        response2 = agent.chat("How do I fix it?")
        assert isinstance(response2, str)
        assert len(response2) > 0

        # Check history
        history = agent.get_history()
        assert len(history) == 4  # 2 user + 2 assistant

        # Clear and restart
        agent.clear_history()
        assert len(agent.get_history()) == 0

        # Can still chat after clearing
        response3 = agent.chat("New conversation")
        assert isinstance(response3, str)

    def test_chat_with_empty_message(self, sample_diagnosis_report):
        """Test that chatting with empty message still works (handled by API layer)."""
        from sklearn_diagnose.server.chat_agent import ChatAgent

        agent = ChatAgent(sample_diagnosis_report)

        # Empty message should still produce a response (error handling is at API level)
        response = agent.chat("")

        # The agent doesn't validate input, that's the API's job
        # So this should work at the agent level
        assert isinstance(response, str)


class TestRealLLMIntegration:
    """
    Integration tests with real LLM providers.

    These tests are skipped by default. To run them, use:
    pytest tests/test_diagnose.py::TestRealLLMIntegration -v -s
    """

    @pytest.fixture
    def test_data(self):
        """Generate synthetic test data with deliberate issues."""
        from sklearn.datasets import make_classification

        # Create dataset with issues (overfitting, class imbalance, redundant features)
        X, y = make_classification(
            n_samples=500,
            n_features=10,
            n_informative=5,
            n_redundant=3,
            n_classes=2,
            weights=[0.9, 0.1],  # Class imbalance
            random_state=42,
        )

        # Add highly correlated features for redundancy detection
        X = np.column_stack([X, X[:, 0] + np.random.normal(0, 0.01, len(X))])
        X = np.column_stack([X, X[:, 1] + np.random.normal(0, 0.01, len(X))])

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        return X_train, X_val, y_train, y_val

    @pytest.fixture
    def trained_model(self, test_data):
        """Train a model that will exhibit overfitting."""
        X_train, X_val, y_train, y_val = test_data

        # Deliberately overfit with weak regularization
        model = LogisticRegression(C=100.0, max_iter=1000, random_state=42)
        model.fit(X_train, y_train)

        return model

    @pytest.mark.skip(reason="Integration test with real LLM - run manually")
    def test_openai_integration(self, test_data, trained_model):
        """Test with OpenAI GPT models."""
        from sklearn_diagnose import setup_llm, diagnose
        from sklearn_diagnose.llm.client import _set_global_client
        import os

        X_train, X_val, y_train, y_val = test_data

        print("\n" + "=" * 70)
        print("TESTING OPENAI INTEGRATION")
        print("=" * 70)

        # Try specified model first, then fallback to known working model
        models_to_try = ["gpt-5.2", "gpt-4o", "gpt-4o-mini"]

        for model_name in models_to_try:
            try:
                print(f"\nTrying OpenAI model: {model_name}")
                setup_llm(
                    provider="openai",
                    model=model_name,
                    api_key=os.getenv("OPENAI_API_KEY")
                )

                print("Running diagnosis...")
                report = diagnose(
                    estimator=trained_model,
                    datasets={"train": (X_train, y_train), "val": (X_val, y_val)},
                    task="classification"
                )

                print("\n✓ OpenAI diagnosis successful!")
                print(f"✓ Model used: {model_name}")
                print(f"✓ Detected {len(report.hypotheses)} issues")
                print(f"✓ Generated {len(report.recommendations)} recommendations")

                # Print summary
                print("\n" + "-" * 70)
                print("DIAGNOSIS SUMMARY")
                print("-" * 70)
                print(report.summary())

                # Success - break out of loop
                break

            except Exception as e:
                print(f"✗ Failed with {model_name}: {str(e)}")
                if model_name == models_to_try[-1]:
                    pytest.fail(f"All OpenAI models failed. Last error: {str(e)}")
                continue

        # Clean up
        _set_global_client(None)

    @pytest.mark.skip(reason="Integration test with real LLM - run manually")
    def test_anthropic_integration(self, test_data, trained_model):
        """Test with Anthropic Claude models."""
        from sklearn_diagnose import setup_llm, diagnose
        from sklearn_diagnose.llm.client import _set_global_client
        import os

        X_train, X_val, y_train, y_val = test_data

        print("\n" + "=" * 70)
        print("TESTING ANTHROPIC INTEGRATION")
        print("=" * 70)

        # Try specified model first, then fallback to known working model
        models_to_try = ["claude-opus-4-5", "claude-opus-4-20250514", "claude-3-5-sonnet-latest"]

        for model_name in models_to_try:
            try:
                print(f"\nTrying Anthropic model: {model_name}")
                setup_llm(
                    provider="anthropic",
                    model=model_name,
                    api_key=os.getenv("ANTHROPIC_API_KEY")
                )

                print("Running diagnosis...")
                report = diagnose(
                    estimator=trained_model,
                    datasets={"train": (X_train, y_train), "val": (X_val, y_val)},
                    task="classification"
                )

                print("\n✓ Anthropic diagnosis successful!")
                print(f"✓ Model used: {model_name}")
                print(f"✓ Detected {len(report.hypotheses)} issues")
                print(f"✓ Generated {len(report.recommendations)} recommendations")

                # Print summary
                print("\n" + "-" * 70)
                print("DIAGNOSIS SUMMARY")
                print("-" * 70)
                print(report.summary())

                # Success - break out of loop
                break

            except Exception as e:
                print(f"✗ Failed with {model_name}: {str(e)}")
                if model_name == models_to_try[-1]:
                    pytest.fail(f"All Anthropic models failed. Last error: {str(e)}")
                continue

        # Clean up
        _set_global_client(None)

    @pytest.mark.skip(reason="Integration test with real LLM - run manually")
    def test_openrouter_integration(self, test_data, trained_model):
        """Test with OpenRouter DeepSeek models."""
        from sklearn_diagnose import setup_llm, diagnose
        from sklearn_diagnose.llm.client import _set_global_client
        import os

        X_train, X_val, y_train, y_val = test_data

        print("\n" + "=" * 70)
        print("TESTING OPENROUTER INTEGRATION")
        print("=" * 70)

        # Try specified model first, then fallback to known working model
        models_to_try = ["deepseek/deepseek-v3.2", "deepseek/deepseek-v3", "deepseek/deepseek-chat"]

        for model_name in models_to_try:
            try:
                print(f"\nTrying OpenRouter model: {model_name}")
                setup_llm(
                    provider="openrouter",
                    model=model_name,
                    api_key=os.getenv("OPENROUTER_API_KEY")
                )

                print("Running diagnosis...")
                report = diagnose(
                    estimator=trained_model,
                    datasets={"train": (X_train, y_train), "val": (X_val, y_val)},
                    task="classification"
                )

                print("\n✓ OpenRouter diagnosis successful!")
                print(f"✓ Model used: {model_name}")
                print(f"✓ Detected {len(report.hypotheses)} issues")
                print(f"✓ Generated {len(report.recommendations)} recommendations")

                # Print summary
                print("\n" + "-" * 70)
                print("DIAGNOSIS SUMMARY")
                print("-" * 70)
                print(report.summary())

                # Success - break out of loop
                break

            except Exception as e:
                print(f"✗ Failed with {model_name}: {str(e)}")
                if model_name == models_to_try[-1]:
                    pytest.fail(f"All OpenRouter models failed. Last error: {str(e)}")
                continue

        # Clean up
        _set_global_client(None)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
