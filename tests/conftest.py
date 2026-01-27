"""
Pytest configuration and fixtures for sklearn-diagnose tests.

This module contains:
- MockLLMClient: A mock LLM client for testing without API calls
- Fixtures for setting up the mock client
"""

from typing import Any, Dict, List

import pytest

from sklearn_diagnose.llm.client import LLMClient, _set_global_client
from sklearn_diagnose.core.schemas import FailureMode, Hypothesis, Recommendation


class MockLLMClient(LLMClient):
    """
    Mock LLM client for testing purposes.
    
    This client returns deterministic responses without making API calls.
    It simulates the behavior of real LLM clients (OpenAI, Anthropic) for testing.
    """
    
    def is_available(self) -> bool:
        return True
    
    def generate_hypotheses(
        self,
        signals: Dict[str, Any],
        task: str
    ) -> List[Hypothesis]:
        """
        Generate mock hypotheses based on signals.
        
        This uses simple deterministic rules to generate hypotheses for testing.
        """
        hypotheses = []
        
        # Check for overfitting
        train_score = signals.get("train_score")
        val_score = signals.get("val_score")
        
        if train_score is not None and val_score is not None:
            gap = train_score - val_score
            if gap >= 0.15:
                hypotheses.append(Hypothesis(
                    name=FailureMode.OVERFITTING,
                    confidence=min(0.95, 0.7 + gap),
                    severity="high" if gap >= 0.25 else "medium",
                    evidence=[
                        f"Train-val gap of {gap:.1%} indicates overfitting",
                        f"Training score: {train_score:.1%}, Validation score: {val_score:.1%}"
                    ]
                ))
        
        # Check for underfitting
        if train_score is not None and train_score < 0.60:
            hypotheses.append(Hypothesis(
                name=FailureMode.UNDERFITTING,
                confidence=0.5 + (0.60 - train_score),
                severity="high" if train_score < 0.50 else "medium",
                evidence=[
                    f"Low training score ({train_score:.1%}) suggests underfitting"
                ]
            ))
        
        # Check for high variance (from CV)
        cv_std = signals.get("cv_std")
        if cv_std is not None and cv_std >= 0.10:
            hypotheses.append(Hypothesis(
                name=FailureMode.HIGH_VARIANCE,
                confidence=min(0.85, 0.5 + cv_std),
                severity="high" if cv_std >= 0.15 else "medium",
                evidence=[
                    f"CV std of {cv_std:.1%} indicates unstable performance"
                ]
            ))
        
        # Check for class imbalance
        minority_ratio = signals.get("minority_class_ratio")
        if task == "classification" and minority_ratio is not None and minority_ratio < 0.20:
            # Build evidence with detailed class information
            evidence = [
                f"Minority class ratio of {minority_ratio:.1%} indicates imbalance"
            ]
            
            # Add class distribution details
            class_dist = signals.get("class_distribution")
            if class_dist:
                evidence.append("Class distribution:")
                for class_label, ratio in class_dist.items():
                    evidence.append(f"  - Class {class_label}: {ratio:.1%}")
            
            # Add per-class recall if available
            per_class_recall = signals.get("per_class_recall")
            if per_class_recall:
                evidence.append("Per-class recall:")
                for class_label, recall in per_class_recall.items():
                    evidence.append(f"  - Class {class_label}: {recall:.1%}")
                
                # Check for recall disparity
                recalls = list(per_class_recall.values())
                if len(recalls) >= 2:
                    max_recall = max(recalls)
                    min_recall = min(recalls)
                    disparity = max_recall - min_recall
                    if disparity > 0.15:
                        evidence.append(f"Recall disparity of {disparity:.1%} shows model bias toward majority class")
            
            # Add per-class precision if available
            per_class_precision = signals.get("per_class_precision")
            if per_class_precision:
                evidence.append("Per-class precision:")
                for class_label, precision in per_class_precision.items():
                    evidence.append(f"  - Class {class_label}: {precision:.1%}")
            
            evidence.append("Consider using class weights, oversampling, or undersampling techniques")
            
            hypotheses.append(Hypothesis(
                name=FailureMode.CLASS_IMBALANCE,
                confidence=min(0.85, 0.5 + (0.20 - minority_ratio) * 2),
                severity="high" if minority_ratio < 0.10 else "medium",
                evidence=evidence
            ))
        
        # Check for label noise (near-perfect training with gap)
        if train_score is not None and train_score >= 0.98:
            if val_score is not None and (train_score - val_score) > 0.10:
                hypotheses.append(Hypothesis(
                    name=FailureMode.LABEL_NOISE,
                    confidence=0.45,
                    severity="low",
                    evidence=[
                        f"Near-perfect training ({train_score:.1%}) with validation gap may indicate label noise"
                    ]
                ))
        
        # Check for feature redundancy (high correlation pairs)
        high_corr_pairs = signals.get("high_correlation_pairs")
        if high_corr_pairs and len(high_corr_pairs) > 0:
            n_pairs = len(high_corr_pairs)
            max_corr = high_corr_pairs[0][2] if high_corr_pairs else 0
            
            # Build evidence with specific pair information
            evidence = [
                f"{n_pairs} highly correlated feature pairs detected (max correlation: {max_corr:.1%})"
            ]
            
            # Add specific pair details (top 5)
            evidence.append("Correlated feature pairs:")
            for feat_i, feat_j, corr in high_corr_pairs[:5]:
                evidence.append(f"  - Feature {feat_i} ↔ Feature {feat_j}: {corr:.1%} correlation")
            
            if n_pairs > 5:
                evidence.append(f"  - ... and {n_pairs - 5} more pairs")
            
            evidence.append("Consider removing one feature from each highly correlated pair")
            
            hypotheses.append(Hypothesis(
                name=FailureMode.FEATURE_REDUNDANCY,
                confidence=min(0.90, 0.5 + n_pairs * 0.1),
                severity="high" if max_corr >= 0.95 else "medium",
                evidence=evidence
            ))
        
        # Check for data leakage
        cv_holdout_gap = signals.get("cv_holdout_gap")
        suspicious_feature_corrs = signals.get("suspicious_feature_correlations")
        
        # Data leakage is suspected if:
        # 1. CV performance significantly exceeds holdout (cv_holdout_gap > 10%)
        # 2. Features are suspiciously correlated with target (> 90%)
        # 3. Performance is "too good to be true" (both train and val > 95% with tiny gap)
        
        leakage_evidence = []
        leakage_confidence = 0.0
        
        # Check CV-holdout gap
        if cv_holdout_gap is not None and cv_holdout_gap > 0.10:
            leakage_evidence.append(f"CV-to-holdout gap of {cv_holdout_gap:.1%} indicates possible data leakage")
            leakage_evidence.append("CV performance significantly exceeds holdout validation")
            leakage_confidence += 0.35
        
        # Check suspicious feature-target correlations
        if suspicious_feature_corrs and len(suspicious_feature_corrs) > 0:
            n_suspicious = len(suspicious_feature_corrs)
            max_corr = abs(suspicious_feature_corrs[0][1]) if suspicious_feature_corrs else 0
            
            leakage_evidence.append(f"{n_suspicious} features with suspiciously high target correlation (max: {max_corr:.1%})")
            leakage_evidence.append("Suspicious feature-target correlations (potential target leakage):")
            for feat_idx, corr in suspicious_feature_corrs[:5]:
                leakage_evidence.append(f"  - Feature {feat_idx}: {abs(corr):.1%} correlation with target")
            
            if n_suspicious > 5:
                leakage_evidence.append(f"  - ... and {n_suspicious - 5} more suspicious features")
            
            leakage_confidence += 0.30 + (n_suspicious * 0.05)
        
        # Check "too good to be true" pattern
        if train_score is not None and val_score is not None:
            if train_score >= 0.95 and val_score >= 0.95:
                gap = train_score - val_score
                if gap < 0.03:
                    leakage_evidence.append(f"Suspiciously high performance: train {train_score:.1%}, val {val_score:.1%}")
                    leakage_evidence.append("Near-perfect scores with minimal gap may indicate data leakage")
                    leakage_confidence += 0.25
        
        if leakage_evidence:
            leakage_evidence.append("Investigate data pipeline for information leakage")
            hypotheses.append(Hypothesis(
                name=FailureMode.DATA_LEAKAGE,
                confidence=min(0.90, leakage_confidence),
                severity="high" if leakage_confidence >= 0.50 else "medium",
                evidence=leakage_evidence
            ))
        
        return hypotheses
    
    def generate_recommendations(
        self,
        hypotheses: List[Hypothesis],
        example_recommendations: Dict[str, List[dict]],
        max_recommendations: int = 5
    ) -> List[Recommendation]:
        """
        Generate mock recommendations based on hypotheses.
        
        Returns recommendations from the example templates for detected failure modes.
        """
        if not hypotheses:
            return []
        
        recommendations = []
        seen_actions = set()
        
        # Sort hypotheses by confidence
        sorted_hypotheses = sorted(hypotheses, key=lambda h: h.confidence, reverse=True)
        
        for h in sorted_hypotheses:
            mode_name = h.name.value
            if mode_name in example_recommendations:
                for rec_template in example_recommendations[mode_name]:
                    if len(recommendations) >= max_recommendations:
                        break
                    
                    action = rec_template.get("action", "")
                    if action in seen_actions:
                        continue
                    
                    recommendations.append(Recommendation(
                        action=action,
                        rationale=rec_template.get("rationale", ""),
                        related_hypothesis=h.name
                    ))
                    seen_actions.add(action)
            
            if len(recommendations) >= max_recommendations:
                break
        
        return recommendations[:max_recommendations]
    
    def generate_summary(
        self,
        hypotheses: List[Hypothesis],
        recommendations: List[Recommendation],
        signals: Dict[str, Any],
        task: str
    ) -> str:
        """Generate a simple mock summary including recommendations."""
        lines = []
        
        # Diagnosis section
        if not hypotheses:
            lines.append("## Diagnosis\n")
            lines.append("No significant issues detected in your model.\n")
        else:
            lines.append("## Diagnosis\n")
            lines.append("Based on the analysis, here are the key findings:\n")
            
            for h in sorted(hypotheses, key=lambda x: x.confidence, reverse=True)[:3]:
                lines.append(f"- **{h.name.value.replace('_', ' ').title()}** ({h.confidence:.0%} confidence, {h.severity} severity)")
                
                # For feature redundancy, class imbalance, and data leakage, show all evidence (includes detailed info)
                if h.name in (FailureMode.FEATURE_REDUNDANCY, FailureMode.CLASS_IMBALANCE, FailureMode.DATA_LEAKAGE):
                    for ev in h.evidence:
                        lines.append(f"  - {ev}")
                elif h.evidence:
                    lines.append(f"  - {h.evidence[0]}")
            lines.append("")
        
        # Recommendations section
        if recommendations:
            lines.append("## Recommendations\n")
            for i, rec in enumerate(recommendations, 1):
                lines.append(f"**{i}. {rec.action}**")
                lines.append(f"   {rec.rationale}")
                if rec.related_hypothesis:
                    lines.append(f"   *(Addresses: {rec.related_hypothesis.value})*")
                lines.append("")
        
        return "\n".join(lines)


@pytest.fixture(autouse=True)
def setup_mock_llm():
    """
    Configure mock LLM client before each test.

    This fixture runs automatically for all tests (autouse=True).
    It sets up the MockLLMClient before each test and clears it after.
    """
    _set_global_client(MockLLMClient())
    yield
    _set_global_client(None)


# ============================================================================
# Chatbot-specific fixtures
# ============================================================================

@pytest.fixture
def sample_diagnosis_report():
    """
    Create a sample diagnosis report for chatbot testing.

    This fixture creates a minimal but realistic DiagnosisReport that can be
    used to test the ChatAgent and API endpoints.
    """
    from sklearn_diagnose.core.schemas import (
        DiagnosisReport, Hypothesis, Recommendation, Signals, TaskType, FailureMode
    )

    # Create sample signals
    signals = Signals(
        train_score=0.95,
        val_score=0.72,
        train_val_gap=0.23,
        cv_mean=0.74,
        cv_std=0.08,
        cv_min=0.65,
        cv_max=0.82,
        cv_range=0.17,
        cv_fold_scores=[0.72, 0.76, 0.65, 0.75, 0.82],
        cv_train_val_gap=0.21,
        n_samples_train=400,
        n_samples_val=100,
        n_features=20,
        feature_to_sample_ratio=0.05,
        residual_mean=None,
        residual_std=None,
        residual_skew=None,
        residual_kurtosis=None,
        class_distribution={0: 0.50, 1: 0.50},
        minority_class_ratio=0.50,
        per_class_recall={0: 0.75, 1: 0.68},
        per_class_precision={0: 0.71, 1: 0.74},
        confusion_matrix=[[75, 25], [32, 68]],
        feature_correlations=None,
        high_correlation_pairs=[(0, 10, 0.95), (1, 11, 0.92)],
        feature_importances=None,
        feature_target_correlations=None,
        cv_holdout_gap=None,
        suspicious_feature_correlations=None,
    )

    # Create sample hypotheses
    hypotheses = [
        Hypothesis(
            name=FailureMode.OVERFITTING,
            confidence=0.85,
            severity="high",
            evidence=[
                "Train-val gap of 23.0% indicates overfitting",
                "Training score: 95.0%, Validation score: 72.0%"
            ]
        ),
        Hypothesis(
            name=FailureMode.FEATURE_REDUNDANCY,
            confidence=0.75,
            severity="medium",
            evidence=[
                "2 highly correlated feature pairs detected",
                "Feature 0 ↔ Feature 10: 95.0% correlation"
            ]
        )
    ]

    # Create sample recommendations
    recommendations = [
        Recommendation(
            action="Increase regularization strength",
            rationale="Stronger regularization penalizes model complexity and reduces overfitting",
            related_hypothesis=FailureMode.OVERFITTING
        ),
        Recommendation(
            action="Remove redundant features",
            rationale="Eliminating correlated features reduces multicollinearity and improves model interpretability",
            related_hypothesis=FailureMode.FEATURE_REDUNDANCY
        )
    ]

    # Create the diagnosis report
    report = DiagnosisReport(
        hypotheses=hypotheses,
        recommendations=recommendations,
        signals=signals,
        task=TaskType.CLASSIFICATION,
        estimator_type="LogisticRegression",
        has_pipeline=False,
        _llm_summary="Mock summary for testing"
    )

    return report
