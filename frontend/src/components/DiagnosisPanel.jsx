/**
 * Diagnosis panel showing report summary in sidebar.
 *
 * Following bundle-dynamic-imports: loaded only when needed.
 */

import { useState, useEffect } from 'react';
import { fetchReport } from '../services/api';
import { AlertCircle, CheckCircle, Info, Lightbulb } from 'lucide-react';

const CONFIDENCE_COLORS = {
  HIGH: 'text-red-600',
  MEDIUM: 'text-yellow-600',
  LOW: 'text-blue-600',
};

const SEVERITY_ICONS = {
  high: AlertCircle,
  medium: Info,
  low: CheckCircle,
};

export function DiagnosisPanel() {
  const [report, setReport] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchReport()
      .then((data) => {
        setReport(data.report);
        setLoading(false);
      })
      .catch((err) => {
        console.error('Failed to fetch report:', err);
        setError('Failed to load diagnosis report');
        setLoading(false);
      });
  }, []);

  if (loading) {
    return (
      <aside className="w-80 bg-gray-50 border-l border-gray-200 p-4 overflow-y-auto">
        <div className="animate-pulse space-y-4">
          <div className="h-6 bg-gray-200 rounded w-3/4"></div>
          <div className="h-4 bg-gray-200 rounded"></div>
          <div className="h-4 bg-gray-200 rounded w-5/6"></div>
        </div>
      </aside>
    );
  }

  if (error || !report) {
    return (
      <aside className="w-80 bg-gray-50 border-l border-gray-200 p-4">
        <p className="text-red-600 text-sm">{error || 'No report available'}</p>
      </aside>
    );
  }

  const { hypotheses, recommendations, signals, task } = report;

  return (
    <aside className="w-80 bg-gray-50 border-l border-gray-200 p-4 overflow-y-auto">
      {/* Model info */}
      <div className="mb-6">
        <h2 className="text-lg font-semibold text-gray-900 mb-2">Model Info</h2>
        <div className="text-sm space-y-1">
          <p>
            <span className="font-medium">Task:</span>{' '}
            <span className="capitalize">{task}</span>
          </p>
          <p>
            <span className="font-medium">Train Score:</span>{' '}
            {signals.train_score?.toFixed(4)}
          </p>
          <p>
            <span className="font-medium">Val Score:</span>{' '}
            {signals.val_score?.toFixed(4)}
          </p>
          {signals.train_val_gap !== null && (
            <p>
              <span className="font-medium">Gap:</span>{' '}
              {signals.train_val_gap?.toFixed(4)}
            </p>
          )}
        </div>
      </div>

      {/* Detected Issues */}
      <div className="mb-6">
        <h2 className="text-lg font-semibold text-gray-900 mb-3">
          Detected Issues
        </h2>
        {hypotheses && hypotheses.length > 0 ? (
          <div className="space-y-3">
            {hypotheses.map((hyp, index) => {
              const SeverityIcon = SEVERITY_ICONS[hyp.severity] || Info;
              const confidencePercent = (hyp.confidence * 100).toFixed(0);

              return (
                <div
                  key={index}
                  className="bg-white rounded-lg p-3 shadow-sm border border-gray-200"
                >
                  <div className="flex items-start gap-2 mb-2">
                    <SeverityIcon className="w-5 h-5 text-gray-600 flex-shrink-0 mt-0.5" />
                    <div className="flex-1">
                      <h3 className="font-medium text-sm text-gray-900 capitalize">
                        {hyp.name.replace(/_/g, ' ')}
                      </h3>
                      <div className="flex items-center gap-2 mt-1">
                        <span
                          className={`text-xs font-semibold ${
                            CONFIDENCE_COLORS[
                              confidencePercent >= 75
                                ? 'HIGH'
                                : confidencePercent >= 50
                                ? 'MEDIUM'
                                : 'LOW'
                            ]
                          }`}
                        >
                          {confidencePercent}% confidence
                        </span>
                        <span className="text-xs text-gray-500 capitalize">
                          {hyp.severity} severity
                        </span>
                      </div>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        ) : (
          <p className="text-sm text-gray-600">No significant issues detected.</p>
        )}
      </div>

      {/* Recommendations */}
      <div>
        <h2 className="text-lg font-semibold text-gray-900 mb-3 flex items-center gap-2">
          <Lightbulb className="w-5 h-5" />
          Recommendations
        </h2>
        {recommendations && recommendations.length > 0 ? (
          <div className="space-y-2">
            {recommendations.slice(0, 5).map((rec, index) => (
              <div
                key={index}
                className="bg-white rounded-lg p-3 shadow-sm border border-gray-200"
              >
                <p className="text-sm font-medium text-gray-900">{rec.action}</p>
                {rec.related_hypothesis && (
                  <p className="text-xs text-gray-500 mt-1 capitalize">
                    Fixes: {rec.related_hypothesis.replace(/_/g, ' ')}
                  </p>
                )}
              </div>
            ))}
          </div>
        ) : (
          <p className="text-sm text-gray-600">No recommendations available.</p>
        )}
      </div>
    </aside>
  );
}
