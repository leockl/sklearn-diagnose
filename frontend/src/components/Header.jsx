/**
 * Application header component.
 *
 * Following rendering-hoist-jsx: static elements defined once.
 */

import { Activity } from 'lucide-react';

export function Header() {
  return (
    <header className="bg-gradient-to-r from-primary-600 to-primary-700 text-white shadow-lg">
      <div className="container mx-auto px-4 py-4">
        <div className="flex items-center gap-3">
          <Activity className="w-8 h-8" />
          <div>
            <h1 className="text-2xl font-bold">sklearn-diagnose</h1>
            <p className="text-sm text-primary-100">Interactive Model Diagnostics</p>
          </div>
        </div>
      </div>
    </header>
  );
}
