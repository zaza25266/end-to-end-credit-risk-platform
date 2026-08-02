import React, { useState } from "import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [formData, setFormData] = useState({
    RevolvingUtilizationOfUnsecuredLines: 0.35,
    age: 42,
    "NumberOfTime30-59DaysPastDueNotWorse": 0,
    DebtRatio: 0.45,
    MonthlyIncome: 5000.0,
    NumberOfOpenCreditLinesAndLoans: 6,
    NumberRealEstateLoansOrLines: 1,
    NumberOfTimes90DaysLate: 0,
    NumberDependingPersons: 1
  });

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: parseFloat(e.target.value) || 0
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await axios.post("http://localhost:8000/predict", formData);
      setResult(response.data);
    } catch (err) {
      setError("Failed to connect to scoring engine or invalid input payload.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-900 text-gray-100 flex flex-col items-center p-6">
      <header className="w-full max-w-4xl mb-8 border-b border-gray-800 pb-4">
        <h1 className="text-3xl font-bold tracking-tight text-blue-400">Credit Risk Decision Platform</h1>
        <p className="text-sm text-gray-400">AI-Powered Enterprise Underwriting & Compliance Workspace</p>
      </header>

      <div className="w-full max-w-4xl grid grid-cols-1 md:grid-cols-2 gap-8">
        {/* Input Form */}
        <div className="bg-gray-800 p-6 rounded-xl shadow-lg border border-gray-700">
          <h2 className="text-xl font-semibold mb-4 text-blue-300">Applicant Financial Profile</h2>
          <form onSubmit={handleSubmit} className="space-y-4">
            {Object.keys(formData).map((key) => (
              <div key={key} className="flex flex-col">
                <label className="text-xs font-medium text-gray-300 mb-1">{key}</label>
                <input
                  type="number"
                  step="any"
                  name={key}
                  value={formData[key]}
                  onChange={handleChange}
                  className="bg-gray-900 border border-gray-700 rounded px-3 py-2 text-sm focus:outline-none focus:border-blue-500"
                  required
                />
              </div>
            ))}
            <button
              type="submit"
              disabled={loading}
              className="w-full bg-blue-600 hover:bg-blue-500 text-white font-semibold py-2.5 rounded transition duration-200 mt-4"
            >
              {loading ? "Scoring Application..." : "Evaluate Risk Score"}
            </button>
          </form>
        </div>

        {/* Results Panel */}
        <div className="bg-gray-800 p-6 rounded-xl shadow-lg border border-gray-700 flex flex-col justify-between">
          <div>
            <h2 className="text-xl font-semibold mb-4 text-blue-300">Underwriting Verdict</h2>
            {error && <div className="p-4 bg-red-900/50 border border-red-700 rounded text-red-200 text-sm">{error}</div>}
            
            {result ? (
              <div className="space-y-6">
                <div className={`p-4 rounded-lg text-center ${result.prediction === "Approved" ? "bg-green-900/40 border border-green-700 text-green-300" : "bg-red-900/40 border border-red-700 text-red-300"}`}>
                  <h3 className="text-sm font-medium uppercase tracking-wider">Decision Outcome</h3>
                  <p className="text-2xl font-bold mt-1">{result.prediction}</p>
                </div>

                <div className="bg-gray-900 p-4 rounded border border-gray-700 space-y-2">
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-400">Default Probability:</span>
                    <span className="font-mono font-bold">{(result.default_probability * 100).toFixed(2)}%</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-400">Decision Threshold:</span>
                    <span className="font-mono">{result.decision_threshold}</span>
                  </div>
                  <div className="w-full bg-gray-800 h-2 rounded-full mt-3 overflow-hidden">
                    <div 
                      className={`h-full ${result.default_probability >= result.decision_threshold ? "bg-red-500" : "bg-green-500"}`}
                      style={{ width: `${result.risk_score_percentage}%` }}
                    ></div>
                  </div>
                </div>
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center h-64 text-gray-500 text-center">
                <p>Submit an applicant profile to generate automated risk scores and decision audits.</p>
              </div>
            )}
          </div>
          <div className="text-xs text-gray-500 border-t border-gray-700 pt-4 text-center">
            Secured Enterprise Underwriting Engine v1.0
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;