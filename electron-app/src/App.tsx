import { useEffect, useState } from "react";
import { FileInput } from "./components/FileInput";
import { ModelSelector } from "./components/ModelSelector";
import { FittingView } from "./components/FittingView";
import { ResultsPanel } from "./components/ResultsPanel";
import { ErrorBoundary } from "./components/ErrorBoundary";
import { api } from "./api/client";
import type { SessionMeta, ModelsResponse, JobStatus } from "../types";

type Step = 1 | 2 | 3 | 4;

type Theme = "default" | "gruvbox" | "cappuccin";
const THEME_ORDER: Theme[] = ["default", "gruvbox", "cappuccin"];
const THEME_GLYPH: Record<Theme, string> = {
  default: "◐",
  gruvbox: "◑",
  cappuccin: "◒",
};

function getInitialTheme(): Theme {
  if (typeof window === "undefined") return "default";
  const saved = window.localStorage.getItem("diffsuite-theme") as Theme | null;
  return saved && THEME_ORDER.includes(saved) ? saved : "default";
}

export default function App() {
  const [step, setStep] = useState<Step>(1);
  const [meta, setMeta] = useState<SessionMeta | null>(null);
  const [models, setModels] = useState<ModelsResponse | null>(null);
  const [selectedModel, setSelectedModel] = useState<string | null>(null);
  const [forcedReason, setForcedReason] = useState<string | null>(null);
  const [job, setJob] = useState<JobStatus | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [theme, setTheme] = useState<Theme>(getInitialTheme);

  // Apply the chosen theme to the document root and persist the choice.
  useEffect(() => {
    const root = document.documentElement;
    if (theme === "default") {
      root.removeAttribute("data-theme");
    } else {
      root.setAttribute("data-theme", theme);
    }
    window.localStorage.setItem("diffsuite-theme", theme);
  }, [theme]);

  function cycleTheme() {
    const i = THEME_ORDER.indexOf(theme);
    setTheme(THEME_ORDER[(i + 1) % THEME_ORDER.length]);
  }

  const steps = [
    { n: 1, label: "Input" },
    { n: 2, label: "Model" },
    { n: 3, label: "Fit" },
    { n: 4, label: "Results" },
  ];

  // After the input file loads, fetch compatible models directly from the
  // detected b-values (no separate B-Values step needed) and advance to Model.
  async function handleLoaded(m: SessionMeta) {
    setMeta(m);
    setLoadError(null);
    try {
      const resp = await api.getModels(m.bvals);
      setModels(resp);
      setStep(2);
    } catch (e: any) {
      setLoadError(`Could not determine models: ${e.message}`);
    }
  }

  return (
    <div className="app">
      <div className="header-wrap">
        <div className="header">
          <button
            className="theme-switch"
            onClick={cycleTheme}
            title={`Theme: ${theme === "default" ? "Default" : theme.charAt(0).toUpperCase() + theme.slice(1)} (switch with this dot)`}
            aria-label="Switch theme"
          >
            {THEME_GLYPH[theme]}
          </button>
          <div className="brand">
            <svg className="brand-mark" viewBox="0 0 24 24" fill="none" aria-hidden="true">
              <path
                d="M4 15l5-6 4 5 3-4 4 5"
                stroke="var(--accent)"
                strokeWidth="2.2"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
            <h1 className="brand-name">
              <span className="brand-diff">Diff</span><span className="brand-suite">Suite</span>
            </h1>
          </div>
        <span className="sub">DWI fitting &amp; b-value extrapolation</span>
      </div>
      </div>

      <div className="stepper">
        {steps.map((s) => (
          <div
            key={s.n}
            className={
              "step" +
              (step === s.n ? " active" : "") +
              (step > s.n ? " done" : "")
            }
            onClick={() => {
              // only allow navigating to completed steps
              if (s.n < step) setStep(s.n as Step);
            }}
          >
            <span className="num">{s.n}</span>
            {s.label}
          </div>
        ))}
      </div>

      <div className="content">
        <ErrorBoundary>
        {step === 1 && (
          <FileInput onLoaded={handleLoaded} />
        )}

        {step === 2 && models && meta && (
          <ModelSelector
            models={models}
            meta={meta}
            selected={selectedModel}
            onSelect={(k, reason) => {
              setSelectedModel(k);
              setForcedReason(reason ?? null);
              setStep(3);
            }}
            onBvalsChange={(next, bvals) => {
              setModels(next);
              setMeta((m) => (m ? { ...m, bvals } : m));
            }}
          />
        )}

        {step === 3 && meta && selectedModel && (
          <FittingView
            sessionId={meta.session_id}
            modelKey={selectedModel}
            forcedReason={forcedReason}
            uniqBvals={models?.uniq_b}
            onComplete={(j) => {
              setJob(j);
              setStep(4);
            }}
          />
        )}

        {step === 4 && meta && job && (
          <ResultsPanel sessionId={meta.session_id} meta={meta} job={job} />
        )}
        </ErrorBoundary>
      </div>

      {loadError && (
        <p style={{ color: "var(--red)", padding: "0 16px" }}>{loadError}</p>
      )}
    </div>
  );
}
