import React, { useEffect, useState } from "react";
import type {
  HistoricalSummaryResponse,
  LeaguesResponse,
  Prediction,
  PredictionsResponse,
  ScrapeReport,
} from "./types";
import { MatchSidebar } from "@/components/MatchSidebar";
import { MatchHeader } from "@/components/MatchHeader";
import { ProbabilityChart } from "@/components/ProbabilityChart";
import { EVAnalysis } from "@/components/EVAnalysis";
import { GoalsDistribution } from "@/components/GoalsDistribution";
import { FormAnalysis } from "@/components/FormAnalysis";
import { ExtraStats } from "@/components/ExtraStats";
import { OddsTable } from "@/components/OddsTable";
import { MatchComparison } from "@/components/MatchComparison";
import { SummaryStats } from "@/components/SummaryStats";
import { ShapExplanations } from "@/components/ShapExplanations";
import { Activity } from "lucide-react";

type View = "predictions" | "historical";

export function App() {
  const [leagues, setLeagues] = useState<string[]>([]);
  const [league, setLeague] = useState(() => localStorage.getItem("league") ?? "");
  const [rows, setRows] = useState<Prediction[]>([]);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [view, setView] = useState<View>("predictions");
  const [selectedMatchIndex, setSelectedMatchIndex] = useState(0);
  const [evModel, setEvModel] = useState<"poisson" | "ml">(() => {
    const saved = localStorage.getItem("evModel");
    return saved === "ml" ? "ml" : "poisson";
  });

  const [histSummary, setHistSummary] = useState<HistoricalSummaryResponse | null>(null);
  const [scrapeReport, setScrapeReport] = useState<ScrapeReport | null>(null);

  async function loadLeagues() {
    setErr(null);
    try {
      const res = await fetch(`/api/leagues`);
      if (!res.ok) throw new Error(await res.text());
      const json = (await res.json()) as LeaguesResponse;
      const list = json.leagues ?? [];
      setLeagues(list);

      const preferred = localStorage.getItem("league") ?? "";
      if (preferred && list.includes(preferred)) {
        setLeague(preferred);
      } else {
        setLeague(list[0] ?? "");
      }
    } catch (e: any) {
      setLeagues([]);
      setLeague("");
      setErr(e?.message ?? String(e));
    }
  }

  async function load() {
    if (!league) return;
    setLoading(true);
    setErr(null);
    try {
      const res = await fetch(`/api/predictions?league=${encodeURIComponent(league)}`);
      if (!res.ok) throw new Error(await res.text());
      const json = (await res.json()) as PredictionsResponse;
      setRows(json.predictions ?? []);
      setSelectedMatchIndex(0);
    } catch (e: any) {
      setErr(e?.message ?? String(e));
      setRows([]);
    } finally {
      setLoading(false);
    }
  }

  async function runPredict() {
    if (!league) return;
    setLoading(true);
    setErr(null);
    try {
      const res = await fetch(`/api/predict?league=${encodeURIComponent(league)}`, { method: "POST" });
      if (!res.ok) throw new Error(await res.text());
      await load();
    } catch (e: any) {
      setErr(e?.message ?? String(e));
    } finally {
      setLoading(false);
    }
  }

  async function loadHistoricalSummary() {
    if (!league) return;
    setErr(null);
    try {
      const res = await fetch(`/api/historical/summary?league=${encodeURIComponent(league)}`);
      if (!res.ok) throw new Error(await res.text());
      const json = (await res.json()) as HistoricalSummaryResponse;
      setHistSummary(json);
    } catch (e: any) {
      setHistSummary(null);
      setErr(e?.message ?? String(e));
    }
  }

  async function runScrape() {
    if (!league) return;
    setLoading(true);
    setErr(null);
    try {
      const res = await fetch(`/api/scrape?league=${encodeURIComponent(league)}&upcoming=true`, { method: "POST" });
      if (!res.ok) throw new Error(await res.text());
      const json = (await res.json()) as ScrapeReport;
      setScrapeReport(json);
      await loadHistoricalSummary();
    } catch (e: any) {
      setErr(e?.message ?? String(e));
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    loadLeagues();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (!league) return;
    localStorage.setItem("league", league);
    if (view === "historical") {
      loadHistoricalSummary();
    } else {
      load();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [league, view]);

  useEffect(() => {
    if (selectedMatchIndex >= rows.length && rows.length > 0) {
      setSelectedMatchIndex(0);
    }
  }, [selectedMatchIndex, rows.length]);

  useEffect(() => {
    localStorage.setItem("evModel", evModel);
  }, [evModel]);

  const selectedMatch = rows[selectedMatchIndex] ?? null;

  return (
    <div className="flex min-h-screen bg-background dark">
      {view === "predictions" && rows.length > 0 && (
        <MatchSidebar matches={rows} selectedIndex={selectedMatchIndex} onSelect={setSelectedMatchIndex} />
      )}

      <div className="flex-1 overflow-auto">
        <header className="sticky top-0 z-50 border-b border-border/50 bg-background/80 backdrop-blur-lg">
          <div className="flex items-center justify-between gap-3 p-4">
            <div className="flex items-center gap-3">
              <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-primary">
                <Activity className="h-5 w-5 text-primary-foreground" />
              </div>
              <div>
                <h1 className="text-lg font-bold text-foreground">Football Predictions</h1>
                <p className="text-xs text-muted-foreground">Analytics Dashboard</p>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <div className="flex items-center gap-2">
                <button
                  className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
                    view === "predictions"
                      ? "bg-primary text-primary-foreground"
                      : "bg-secondary text-muted-foreground hover:bg-secondary/80"
                  }`}
                  disabled={loading}
                  onClick={() => setView("predictions")}
                >
                  Predictions
                </button>
                <button
                  className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
                    view === "historical"
                      ? "bg-primary text-primary-foreground"
                      : "bg-secondary text-muted-foreground hover:bg-secondary/80"
                  }`}
                  disabled={loading}
                  onClick={() => setView("historical")}
                >
                  Historical
                </button>
              </div>
              <select
                className="bg-input border border-border rounded-lg px-3 py-1.5 text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-ring"
                value={league}
                onChange={(e) => setLeague(e.target.value)}
                disabled={loading || leagues.length === 0}
              >
                {leagues.length === 0 ? (
                  <option value="">No leagues found</option>
                ) : (
                  leagues.map((l) => (
                    <option key={l} value={l}>
                      {l}
                    </option>
                  ))
                )}
              </select>
              {view === "predictions" ? (
                <button
                  className="bg-primary text-primary-foreground px-3 py-1.5 rounded-lg text-sm font-medium hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                  disabled={loading || !league}
                  onClick={runPredict}
                >
                  Recompute
                </button>
              ) : (
                <button
                  className="bg-primary text-primary-foreground px-3 py-1.5 rounded-lg text-sm font-medium hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                  disabled={loading || !league}
                  onClick={runScrape}
                >
                  Scrape now
                </button>
              )}
            </div>
          </div>
        </header>

        <main className="space-y-6 p-4 md:p-6">
          {err && (
            <div className="border border-destructive/50 bg-destructive/10 rounded-lg p-4">
              <div className="font-semibold text-destructive mb-2">Error</div>
              <div className="text-sm text-muted-foreground whitespace-pre-wrap">{err}</div>
            </div>
          )}

          {loading && (
            <div className="text-center text-muted-foreground py-8">Loading...</div>
          )}

          {view === "historical" ? (
            <div className="border border-border/50 bg-card/50 backdrop-blur rounded-lg p-6">
              <div className="font-bold mb-4">Historical summary</div>
              {histSummary ? (
                <div className="text-muted-foreground space-y-2 leading-relaxed">
                  <div>
                    <span className="font-bold">League:</span> {histSummary.league}
                  </div>
                  <div>
                    <span className="font-bold">Matches:</span> {histSummary.count}
                  </div>
                  <div>
                    <span className="font-bold">First match date:</span> {histSummary.first_match_date ?? "—"}
                  </div>
                  <div>
                    <span className="font-bold">Last match date:</span> {histSummary.last_match_date ?? "—"}
                  </div>
                  <div>
                    <span className="font-bold">Last updated (UTC):</span> {histSummary.last_updated_utc ?? "—"}
                  </div>
                </div>
              ) : (
                <div className="text-muted-foreground">No historical summary loaded yet.</div>
              )}

              {scrapeReport && (
                <div className="mt-6">
                  <div className="font-bold mb-2">Last scrape</div>
                  <pre className="text-xs text-muted-foreground whitespace-pre-wrap bg-secondary/50 p-4 rounded-lg">
                    {JSON.stringify(scrapeReport, null, 2)}
                  </pre>
                </div>
              )}
            </div>
          ) : rows.length > 0 && selectedMatch ? (
            <>
              <SummaryStats matches={rows} />
              <MatchHeader match={selectedMatch} evModel={evModel} onEvModelChange={setEvModel} />
              <div className="grid gap-6 lg:grid-cols-2">
                <ProbabilityChart match={selectedMatch} />
                <EVAnalysis match={selectedMatch} evModel={evModel} />
              </div>
              <div className="grid gap-6 lg:grid-cols-2">
                <GoalsDistribution match={selectedMatch} />
                <FormAnalysis match={selectedMatch} />
              </div>
              <OddsTable match={selectedMatch} />
              <ExtraStats match={selectedMatch} />
              {selectedMatch.ml_probs && <ShapExplanations match={selectedMatch} />}
              <MatchComparison matches={rows} />
            </>
          ) : !loading && rows.length === 0 ? (
            <div className="text-center text-muted-foreground py-8">No predictions available. Select a league and click "Recompute".</div>
          ) : null}
        </main>
      </div>
    </div>
  );
}
