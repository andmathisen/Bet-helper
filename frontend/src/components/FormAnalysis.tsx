import type { Prediction } from "@/ui/types";
import { cn } from "@/lib/utils";

interface FormAnalysisProps {
  match: Prediction;
}

function FormBadge({ result }: { result: string }) {
  const colorMap: Record<string, string> = {
    W: "bg-primary text-primary-foreground",
    D: "bg-chart-3 text-primary-foreground",
    L: "bg-destructive text-destructive-foreground",
  };

  return (
    <span className={cn("inline-flex h-7 w-7 items-center justify-center rounded-md text-xs font-bold", colorMap[result] ?? "bg-secondary text-foreground")}>
      {result}
    </span>
  );
}

function TeamFormCard({ form, teamName, isHome }: { form: any; teamName: string; isHome: boolean }) {
  if (!form) return null;
  
  const sequence = typeof form.sequence === "string" ? form.sequence.split("") : [];
  const ppg = typeof form.ppg === "number" ? form.ppg : 0;
  const record = form.record ?? { W: 0, D: 0, L: 0 };
  const gf = typeof form.gf === "number" ? form.gf : 0;
  const ga = typeof form.ga === "number" ? form.ga : 0;
  const as_home = form.as_home;
  const as_away = form.as_away;

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h4 className="font-semibold text-foreground">{teamName}</h4>
        <span className="text-sm text-muted-foreground">{ppg.toFixed(2)} PPG</span>
      </div>

      <div className="flex items-center gap-1.5">
        {sequence.map((result: string, idx: number) => (
          <FormBadge key={idx} result={result} />
        ))}
      </div>

      <div className="grid grid-cols-3 gap-2 text-center text-sm">
        <div className="rounded-lg bg-secondary/50 p-2">
          <p className="text-xs text-muted-foreground">W</p>
          <p className="font-bold text-primary">{record.W ?? 0}</p>
        </div>
        <div className="rounded-lg bg-secondary/50 p-2">
          <p className="text-xs text-muted-foreground">D</p>
          <p className="font-bold text-chart-3">{record.D ?? 0}</p>
        </div>
        <div className="rounded-lg bg-secondary/50 p-2">
          <p className="text-xs text-muted-foreground">L</p>
          <p className="font-bold text-destructive">{record.L ?? 0}</p>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-2 text-sm">
        <div className="flex justify-between rounded-lg bg-secondary/50 p-2">
          <span className="text-muted-foreground">GF</span>
          <span className="font-medium text-foreground">{gf.toFixed(1)}</span>
        </div>
        <div className="flex justify-between rounded-lg bg-secondary/50 p-2">
          <span className="text-muted-foreground">GA</span>
          <span className="font-medium text-foreground">{ga.toFixed(1)}</span>
        </div>
      </div>

      {isHome && as_home && (
        <div className="border-t border-border/50 pt-3">
          <p className="text-xs text-muted-foreground mb-2">Home Form: {as_home.sequence ?? ""}</p>
          <div className="flex items-center gap-1.5">
            {typeof as_home.sequence === "string"
              ? as_home.sequence.split("").map((result: string, idx: number) => (
                  <FormBadge key={idx} result={result} />
                ))
              : null}
          </div>
          <p className="text-xs text-muted-foreground mt-2">
            {typeof as_home.ppg === "number" ? as_home.ppg.toFixed(2) : "—"} PPG at home
          </p>
        </div>
      )}

      {!isHome && as_away && (
        <div className="border-t border-border/50 pt-3">
          <p className="text-xs text-muted-foreground mb-2">Away Form: {as_away.sequence ?? ""}</p>
          <div className="flex items-center gap-1.5">
            {typeof as_away.sequence === "string"
              ? as_away.sequence.split("").map((result: string, idx: number) => (
                  <FormBadge key={idx} result={result} />
                ))
              : null}
          </div>
          <p className="text-xs text-muted-foreground mt-2">
            {typeof as_away.ppg === "number" ? as_away.ppg.toFixed(2) : "—"} PPG away
          </p>
        </div>
      )}
    </div>
  );
}

export function FormAnalysis({ match }: FormAnalysisProps) {
  const form: any = match.form ?? {};
  const homeForm = form.home;
  const awayForm = form.away;
  const n = typeof form.n === "number" ? form.n : 5;

  return (
    <div className="border border-border/50 bg-card/50 backdrop-blur rounded-lg">
      <div className="pb-2 px-6 pt-6">
        <div className="text-lg font-semibold">
          Form Analysis <span className="text-sm font-normal text-muted-foreground">(Last {n} games)</span>
        </div>
      </div>
      <div className="p-6 pt-2">
        <div className="grid gap-6 md:grid-cols-2">
          <TeamFormCard form={homeForm} teamName={match.match.home} isHome={true} />
          <TeamFormCard form={awayForm} teamName={match.match.away} isHome={false} />
        </div>
      </div>
    </div>
  );
}
