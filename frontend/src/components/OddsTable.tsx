import type { Prediction } from "@/ui/types";

interface OddsTableProps {
  match: Prediction;
}

export function OddsTable({ match }: OddsTableProps) {
  const rows = [
    {
      label: "Home",
      odds: match.odds.home,
      prob: match.probs.home,
      mlProb: match.ml_probs?.home ?? null,
      market: match.market_implied?.home ?? null,
      blended: match.blended?.home ?? null,
      ev: match.ev.home ?? null,
      mlEv: match.ml_ev?.home ?? null,
    },
    {
      label: "Draw",
      odds: match.odds.draw,
      prob: match.probs.draw,
      mlProb: match.ml_probs?.draw ?? null,
      market: match.market_implied?.draw ?? null,
      blended: match.blended?.draw ?? null,
      ev: match.ev.draw ?? null,
      mlEv: match.ml_ev?.draw ?? null,
    },
    {
      label: "Away",
      odds: match.odds.away,
      prob: match.probs.away,
      mlProb: match.ml_probs?.away ?? null,
      market: match.market_implied?.away ?? null,
      blended: match.blended?.away ?? null,
      ev: match.ev.away ?? null,
      mlEv: match.ml_ev?.away ?? null,
    },
  ];

  return (
    <div className="border border-border/50 bg-card/50 backdrop-blur rounded-lg">
      <div className="pb-2 px-6 pt-6">
        <div className="text-lg font-semibold">Odds & Probabilities Summary</div>
      </div>
      <div className="p-6 pt-2">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border/50">
                <th className="text-left text-muted-foreground py-2 px-4">Outcome</th>
                <th className="text-right text-muted-foreground py-2 px-4">Odds</th>
                <th className="text-right text-muted-foreground py-2 px-4">Poisson %</th>
                <th className="text-right text-muted-foreground py-2 px-4">ML Model %</th>
                <th className="text-right text-muted-foreground py-2 px-4">Market %</th>
                <th className="text-right text-muted-foreground py-2 px-4">Blended %</th>
                <th className="text-right text-muted-foreground py-2 px-4">EV (Poisson) %</th>
                <th className="text-right text-muted-foreground py-2 px-4">EV (ML) %</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr key={row.label} className="border-b border-border/50">
                  <td className="font-medium text-foreground py-2 px-4">{row.label}</td>
                  <td className="text-right text-foreground font-mono py-2 px-4">{row.odds.toFixed(2)}</td>
                  <td className="text-right text-foreground py-2 px-4">{(row.prob * 100).toFixed(1)}%</td>
                  <td className="text-right text-foreground py-2 px-4">
                    {row.mlProb !== null ? (row.mlProb * 100).toFixed(1) + "%" : "—"}
                  </td>
                  <td className="text-right text-foreground py-2 px-4">
                    {row.market !== null ? (row.market * 100).toFixed(1) + "%" : "—"}
                  </td>
                  <td className="text-right text-foreground py-2 px-4">
                    {row.blended !== null ? (row.blended * 100).toFixed(1) + "%" : "—"}
                  </td>
                  <td
                    className={`text-right font-semibold py-2 px-4 ${
                      (row.ev ?? 0) > 0 ? "text-primary" : "text-destructive"
                    }`}
                  >
                    {row.ev !== null ? ((row.ev > 0 ? "+" : "") + (row.ev * 100).toFixed(2) + "%") : "—"}
                  </td>
                  <td
                    className={`text-right font-semibold py-2 px-4 ${
                      (row.mlEv ?? 0) > 0 ? "text-primary" : "text-destructive"
                    }`}
                  >
                    {row.mlEv !== null ? ((row.mlEv > 0 ? "+" : "") + (row.mlEv * 100).toFixed(2) + "%") : "—"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
