import type { Prediction } from "./types";

export function fmt(n: number | null | undefined, digits = 2): string {
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  return n.toFixed(digits);
}

export function maxEv(p: Prediction): { outcome: "Home" | "Draw" | "Away"; ev: number } | null {
  const c: Array<{ outcome: "Home" | "Draw" | "Away"; ev: number }> = [];
  if (typeof p.ev.home === "number") c.push({ outcome: "Home", ev: p.ev.home });
  if (typeof p.ev.draw === "number") c.push({ outcome: "Draw", ev: p.ev.draw });
  if (typeof p.ev.away === "number") c.push({ outcome: "Away", ev: p.ev.away });
  if (!c.length) return null;
  return c.reduce((a, b) => (b.ev > a.ev ? b : a));
}

export function edge(p: Prediction): { home: number | null; draw: number | null; away: number | null } {
  const m = p.market_implied;
  if (!m) return { home: null, draw: null, away: null };
  return {
    home: p.probs.home - m.home,
    draw: p.probs.draw - m.draw,
    away: p.probs.away - m.away,
  };
}

