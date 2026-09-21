import type { ComponentRecord, LotMetrics } from '../types/assessment';

export function Verdict({ value }: { value: string }) { const kind = value.includes('QUALIFIED') || value === 'PASS' ? 'pass' : value.includes('REJECT') ? 'critical' : 'warning'; return <span className={`verdict ${kind}`}>{value}</span>; }

export function TelemetryChart({ component }: { component: ComponentRecord }) {
  const points = component.telemetry.filter(p => !p.forecast); const all = component.telemetry;
  const max = Math.max(50, ...all.map(p => p.value)) * 1.07; const x = (h: number) => 42 + h / 168 * 540; const y = (v: number) => 186 - v / max * 150;
  const measured = points.map(p => `${x(p.hour)},${y(p.value)}`).join(' '); const forecast = component.forecast168;
  return <div className="chart-wrap"><div className="chart-key"><span><i className="line blue"/>Measured telemetry</span><span><i className="line amber"/>168h forecast</span><span><i className="line limit"/>Qualification ceiling</span></div><svg viewBox="0 0 620 220" role="img" aria-label="Burn-in telemetry trajectory"><g className="grid"><line x1="42" y1="36" x2="590" y2="36"/><line x1="42" y1="86" x2="590" y2="86"/><line x1="42" y1="136" x2="590" y2="136"/><line x1="42" y1="186" x2="590" y2="186"/></g><line className="limit-line" x1="42" y1={y(50)} x2="590" y2={y(50)}/><polyline className="measured" points={measured}/><line className="forecast" x1={x(24)} y1={y(component.iddq24)} x2={x(168)} y2={y(forecast)}/>{points.map((p, i) => <circle key={i} cx={x(p.hour)} cy={y(p.value)} r="4" className="dot"/>)}<circle cx={x(168)} cy={y(forecast)} r="4" className="forecast-dot"/><g className="axis"><text x="42" y="210">0h</text><text x="112" y="210">24h</text><text x="348" y="210">96h</text><text x="570" y="210">168h</text><text x="4" y="190">0</text><text x="0" y="40">{max.toFixed(0)}µA</text></g></svg></div>;
}

export function DistributionChart({ lot, component }: { lot: LotMetrics; component?: ComponentRecord }) {
  const bars = [15, 29, 48, 76, 100, 88, 65, 38, 19, 8]; const selected = component ? Math.min(94, Math.max(5, ((component.iddq0 - lot.lowerPat) / (lot.upperPat - lot.lowerPat)) * 90)) : 50;
  return <div className="distribution"><div className="dist-scale"><span>{lot.lowerPat.toFixed(2)} µA</span><span>Dynamic PAT range</span><span>{lot.upperPat.toFixed(2)} µA</span></div><div className="histogram">{bars.map((b, i) => <div key={i} className="bar" style={{height: `${b}%`}}/>)}<div className="median-marker" style={{left:'50%'}}><span>MEDIAN</span></div><div className="component-marker" style={{left:`${selected}%`}}><span>SELECTED DIE</span></div></div><div className="dist-note">Population {lot.population} · {lot.anomalies} anomalous components · robust σ {lot.robustSigma.toFixed(2)} µA</div></div>;
}
