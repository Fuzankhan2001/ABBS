import type { Assessment, ComponentRecord, LotMetrics } from '../types/assessment';

const lots: LotMetrics[] = [
  { id: 'LOT_ALPHA', population: 760, median: 10.18, robustSigma: 1.08, upperPat: 13.42, lowerPat: 6.94, anomalies: 31 },
  { id: 'LOT_BETA', population: 609, median: 13.48, robustSigma: 1.37, upperPat: 17.59, lowerPat: 9.37, anomalies: 42 },
  { id: 'LOT_GAMMA', population: 627, median: 8.79, robustSigma: 0.89, upperPat: 11.46, lowerPat: 6.12, anomalies: 28 },
  { id: 'LOT_DELTA', population: 504, median: 14.97, robustSigma: 1.55, upperPat: 19.62, lowerPat: 10.32, anomalies: 37 }
];
const make = (i: number, overrides: Partial<ComponentRecord> = {}): ComponentRecord => {
  const lot = lots[i % lots.length]; const base = lot.median + (((i * 17) % 13) - 6) * 0.11;
  const drift = 0.28 + (i % 5) * 0.035; const at24 = base + drift; const forecast = at24 + 1.2 + (i % 4) * .2;
  return { id: `ISRO_IC_${String(i).padStart(5, '0')}`, lot: lot.id, verdict: 'QUALIFIED', moduleA: 'PASS', moduleB: 'PASS', iddq0: +base.toFixed(3), iddq24: +at24.toFixed(3), iddq96: +(at24 + .54).toFixed(3), iddq168: +forecast.toFixed(3), forecast168: +forecast.toFixed(3), slope: +((forecast - at24) / 144).toFixed(4), mahalanobis: +(0.31 + (i % 10) * .18).toFixed(2), patViolated: false, covarianceViolated: false, reason: 'NOMINAL: Conforms to dynamic lot distribution envelopes.', moduleBReason: 'Projected leakage growth remains below the early-rejection threshold.', telemetry: [{hour: 0, value: +base.toFixed(3)}, {hour: 24, value: +at24.toFixed(3)}, {hour: 96, value: +(at24 + .54).toFixed(3)}, {hour: 168, value: +forecast.toFixed(3)}], ...overrides };
};
const records = Array.from({ length: 42 }, (_, i) => make(i));
records[3] = make(3, { id: 'ISRO_IC_00003', lot: 'LOT_GAMMA', verdict: 'FINAL REJECT', moduleA: 'FLAGGED', moduleB: 'REJECT', iddq0: 8.142, iddq24: 11.261, iddq96: 18.899, iddq168: 47.631, forecast168: 45.882, slope: .24, mahalanobis: 4.67, patViolated: false, covarianceViolated: true, reason: 'Covariance Breakdown: abnormal 0h–24h trajectory vector (Dₘ=4.67) relative to the LOT_GAMMA population.', moduleBReason: 'Projected degradation slope of 0.2400 µA/hr exceeds the 0.0400 µA/hr early-rejection threshold.', telemetry: [{hour:0,value:8.142},{hour:24,value:11.261},{hour:96,value:18.899},{hour:168,value:47.631},{hour:168,value:45.882,forecast:true}] });
records[11] = make(11, { verdict: 'MODULE A ANOMALY', moduleA: 'FLAGGED', iddq0: 19.91, mahalanobis: 3.44, patViolated: true, covarianceViolated: true, reason: 'CRITICAL: baseline exceeds the dynamic upper PAT bound and the 0h–24h covariance envelope.', moduleBReason: 'Forecast remains within the drift threshold.' });
records[19] = make(19, { verdict: 'EARLY DRIFT REJECT', moduleB: 'REJECT', forecast168: 23.44, slope: .071, moduleBReason: 'Projected degradation slope of 0.0710 µA/hr exceeds the early-rejection threshold.' });
records[27] = make(27, { verdict: 'FINAL REJECT', moduleA: 'FLAGGED', moduleB: 'REJECT', mahalanobis: 3.8, covarianceViolated: true, slope: .093, forecast168: 21.2, reason: 'Covariance anomaly detected outside the lot trajectory envelope.', moduleBReason: 'Forecasted leakage increase requires an early screening reject.' });
export const mockAssessment: Assessment = { id: 'ABSS-2026-09-21-001', mission: 'EOS-08', scenario: 'Payload Subsystem Burn-In', dataset: 'Mission Assessment Dataset', assessedAt: '21 SEP 2026 · 14:32 IST', status: 'COMPLETE', components: records, lots };
