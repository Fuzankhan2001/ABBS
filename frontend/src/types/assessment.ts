export type Verdict = 'QUALIFIED' | 'MODULE A ANOMALY' | 'EARLY DRIFT REJECT' | 'FINAL REJECT';
export type ModuleStatus = 'PASS' | 'FLAGGED' | 'REJECT' | 'NOT EVALUATED';
export interface TelemetryPoint { hour: number; value: number; forecast?: boolean }
export interface LotMetrics { id: string; population: number; median: number; robustSigma: number; upperPat: number; lowerPat: number; anomalies: number }
export interface ComponentRecord {
  id: string; lot: string; verdict: Verdict; moduleA: ModuleStatus; moduleB: ModuleStatus;
  iddq0: number; iddq24: number; iddq96?: number; iddq168?: number; forecast168?: number;
  slope?: number; mahalanobis: number; patViolated: boolean; covarianceViolated: boolean;
  moduleAReject?: boolean; moduleBEarlyReject?: boolean; moduleBEvaluated?: boolean; finalSystemReject?: boolean;
  reason: string; moduleBReason: string; telemetry: TelemetryPoint[];
}
export interface Assessment { id: string; mission: string; scenario: string; dataset: string; assessedAt: string; status: 'COMPLETE'; source?: 'LIVE' | 'MOCK'; source_type?: 'MISSION_ASSESSMENT' | 'ATE_CSV'; source_filename?: string | null; components: ComponentRecord[]; lots: LotMetrics[] }
export interface UploadMetadata { name: string; size: number; rows: number; valid: boolean; error?: string; errorCode?: string; missingColumns?: string[]; row?: number; column?: string }
