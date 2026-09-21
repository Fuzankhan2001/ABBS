import type { AssessmentAdapter } from './assessmentAdapter';
import type { Assessment, ComponentRecord, LotMetrics, UploadMetadata } from '../types/assessment';
const API_URL = import.meta.env.VITE_ABSS_API_URL ?? 'http://localhost:8000';
export class ApiError extends Error { constructor(message: string, public readonly status: number, public readonly code?: string, public readonly missingColumns: string[] = []) { super(message); } }
async function request<T>(path: string, init?: RequestInit): Promise<T> { let response: Response; try { response = await fetch(`${API_URL}${path}`, init); } catch { throw new ApiError('Assessment service unavailable.', 0); } if (response.ok) return response.json() as Promise<T>; const payload = await response.json().catch(() => null) as { error?: string; message?: string; missing_columns?: string[]; detail?: { error?: string; message?: string; missing_columns?: string[] } } | null; const detail = payload?.detail; throw new ApiError(payload?.message ?? detail?.message ?? 'Assessment service unavailable.', response.status, payload?.error ?? detail?.error, payload?.missing_columns ?? detail?.missing_columns ?? []); }
export const pythonAssessmentAdapter: AssessmentAdapter = {
  async runMissionAssessment(profile: string): Promise<Assessment> { const assessment = await request<Assessment>('/api/assessment/mission', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({ mission: profile }) }); return {...assessment, source: 'LIVE'}; },
  async validateUpload(file: File): Promise<UploadMetadata> { const body = new FormData(); body.append('file', file); return request<UploadMetadata>('/api/assessment/validate', { method: 'POST', body }); },
  async runUploadedAssessment(file: File): Promise<Assessment> { const body = new FormData(); body.append('file', file); const assessment = await request<Assessment>('/api/assessment/upload', { method: 'POST', body }); return {...assessment, source: 'LIVE'}; },
  getComponent(componentId: string): Promise<ComponentRecord> { return request<ComponentRecord>(`/api/components/${encodeURIComponent(componentId)}`); },
  getLot(lotId: string): Promise<LotMetrics> { return request<LotMetrics>(`/api/lots/${encodeURIComponent(lotId)}`); }
};
