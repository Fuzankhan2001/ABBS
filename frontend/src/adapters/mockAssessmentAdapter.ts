import { mockAssessment } from '../data/mockAssessment';
import type { AssessmentAdapter } from './assessmentAdapter';
import type { Assessment, UploadMetadata } from '../types/assessment';
const pause = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));
export const mockAssessmentAdapter: AssessmentAdapter = {
  async runMissionAssessment(profile) { await pause(500); return { ...mockAssessment, source: 'MOCK', mission: profile, scenario: profile === 'Gaganyaan' ? 'Avionics Tier-1 Screening' : 'Payload Subsystem Burn-In' }; },
  async validateUpload(file): Promise<UploadMetadata> { const text = await file.text(); const [header, ...rows] = text.trim().split(/\r?\n/); const required = ['Component_ID','Lot_ID','Iddq_0h_uA','Iddq_24h_uA']; const valid = required.every(c => header.split(',').includes(c)); return { name: file.name, size: file.size, rows: Math.max(0, rows.length), valid, error: valid ? undefined : `Required channels: ${required.join(', ')}` }; },
  async runUploadedAssessment() { await pause(500); return { ...mockAssessment, source: 'MOCK', dataset: 'Imported ATE Dataset' }; },
  async getComponent(componentId) { await pause(80); const component = mockAssessment.components.find(item => item.id === componentId); if (!component) throw new Error('Component not found'); return component; },
  async getLot(lotId) { await pause(80); const lot = mockAssessment.lots.find(item => item.id === lotId); if (!lot) throw new Error('Lot not found'); return lot; }
};
