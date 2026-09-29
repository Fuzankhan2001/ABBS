import { mockAssessmentAdapter } from '../adapters/mockAssessmentAdapter';
import { ApiError, pythonAssessmentAdapter } from '../adapters/pythonAssessmentAdapter';
import type { AssessmentAdapter } from '../adapters/assessmentAdapter';
const fallback = async <T>(live: () => Promise<T>, mock: () => Promise<T>): Promise<T> => { try { return await live(); } catch (error) { if (error instanceof ApiError && error.status !== 0) throw error; console.warn('ABSS API unavailable; using mock assessment fallback.', error); return mock(); } };
export const assessmentService: AssessmentAdapter = {
  runMissionAssessment: profile => fallback(() => pythonAssessmentAdapter.runMissionAssessment(profile), () => mockAssessmentAdapter.runMissionAssessment(profile)),
  // Uploads must never be replaced by mock data: callers need the real validation/result.
  validateUpload: file => pythonAssessmentAdapter.validateUpload(file),
  runUploadedAssessment: file => pythonAssessmentAdapter.runUploadedAssessment(file),
  getComponent: componentId => fallback(() => pythonAssessmentAdapter.getComponent(componentId), () => mockAssessmentAdapter.getComponent(componentId)),
  getLot: lotId => fallback(() => pythonAssessmentAdapter.getLot(lotId), () => mockAssessmentAdapter.getLot(lotId)),
};
