import { ProcessingStatus } from '../types';

export const getProcessingBadgeClass = (state?: string): string => {
  switch (state) {
    case 'queued':
      return 'badge badge-queued';
    case 'processing':
      return 'badge badge-processing';
    case 'retry_scheduled':
      return 'badge badge-retry';
    case 'completed':
      return 'badge badge-completed';
    case 'failed':
      return 'badge badge-failed';
    default:
      return 'badge badge-not-found';
  }
};

interface ProcessingBadgeProps {
  status?: ProcessingStatus | null;
}

export const ProcessingBadge = ({ status }: ProcessingBadgeProps) => {
  const state = status?.state || 'not_found';
  const label = status?.label || 'No Job Found';
  const badgeClass = getProcessingBadgeClass(state);

  return <span className={badgeClass}>{label}</span>;
};

