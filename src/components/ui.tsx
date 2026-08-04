import type { ReactNode } from 'react';
import type { Severity } from '@/types/analysis';
import { SEVERITY_TONE, SIGNAL_HEX, type SignalTone } from '@/lib/scoring';

/**
 * Shared primitives.
 *
 * Every panel on the deck is built from these three pieces, which is what keeps
 * the surface reading as one instrument rather than thirteen separate widgets.
 */

/* ------------------------------------------------------------------ */
/* Panel                                                               */
/* ------------------------------------------------------------------ */

interface PanelProps {
  title?: string;
  /** Rendered at the right edge of the title row. */
  aside?: ReactNode;
  children: ReactNode;
  className?: string;
  /** Removes body padding — for panels that manage their own scroll region. */
  flush?: boolean;
}

export function Panel({ title, aside, children, className = '', flush }: PanelProps) {
  return (
    <section
      className={`flex min-h-0 min-w-0 flex-col rounded-panel border border-edge bg-panel ${className}`}
      style={{ boxShadow: 'var(--elev-1)' }}
    >
      {title && (
        <header className="flex h-9 shrink-0 items-center justify-between gap-3 border-b border-edge px-4">
          <h2 className="text-panel uppercase text-inkDim">{title}</h2>
          {aside}
        </header>
      )}
      <div className={`flex min-h-0 flex-1 flex-col ${flush ? '' : 'p-4'}`}>{children}</div>
    </section>
  );
}

/* ------------------------------------------------------------------ */
/* Chip                                                                */
/* ------------------------------------------------------------------ */

interface ChipProps {
  children: ReactNode;
  /** Hex from the signal ramp, or undefined for a neutral chip. */
  tone?: string;
  className?: string;
  size?: 'sm' | 'md';
  title?: string;
}

export function Chip({ children, tone, className = '', size = 'sm', title }: ChipProps) {
  const pad = size === 'md' ? 'px-2.5 py-1.5' : 'px-1.5 py-1';
  if (!tone) {
    return (
      <span
        title={title}
        className={`inline-flex items-center gap-1 rounded-chip border border-edge bg-panelHi text-micro uppercase text-inkDim ${pad} ${className}`}
      >
        {children}
      </span>
    );
  }
  return (
    <span
      title={title}
      className={`inline-flex items-center gap-1 rounded-chip border text-micro uppercase ${pad} ${className}`}
      style={{
        color: tone,
        borderColor: `${tone}66`,
        backgroundColor: `${tone}1F`,
      }}
    >
      {children}
    </span>
  );
}

export function SeverityChip({ severity, className }: { severity: Severity; className?: string }) {
  // Meaning is never carried by colour alone — the label is always spelled out.
  return (
    <Chip tone={SIGNAL_HEX[SEVERITY_TONE[severity]]} className={className}>
      {severity}
    </Chip>
  );
}

/* ------------------------------------------------------------------ */
/* Bar                                                                 */
/* ------------------------------------------------------------------ */

interface BarProps {
  /** 0..1 */
  value: number;
  tone: string;
  /** Track height in px. 4 for sub-scores, 3 for confidence micro-bars. */
  height?: number;
  className?: string;
}

export function Bar({ value, tone, height = 4, className = '' }: BarProps) {
  return (
    <div
      className={`w-full overflow-hidden rounded-bar bg-edge ${className}`}
      style={{ height }}
      role="presentation"
    >
      <div
        className="h-full rounded-bar"
        style={{ width: `${Math.max(0, Math.min(1, value)) * 100}%`, background: tone }}
      />
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Field                                                               */
/* ------------------------------------------------------------------ */

export function Field({
  label,
  value,
  className = '',
  title,
}: {
  label: string;
  value: ReactNode;
  className?: string;
  title?: string;
}) {
  // Both label and value truncate independently. A label that overflows its
  // cell runs into the next one and reads as a single word — "RESOLUTIONFPS".
  return (
    <div className={`flex min-w-0 flex-col gap-1.5 ${className}`} title={title}>
      <span className="block truncate text-label uppercase text-inkFaint">{label}</span>
      <span className="num block truncate text-data text-ink">{value}</span>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Dot                                                                 */
/* ------------------------------------------------------------------ */

export function Dot({ tone, size = 6 }: { tone: string; size?: number }) {
  return (
    <span
      className="inline-block shrink-0 rounded-full"
      style={{ width: size, height: size, background: tone }}
    />
  );
}

export const TONE_LABELS: [SignalTone, string][] = [
  ['clear', 'Very low'],
  ['low', 'Low'],
  ['medium', 'Medium'],
  ['high', 'High'],
  ['critical', 'Critical'],
];
