/**
 * The falcon mark. Drawn, not an emoji, not an imported asset.
 *
 * Shared between the header and the landing hero so the brand is drawn once —
 * two copies of the same path data would drift the first time either one got
 * a tweak.
 */
export function Falcon({ className = '' }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 16" className={className} fill="none" aria-hidden="true">
      <path
        d="M1 4.5 L11.2 7.4 L12 3 L12.8 7.4 L23 4.5 L14.4 9.6 L12 15 L9.6 9.6 Z"
        fill="currentColor"
      />
    </svg>
  );
}
