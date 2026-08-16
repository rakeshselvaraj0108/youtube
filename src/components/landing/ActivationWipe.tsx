import { useEffect } from 'react';
import { motion } from 'framer-motion';

/**
 * The transition from the hero into the working deck.
 *
 * A hard cut from a cinematic landing page into a dense instrument panel
 * reads as two different products stapled together. This buys one beat of
 * continuity: a shockwave expands from wherever the visitor actually
 * clicked, a scanline sweeps once, and the same terminal vocabulary the
 * dashboard's own TerminalColumn uses ("AGENTS ONLINE") appears before the
 * flash hides the DOM swap underneath it.
 *
 * Timing is a fixed `setTimeout` rather than `onAnimationComplete` on one of
 * the child motions — several elements animate on independent schedules
 * here, and tying completion to "whichever finishes last" is a more fragile
 * contract than a single duration this component owns outright.
 */

const DURATION_MS = 900;

const LINES = ['ENGAGING PREFLIGHT…', 'AGENTS ONLINE', 'SYSTEMS NOMINAL'];

export function ActivationWipe({
  origin,
  onComplete,
}: {
  origin: { x: number; y: number };
  onComplete: () => void;
}) {
  useEffect(() => {
    const timer = window.setTimeout(onComplete, DURATION_MS);
    return () => window.clearTimeout(timer);
  }, [onComplete]);

  return (
    <div className="pointer-events-none fixed inset-0 z-[100] overflow-hidden">
      {/* Shockwave from the click point. Sized in vmax rather than a fixed
          pixel scale, so it fully covers the viewport regardless of screen
          size or where within it the visitor actually clicked. */}
      <motion.div
        initial={{ scale: 0, opacity: 0.9 }}
        animate={{ scale: 1, opacity: 0 }}
        transition={{ duration: DURATION_MS / 1000, ease: [0.16, 1, 0.3, 1] }}
        className="absolute rounded-full"
        style={{
          left: origin.x,
          top: origin.y,
          width: '300vmax',
          height: '300vmax',
          translate: '-50% -50%',
          background:
            'radial-gradient(circle, rgba(139,199,255,0.9) 0%, rgba(52,211,153,0.4) 25%, rgba(4,6,10,0) 45%)',
        }}
      />

      {/* One scanline sweep, full width. */}
      <motion.div
        initial={{ top: '-2%', opacity: 0 }}
        animate={{ top: '102%', opacity: [0, 1, 1, 0] }}
        transition={{ duration: DURATION_MS / 1000, ease: [0.16, 1, 0.3, 1] }}
        className="absolute left-0 h-px w-full"
        style={{
          background:
            'linear-gradient(90deg, transparent, #8BC7FF 20%, #E8EDF7 50%, #8BC7FF 80%, transparent)',
          boxShadow: '0 0 24px 2px rgba(139,199,255,0.7)',
        }}
      />

      {/* Terminal boot lines — the same vocabulary the dashboard itself uses. */}
      <div className="absolute left-1/2 top-1/2 flex -translate-x-1/2 -translate-y-1/2 flex-col items-center gap-1.5">
        {LINES.map((line, i) => (
          <motion.span
            key={line}
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: [0, 1, 1, 0], y: 0 }}
            transition={{
              duration: DURATION_MS / 1000 - 0.1,
              delay: i * 0.09,
              ease: 'easeOut',
            }}
            className="num text-[11px] uppercase tracking-[0.2em] text-ink"
          >
            {line}
          </motion.span>
        ))}
      </div>

      {/* Final flash, timed to peak exactly when `onComplete` fires and the
          DOM swaps underneath it — not before. A flash that has already
          faded back to zero by the time the cut happens hides nothing. */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: [0, 0, 0.96] }}
        transition={{ duration: DURATION_MS / 1000, times: [0, 0.6, 1], ease: 'easeIn' }}
        className="absolute inset-0 bg-void"
      />
    </div>
  );
}
