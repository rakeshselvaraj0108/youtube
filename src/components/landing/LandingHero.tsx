import { lazy, Suspense, useEffect, useState } from 'react';
import { motion } from 'framer-motion';

import { Falcon } from '@/components/Falcon';
import { ActivateButton } from '@/components/landing/ActivateButton';
import { PipelineSteps } from '@/components/landing/PipelineSteps';
import { fetchHealth, type Health } from '@/lib/api';
import { AGENT_HEX } from '@/lib/agents';

/**
 * The front door.
 *
 * Everything above the fold is a single fixed hero — the product itself has
 * no page scroll, and the door to it should not either. Below the fold is a
 * short, honest account of the four things the engine actually does, for a
 * visitor who scrolls before they click.
 *
 * The live status line is real or absent, never invented. This project's
 * central discipline — a number is either measured or it says so — does not
 * stop at the dashboard's edge; a landing page with a fabricated "10,000+
 * videos scanned" counter would be lying in the one place a visitor has no
 * way yet to check.
 */

const EASE = [0.16, 1, 0.3, 1] as const;

// three.js + @react-three/fiber + drei is ~1MB before gzip — real weight the
// dashboard itself never needs. Splitting it into its own chunk means a
// returning visitor whose session already skips the hero (`sessionStorage`,
// or an injected report.html) never fetches it at all, and a first-time
// visitor sees the headline and the button immediately while the
// constellation streams in behind them a beat later. `Suspense` falls back
// to `null` on purpose: the gradient and grid layered under it already read
// as an intentional dark backdrop before the canvas mounts.
const Scene3D = lazy(() =>
  import('@/components/landing/Scene3D').then((m) => ({ default: m.Scene3D })),
);

function LiveStatus() {
  const [health, setHealth] = useState<Health | null | 'loading'>('loading');

  useEffect(() => {
    let cancelled = false;
    void fetchHealth().then((result) => {
      if (!cancelled) setHealth(result);
    });
    return () => {
      cancelled = true;
    };
  }, []);

  const agentCount = Object.keys(AGENT_HEX).length;

  return (
    <div className="flex flex-wrap items-center justify-center gap-x-3 gap-y-1.5 text-[10px]">
      <span className="num flex items-center gap-1.5 uppercase tracking-[0.1em] text-inkFaint">
        <span
          className="h-1.5 w-1.5 rounded-full"
          style={{
            background:
              health && health !== 'loading' ? '#34D399' : health === 'loading' ? '#E3C77B' : '#4E5A70',
          }}
        />
        {health === 'loading'
          ? 'reaching the engine…'
          : health
            ? `engine ${health.engineVersion} · ${health.ffmpegAvailable ? 'ffmpeg ready' : 'ffmpeg missing'}`
            : 'standalone — run `preflight serve` for a live connection'}
      </span>
      <span className="hidden h-3 w-px bg-edge sm:block" />
      <span className="num uppercase tracking-[0.1em] text-inkFaint">
        {agentCount} agents · 4 pipeline tiers · adversarial triad on every finding
      </span>
    </div>
  );
}

export function LandingHero({
  onActivate,
}: {
  onActivate: (origin: { x: number; y: number }) => void;
}) {
  return (
    <div className="relative h-full w-full overflow-y-auto overflow-x-hidden bg-void">
      {/* ---- Hero ------------------------------------------------------ */}
      <section className="relative flex min-h-full w-full flex-col items-center justify-center px-6 py-24">
        <div className="absolute inset-0 -z-10">
          <Suspense fallback={null}>
            <Scene3D />
          </Suspense>
          {/* Vignette so the constellation recedes behind the copy rather
              than competing with it. */}
          <div
            className="absolute inset-0"
            style={{
              background:
                'radial-gradient(ellipse at center, transparent 0%, transparent 35%, #04060A 88%)',
            }}
          />
          {/* Faint HUD grid — cheap, static, no texture asset. */}
          <div
            className="absolute inset-0 opacity-[0.05]"
            style={{
              backgroundImage:
                'linear-gradient(#8BC7FF 1px, transparent 1px), linear-gradient(90deg, #8BC7FF 1px, transparent 1px)',
              backgroundSize: '64px 64px',
            }}
          />
        </div>

        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.9, ease: EASE }}
          className="relative mb-6 flex h-14 w-14 items-center justify-center"
        >
          <span
            className="absolute inset-0 rounded-full"
            style={{
              background: 'radial-gradient(circle, rgba(52,211,153,0.25), transparent 70%)',
            }}
          />
          <span className="absolute inset-1.5 rounded-full border border-sig-clear/30" />
          <Falcon className="relative h-6 w-9 text-ink" />
        </motion.div>

        <motion.span
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, delay: 0.15, ease: EASE }}
          className="num mb-4 rounded-chip border border-edge px-3 py-1 text-[9px] uppercase tracking-[0.24em] text-inkFaint"
        >
          Closed-loop video compliance verification
        </motion.span>

        <motion.h1
          initial={{ opacity: 0, y: 16, filter: 'blur(8px)' }}
          animate={{ opacity: 1, y: 0, filter: 'blur(0px)' }}
          transition={{ duration: 0.9, delay: 0.25, ease: EASE }}
          className="select-none text-center font-sans leading-none text-ink"
          style={{
            fontSize: 'clamp(3.2rem, 10vw, 7.5rem)',
            fontWeight: 300,
            letterSpacing: '-0.02em',
          }}
        >
          PREFLIGHT
        </motion.h1>

        <motion.p
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, delay: 0.4, ease: EASE }}
          className="mt-5 max-w-xl text-center text-[14px] leading-relaxed text-inkDim"
        >
          Predicts YouTube monetization risk before you publish — cites the
          exact policy clause, compiles an executable fix, then re-analyses
          the rendered file to prove the fix actually worked.
        </motion.p>

        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, delay: 0.55, ease: EASE }}
          className="mt-9"
        >
          <ActivateButton onActivate={onActivate} />
        </motion.div>

        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.7, delay: 0.75 }}
          className="mt-8"
        >
          <LiveStatus />
        </motion.div>

        <motion.div
          animate={{ y: [0, 6, 0] }}
          transition={{ duration: 2.2, repeat: Infinity, ease: 'easeInOut' }}
          className="absolute bottom-8 flex flex-col items-center gap-1 text-inkFaint"
        >
          <span className="text-[9px] uppercase tracking-[0.2em]">how it works</span>
          <span className="h-6 w-px bg-gradient-to-b from-inkFaint to-transparent" />
        </motion.div>
      </section>

      {/* ---- How it works ------------------------------------------------ */}
      <section className="relative mx-auto w-full max-w-5xl px-6 pb-24">
        <motion.h2
          initial={{ opacity: 0, y: 12 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
          className="mb-2 text-center text-[11px] uppercase tracking-[0.2em] text-inkFaint"
        >
          The closed loop
        </motion.h2>
        <motion.p
          initial={{ opacity: 0, y: 12 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6, delay: 0.05 }}
          className="mx-auto mb-10 max-w-lg text-center text-[13px] leading-relaxed text-inkDim"
        >
          Every run passes through the same four movements — and the last one
          is what most compliance tools skip.
        </motion.p>
        <PipelineSteps />
      </section>

      <footer className="relative flex w-full flex-col items-center gap-1 border-t border-edge px-6 py-8 text-center">
        <span className="text-[10px] uppercase tracking-[0.14em] text-inkFaint">
          Built for the YouTube Automation Hackathon
        </span>
        <span className="text-[10px] text-inkFaint">
          Rakesh Selvaraj · Chennai Institute of Technology
        </span>
      </footer>
    </div>
  );
}
