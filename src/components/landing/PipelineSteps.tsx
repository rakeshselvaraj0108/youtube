import { Eye, Film, ShieldCheck, Wand2 } from 'lucide-react';
import { motion } from 'framer-motion';
import type { LucideIcon } from 'lucide-react';

import { AGENT_HEX } from '@/lib/agents';

/**
 * How the loop actually works — four real phases, not marketing filler.
 *
 * These are the same four movements described throughout the engine's own
 * code and this session's build log: observe-and-argue, simulate, remediate,
 * verify. Nothing here is aspirational copy; every sentence names something
 * the codebase actually does (the adversarial triad, the renderable-ceiling
 * simulation, the compiled EDL, the closed-loop re-analysis).
 */

interface Step {
  icon: LucideIcon;
  tone: string;
  title: string;
  body: string;
}

const STEPS: Step[] = [
  {
    icon: Eye,
    tone: AGENT_HEX.vision,
    title: 'Observe & argue',
    body: 'Twelve agents scan frame, transcript and audio. Every finding is prosecuted, defended and adjudicated by an independent triad before it counts.',
  },
  {
    icon: Wand2,
    tone: AGENT_HEX.score,
    title: 'Simulate',
    body: 'Every candidate fix is scored with the same readiness scorer the report uses, before a single frame of the source is touched.',
  },
  {
    icon: Film,
    tone: AGENT_HEX.remedy,
    title: 'Remediate',
    body: 'The chosen edit is compiled into a real ffmpeg program and rendered — the plan on screen is the command that actually runs.',
  },
  {
    icon: ShieldCheck,
    tone: AGENT_HEX.policy,
    title: 'Verify',
    body: 'The rendered file is put back through the full pipeline and re-analysed from scratch. A verdict is a measurement, never an assumption.',
  },
];

function StepCard({ step, index }: { step: Step; index: number }) {
  const Icon = step.icon;
  return (
    <motion.div
      initial={{ opacity: 0, y: 28 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: '-80px' }}
      transition={{ duration: 0.6, delay: index * 0.08, ease: [0.16, 1, 0.3, 1] }}
      whileHover={{ rotateX: -4, rotateY: 4, translateY: -3 }}
      style={{ transformPerspective: 800 }}
      className="group relative flex flex-col gap-3 rounded-panel border border-edge bg-panel/60 p-5 backdrop-blur-sm transition-colors duration-base hover:border-edgeHi"
    >
      <div
        className="flex h-9 w-9 items-center justify-center rounded-chip border"
        style={{ borderColor: `${step.tone}55`, background: `${step.tone}14` }}
      >
        <Icon className="h-4 w-4" style={{ color: step.tone }} strokeWidth={1.6} />
      </div>
      <span className="num text-[10px] uppercase tracking-[0.1em] text-inkFaint">
        {String(index + 1).padStart(2, '0')}
      </span>
      <h3 className="text-[14px] font-semibold text-ink">{step.title}</h3>
      <p className="text-[12px] leading-relaxed text-inkDim">{step.body}</p>
    </motion.div>
  );
}

export function PipelineSteps() {
  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
      {STEPS.map((step, i) => (
        <StepCard key={step.title} step={step} index={i} />
      ))}
    </div>
  );
}
