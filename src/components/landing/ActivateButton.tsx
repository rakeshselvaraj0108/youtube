import { useRef } from 'react';
import { ArrowRight } from 'lucide-react';
import { motion, useMotionValue, useSpring, useTransform } from 'framer-motion';

/**
 * The activation switch.
 *
 * A magnetic tilt card: the button leans toward the pointer within a small
 * bound and a radial highlight tracks the cursor underneath the label,
 * spring-damped rather than snapping directly to the pointer so the motion
 * reads as weight rather than as a hover state flipping on and off.
 */
export function ActivateButton({
  onActivate,
}: {
  onActivate: (origin: { x: number; y: number }) => void;
}) {
  const ref = useRef<HTMLButtonElement>(null);
  const px = useMotionValue(0.5);
  const py = useMotionValue(0.5);
  const spring = { stiffness: 220, damping: 18, mass: 0.6 };
  const rotateX = useSpring(useTransform(py, [0, 1], [10, -10]), spring);
  const rotateY = useSpring(useTransform(px, [0, 1], [-12, 12]), spring);
  const glowX = useTransform(px, (v) => `${v * 100}%`);
  const glowY = useTransform(py, (v) => `${v * 100}%`);

  return (
    <motion.button
      ref={ref}
      type="button"
      onPointerMove={(event) => {
        const rect = event.currentTarget.getBoundingClientRect();
        px.set((event.clientX - rect.left) / rect.width);
        py.set((event.clientY - rect.top) / rect.height);
      }}
      onPointerLeave={() => {
        px.set(0.5);
        py.set(0.5);
      }}
      onClick={(event) => onActivate({ x: event.clientX, y: event.clientY })}
      style={{ rotateX, rotateY, transformPerspective: 700 }}
      whileTap={{ scale: 0.96 }}
      className="group relative isolate overflow-hidden rounded-panel border border-[#2A3A55] bg-[#0B0F17] px-7 py-3.5 text-left shadow-elev-3 transition-colors duration-fast hover:border-[#4A6A9E]"
    >
      {/* Pointer-tracked glow, spring-damped so it reads as weight. */}
      <motion.span
        aria-hidden
        className="pointer-events-none absolute inset-0 opacity-0 transition-opacity duration-base group-hover:opacity-100"
        style={{
          background: useTransform(
            [glowX, glowY],
            ([x, y]) =>
              `radial-gradient(180px circle at ${x} ${y}, rgba(52,211,153,0.22), transparent 70%)`,
          ),
        }}
      />
      <span className="relative flex items-center gap-3">
        <span className="num text-[13px] uppercase tracking-[0.16em] text-ink">
          Get Started
        </span>
        <ArrowRight className="h-4 w-4 text-sig-clear transition-transform duration-fast group-hover:translate-x-1" />
      </span>
      <span className="relative mt-1 block text-[9px] uppercase tracking-[0.14em] text-inkFaint">
        engage the closed loop
      </span>
    </motion.button>
  );
}
