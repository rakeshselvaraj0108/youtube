import { Component, Suspense, useEffect, useState, type ReactNode } from 'react';
import { Canvas } from '@react-three/fiber';

import { AgentConstellation } from '@/components/landing/AgentConstellation';

/**
 * The WebGL background, isolated behind an error boundary.
 *
 * A landing page is the one screen every visitor sees regardless of what
 * they came to do, so it is the one place a missing GPU, a locked-down
 * sandbox, or a `prefers-reduced-motion` visitor must never see a blank
 * page or a crashed React tree. If the canvas cannot render, the hero still
 * reads perfectly on the CSS gradient behind it — the 3D scene is
 * atmosphere, never the only carrier of the content.
 */

class CanvasBoundary extends Component<{ children: ReactNode }, { failed: boolean }> {
  state = { failed: false };
  static getDerivedStateFromError() {
    return { failed: true };
  }
  componentDidCatch(error: unknown) {
    // eslint-disable-next-line no-console
    console.warn('[landing] 3D scene disabled:', error);
  }
  render() {
    return this.state.failed ? null : this.props.children;
  }
}

function usePrefersReducedMotion(): boolean {
  const [reduced, setReduced] = useState(
    () => window.matchMedia?.('(prefers-reduced-motion: reduce)').matches ?? false,
  );
  useEffect(() => {
    const query = window.matchMedia('(prefers-reduced-motion: reduce)');
    const onChange = () => setReduced(query.matches);
    query.addEventListener('change', onChange);
    return () => query.removeEventListener('change', onChange);
  }, []);
  return reduced;
}

/** Pauses the render loop while the tab is hidden — a hero nobody is
 * looking at should not spend a laptop's battery animating particles. */
function useTabVisible(): boolean {
  const [visible, setVisible] = useState(() => document.visibilityState === 'visible');
  useEffect(() => {
    const onChange = () => setVisible(document.visibilityState === 'visible');
    document.addEventListener('visibilitychange', onChange);
    return () => document.removeEventListener('visibilitychange', onChange);
  }, []);
  return visible;
}

export function Scene3D() {
  const reducedMotion = usePrefersReducedMotion();
  const visible = useTabVisible();

  return (
    <CanvasBoundary>
      <Canvas
        dpr={[1, 1.75]}
        gl={{
          antialias: true,
          alpha: true,
          powerPreference: 'low-power',
          // Without this, a WebGL implementation is allowed to discard the
          // drawing buffer the instant it is composited, which a continuous
          // render loop never notices — but any out-of-band capture of the
          // canvas (a screenshot tool, screen recording, print-to-PDF) can
          // land in the gap and read back nothing. Negligible cost for a
          // scene this size; the alternative is a hero that photographs as
          // a black rectangle for reasons nobody watching it would guess.
          preserveDrawingBuffer: true,
        }}
        camera={{ position: [0, 0.4, 8.5], fov: 42 }}
        frameloop={visible ? 'always' : 'never'}
        aria-hidden
        className="!absolute inset-0"
      >
        <fog attach="fog" args={['#04060A', 9, 17]} />
        <ambientLight intensity={0.3} />
        <pointLight position={[4, 3, 6]} intensity={45} color="#5B7CFF" />
        <Suspense fallback={null}>
          <AgentConstellation reducedMotion={reducedMotion} />
        </Suspense>
      </Canvas>
    </CanvasBoundary>
  );
}
