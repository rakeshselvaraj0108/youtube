import { useState } from 'react';
import { AnimatePresence, motion } from 'framer-motion';

import App from '@/App';
import { LandingHero } from '@/components/landing/LandingHero';
import { ActivationWipe } from '@/components/landing/ActivationWipe';
import { injectedReport } from '@/lib/reportSource';

/**
 * Front door, then the deck.
 *
 * A visitor who has already clicked through once in this browser session
 * should not sit through the hero again on every reload while they are in
 * the middle of a run — `sessionStorage` remembers the click for the tab's
 * lifetime and forgets it the moment the tab closes, which is the difference
 * between "seen it" and "always skip it".
 *
 * `report.html` opened standalone from an email carries an injected report
 * and has no "Get Started" moment to offer — it goes straight to the deck.
 */

const ENTRY_KEY = 'preflight.entered';

type Phase = 'landing' | 'activating' | 'app';

function initialPhase(): Phase {
  if (injectedReport() !== null) return 'app';
  try {
    return sessionStorage.getItem(ENTRY_KEY) === '1' ? 'app' : 'landing';
  } catch {
    // Storage can throw in a locked-down sandbox; falling back to the hero
    // every time is the safe direction, not a hard failure.
    return 'landing';
  }
}

export default function Root() {
  const [phase, setPhase] = useState<Phase>(initialPhase);
  const [origin, setOrigin] = useState({ x: 0, y: 0 });

  const enter = () => {
    try {
      sessionStorage.setItem(ENTRY_KEY, '1');
    } catch {
      /* best-effort — the session simply replays the hero on reload */
    }
    setPhase('app');
  };

  return (
    <>
      <AnimatePresence>
        {phase !== 'app' && (
          <motion.div
            key="landing"
            exit={{ opacity: 0, transition: { duration: 0.35 } }}
            className="fixed inset-0 z-40"
          >
            <LandingHero
              onActivate={(clickOrigin) => {
                setOrigin(clickOrigin);
                setPhase('activating');
              }}
            />
          </motion.div>
        )}
      </AnimatePresence>

      {phase === 'activating' && <ActivationWipe origin={origin} onComplete={enter} />}

      {phase === 'app' && <App />}
    </>
  );
}
