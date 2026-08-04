import type { Config } from 'tailwindcss';

/**
 * PREFLIGHT design tokens.
 *
 * Discipline rule (§4.1): the `sig` ramp is the ONLY place saturated colour is
 * allowed to appear in data. `agent` colours appear ONLY in the terminal log and
 * the agent graph. Everything else is greyscale. Colour always means something.
 */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        void: '#04060A',
        abyss: '#070A11',
        panel: '#0B0F17',
        panelHi: '#111725',
        edge: '#1A2233',
        edgeHi: '#26324A',
        ink: '#E8EDF7',
        inkDim: '#8A97AE',
        inkFaint: '#4E5A70',

        // signal ramp — scores, severities, risk bands
        sig: {
          critical: '#FF3B5C',
          high: '#FF7A45',
          medium: '#FFB020',
          low: '#A3E635',
          clear: '#34D399',
        },

        // agent identity — terminal log + agent graph only
        agent: {
          ingest: '#38BDF8',
          speech: '#818CF8',
          vision: '#E879F9',
          ocr: '#F472B6',
          audio: '#2DD4BF',
          access: '#FBBF24',
          meta: '#A78BFA',
          policy: '#22D3EE',
          score: '#4ADE80',
          remedy: '#FB7185',
          report: '#94A3B8',
        },
      },

      fontFamily: {
        mono: ['"JetBrains Mono"', 'ui-monospace', 'SFMono-Regular', 'monospace'],
        sans: ['Inter', 'ui-sans-serif', 'system-ui', 'sans-serif'],
      },

      // §4.3 type scale. [size, { lineHeight, letterSpacing, fontWeight }]
      fontSize: {
        display: ['56px', { lineHeight: '1', letterSpacing: '-0.01em', fontWeight: '300' }],
        h1: ['22px', { lineHeight: '1.2', letterSpacing: '-0.02em', fontWeight: '600' }],
        panel: ['11px', { lineHeight: '1', letterSpacing: '0.12em', fontWeight: '600' }],
        label: ['10px', { lineHeight: '1', letterSpacing: '0.08em', fontWeight: '500' }],
        body: ['13px', { lineHeight: '1.5', fontWeight: '400' }],
        data: ['12px', { lineHeight: '1.4', fontWeight: '400' }],
        code: ['11px', { lineHeight: '1.6', fontWeight: '400' }],
        micro: ['9px', { lineHeight: '1', letterSpacing: '0.1em', fontWeight: '600' }],
      },

      // §4.4 geometry — nothing rounder than 4px.
      borderRadius: {
        panel: '4px',
        chip: '3px',
        bar: '2px',
      },

      // §4.2 four fixed elevation levels. No ad-hoc shadows anywhere.
      boxShadow: {
        'elev-0': 'none',
        'elev-1': '0 1px 0 rgba(255,255,255,.03) inset, 0 1px 2px rgba(0,0,0,.6)',
        'elev-2': '0 1px 0 rgba(255,255,255,.05) inset, 0 8px 24px -8px rgba(0,0,0,.8)',
        'elev-3':
          '0 1px 0 rgba(255,255,255,.07) inset, 0 20px 50px -12px rgba(0,0,0,.9), 0 0 0 1px #26324A',
      },

      spacing: {
        // 4px base unit is Tailwind's default scale; these are the named ones.
        panel: '16px',
        gutter: '12px',
        rail: '340px',
      },

      transitionTimingFunction: {
        'expo-out': 'cubic-bezier(0.16, 1, 0.3, 1)',
        'in-out-soft': 'cubic-bezier(0.65, 0, 0.35, 1)',
      },

      transitionDuration: {
        instant: '120ms',
        fast: '200ms',
        base: '320ms',
        slow: '520ms',
        epic: '1100ms',
      },
    },
  },
  plugins: [],
} satisfies Config;
