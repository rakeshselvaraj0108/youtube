import { useEffect, useRef, useState } from 'react';

import { fetchHealth, fetchRun, fetchRuns, startJob, type Health, type RunSummary } from '@/lib/api';
import { useAnalysis } from '@/store/analysis';

/**
 * The control surface. Everything else on this page reports on an analysis
 * that already happened; this is where one starts.
 *
 * It degrades to a status line when `preflight serve` is not running, which
 * is the normal state for a report.html opened from an email — that page is
 * a finished artifact and has nothing to run against.
 */
export function RunBar() {
  const [health, setHealth] = useState<Health | null>(null);
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [video, setVideo] = useState('data/corpus/clips/g001.mp4');
  // Defaults to the full run. Defaulting to offline meant clicking "analyse"
  // skipped the triad, retrieval and the key entirely and returned in three
  // seconds — which reads as a broken tool rather than a deliberate mode,
  // because the deck gave no sign the interesting half had been switched off.
  const [offline, setOffline] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [elapsed, setElapsed] = useState(0);

  const setReport = useAnalysis((s) => s.setReport);
  const source = useAnalysis((s) => s.source);
  const timer = useRef<number | null>(null);

  useEffect(() => {
    void (async () => {
      setHealth(await fetchHealth());
      setRuns(await fetchRuns());
    })();
  }, []);

  // A run can take minutes online. A counter is the difference between
  // "working" and "hung" when there is no token stream to show.
  useEffect(() => {
    if (!busy) {
      if (timer.current) window.clearInterval(timer.current);
      return;
    }
    setElapsed(0);
    timer.current = window.setInterval(() => setElapsed((s) => s + 1), 1000);
    return () => {
      if (timer.current) window.clearInterval(timer.current);
    };
  }, [busy]);

  async function run() {
    setBusy(true);
    setError(null);
    useAnalysis.getState().resetLive();
    try {
      await startJob(video.trim(), { offline }, (event) => {
        useAnalysis.getState().applyEvent(event);
        if (event.type === 'run.error') {
          setError(event.error ?? 'run failed');
          setBusy(false);
        }
        if (event.type === 'run.complete' && event.id) {
          // The stream carries progress; the report itself is fetched once
          // at the end rather than pushed through the event channel, which
          // keeps a 60 kB document out of every frame.
          void (async () => {
            const report = await fetchRun(event.id!);
            if (report) setReport(report, 'api');
            setRuns(await fetchRuns());
            setBusy(false);
          })();
        }
      });
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setBusy(false);
    }
  }

  async function load(id: string) {
    if (!id) return;
    setBusy(true);
    try {
      const report = await fetchRun(id);
      if (report) setReport(report, 'api');
    } finally {
      setBusy(false);
    }
  }

  const offlineForced = health !== null && !health.online;

  return (
    <div className="flex shrink-0 flex-wrap items-center gap-2 border-b border-edge bg-panel px-gutter py-2 text-[10px]">
      <span
        className={`num rounded px-1.5 py-0.5 uppercase tracking-[0.08em] ${
          source === 'fixture'
            ? 'bg-amber-500/15 text-amber-400'
            : 'bg-emerald-500/15 text-emerald-400'
        }`}
        title={
          source === 'fixture'
            ? 'Demo data — no engine attached'
            : source === 'injected'
              ? 'Real run, embedded by the CLI'
              : 'Real run, served by the API'
        }
      >
        {source === 'fixture' ? 'DEMO DATA' : source === 'injected' ? 'LIVE · EMBEDDED' : 'LIVE · API'}
      </span>

      {health === null ? (
        <span className="text-inkFaint">
          engine offline — start it with <code className="text-inkDim">preflight serve</code>
        </span>
      ) : (
        <>
          <input
            value={video}
            onChange={(e) => setVideo(e.target.value)}
            spellCheck={false}
            placeholder="path/to/video.mp4"
            className="num min-w-0 flex-1 rounded border border-edge bg-void px-2 py-1 text-inkDim outline-none focus:border-inkFaint"
          />

          <label className="flex items-center gap-1 text-inkFaint" title={
            offlineForced ? 'No API key configured — offline is the only mode available' : ''
          }>
            <input
              type="checkbox"
              checked={offline || offlineForced}
              disabled={offlineForced}
              onChange={(e) => setOffline(e.target.checked)}
            />
            offline
          </label>

          <button
            onClick={() => void run()}
            disabled={busy || !video.trim() || !health.ffmpegAvailable}
            className="num rounded border border-edge px-2.5 py-1 uppercase tracking-[0.08em] text-inkDim hover:border-inkFaint disabled:opacity-40"
            title={health.ffmpegAvailable ? '' : 'ffmpeg is not on PATH'}
          >
            {busy ? `analysing ${elapsed}s` : 'analyse'}
          </button>

          {runs.length > 0 && (
            <select
              onChange={(e) => void load(e.target.value)}
              defaultValue=""
              className="num max-w-[180px] rounded border border-edge bg-void px-1.5 py-1 text-inkFaint outline-none"
            >
              <option value="">{runs.length} past run{runs.length === 1 ? '' : 's'}</option>
              {runs.map((r) => (
                <option key={r.id} value={r.id}>
                  {r.filename} · {r.overall ?? '—'}
                </option>
              ))}
            </select>
          )}

          <span className="num text-inkFaint">
            {health.capabilitySummary.preferred + health.capabilitySummary.fallback}/8 capabilities
          </span>
        </>
      )}

      {error && <span className="text-red-400">{error}</span>}
    </div>
  );
}
