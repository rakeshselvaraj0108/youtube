import { useEffect, useRef, useState } from 'react';
import { Film, Pause, Play, SkipForward, Volume2 } from 'lucide-react';
import { Panel } from '@/components/ui';
import { useAnalysis } from '@/store/analysis';
import { readinessHex, SIGNAL_HEX, VERDICT_META } from '@/lib/scoring';
import { formatTimecode } from '@/lib/time';
import type { AnalysisReport } from '@/types/analysis';

/**
 * The proof panel.
 *
 * Phase 3 links the two scrub bars so dragging one drags the other — that is
 * what demonstrates the fix is the same footage rather than a second clip.
 * Phase 2 renders both players with the remediated spans marked.
 */

// The playhead now starts at 0 and is driven by whatever the reader clicks,
// so the parked demo position no longer applies — a hardcoded start would
// fight the first seek.

function PlayerFrame({
  report,
  variant,
  positionT,
  onSeek,
}: {
  report: AnalysisReport;
  variant: 'BEFORE' | 'AFTER';
  positionT: number;
  onSeek: (t: number) => void;
}) {
  const [failed, setFailed] = useState(false);
  const [playing, setPlaying] = useState(false);
  const [muted, setMuted] = useState(true);
  const videoRef = useRef<HTMLVideoElement>(null);
  const scrubRef = useRef<HTMLDivElement>(null);
  const duration = report.video.durationMs;
  const isAfter = variant === 'AFTER';
  const hasVerifiedAfter = useAnalysis((s) => s.hasVerifiedAfter);
  // From the live store, so the counts describe the run on screen rather
  // than the demo fixture this component used to read regardless.
  const ops = useAnalysis((s) => s.before.remediation.ops);
  const findingCount = useAnalysis((s) => s.before.findings.length);

  /* Both scrub bars write to one shared position, and both <video> elements
     read from it. Dragging either one moves both — that is what demonstrates
     the safe render is the same footage rather than a second clip. */
  useEffect(() => {
    const video = videoRef.current;
    if (!video || !Number.isFinite(video.duration)) return;
    const target = positionT * video.duration;
    if (Math.abs(video.currentTime - target) > 0.05) video.currentTime = target;
  }, [positionT]);

  const seekFromEvent = (clientX: number) => {
    const rect = scrubRef.current?.getBoundingClientRect();
    if (!rect) return;
    onSeek(Math.max(0, Math.min(1, (clientX - rect.left) / rect.width)));
  };

  const onScrubDown = (event: React.PointerEvent<HTMLDivElement>) => {
    event.currentTarget.setPointerCapture(event.pointerId);
    seekFromEvent(event.clientX);
  };

  const onScrubMove = (event: React.PointerEvent<HTMLDivElement>) => {
    if (event.buttons !== 1) return;
    seekFromEvent(event.clientX);
  };

  const togglePlayback = async () => {
    const video = videoRef.current;
    if (!video) return;
    if (video.paused) {
      try {
        await video.play();
      } catch {
        // Autoplay policy or a missing source leaves an honest paused player.
      }
    } else {
      video.pause();
    }
  };

  const skip = () => {
    const video = videoRef.current;
    if (!video || !Number.isFinite(video.duration)) return;
    onSeek(Math.min(1, (video.currentTime + 10) / video.duration));
  };

  return (
    <Panel className="min-w-0" flush>
      <div className="flex shrink-0 items-center justify-between gap-2 border-b border-edge px-3 py-2">
        <span
          className="text-micro uppercase"
          style={{ color: isAfter ? SIGNAL_HEX.clear : '#8A97AE' }}
        >
          {isAfter ? (hasVerifiedAfter ? 'After (Verified)' : 'After (Pending)') : 'Before (Original)'}
        </span>
        <span className="num text-[10px] text-inkFaint">
          {formatTimecode(duration * positionT)} / {formatTimecode(duration)}
        </span>
      </div>

      <div className="relative aspect-video w-full shrink-0 overflow-hidden bg-abyss">
        {!failed ? (
          <video
            ref={videoRef}
            src={report.video.srcUrl}
            poster={report.video.posterUrl}
            className="h-full w-full object-cover"
            preload="metadata"
            muted={muted}
            playsInline
            onPlay={() => setPlaying(true)}
            onPause={() => setPlaying(false)}
            onTimeUpdate={(event) => {
              const video = event.currentTarget;
              if (Number.isFinite(video.duration) && video.duration > 0) {
                onSeek(video.currentTime / video.duration);
              }
            }}
            onError={() => setFailed(true)}
          />
        ) : (
          <div className="flex h-full w-full flex-col items-center justify-center gap-1.5 text-inkFaint">
            <Film className="h-5 w-5" strokeWidth={1.5} />
            <span className="num text-[9px] uppercase tracking-[0.1em]">
              {report.video.filename}
            </span>
          </div>
        )}
      </div>

      {/* scrub bar — seek-linked to its counterpart */}
      <div className="shrink-0 px-3 pt-2.5">
        <div
          ref={scrubRef}
          onPointerDown={onScrubDown}
          onPointerMove={onScrubMove}
          role="slider"
          tabIndex={0}
          aria-label={`${isAfter ? 'Safe' : 'Original'} render position`}
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={Math.round(positionT * 100)}
          onKeyDown={(event) => {
            if (event.key === 'ArrowRight') onSeek(Math.min(1, positionT + 0.01));
            if (event.key === 'ArrowLeft') onSeek(Math.max(0, positionT - 0.01));
          }}
          className="relative -my-2 cursor-pointer py-2"
        >
          <div className="relative h-1 w-full rounded-bar bg-edge">
          <div
            className="absolute inset-y-0 left-0 rounded-bar"
            style={{
              width: `${positionT * 100}%`,
              background: isAfter ? SIGNAL_HEX.clear : SIGNAL_HEX.critical,
            }}
          />
          {/* remediated spans — only meaningful on the safe render */}
          {isAfter && hasVerifiedAfter &&
            ops.map((op) => (
              <span
                key={op.index}
                className="absolute inset-y-0 rounded-bar"
                style={{
                  left: `${(op.startMs / duration) * 100}%`,
                  width: `${Math.max(0.5, ((op.endMs - op.startMs) / duration) * 100)}%`,
                  background: SIGNAL_HEX.clear,
                }}
                title={`${op.op} · ${formatTimecode(op.startMs)}`}
              />
            ))}
          <span
            className="absolute top-1/2 h-2.5 w-2.5 -translate-x-1/2 -translate-y-1/2 rounded-full border border-void"
            style={{
              left: `${positionT * 100}%`,
              background: isAfter ? SIGNAL_HEX.clear : SIGNAL_HEX.critical,
            }}
          />
          </div>
        </div>
      </div>

      <div className="flex shrink-0 items-center gap-3 px-3 py-2.5 text-inkFaint">
        <button type="button" onClick={() => void togglePlayback()} title={playing ? 'Pause' : 'Play'}>
          {playing ? <Pause className="h-3.5 w-3.5" fill="currentColor" /> : <Play className="h-3.5 w-3.5" fill="currentColor" />}
        </button>
        <button type="button" onClick={skip} title="Skip forward 10 seconds">
          <SkipForward className="h-3.5 w-3.5" />
        </button>
        <button
          type="button"
          onClick={() => {
            const video = videoRef.current;
            if (!video) return;
            video.muted = !muted;
            setMuted(video.muted);
          }}
          title={muted ? 'Unmute' : 'Mute'}
        >
          <Volume2 className={`h-3.5 w-3.5 ${muted ? 'opacity-50' : ''}`} />
        </button>
        <span className="ml-auto num text-[9px] uppercase tracking-[0.08em]">
          {isAfter ? (hasVerifiedAfter ? `${ops.length} spans remediated` : 'verification pending') : `${findingCount} findings`}
        </span>
      </div>
    </Panel>
  );
}

function FixBridge() {
  const applied = useAnalysis((s) => s.applied);
  const hasVerifiedAfter = useAnalysis((s) => s.hasVerifiedAfter);
  const setApplied = useAnalysis((s) => s.setApplied);

  const before = useAnalysis((s) => s.before.scores);
  const after = useAnalysis((s) => s.after.scores);
  const delta = after.overall - before.overall;
  const ops = useAnalysis((s) => s.before.remediation.ops.length);

  return (
    <div className="flex min-w-0 flex-col items-center justify-center gap-3 px-2">
      <span className="num text-data text-inkDim">{ops} operations</span>

      {/* a thin arrow, not a glowing shield */}
      <div className="relative h-px w-full bg-edge">
        <span
          className="absolute -top-0.5 h-1.5 w-1.5 rounded-full"
          style={{ left: '20%', background: SIGNAL_HEX.clear }}
        />
        <span
          className="absolute -top-0.5 h-1.5 w-1.5 rounded-full opacity-50"
          style={{ left: '55%', background: SIGNAL_HEX.clear }}
        />
        <span
          className="absolute -right-1 -top-1 h-0 w-0"
          style={{
            borderTop: '4px solid transparent',
            borderBottom: '4px solid transparent',
            borderLeft: `5px solid ${SIGNAL_HEX.clear}`,
          }}
        />
      </div>

      <div className="flex flex-col items-center gap-0.5">
        <span className="num text-data" style={{ color: SIGNAL_HEX.clear }}>
          {hasVerifiedAfter ? `${delta >= 0 ? '+' : ''}${delta} readiness` : 'verification required'}
        </span>
        <span className={`num text-[9px] text-inkFaint ${hasVerifiedAfter ? '' : 'invisible'}`}>
          <span style={{ color: readinessHex(before.overall) }}>{before.overall}</span>
          {' → '}
          <span style={{ color: readinessHex(after.overall) }}>{after.overall}</span>
        </span>
      </div>

      <button
        type="button"
        onClick={() => setApplied(!applied)}
        disabled={!hasVerifiedAfter}
        className="flex w-full items-center justify-center gap-1.5 rounded-chip border px-2.5 py-2 text-micro uppercase transition-colors duration-fast"
        style={{
          color: !hasVerifiedAfter ? '#58647A' : applied ? '#8A97AE' : SIGNAL_HEX.clear,
          borderColor: !hasVerifiedAfter || applied ? '#26324A' : `${SIGNAL_HEX.clear}66`,
          background: !hasVerifiedAfter || applied ? 'transparent' : `${SIGNAL_HEX.clear}14`,
        }}
      >
        {applied ? <Pause className="h-3 w-3" /> : <Play className="h-3 w-3" />}
        {!hasVerifiedAfter ? 'Await verification' : applied ? 'View original' : 'View verified render'}
      </button>

      <span className={`num text-center text-[9px] uppercase leading-relaxed tracking-[0.06em] text-inkFaint ${hasVerifiedAfter ? '' : 'invisible'}`}>
        {VERDICT_META[before.verdict].label}
        <br />↓<br />
        {VERDICT_META[after.verdict].label}
      </span>
    </div>
  );
}

export function BeforeAfterPlayers() {
  // One position, two players — and now one position shared with the whole
  // deck. Held in the store rather than here so a finding, a timeline band
  // or a piece of evidence can move the video; while it was local state,
  // clicking a finding could highlight it and nothing else.
  const before = useAnalysis((s) => s.before);
  const after = useAnalysis((s) => s.after);
  const playheadMs = useAnalysis((s) => s.playheadMs);
  const seekTo = useAnalysis((s) => s.seekTo);

  const duration = before.video.durationMs || 1;
  const positionT = Math.min(1, Math.max(0, playheadMs / duration));
  const setPositionT = (t: number) => seekTo(t * duration);

  return (
    <>
      <PlayerFrame
        report={before}
        variant="BEFORE"
        positionT={positionT}
        onSeek={setPositionT}
      />
      <FixBridge />
      <PlayerFrame
        report={after}
        variant="AFTER"
        positionT={positionT}
        onSeek={setPositionT}
      />
    </>
  );
}
