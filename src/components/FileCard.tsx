import { useState } from 'react';
import { Check, Copy, Film, Play } from 'lucide-react';
import { Field, Panel } from '@/components/ui';
import { useReport } from '@/store/analysis';
import { formatBytes, formatDuration, truncateHash } from '@/lib/time';

function Poster({ src, alt }: { src: string; alt: string }) {
  const [failed, setFailed] = useState(false);

  return (
    <div className="relative aspect-video w-full shrink-0 overflow-hidden rounded-panel border border-edge bg-abyss">
      {!failed ? (
        <img
          src={src}
          alt={alt}
          className="h-full w-full object-cover"
          onError={() => setFailed(true)}
        />
      ) : (
        <div className="flex h-full w-full flex-col items-center justify-center gap-1.5 text-inkFaint">
          <Film className="h-5 w-5" strokeWidth={1.5} />
          <span className="num text-[9px] uppercase tracking-[0.1em]">no poster frame</span>
        </div>
      )}
      <div className="absolute inset-0 flex items-center justify-center">
        <span className="flex h-8 w-8 items-center justify-center rounded-full border border-edgeHi bg-void/70">
          <Play className="ml-0.5 h-3 w-3 text-ink" fill="currentColor" />
        </span>
      </div>
    </div>
  );
}

function HashField({ hash }: { hash: string }) {
  const [copied, setCopied] = useState(false);

  return (
    <div className="flex min-w-0 flex-col gap-1.5">
      <span className="text-label uppercase text-inkFaint">Attestation Hash</span>
      <button
        type="button"
        onClick={() => {
          void navigator.clipboard?.writeText(hash);
          setCopied(true);
          window.setTimeout(() => setCopied(false), 1400);
        }}
        className="group flex items-center gap-1.5 text-left"
        title={hash}
      >
        <span className="num truncate text-data text-ink">{truncateHash(hash, 14, 6)}</span>
        {copied ? (
          <Check className="h-3 w-3 shrink-0 text-sig-clear" />
        ) : (
          <Copy className="h-3 w-3 shrink-0 text-inkFaint transition-colors duration-instant group-hover:text-ink" />
        )}
      </button>
    </div>
  );
}

export function FileCard() {
  const report = useReport();
  const { video, meta } = report;

  const analyzed = new Date(meta.analyzedAt);
  const analyzedLabel = `${analyzed.toLocaleDateString('en-GB', {
    day: '2-digit',
    month: 'short',
    year: 'numeric',
  })} · ${analyzed.toISOString().slice(11, 19)}`;

  return (
    <Panel className="min-w-0">
      <div className="flex min-w-0 gap-4">
        <div className="w-[38%] max-w-[220px] shrink-0">
          <Poster src={video.posterUrl} alt={`Poster frame from ${video.filename}`} />
        </div>

        <div className="flex min-w-0 flex-1 flex-col justify-between gap-4">
          <h1 className="truncate text-h1 text-ink">{video.filename}</h1>

          <div className="grid grid-cols-2 gap-x-4 gap-y-3 sm:grid-cols-3 xl:grid-cols-5">
            <Field label="Duration" value={formatDuration(video.durationMs)} />
            <Field label="Resolution" value={`${video.width}×${video.height}`} />
            <Field label="FPS" value={video.fps} />
            <Field label="Audio" value={`${video.sampleRate / 1000} kHz ${video.audioCodec}`} />
            <Field label="Size" value={formatBytes(video.sizeBytes)} />
          </div>

          <div className="grid grid-cols-2 gap-x-4 gap-y-3 border-t border-edge pt-3 xl:grid-cols-4">
            <Field label="Analyzed On" value={analyzedLabel} />
            <Field label="Policy Version" value={meta.policyVersion} />
            <Field label="Engine Version" value={`PREFLIGHT v${meta.engineVersion}`} />
            <HashField hash={meta.attestationHash} />
          </div>
        </div>
      </div>
    </Panel>
  );
}
