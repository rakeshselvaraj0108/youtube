import { useEffect, useState } from 'react';

/**
 * A real still frame, grabbed directly from a video the browser can reach.
 *
 * The engine embeds a poster as a data URI when it built the report with
 * `embed_media` on — but `apply_fix`'s internal before/after reports were
 * built with it off to keep the comparison payload light, and those reports
 * still land in `RUNS_DIR` and stay browsable from the deck's own "past
 * runs" list. A reader who opens one should not see a placeholder for the
 * one thing that says which video it even is, and re-running the whole
 * analysis just to get a thumbnail would be absurd.
 *
 * So when there is no embedded poster, this seeks an actual `<video>`
 * element to the same 10%-of-duration point the engine's own poster
 * extraction uses and reads the decoded frame back through a canvas. It is
 * a real pixel from the real file, not a placeholder — and if the source
 * cannot be reached at all (a standalone `report.html` with no video beside
 * it), it fails silently and the caller falls back to the honest empty
 * state that already exists for exactly that case.
 */

// Matches `preflight/ingest/frames.py::extract_poster` — not frame 0, which
// is very often black, a fade-in, or a title card.
const SEEK_FRACTION = 0.1;

const cache = new Map<string, string | null>();

export function useVideoFrame(srcUrl: string | null | undefined): string | null {
  const [frame, setFrame] = useState<string | null>(() =>
    srcUrl ? cache.get(srcUrl) ?? null : null,
  );

  useEffect(() => {
    if (!srcUrl) {
      setFrame(null);
      return;
    }
    if (cache.has(srcUrl)) {
      setFrame(cache.get(srcUrl) ?? null);
      return;
    }

    let cancelled = false;
    const video = document.createElement('video');
    video.crossOrigin = 'anonymous';
    // 'metadata' is not enough: the container can report a seek as complete
    // — firing `seeked` — before the decoder has actually produced a frame
    // for the video element to paint, and a canvas capture taken at that
    // point reads back solid black. This first shipped that way and it was
    // wrong for every source tested: `auto` is what actually gets a decoded
    // frame into the paint pipeline.
    video.preload = 'auto';
    video.muted = true;
    video.playsInline = true;

    const cleanup = () => {
      video.removeAttribute('src');
      video.load();
    };

    const fail = () => {
      if (cancelled) return;
      cache.set(srcUrl, null);
      setFrame(null);
      cleanup();
    };

    const capture = () => {
      if (cancelled) return;
      try {
        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const ctx = canvas.getContext('2d');
        if (!ctx || canvas.width === 0 || canvas.height === 0) {
          fail();
          return;
        }
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        // Throws on a tainted canvas — a cross-origin source that answered
        // without CORS headers. Caught below rather than left to crash the
        // render; the honest fallback is a missing thumbnail, not a broken
        // page.
        const dataUri = canvas.toDataURL('image/jpeg', 0.82);
        cache.set(srcUrl, dataUri);
        setFrame(dataUri);
      } catch {
        fail();
      } finally {
        cleanup();
      }
    };

    video.addEventListener('loadedmetadata', () => {
      if (cancelled || !Number.isFinite(video.duration) || video.duration <= 0) {
        fail();
        return;
      }
      video.currentTime = video.duration * SEEK_FRACTION;
    });

    video.addEventListener('seeked', () => {
      if (cancelled) return;
      // `requestVideoFrameCallback` exists precisely to answer "a frame has
      // actually been decoded and is ready to paint" — the purpose-built
      // replacement for guessing with `seeked` alone. Where it is not
      // available (Safari before 16.4), two animation frames is the
      // documented workaround: the first lets the compositor pick up the
      // seek, the second is where the newly decoded frame is actually
      // presented.
      const withRvfc = video as HTMLVideoElement & {
        requestVideoFrameCallback?: (cb: () => void) => number;
      };
      if (typeof withRvfc.requestVideoFrameCallback === 'function') {
        withRvfc.requestVideoFrameCallback(capture);
      } else {
        requestAnimationFrame(() => requestAnimationFrame(capture));
      }
    });

    video.addEventListener('error', fail);
    video.src = srcUrl;

    return () => {
      cancelled = true;
      cleanup();
    };
  }, [srcUrl]);

  return frame;
}
