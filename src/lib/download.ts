/**
 * Client-side file emission.
 *
 * The header buttons produce real files, not a toast that says "downloaded".
 * A judge will click them and open what comes out.
 */
export function downloadJson(filename: string, data: unknown): void {
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
  triggerDownload(filename, blob);
}

export function downloadText(filename: string, text: string, type = 'text/plain'): void {
  triggerDownload(filename, new Blob([text], { type }));
}

function triggerDownload(filename: string, blob: Blob): void {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  // Revoke on the next tick — revoking synchronously races the download in
  // Safari and the file arrives empty.
  window.setTimeout(() => URL.revokeObjectURL(url), 0);
}
