"""List the model slugs the configured endpoint actually serves.

NVIDIA's catalogue rotates — entries are renamed and retired — so a slug that
worked last month can 404 or 503 today. Run this on build day and put the
survivors in .env, never in a call site.

    python scripts/list_models.py [filter]
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request

from preflight.config import Settings


def main() -> int:
    settings = Settings.load()
    if not settings.api_key:
        print("NVIDIA_API_KEY is not set.")
        return 2

    needle = (sys.argv[1] if len(sys.argv) > 1 else "").lower()
    url = f"{settings.base_url.rstrip('/')}/models"
    request = urllib.request.Request(
        url, headers={"Authorization": f"Bearer {settings.api_key}"}
    )

    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        print(f"HTTP {exc.code}: {exc.read().decode('utf-8', errors='replace')[:300]}")
        return 3

    ids = sorted(item.get("id", "") for item in payload.get("data", []))
    matched = [i for i in ids if needle in i.lower()] if needle else ids

    print(f"{len(ids)} models served; showing {len(matched)}")
    print()
    for model_id in matched:
        print(f"  {model_id}")

    configured = settings.models.to_json()
    print()
    print("configured slugs:")
    for role, slug in configured.items():
        if role == "asr":
            continue
        mark = "OK  " if slug in ids else "GONE"
        print(f"  [{mark}] {role:<12} {slug}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
