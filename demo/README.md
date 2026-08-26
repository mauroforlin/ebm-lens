# Demo

A read-only, static build of the app: fifteen real, pre-recorded runs instead
of a live backend, so anyone can see EBM Lens work without cloning the repo
or holding an API key.

It's the same frontend as the real app - only the search call is swapped for
a recorded answer, replayed at a watchable pace. The sources, summaries,
cost and timings shown are exactly what the pipeline produced; only the
progress animation is sped up. A change to the real UI reaches the demo
automatically on the next build.

Every one of the fifteen questions is read and judged by hand before being
committed - relevant sources, grounded claims, nothing broken or empty.

```bash
python demo/build.py
python -m http.server -d _site 8080   # http://localhost:8080
```

Re-recording fixtures needs an `OPENROUTER_API_KEY`, costs a few dollars, and
takes 30-60 minutes: `python demo/scripts/record_demo.py`.

Deploys automatically to GitHub Pages on every push to `main`.
