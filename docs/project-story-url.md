# Project story slideshow — published URL

**Live URL:** http://dorby-project-story-411960113601.s3-website-us-east-1.amazonaws.com/

Plain-language, chronological walkthrough of the project (July 16–August 4, 2026), built as a
keynote-style slideshow with light/dark mode (respects `prefers-color-scheme`, persists in
`localStorage`). Content refreshed 2026-08-04 (query-weighted lever, train-vs-eval text,
split/field-gate losses, nomad-drift calibration, judge prompt evolution, experiment deep
links). Source file: [`docs/html/project-story.html`](html/project-story.html).

## Hosting details

| | |
|---|---|
| Bucket | `dorby-project-story-411960113601` |
| Region | `us-east-1` |
| AWS account | `411960113601` (`tf_provisioner` profile) |
| Mode | S3 static website hosting, public read on objects |
| Object | `index.html` (single self-contained file, no assets) |
| Cache | `no-cache, max-age=60` so redeploys show up immediately |

Cost is effectively zero — one HTML object plus request charges.

Note this is an unencrypted `http://` S3 website endpoint (the website-hosting endpoint does
not support HTTPS). Put CloudFront in front of it if an `https://` link is ever needed.

## Redeploy after editing the slideshow

```bash
export AWS_PROFILE=tf_provisioner AWS_DEFAULT_REGION=us-east-1
aws s3 cp docs/html/project-story.html \
  s3://dorby-project-story-411960113601/index.html \
  --content-type "text/html; charset=utf-8" \
  --cache-control "no-cache, max-age=60"
```
