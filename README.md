# Vehicle Number Extractor

Upload a video → get every vehicle number plate detected in it as an Excel file.
Built for ground auditors across different cities: they open a link on their
phone, upload a video, and download an Excel sheet with the vehicle numbers.

## What this can and can't do

This uses computer vision + OCR running on the server — no manual review needed
unless a plate is unclear. It works best when the plate is reasonably visible,
not too small, and not badly blurred, in decent lighting.

**It will not be 100% accurate.** Automatic plate recognition from casual phone
video never is. The output Excel highlights low-confidence reads in yellow so
you know exactly which ones to double check by eye.

## Deploying for free (Render.com)

Render's free tier costs nothing and needs no credit card, with two trade-offs
worth knowing before you commit auditors to it:
- **Cold start:** if nobody's used it in the last 15 minutes, the next upload
  takes 30–60 seconds to "wake up" the server before it starts. The app shows
  a "waking up" message during this so it doesn't look frozen.
- **Limited resources:** 512 MB RAM, a fraction of a CPU core. Fine for a
  handful of videos a day, but processing will be slower than a paid tier, and
  very large or very long videos may fail. Upload size is capped at 150 MB —
  ask auditors to trim videos to just the relevant footage if needed.

If this ever outgrows the free tier (more auditors, bigger videos, faster
turnaround needed), the same files deploy directly to Render's $7/month
Starter tier with no code changes — just pick "Starter" instead of "Free"
when creating the service.

### Step-by-step

**1. Put the code on GitHub**
1. Go to github.com and sign up if you don't have an account (free)
2. Click "New repository," name it e.g. `vehicle-number-extractor`, keep it
   Public or Private (either works), and click "Create repository"
3. On the new repo page, click "uploading an existing file"
4. Drag in **all the files and folders** from this zip (unzip it first) and
   commit

**2. Deploy to Render**
1. Go to render.com and sign up free (using "Sign in with GitHub" is easiest)
2. Click "New +" → "Web Service"
3. Connect your GitHub account if asked, then select the
   `vehicle-number-extractor` repo you just created
4. Render will detect the `Dockerfile` automatically — leave build settings
   as-is
5. Under **Instance Type**, select **Free**
6. Click "Create Web Service"
7. Wait for the first build to finish (5–10 minutes the first time — it's
   installing Tesseract OCR and Python packages)

**3. Get your link**
Once deployed, Render shows a URL like:
```
https://vehicle-number-extractor.onrender.com
```
That's the link you send auditors in any city. They open it on their phone
browser, no app install needed, upload the video, and download the Excel file
straight to their phone — from there they can WhatsApp or email it to you.

### Updating the app later
If you ever want to change something (e.g. plate format rules), edit the file
in your GitHub repo (or push a new version) — Render automatically redeploys
whenever the repo changes.

## Notes
- The app currently expects Indian-style plate formats (e.g. `MH12AB1234`).
  Say the word if you need this adjusted.
- Each auditor's upload is processed one at a time on the free tier — if two
  people upload at the exact same moment, the second one queues briefly.
  This is a non-issue at 1–5 videos/day.
- If a video is failing to process (too large, too long, or the server runs
  out of memory), trimming it to a shorter clip usually fixes it.
