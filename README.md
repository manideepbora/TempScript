Setup:

bash
python3 -m venv venv && source venv/bin/activate
pip install torch torchvision open_clip_torch pillow pillow-heif imagehash opencv-python numpy tqdm
python photo_triage.py ~/Pictures

It defaults to a dry run: it writes photo_triage_report.csv next to your photos and moves nothing. Open that in Numbers, sort by verdict, and see whether the calls look right. When you're happy, re-run with --apply and rejects get moved into _photo_review/, sorted by reason, with duplicate groups kept together in their own subfolders so you can compare the rejected shot against the one it kept.

Two things worth tuning after the first pass. --blur-threshold is the one that'll need adjusting — 45 is a reasonable default, but soft-focus portraits and intentionally shallow depth-of-field shots score low, so check what landed in low_quality/ and raise or lower it. And --clip-threshold at 0.94 is deliberately strict; drop it to 0.90 if burst sequences aren't getting grouped, but below about 0.88 it starts lumping together photos that merely share a subject.

The CLIP weights download once from Hugging Face on first run. After that, unplug your network and it still works — no image data leaves your machine at any point.