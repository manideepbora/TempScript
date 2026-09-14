## macOS setup

```bash
python3 -m venv venv && source venv/bin/activate
python -m pip install torch torchvision open_clip_torch pillow pillow-heif imagehash opencv-python numpy tqdm
python photo_triage.py ~/Pictures
```

## Windows setup (PowerShell)

```powershell
py -m venv venv
.\venv\Scripts\Activate.ps1
py -m pip install torch torchvision open_clip_torch pillow pillow-heif imagehash opencv-python numpy tqdm
py photo_triage.py "$env:USERPROFILE\Pictures"
```

If PowerShell blocks the activation script, run the following once for the
current terminal, then repeat the activation command:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Use `--no-clip` to skip the optional CLIP stage and its first-run model
download:

```powershell
py photo_triage.py "$env:USERPROFILE\Pictures" --no-clip
```

## Optional local LLM judge

Install [Ollama](https://ollama.com) on the machine that will process your
photos, then download a vision-capable model. This model runs on your computer;
the script sends resized previews only to Ollama's loopback API at
`http://127.0.0.1:11434`, never to a cloud service.

### macOS

```bash
ollama pull qwen2.5vl:7b
ollama serve  # only when the Ollama service is not already running
python photo_triage.py ~/Pictures --llm-judge          # dry run
python photo_triage.py ~/Pictures --llm-judge --apply  # move reviewed rejects
```

### Windows (PowerShell)

```powershell
ollama pull qwen2.5vl:7b
ollama serve  # only when the Ollama service is not already running
py photo_triage.py "$env:USERPROFILE\Pictures" --llm-judge          # dry run
py photo_triage.py "$env:USERPROFILE\Pictures" --llm-judge --apply  # move reviewed rejects
```

The model subjectively selects the strongest composition, expression, moment,
focus, and exposure. Use another downloaded local vision model with
`--llm-model MODEL_NAME`, for example `--llm-model llava`. Each group is capped
at the eight strongest metric candidates, and normal metric scoring is used if
the local model cannot respond.

### Reject individually bad photos

`--llm-judge` only compares similar images, so it cannot reject a poor photo
that is not in a group. Add `--llm-quality` to inspect every remaining photo for
clear failures such as closed eyes, missed focus, obstructed subjects,
accidental framing, or severe exposure problems:

```bash
python photo_triage.py ~/Pictures --llm-judge --llm-quality
```

```powershell
py photo_triage.py "$env:USERPROFILE\Pictures" --llm-judge --llm-quality
```

This makes one local model request per image, so it can be slow. Always begin
with a dry run and review `photo_triage_report.csv`; subjective LLM rejections
are recorded as `LLM: ...` in the report's reason column.

It defaults to a dry run: it writes photo_triage_report.csv next to your photos and moves nothing. Open that in Numbers, sort by verdict, and see whether the calls look right. When you're happy, re-run with --apply and rejects get moved into _photo_review/, sorted by reason, with duplicate groups kept together in their own subfolders so you can compare the rejected shot against the one it kept.

Two things worth tuning after the first pass. --blur-threshold is the one that'll need adjusting — 45 is a reasonable default, but soft-focus portraits and intentionally shallow depth-of-field shots score low, so check what landed in low_quality/ and raise or lower it. And --clip-threshold at 0.94 is deliberately strict; drop it to 0.90 if burst sequences aren't getting grouped, but below about 0.88 it starts lumping together photos that merely share a subject.

The CLIP weights download once from Hugging Face on first run. After that, unplug your network and it still works — no image data leaves your machine at any point.