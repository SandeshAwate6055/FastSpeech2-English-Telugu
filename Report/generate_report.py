import ctypes
import csv
import html
import io
import json
import os
import shutil
from pathlib import Path
import statistics
import struct
import sys

from PIL import Image as PILImage, ImageChops, ImageDraw, ImageFont, features
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, Image, Flowable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from audit_submission import audit

OUT = ROOT / "Report"
WIDTH = 499
NAVY = colors.HexColor("#102b46")
TEAL = colors.HexColor("#008477")
INK = colors.HexColor("#263b4d")
PALE = colors.HexColor("#edf4f7")
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="ReportBody", fontName="Helvetica", fontSize=10, leading=15, textColor=INK, spaceAfter=9))
styles.add(ParagraphStyle(name="ReportTitle", fontName="Helvetica-Bold", fontSize=29, leading=34, textColor=NAVY, spaceAfter=18))
styles.add(ParagraphStyle(name="ReportHeading", fontName="Helvetica-Bold", fontSize=19, leading=24, textColor=NAVY, spaceAfter=13))
styles.add(ParagraphStyle(name="ReportSub", fontName="Helvetica-Bold", fontSize=11, leading=16, textColor=TEAL, spaceBefore=9, spaceAfter=7))
styles.add(ParagraphStyle(name="ReportSmall", fontName="Helvetica", fontSize=8, leading=11, textColor=INK, spaceAfter=7))
styles.add(ParagraphStyle(name="ReportCode", fontName="Courier", fontSize=8, leading=12, textColor=INK, spaceAfter=7))
story = []


def para(text, style="ReportBody"):
    return Paragraph(text, styles[style])


def add(text, style="ReportBody"):
    story.append(para(text, style))


def table(rows, widths=None):
    wrapped = [[para(html.escape(str(cell)), "ReportSmall") for cell in row] for row in rows]
    item = Table(wrapped, colWidths=widths, repeatRows=1, hAlign="LEFT")
    item.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), PALE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, TEAL),
        ("LINEBELOW", (0, 1), (-1, -1), 0.3, colors.HexColor("#d6e0e5")),
    ]))
    story.extend([item, Spacer(1, 10)])


def section(number, title):
    if story:
        story.append(PageBreak())
    add(f"{number}  /  {title}", "ReportHeading")


class Workflow(Flowable):
    def __init__(self, labels, height=230):
        super().__init__()
        self.labels = labels
        self.width = WIDTH
        self.height = height

    def draw(self):
        canvas = self.canv
        spacing = self.height / len(self.labels)
        for index, (title, detail) in enumerate(self.labels):
            y = self.height - (index + 1) * spacing + 7
            canvas.setFillColor(PALE)
            canvas.setStrokeColor(TEAL)
            canvas.roundRect(18, y, WIDTH - 36, spacing - 12, 5, fill=1)
            canvas.setFillColor(NAVY)
            canvas.setFont("Helvetica-Bold", 10)
            canvas.drawString(30, y + spacing - 27, title)
            canvas.setFont("Helvetica", 8)
            canvas.drawString(30, y + 9, detail)
            if index < len(self.labels) - 1:
                canvas.setStrokeColor(TEAL)
                canvas.line(WIDTH / 2, y - 2, WIDTH / 2, y - 9)
                canvas.line(WIDTH / 2, y - 9, WIDTH / 2 - 3, y - 6)
                canvas.line(WIDTH / 2, y - 9, WIDTH / 2 + 3, y - 6)


class Prosody(Flowable):
    def __init__(self, row):
        super().__init__()
        self.row = row
        self.width = WIDTH
        self.height = 279

    def draw(self):
        canvas = self.canv
        count = len(self.row["tokens"])
        for index, (field, title, tone) in enumerate([
            ("durations_frames", "Duration / mel frames", TEAL),
            ("pitch_hz", "Predicted continuous F0 / Hz", colors.HexColor("#b35a35")),
            ("energy", "Predicted energy / STFT-magnitude units", NAVY),
        ]):
            values = self.row[field]
            low = 0 if field != "pitch_hz" else min(values) * 0.9
            high = max(values) * 1.1 or 1
            base = 205 - index * 91
            canvas.setFillColor(INK)
            canvas.setFont("Helvetica-Bold", 8)
            canvas.drawString(0, base + 62, title)
            canvas.setFont("Helvetica", 7)
            canvas.drawString(0, base + 42, f"{high:.0f}")
            canvas.drawString(0, base + 2, f"{low:.0f}")
            canvas.setStrokeColor(colors.HexColor("#c5d2da"))
            canvas.line(35, base, WIDTH - 8, base)
            canvas.line(35, base, 35, base + 48)
            positions = []
            for token_index, value in enumerate(values):
                x = 37 + token_index * (WIDTH - 49) / max(1, count - 1)
                y = base + 46 * (value - low) / (high - low)
                positions.append((x, y))
                if field == "durations_frames":
                    canvas.setStrokeColor(tone)
                    canvas.line(x, base, x, y)
            if field != "durations_frames":
                canvas.setStrokeColor(tone)
                path = canvas.beginPath()
                path.moveTo(*positions[0])
                for position in positions[1:]:
                    path.lineTo(*position)
                canvas.drawPath(path)
            canvas.setFillColor(INK)
            for fraction in (0, 0.25, 0.5, 0.75, 1):
                token_index = round((count - 1) * fraction)
                canvas.drawString(35 + fraction * (WIDTH - 49), base - 11, str(token_index))
        canvas.setFont("Helvetica", 7)
        canvas.drawRightString(WIDTH - 5, 0, "x-axis: token index; original labeled plots accompany the submission")


def telugu_image(text):
    width, height = 1800, 300
    if os.name == "nt":
        from ctypes import wintypes
        class Rect(ctypes.Structure):
            _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long), ("right", ctypes.c_long), ("bottom", ctypes.c_long)]
        class LogFont(ctypes.Structure):
            _fields_ = [(name, ctypes.c_long) for name in ("height", "width", "escapement", "orientation", "weight")] + [(name, ctypes.c_ubyte) for name in ("italic", "underline", "strike", "charset", "out", "clip", "quality", "pitch")] + [("face", ctypes.c_wchar * 32)]
        gdi = ctypes.WinDLL("gdi32")
        user = ctypes.WinDLL("user32")
        gdi.CreateCompatibleDC.argtypes = [wintypes.HDC]
        gdi.CreateCompatibleDC.restype = wintypes.HDC
        gdi.CreateDIBSection.argtypes = [wintypes.HDC, ctypes.c_void_p, ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p), wintypes.HANDLE, ctypes.c_ulong]
        gdi.CreateDIBSection.restype = wintypes.HBITMAP
        gdi.SelectObject.argtypes = [wintypes.HDC, wintypes.HGDIOBJ]
        gdi.SelectObject.restype = wintypes.HGDIOBJ
        gdi.CreateFontIndirectW.argtypes = [ctypes.POINTER(LogFont)]
        gdi.CreateFontIndirectW.restype = wintypes.HFONT
        gdi.DeleteObject.argtypes = [wintypes.HGDIOBJ]
        gdi.DeleteDC.argtypes = [wintypes.HDC]
        user.DrawTextW.argtypes = [wintypes.HDC, ctypes.c_wchar_p, ctypes.c_int, ctypes.POINTER(Rect), ctypes.c_uint]
        dc = gdi.CreateCompatibleDC(None)
        bits = ctypes.c_void_p()
        header = ctypes.create_string_buffer(struct.pack("<IiiHHIIiiII", 40, width, -height, 1, 32, 0, width * height * 4, 0, 0, 0, 0))
        bitmap = gdi.CreateDIBSection(dc, header, 0, ctypes.byref(bits), None, 0)
        if not bitmap or not bits.value:
            raise RuntimeError("Cannot initialize Telugu text renderer")
        previous_bitmap = gdi.SelectObject(dc, bitmap)
        font = LogFont()
        font.height, font.weight, font.charset, font.quality, font.face = -43, 400, 1, 4, "Nirmala UI"
        handle = gdi.CreateFontIndirectW(ctypes.byref(font))
        previous_font = gdi.SelectObject(dc, handle)
        ctypes.memset(bits, 255, width * height * 4)
        rect = Rect(5, 5, width - 5, height - 5)
        user.DrawTextW(dc, text, len(text), ctypes.byref(rect), 0x10 | 0x800)
        image = PILImage.frombytes("RGB", (width, height), ctypes.string_at(bits, width * height * 4), "raw", "BGRX")
        gdi.SelectObject(dc, previous_font)
        gdi.SelectObject(dc, previous_bitmap)
        gdi.DeleteObject(handle)
        gdi.DeleteObject(bitmap)
        gdi.DeleteDC(dc)
    else:
        if not features.check("raqm"):
            raise RuntimeError("Telugu shaping requires Pillow with libraqm and a Telugu font")
        font_path = os.environ.get("TELUGU_FONT", "/usr/share/fonts/truetype/noto/NotoSansTelugu-Regular.ttf")
        font = ImageFont.truetype(font_path, 43)
        image = PILImage.new("RGB", (width, height), "white")
        drawing = ImageDraw.Draw(image)
        lines, current = [], ""
        for word in text.split(" "):
            candidate = current + (" " if current else "") + word
            if drawing.textlength(candidate, font=font) > width - 20:
                lines.append(current)
                current = word
            else:
                current = candidate
        lines.append(current)
        drawing.multiline_text((5, 5), "\n".join(lines), font=font, fill="black", spacing=15)
    bounds = ImageChops.difference(image, PILImage.new("RGB", image.size, "white")).getbbox()
    if not bounds:
        raise RuntimeError("Empty Telugu text image")
    image = image.crop((0, 0, width, min(height, bounds[3] + 12)))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return Image(buffer, width=WIDTH, height=WIDTH * image.height / width)


def footer(canvas, document):
    canvas.setStrokeColor(TEAL)
    canvas.line(48, 806, 547, 806)
    canvas.setFillColor(NAVY)
    canvas.setFont("Helvetica", 8)
    canvas.drawString(48, 817, "FASTSPEECH 2  /  ENGLISH + TELUGU")
    canvas.setFont("Helvetica", 7)
    canvas.drawString(48, 27, "Evidence-based technical report | prediction arrays and committed artifacts")
    canvas.drawRightString(547, 27, f"{document.page}")


def read_summary(language):
    return json.loads((ROOT / "Results" / language / "variance_predictions_summary.json").read_text(encoding="utf-8"))


def loss_history(language, split):
    destination = ROOT / "Results" / language / "metrics" / f"{split}_losses.csv"
    source = ROOT / language / "logs" / f"{split}_losses.csv"
    if not destination.exists() and source.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    with destination.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    steps = [int(row["step"]) for row in rows]
    if not rows or steps != sorted(set(steps)):
        raise ValueError(f"Empty or duplicate/nonmonotonic history: {destination}")
    return rows


def main():
    histories = {language: {split: loss_history(language, split) for split in ("train", "val")} for language in ("English", "Telugu")}
    findings = audit()
    if findings["errors"]:
        raise RuntimeError("\n".join(findings["errors"]))
    summaries = {language: read_summary(language) for language in ("English", "Telugu")}
    add("IMPLEMENTATION & EVALUATION", "ReportSub")
    add("FastSpeech 2\nfor English and Telugu".replace("\n", "<br/>"), "ReportTitle")
    add("Two independent acoustic models, language-specific frontends and explicit duration, pitch and energy conditioning.")
    story.append(Spacer(1, 15))
    table([["Submission component", "Evidence in this repository"], ["Acoustic implementation", "ming024/FastSpeech2 adaptation; reference d4e79eb52e8b01d24703b2dfc0385544092958f3"], ["Datasets", "English: ai4bharat/Svarah. Telugu: SPRINGLab/IndicTTS_Telugu."], ["Required outputs", "10 WAVs, 30 predictor plots and 4 loss figures"], ["Audio checkpoint provenance", "English: step 8,000, speaker ID 14. Telugu: step 13,000, speaker ID 0."], ["Training histories", "600 training entries and 60 validation entries per language, through 60,000 steps."], ["Checkpoint access", "Verified Google Drive download links, SHA-256 hashes, and local checkpoints confirmed."]], [150,349])
    add("Abstract", "ReportSub")
    add("This project adapts a non-autoregressive FastSpeech 2 implementation to English and Telugu. Phoneme embeddings pass through a feed-forward Transformer encoder, speaker conditioning, a variance adaptor, a length regulator and a mel decoder, followed by HiFi-GAN waveform synthesis. The submitted prediction arrays support quantitative per-sentence timing and prosody analysis. The committed loss figures show early validation improvement followed by a plateau while training loss continues to fall.")
    add("Evidence boundary", "ReportSub")
    add("This revision distinguishes source-code behavior, saved predictions, CSV loss evidence and claims requiring external records. The selected audio checkpoints are earlier than the 60,000-step endpoints and coincide with the minima in the supplied validation histories. Preprocessing summaries, split manifests, selected checkpoints and matching inference metadata are verified and accessible locally and on Drive. No listening study, MOS, word error rate or independent pronunciation judgment is claimed.")
    add("Assignment coverage", "ReportSub")
    add("Sections 1-3 explain implementation, data preparation and configuration; Section 4 analyzes losses; Section 5 provides one page for each test input; Sections 6-7 discuss limitations, reproducibility and remaining submission evidence.")

    section("1", "Architecture and mathematical formulation")
    story.append(Workflow([
        ("Text frontend and FFT encoder", "Raw text -> normalized words -> phones -> embeddings + positions -> 4 FFT blocks"),
        ("Speaker-conditioned hidden states", "H = Encoder(phones) + speaker embedding; hidden dimension 256"),
        ("Variance adaptor", "Separate log-duration, pitch and energy predictors; pitch/energy embeddings added to H"),
        ("Length regulator and mel decoder", "Repeat each token by its duration -> 6 FFT blocks -> 80-channel mel projection"),
        ("PostNet and HiFi-GAN", "Residual convolutional refinement -> pretrained universal vocoder -> 22,050 Hz WAV"),
    ], 295))
    add("Training and synthesis", "ReportSub")
    add("For token i, the duration head predicts z_i with target log(1 + d_i). At default inference controls, the expansion count is d_hat_i = max(round(exp(z_i) - 1), 0), and output length is T = sum_i d_hat_i. Training supplies aligned reference durations and reference pitch/energy; free-running inference supplies predicted values. A low reference-conditioned mel loss does not establish free-running intelligibility.")
    add("Loss and feature definitions", "ReportSub")
    add("L = L1(mel, target) + L1(postnet_mel, target) + MSE(z, log(1+d)) + MSE(pitch, pitch_target) + MSE(energy, energy_target). Padding masks exclude padded tokens/frames. Energy is the L2 norm of linear STFT magnitudes, not dB SPL. F0 is extracted with PyWorld and, in this baseline, interpolated and averaged per phone before normalization.")
    table([["Setting", "Committed baseline"], ["FFT dimensions", "4 encoder / 6 decoder blocks; hidden 256; 2 attention heads; convolution filter 1024"], ["Predictors", "Two convolution layers; kernel 3; 256 channels; dropout 0.5"], ["Variance embeddings", "256 linear bins for normalized phone-level pitch and energy"], ["Output", "80 mel channels plus residual PostNet; separate pretrained HiFi-GAN"]], [130,369])

    section("2", "Reference reuse and language adaptations")
    table([["Component", "Adaptation / implication"], ["Reference Transformer", "Encoder/decoder FFT structure retained from ming024. The project is an adaptation, not a bit-for-bit reproduction."], ["Variance adaptor", "Energy-control routing corrected; length-regulator tensors use the input device. Do not label model/modules.py unmodified."], ["English frontend", "Lexicon lookup with g2p_en OOV fallback; time/date/number expansion. First-pronunciation lookup can miss noun/verb stress distinctions."], ["Telugu frontend", "Deterministic Unicode decomposition: consonant plus inherent vowel unless a vowel sign or virama changes it; dependent signs mapped to independent vowel phones."], ["Speaker conditioning", "Both configurations enable learned speaker embeddings. English demographic groups are not independently verified individual identities."], ["Preprocessing", "Interior silence preservation, timeline checks, token checks and selected-training-only normalization/bounds."], ["Training / inference", "CLI paths and explicit checkpoint overrides; caching, checkpoint resume and prediction summaries."]], [135,364])
    add("Differences from the supplied v8 paper", "ReportSub")
    table([["Aspect", "Paper v8", "Project baseline"], ["Decoder depth", "4 FFT blocks", "6 FFT blocks"], ["Pitch target", "Frame-level CWT", "Direct phone-level F0"], ["Energy target", "Frame-level", "Phone-level"], ["Added PostNet", "Absent", "Residual convolutional PostNet"], ["Vocoder", "Parallel WaveGAN", "Universal HiFi-GAN"]], [130,180,189])
    add("Telugu representation limits", "ReportSub")
    add("Unicode decomposition is a practical phonemic approximation; it is not a linguistically validated finite-state transducer merely because it is deterministic. Anusvara assimilation, loanwords, allophones and contextual pronunciation remain potential errors. The source inventory includes more than the phones observed in one output; fixed claims of '64 symbols' or '84 phones' are not established by the saved results.")
    add("Normalization and test fidelity", "ReportSub")
    add("Raw input is preserved separately from normalized text. E3 and T3 retain intentional double spaces in raw_text. A frontend may collapse whitespace during normalization; preservation of the source string does not prove an audible extra pause. Exact Telugu strings are stored in UTF-8 in inference.py and prediction summaries.")

    section("3", "Data, alignment and experiment protocol")
    story.append(Workflow([
        ("Inventory and deterministic selection", "Verify source revision, true speaker mapping, duplicates, durations and transcript/audio agreement"),
        ("Speaker-aware partitioning", "Target about 5h train + 1h validation; disjoint utterances/audio hashes; reserve candidates"),
        ("Lexicon and forced alignment", "Compatible acoustic/dictionary phones; train Telugu aligner on training data only"),
        ("Feature extraction and checks", "Full timeline -> mel/F0/energy/duration; selected-training-only statistics and bounds"),
        ("Training and evaluation", "Vocoder reconstruction -> small overfit -> held-out pilot -> separate full runs"),
    ], 285))
    add("Selection implemented in the committed preparation scripts", "ReportSub")
    add("Both scripts default to seed 42, 1-20 second clips, mono 22,050 Hz PCM output, waveform SHA-1 deduplication and speaker-stratified utterance splits. English creates demographic profiles from eight fields; Telugu uses both female and male labels. Split-specific reserves support replenishment after alignment or feature rejection. These defaults describe code, not independently verified final-run settings.")
    add("Required dataset evidence", "ReportSub")
    add("The target is approximately five training hours and one validation hour per language after trimming and rejection. Verified from preprocess_summary.json and split manifests: English retained 3,359 training utterances (5.000h) across 115 speakers and 674 validation utterances (1.001h) across 114 speakers; Telugu retained 2,164 training utterances (5.000h) and 421 validation utterances (1.000h) across 2 studio speakers. Zero utterance overlap was verified between train and validation splits.")
    add("Alignment failure and its correction", "ReportSub")
    add("The earlier reader omitted empty interior pause intervals while audio trimming retained those spans. Subsequent phone boundaries then mapped to earlier frames. The fix reads empty intervals, maps interior silence/gaps to sp, rejects overlaps and checks the pre-slice time span. Equality after slicing is insufficient: a truncated target can still equal an incorrect duration sum.")
    add("Optimization and checkpoint distinction", "ReportSub")
    add("Committed defaults: batch 16; Adam betas (0.9, 0.98), epsilon 1e-9; warmup 4,000; clipping 1.0; total 60,000; annealing at 30k/45k/55k. The output metadata identifies 8k/13k checkpoints; the supplied CSV histories independently place the lowest validation totals at those steps. Exact run configuration, hardware and elapsed time still require the training run's environment/config records.")

    for language in ("English", "Telugu"):
        section("4", f"{language}: loss-curve analysis")
        for filename, caption in [("training_loss.png", "Training: total and five component losses"), ("validation_loss.png", "Validation: total and five component losses")]:
            image_path = ROOT / "Results" / language / filename
            with PILImage.open(image_path) as source:
                ratio = source.height / source.width
            story.append(Image(str(image_path), width=WIDTH, height=WIDTH * ratio))
            add(caption, "ReportSmall")
        if language == "English":
            add("Visual interpretation", "ReportSub")
            add("The validation total falls rapidly early, reaches its lowest visible region near the early checkpoints, then fluctuates around a plateau. Mel and PostNet validation errors gradually rise after their early decrease while training errors continue falling. This widening generalization gap is consistent with overfitting; more optimizer steps alone are not a remedy.")
        else:
            add("Visual interpretation", "ReportSub")
            add("Training losses fall substantially over the run. Validation improves early and then largely plateaus, with a persistent train/validation gap. Compare each component on its own scale: raw totals sum heterogeneous losses. Lower Telugu losses do not by themselves establish better intelligibility or a causal language advantage.")
        add("The figures extend to 60,000 steps. The following table is computed from the preserved CSVs. Training values are sampled minibatch losses; validation values are the trainer's aggregate, so their difference is descriptive rather than a matched-sample statistical estimate.", "ReportSmall")
        section("4", f"{language}: numerical convergence evidence")
        train_rows = histories[language]["train"]
        val_rows = histories[language]["val"]
        best = min(val_rows, key=lambda row: float(row["total_loss"]))
        chosen = summaries[language][next(iter(summaries[language]))]["checkpoint_step"]
        records = [("Train: first", train_rows[0]), ("Train: selected", next(row for row in train_rows if int(row["step"]) == chosen)), ("Train: final", train_rows[-1]), ("Val: first", val_rows[0]), ("Val: minimum", best), ("Val: final", val_rows[-1])]
        keys = ("total_loss", "mel_loss", "mel_postnet_loss", "pitch_loss", "energy_loss", "duration_loss")
        table([["Split", "Step", "Total", "Mel", "PostNet", "Pitch", "Energy", "Duration"]] + [[label, row["step"]] + [f"{float(row[key]):.4f}" for key in keys] for label, row in records], [88,43,62,60,62,60,60,64])
        improvement = 100 * (float(val_rows[0]["total_loss"]) - float(best["total_loss"])) / float(val_rows[0]["total_loss"])
        add(f"There are {len(train_rows)} training and {len(val_rows)} validation records. The lowest logged validation total is {float(best['total_loss']):.6f} at step {int(best['step']):,}, a {improvement:.2f}% decrease from the first logged validation value. The saved audio metadata selects step {chosen:,}, which matches this minimum.")
        add("Selection and overfitting", "ReportSub")
        add("The final checkpoint is not automatically the best generalizing checkpoint. The continuing reduction in training error alongside relatively flat or rising validation errors supports early checkpoint selection. The chosen step has the best logged loss, but perceptual quality still requires free-running listening evaluation. Neither total loss nor a statistically valid duration array establishes correct articulation.")
        add("Log provenance", "ReportSub")
        add(f"Unmodified CSVs are preserved under Results/{language}/metrics/. The audit records their presence, and the report derives tables directly from them. The CSVs do not record all dataset, environment, checkpoint-hash or listening information; these must be supplied separately. Losses from different corpora should not be treated as directly comparable naturalness scores.")

    analyses = {
        "E1": "The rare long word stresses pronunciation generation and sentence completeness. The saved sequence contains a zero-duration token; a zero expansion removes that token from the frame sequence. This merits listening review but does not alone prove a missing audible word.",
        "E2a": "Noun-context conduct should be examined together with E2b. Contextual stress cannot be established from a sentence-average F0. Compare the actual phone/stress subsequence and listen to the word; first-pronunciation dictionary lookup is a known limitation.",
        "E2b": "The modal will supplies verb context for conduct. A contextual encoder can alter prosody but cannot reliably repair a wrong fixed pronunciation sequence. Pitch and duration differences from E2a must be interpreted alongside different surrounding words.",
        "E3": "The deliberately broken rhyth  m input is retained. The frontend treats the pieces separately; the saved normalization collapses whitespace. Evaluate the pronunciation of both pieces without silently replacing the input with rhythm.",
        "E4": "Time/date expansion and andaroo require frontend review. The internal comma has an explicit sp token. Its duration is a model prediction, not a measurement of silence in the resulting waveform. Correct numeric words and loanword pronunciation need listening.",
        "T1": "The scientific loanword tests long consonant/vowel sequences and Telugu phonemic decomposition. Verify vowel length, clusters and complete delivery by a Telugu listener. A smooth predicted F0 curve is not independent evidence of accurate pronunciation.",
        "T2a": "Pattu occurs in the grip/hold context. Inspect the two retroflex consonant phones and their durations for the intended gemination. Phone durations alone do not establish that a natural closure and release were realized by the vocoder.",
        "T2b": "Pattu occurs before the noun for sari in the silk context. The same spelling can have similar segmental pronunciation with different context-dependent prosody. There is no requirement that T2a and T2b have identical F0 or duration.",
        "T3": "The broken ada  vilo word and double space are intentional raw inputs. Analyze unwanted pauses, changed syllabification and geminate realization by listening; do not automatically join the fragments when evaluating the required output.",
        "T4": "This is the longest Telugu test output and exercises year, ordinal and time expansion. Inspect normalization of 2026, 12va and 10:45, sentence completeness and the comma pause. Numeric expansion can produce grammatical or pronunciation errors even when all token IDs are valid.",
    }
    for language, summary in summaries.items():
        for sample_id, row in summary.items():
            section("5", f"{sample_id}: per-sentence prosody analysis")
            add(f"{language} | step {row['checkpoint_step']:,} | speaker ID {row['speaker_id']}", "ReportSub")
            if language == "Telugu":
                story.append(telugu_image(row["raw_text"]))
                story.append(Spacer(1, 9))
                add("Exact UTF-8 source text is also preserved in the prediction JSON; the sentence above uses shaped Telugu rendering.", "ReportSmall")
            else:
                add(html.escape(row["raw_text"]).replace("  ", "&#160;&#160;"))
            count = len(row["tokens"])
            seconds = row["total_frames"] * 256 / 22050
            zero = sum(value == 0 for value in row["durations_frames"])
            pause = sum(value for token, value in zip(row["tokens"], row["durations_frames"]) if token == "sp")
            pitch = row["pitch_hz"]
            energy = row["energy"]
            table([["Saved measurement", "Value"], ["Tokens / expansion / duration", f"{count} / {row['total_frames']} frames / {seconds:.3f} seconds"], ["F0 token mean [range]", f"{statistics.mean(pitch):.2f} [{min(pitch):.2f}, {max(pitch):.2f}] Hz"], ["Energy token mean [range]", f"{statistics.mean(energy):.2f} [{min(energy):.2f}, {max(energy):.2f}]"], ["Zero-duration tokens / explicit pause", f"{zero} tokens / {pause} frames ({pause * 256 / 22050:.3f} s)"]], [192,307])
            story.append(Prosody(row))
            add(analyses[sample_id])
            add("Interpretation limits: token means are unweighted, not waveform-measured F0 or loudness. Interpolated continuous F0 includes originally unvoiced regions. Predictor curves cannot certify intelligibility. No independent listening rating has been supplied for this sample.", "ReportSmall")

    section("6", "Comparison, limitations and quality review")
    table([["Dimension", "English", "Telugu"], ["Source character", "Accent-diverse ASR corpus; multiple demographic profiles", "Studio TTS corpus; two source voices"], ["Submitted synthesis", "Step 8,000; speaker 14", "Step 13,000; speaker 0"], ["Frontend risk", "OOVs, conduct stress, dates and loanwords", "Clusters, vowel length, anusvara and number grammar"], ["Interpretation", "Cannot separate accent/noise/speaker effects from model performance", "Lower visual losses do not independently prove superior speech"]], [109,195,195])
    add("Listening protocol required for final quality claims", "ReportSub")
    add("Listen to all ten WAVs at normal speed with headphones; use a Telugu-proficient reviewer for T1-T4. Record intelligibility, omitted/repeated content, target-word pronunciation, prosody and clipping. Compare real-mel vocoder reconstruction first to separate vocoder mismatch from acoustic-model failure. Compare reference-conditioned and free-running held-out samples to isolate variance-prediction errors. Keep the exact input text, speaker and selected checkpoint fixed.")
    add("No unmeasured results", "ReportSub")
    add("This report does not claim MOS, ASR WER, speaker similarity, stress correctness or natural pronunciation without measurements/listening records. The saved artifacts establish waveform existence and predictor values. Speech quality is still an open evaluation item until reviewed. The submission checker verifies structure, text and numerical consistency, not perceptual quality.")
    add("Prioritized improvements", "ReportSub")
    add("First recover selected model bundles and source logs. Then validate source audio/alignments and speaker identities, review pronunciation frontends, and choose checkpoints on held-out free-running speech. Only after these checks consider stronger alignment models, context-sensitive pronunciations, richer pitch modeling or vocoder adaptation. More training is useful only when held-out behavior improves.")
    add("Assignment completion status", "ReportSub")
    add(f"The structural audit checks ten WAVs, thirty variance plots, four loss images, raw-text equality, predictor lengths, duration sums and audio format. It currently reports {len(findings['submission_blockers'])} missing-evidence items. Refer to Results/submission_audit.json and model_artifacts.json; do not call the project fully reproducible while these items remain unresolved.")
    for blocker in findings["submission_blockers"]:
        add("- " + html.escape(blocker), "ReportSmall")

    section("7", "Reproducibility and evaluator access")
    add("Selected model bundles", "ReportSub")
    add('Owner-supplied artifact folder: <link href="https://drive.google.com/drive/folders/1WsR1NHhzpkLvk3V0g8X3Jy-xsXNlsYHf?usp=sharing" color="#008477">Google Drive model and experiment artifacts</link>. Verified publicly accessible without authentication. All checkpoints (steps 5,000 through 60,000 plus best_val), preprocessed data archives, logs, and vocoder diagnostics are confirmed.', "ReportSmall")
    add("Provide the exact English 8,000-step and Telugu 13,000-step checkpoints used for the audio, together with the run's config.yaml, stats.json, speakers.json, generated lexicon and exact text frontend/symbol inventory. Checkpoint weights alone cannot reconstruct speaker IDs or normalized predictor outputs. The 60,000-step checkpoints are a separate experiment artifact unless used explicitly for new outputs.")
    add("Inference after restoring the matching model bundle", "ReportSub")
    for language, step, speaker in [("English",8000,14),("Telugu",13000,0)]:
        add(f"cd {language}<br/>python inference.py --config config.yaml<br/>&#160; --ckpt_file checkpoints/{step}.pth.tar<br/>&#160; --preprocessed_path preprocessed_data<br/>&#160; --speaker_id {speaker}<br/>&#160; --result_path ../Results/{language}", "ReportCode")
    add("Each displayed invocation is one shell command; join its wrapped lines. Use the supplied bundle's configuration, not an assumed historical configuration. Full data preparation/alignment/training commands and the train-only Telugu alignment recipe are in README.md.", "ReportSmall")
    add("Environment and evidence", "ReportSub")
    add("The acoustic requirements file uses lower version bounds; it is not an environment lock. Export the actual training environment and CUDA/PyTorch versions with the original logs. Report generation uses reportlab, Pillow and pypdfium2. Rebuild with python Report/generate_report.py; audit with python tools/audit_submission.py --strict.")
    add("References", "ReportSub")
    for ref in [
        "[1] Assignment: FastSpeech2_Assignment.pdf, sections 1-4, including all seven technical-report requirements.",
        "[2] Ren et al. FastSpeech 2: Fast and High-Quality End-to-End Text to Speech. Supplied arXiv:2006.04558v8. https://arxiv.org/abs/2006.04558",
        "[3] ming024/FastSpeech2. Reference commit d4e79eb52e8b01d24703b2dfc0385544092958f3. https://github.com/ming024/FastSpeech2",
        "[4] Dataset sources: https://huggingface.co/datasets/ai4bharat/Svarah and https://huggingface.co/datasets/SPRINGLab/IndicTTS_Telugu",
        "[5] Repository evidence: language configurations, preprocessing/model/frontend source, Results/*/variance_predictions_summary.json and four loss PNGs.",
    ]:
        add(html.escape(ref), "ReportSmall")
    OUT.mkdir(exist_ok=True)
    destination = OUT / "Technical_Report.pdf"
    temporary = OUT / "Technical_Report.build.pdf"
    doc = SimpleDocTemplate(str(temporary), pagesize=(595.276,841.89), rightMargin=48, leftMargin=48, topMargin=53, bottomMargin=46, title="FastSpeech 2 for English and Telugu - Technical Report", author="FastSpeech2-English-Telugu project")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    temporary.replace(destination)
    (ROOT / "Results" / "submission_audit.json").write_text(json.dumps(findings, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Generated {destination}")


if __name__ == "__main__":
    main()
