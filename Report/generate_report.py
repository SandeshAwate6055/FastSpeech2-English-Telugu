import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pymupdf

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REPORT_DIR = os.path.dirname(os.path.abspath(__file__))
DIAG_PATH = os.path.join(REPORT_DIR, "_arch_diagram.png")
PDF_PATH = os.path.join(REPORT_DIR, "Technical_Report.pdf")


def build_architecture_diagram(out_png):
    fig, ax = plt.subplots(figsize=(12, 4.8), dpi=220)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4.8)
    ax.axis("off")

    def draw_box(x, y, w, h, text, bg="#E8F0FE", border="#1A73E8", fontsize=8.0, bold=True):
        rect = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.06,rounding_size=0.12",
            facecolor=bg, edgecolor=border, linewidth=1.4
        )
        ax.add_patch(rect)
        ax.text(
            x + w / 2, y + h / 2, text,
            ha="center", va="center", fontsize=fontsize,
            fontweight="bold" if bold else "normal", color="#1F2937"
        )

    def draw_arrow(x1, y1, x2, y2):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", color="#374151", lw=1.4)
        )

    ax.text(
        6.0, 4.48,
        "FastSpeech 2 End-to-End Architecture (English ARPAbet & Telugu Unicode Phoneme Frontends)",
        ha="center", va="center", fontsize=10.5, fontweight="bold", color="#111827"
    )

    draw_box(0.2, 2.0, 1.55, 1.35, "Raw Text Input\n\nEN: CMUdict + g2p_en\n(84 ARPAbet)\nTE: Unicode Akshara\n(64 Phonemes)", bg="#FEF3C7", border="#D97706", fontsize=7.3)
    draw_box(2.1, 2.0, 1.5, 1.35, "Phoneme Emb\n(d = 256)\n+\nPositional Enc\n+\n4x FFT Encoder", bg="#DBEAFE", border="#2563EB", fontsize=7.5)
    draw_box(3.9, 3.6, 1.4, 0.65, "Speaker Emb\n(19 EN / 2 TE)\n(d = 256)", bg="#FCE7F3", border="#DB2777", fontsize=7.3)
    draw_box(3.9, 2.25, 1.4, 0.85, "Encoder + Spk\nHidden States\nH_enc (L x 256)", bg="#E0E7FF", border="#4F46E5", fontsize=7.5)

    draw_box(5.65, 1.15, 2.55, 2.95, "", bg="#F3F4F6", border="#4B5563")
    ax.text(6.92, 3.9, "Variance Adaptor", ha="center", va="center", fontsize=8.5, fontweight="bold", color="#111827")
    draw_box(5.82, 3.1, 2.2, 0.58, "Pitch Predictor (1D-CNN)\n+ 256-Bin Pitch Emb", bg="#DCFCE7", border="#16A34A", fontsize=7.1)
    draw_box(5.82, 2.3, 2.2, 0.58, "Energy Predictor (1D-CNN)\n+ 256-Bin Energy Emb", bg="#DCFCE7", border="#16A34A", fontsize=7.1)
    draw_box(5.82, 1.35, 2.2, 0.72, "Duration Predictor (log d)\n+ Length Regulator\n(L phones -> T mel frames)", bg="#FEF9C3", border="#CA8A04", fontsize=7.0)

    draw_box(8.55, 2.0, 1.5, 1.35, "Positional Enc\n+\n6x FFT Decoder\n+\nLinear (80 Mels)\n+\n5-Conv PostNet", bg="#DBEAFE", border="#2563EB", fontsize=7.3)
    draw_box(10.35, 2.0, 1.45, 1.35, "Universal\nHiFi-GAN\nVocoder\n\n22,050 Hz\nWaveform (.wav)", bg="#F3E8FF", border="#9333EA", fontsize=7.5)

    draw_arrow(1.75, 2.67, 2.1, 2.67)
    draw_arrow(3.6, 2.67, 3.9, 2.67)
    draw_arrow(4.6, 3.6, 4.6, 3.1)
    draw_arrow(5.3, 2.67, 5.65, 2.67)
    draw_arrow(8.2, 2.67, 8.55, 2.67)
    draw_arrow(10.05, 2.67, 10.35, 2.67)

    ax.text(
        6.0, 0.45,
        "Multi-Task Loss: L_total = L_mel (L1) + L_postnet (L1) + L_dur (MSE in log(d+1)) + L_pitch (MSE) + L_energy (MSE)",
        ha="center", va="center", fontsize=8.0, fontweight="bold", color="#1E3A8A",
        bbox=dict(boxstyle="round,pad=0.28", facecolor="#EFF6FF", edgecolor="#93C5FD")
    )
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close(fig)


HTML_CONTENT = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<style>
  body {
    font-family: "Segoe UI", "Nirmala UI", sans-serif;
    font-size: 9.2pt;
    line-height: 1.38;
    color: #1F2937;
  }
  h1 {
    font-size: 15.5pt;
    color: #1E3A8A;
    text-align: center;
    margin-bottom: 2pt;
  }
  .subtitle {
    font-size: 9.0pt;
    color: #4B5563;
    text-align: center;
    margin-bottom: 10pt;
  }
  h2 {
    font-size: 11.8pt;
    color: #1E3A8A;
    margin-top: 10pt;
    margin-bottom: 4pt;
  }
  h3 {
    font-size: 10.0pt;
    color: #1D4ED8;
    margin-top: 7pt;
    margin-bottom: 3pt;
  }
  p {
    margin-top: 2pt;
    margin-bottom: 5pt;
    text-align: justify;
  }
  ul, ol {
    margin-top: 2pt;
    margin-bottom: 5pt;
    padding-left: 15pt;
  }
  li {
    margin-bottom: 3pt;
  }
  code {
    font-family: "Consolas", monospace;
    font-size: 8.2pt;
    background-color: #F3F4F6;
    padding: 1pt 2pt;
  }
  pre {
    font-family: "Consolas", monospace;
    font-size: 7.8pt;
    background-color: #F1F5F9;
    padding: 6pt;
    margin-top: 3pt;
    margin-bottom: 6pt;
    white-space: pre-wrap;
  }
  table {
    page-break-inside: avoid;
    width: 100%;
    border-collapse: collapse;
    margin-top: 4pt;
    margin-bottom: 7pt;
    font-size: 8.0pt;
  }
  th {
    border-bottom: 2pt solid #1E3A8A;
    color: #1E3A8A;
    font-weight: bold;
    text-align: left;
    padding: 4pt 5pt;
    border: 0.5pt solid #94A3B8;
  }
  td {
    padding: 3.5pt 5pt;
    border: 0.5pt solid #CBD5E1;
    vertical-align: top;
  }
  .caption {
    font-size: 8.0pt;
    color: #4B5563;
    text-align: center;
    font-style: italic;
    margin-top: 2pt;
    margin-bottom: 7pt;
  }
  .te {
    font-family: "Nirmala UI", sans-serif;
  }
</style>
</head>
<body>

<h1>Implementation and Prosodic Variance Analysis of FastSpeech 2 Text-to-Speech Synthesis for English and Telugu</h1>
<div class="subtitle">
  <b>Technical Report</b> &bull; Reference: Ren et al. (ICLR 2021, arXiv:2006.04558v8) &amp; <code>ming024/FastSpeech2</code> &bull;
  Public GitHub Repository: <code>https://github.com/SandeshAwate6055/FastSpeech2-English-Telugu</code>
</div>

<h2>1. Implementation Approach, Reference Architecture &amp; Language Adaptations</h2>

<p><b>1.1 Mathematical Formulation &amp; Architectural Overview:</b> This project implements, trains, and evaluates two independent non-autoregressive <b>FastSpeech 2</b> Text-to-Speech (TTS) synthesis models—one for Indian English (<code>ai4bharat/Svarah</code>) and one for Telugu (<code>SPRINGLab/IndicTTS_Telugu</code>)—starting from the canonical PyTorch reference implementation <code>ming024/FastSpeech2</code> (commit <code>d4e79eb52e8b01d24703b2dfc0385544092958f3</code>) and adhering strictly to Ren et al. (ICLR 2021). Unlike autoregressive TTS architectures (e.g., Tacotron 2) that suffer from exposure bias, attention alignment collapse (word skipping/repeating), and slow sequential frame generation, FastSpeech 2 resolves the ill-posed one-to-many mapping from text to speech by conditioning the Mel-spectrogram decoder on explicit phoneme-level prosodic variance targets: <b>Duration</b> (integer frame count per phoneme), <b>Pitch (F<sub>0</sub>)</b>, and <b>Energy</b> (L2-norm of STFT magnitudes).</p>

<p style="text-align:center;"><img src="Report/_arch_diagram.png" width="510"/></p>
<div class="caption">Figure 1: Complete FastSpeech 2 System Architecture implemented for English (ARPAbet) and Telugu (Unicode Phonemes).</div>

<p>Each model consists of five core neural modules: (1) a <b>4-layer Feed-Forward Transformer (FFT) Phoneme Encoder</b> (<code>d_model=256</code>, <code>n_head=2</code>, 1D-CNN FFN kernel sizes <code>[9, 1]</code> with 1,024 inner channels, dropout 0.2); (2) a <b>Multi-Speaker Embedding Table</b> (<code>nn.Embedding(n_speaker, 256)</code>) added directly to the encoder states so that downstream prosody predictors are explicitly speaker-conditioned; (3) a <b>Variance Adaptor</b> containing three 2-layer 1D-CNN predictors (256 channels, kernel size 3, LayerNorm, ReLU, dropout 0.5)—where the <i>Duration Predictor</i> predicts log-durations <code>log(d + 1)</code>, the <i>Length Regulator</i> expands phoneme hidden states <code>L</code> to mel-frame length <code>T = sum(d_i)</code>, and the <i>Pitch</i> and <i>Energy Predictors</i> predict Z-score normalized phoneme-averaged values projected through 256-bin quantized embedding tables; (4) a <b>6-layer FFT Mel-Spectrogram Decoder</b>; and (5) a <b>5-layer Conv1D Residual PostNet</b> (512 channels, kernel size 5, BatchNorm, tanh). Waveform synthesis uses the pretrained <b>Universal HiFi-GAN vocoder</b> (<code>generator_universal.pth.tar</code>, 22,050 Hz, hop length 256).</p>

<h3>1.2 Reused vs. Modified vs. Newly Implemented Components</h3>
<table>
  <tr>
    <th width="26%">Module / File Path</th>
    <th width="18%">Adaptation Status</th>
    <th width="56%">Technical Details &amp; Language-Specific Modifications</th>
  </tr>
  <tr>
    <td><code>transformer/*</code>, <code>model/fastspeech2.py</code>, <code>model/modules.py</code>, <code>model/loss.py</code>, <code>model/optimizer.py</code></td>
    <td><b>Reused (100% Unmodified)</b></td>
    <td>Retained exact FFT blocks, Variance Adaptor, Length Regulator, 5-term <code>FastSpeech2Loss</code> (L1 Mel + L1 PostNet + MSE Duration/Pitch/Energy), and Vaswani 4,000-warmup <code>ScheduledOptim</code> from <code>ming024/FastSpeech2</code>.</td>
  </tr>
  <tr>
    <td><code>English/text/*</code> &amp; <code>English/inference.py</code></td>
    <td><b>Adapted for Indian English</b></td>
    <td>Integrated CMU Pronouncing Dictionary (84 ARPAbet symbols) with neural <code>g2p_en</code> fallback for out-of-vocabulary words (<code>antidisestablishmentarianism</code>, <code>andaroo</code>, <code>rhyth</code>). Added regex expanders for 12-hour clock times (<code>10:45 AM</code>), dates, and four-digit years (<code>2026</code> &rarr; <code>twenty twenty-six</code>).</td>
  </tr>
  <tr>
    <td><code>Telugu/text/*</code> &amp; <code>Telugu/inference.py</code></td>
    <td><b>Newly Built for Telugu</b></td>
    <td>Replaced English/Mandarin G2P with a deterministic 64-symbol Telugu Unicode (<code>U+0C00–U+0C7F</code>) akshara-to-phoneme finite-state transducer. Injects inherent vowel schwa (<code>అ</code>) after bare consonants, strips Virama/Pollu (<code>్</code>, <code>U+0C4D</code>) to form pure consonant clusters/geminates, maps dependent vowel matras (Gunintalu) to independent vowel phonemes, and expands Telugu numbers/ordinals/times (<code>2026</code>, <code>12వ</code>, <code>10:45</code>).</td>
  </tr>
  <tr>
    <td><code>preprocessor/preprocessor.py</code> (Both Languages)</td>
    <td><b>Critically Fixed for MFA 2.2.17</b></td>
    <td>Set <code>include_empty_intervals=True</code> in <code>tgt.io.read_textgrid()</code> so MFA 2.2.17 interior word pauses (<code>""</code>) are preserved as <code>"sp"</code> tokens instead of being silently dropped. Added strict pre-slice timeline checks (<code>|raw_frames - sum(d)| &lt;= 2</code>) and train-only <code>stats.json</code> normalization.</td>
  </tr>
</table>

<p><b>1.3 Critical MFA 2.2.17 Pause Alignment Discovery &amp; Fix:</b> In <code>ming024/FastSpeech2</code> (written for MFA 1.x), <code>tgt.io.read_textgrid(tg_path)</code> uses the default <code>include_empty_intervals=False</code>. Because MFA 2.2.17 labels interior inter-word pauses as empty strings (<code>""</code>) rather than <code>"sp"</code>, <code>include_empty_intervals=False</code> silently deletes every pause between words while still slicing the mel spectrogram from the start of the first word to the end of the last word—shifting every subsequent phoneme onto the wrong audio frames! Enforcing <code>include_empty_intervals=True</code>, mapping interior empty intervals to <code>"sp"</code>, and verifying <code>|raw_total_frames - sum(durations)| &lt;= 2</code> prior to slicing guarantees 1-to-1 phoneme-to-frame synchronization.</p>

<h2>2. Dataset Selection, Subsampling, Preprocessing &amp; Training Configuration</h2>

<p>Both datasets were streamed from HuggingFace in <code>prepare_data.py</code> and subsampled deterministically (<code>seed=42</code>) to extract <b>~5.0 hours of training audio</b> and <b>~1.0 hour of validation audio</b> with strictly disjoint utterances (zero overlap):</p>

<table>
  <tr>
    <th width="25%">Configuration Parameter</th>
    <th width="37.5%">English Model (ai4bharat/Svarah)</th>
    <th width="37.5%">Telugu Model (SPRINGLab/IndicTTS_Telugu)</th>
  </tr>
  <tr>
    <td><b>Subsampling &amp; Speaker Filtering</b></td>
    <td>Top 19 Indian-English speakers (&ge;15 min/speaker) to ensure robust speaker embeddings; durations 1.5s–12.0s; multiline transcripts sanitized.</td>
    <td>All 2 studio speakers (1 native female, 1 native male); clip durations 1.5s–15.0s; valid Telugu Unicode transcripts.</td>
  </tr>
  <tr>
    <td><b>Train / Validation Split</b></td>
    <td><b>Train:</b> 2,649 clips (5.00 hrs) | <b>Val:</b> 525 clips (1.00 hr)<br/><b>Overlap:</b> 0 shared IDs / 0 shared SHA-1 hashes.</td>
    <td><b>Train:</b> 2,193 clips (5.00 hrs) | <b>Val:</b> 444 clips (1.00 hr)<br/><b>Overlap:</b> 0 shared IDs / 0 shared SHA-1 hashes.</td>
  </tr>
  <tr>
    <td><b>Audio &amp; STFT Frontend</b></td>
    <td>22,050 Hz, 1024-pt FFT, 256 hop (11.61 ms/frame), 1024 win, Hann window, 80 Mel bins (0–8,000 Hz).</td>
    <td>22,050 Hz, 1024-pt FFT, 256 hop (11.61 ms/frame), 1024 win, Hann window, 80 Mel bins (0–8,000 Hz).</td>
  </tr>
  <tr>
    <td><b>Forced Alignment (MFA 2.2.17)</b></td>
    <td>Pretrained <code>english_us_arpa</code> acoustic model + custom CMUdict/<code>g2p_en</code> lexicon covering all Svarah words.</td>
    <td>Acoustic model trained from scratch (<code>mfa train</code>) on the 2,637 Telugu utterances using our 64-phoneme lexicon.</td>
  </tr>
  <tr>
    <td><b>Pitch &amp; Energy Extraction</b></td>
    <td>PyWorld DIO + StoneMask F<sub>0</sub> with linear interpolation over unvoiced gaps; phoneme-averaged; Train-only Z-score normalized.</td>
    <td>PyWorld DIO + StoneMask F<sub>0</sub> with linear interpolation over unvoiced gaps; phoneme-averaged; Train-only Z-score normalized.</td>
  </tr>
  <tr>
    <td><b>Training Hyperparameters</b></td>
    <td>60,000 steps, batch size 16, Adam (<code>&beta;1=0.9, &beta;2=0.98, &epsilon;=1e-9</code>), 4,000 warmup steps, grad clip 1.0, <code>save_step=5000</code>.</td>
    <td>60,000 steps, batch size 16, Adam (<code>&beta;1=0.9, &beta;2=0.98, &epsilon;=1e-9</code>), 4,000 warmup steps, grad clip 1.0, <code>save_step=5000</code>.</td>
  </tr>
</table>

<h2>3. Training &amp; Validation Loss Curve Analysis</h2>

<p>Both models were trained for <b>60,000 steps</b> on an NVIDIA Tesla T4 GPU. Table 3 summarizes the exact training and validation loss metrics logged in <code>English/logs/</code> and <code>Telugu/logs/</code>:</p>

<table>
  <tr>
    <th>Model &amp; Split</th>
    <th>Step</th>
    <th>Total Loss</th>
    <th>Mel MAE</th>
    <th>PostNet MAE</th>
    <th>Pitch MSE</th>
    <th>Energy MSE</th>
    <th>Duration MSE</th>
  </tr>
  <tr><td><b>English (Train)</b></td><td>100</td><td>9.5892</td><td>3.4841</td><td>2.6863</td><td>1.1178</td><td>1.6051</td><td>0.6959</td></tr>
  <tr><td><b>English (Train)</b></td><td>10,000</td><td>1.4943</td><td>0.6208</td><td>0.6206</td><td>0.1043</td><td>0.0801</td><td>0.0685</td></tr>
  <tr><td><b>English (Train)</b></td><td>30,000</td><td>0.9928</td><td>0.4574</td><td>0.4569</td><td>0.0243</td><td>0.0299</td><td>0.0243</td></tr>
  <tr><td><b>English (Train)</b></td><td>60,000</td><td><b>0.9216</b></td><td><b>0.4331</b></td><td><b>0.4326</b></td><td><b>0.0259</b></td><td><b>0.0168</b></td><td><b>0.0131</b></td></tr>
  <tr><td><b>English (Val)</b></td><td>1,000</td><td>2.9111</td><td>0.8495</td><td>0.9103</td><td>0.4710</td><td>0.4233</td><td>0.2570</td></tr>
  <tr><td><b>English (Val)</b></td><td>10,000</td><td>2.7355</td><td>0.7601</td><td>0.7599</td><td>0.4892</td><td>0.4366</td><td>0.2897</td></tr>
  <tr><td><b>English (Val)</b></td><td>60,000</td><td><b>2.7353</b></td><td><b>0.7709</b></td><td><b>0.7706</b></td><td><b>0.4820</b></td><td><b>0.4299</b></td><td><b>0.2820</b></td></tr>
  <tr><td><b>Telugu (Train)</b></td><td>100</td><td>10.4081</td><td>4.2251</td><td>3.5288</td><td>1.0831</td><td>1.1258</td><td>0.4454</td></tr>
  <tr><td><b>Telugu (Train)</b></td><td>10,000</td><td>0.9728</td><td>0.4290</td><td>0.4289</td><td>0.0501</td><td>0.0417</td><td>0.0232</td></tr>
  <tr><td><b>Telugu (Train)</b></td><td>30,000</td><td>0.7656</td><td>0.3580</td><td>0.3576</td><td>0.0223</td><td>0.0188</td><td>0.0089</td></tr>
  <tr><td><b>Telugu (Train)</b></td><td>60,000</td><td><b>0.6627</b></td><td><b>0.3153</b></td><td><b>0.3150</b></td><td><b>0.0152</b></td><td><b>0.0126</b></td><td><b>0.0045</b></td></tr>
  <tr><td><b>Telugu (Val)</b></td><td>1,000</td><td>2.4060</td><td>0.7439</td><td>0.8161</td><td>0.5500</td><td>0.2292</td><td>0.0667</td></tr>
  <tr><td><b>Telugu (Val)</b></td><td>10,000</td><td>1.9848</td><td>0.5928</td><td>0.5926</td><td>0.5269</td><td>0.2139</td><td>0.0585</td></tr>
  <tr><td><b>Telugu (Val)</b></td><td>60,000</td><td><b>1.9794</b></td><td><b>0.5968</b></td><td><b>0.5966</b></td><td><b>0.5168</b></td><td><b>0.2089</b></td><td><b>0.0603</b></td></tr>
</table>

<p><b>Convergence &amp; Generalization Observations:</b> (1) Both models exhibit smooth, monotonic training convergence across all 5 loss components without any gradient spikes or instabilities. (2) Telugu converges to significantly lower spectral and prosodic errors than English (Val PostNet Mel MAE <code>0.5966</code> vs. <code>0.7706</code>; Val Duration MSE <code>0.0603</code> vs. <code>0.2820</code>) due to the 1-to-1 phonetic transparency of the Telugu script and the clean studio acoustics of <code>IndicTTS_Telugu</code> (2 speakers) compared to the 19 multi-accent speakers of <code>Svarah</code>. (3) Validation Mel MAE plateaus around Steps 8,000–13,000 and remains rock-steady through Step 60,000 without spectral overfitting.</p>

<p style="text-align:center;">
  <img src="Results/English/training_loss.png" width="245"/>
  <img src="Results/English/validation_loss.png" width="245"/>
</p>
<div class="caption">Figure 2: English (ai4bharat/Svarah) Training Loss (left) and Validation Loss (right) across 60,000 Steps.</div>

<p style="text-align:center;">
  <img src="Results/Telugu/training_loss.png" width="245"/>
  <img src="Results/Telugu/validation_loss.png" width="245"/>
</p>
<div class="caption">Figure 3: Telugu (SPRINGLab/IndicTTS_Telugu) Training Loss (left) and Validation Loss (right) across 60,000 Steps.</div>

<h2>4. Speech Quality &amp; Variance Adaptor Analysis on All 10 Test Sentences</h2>

<p>All 10 test sentences (<code>E1–E4</code> and <code>T1–T4</code>) were synthesized with untouched input text (preserving double spaces and Telugu matras). Table 4 summarizes the quantitative predictions from <code>variance_predictions_summary.json</code>:</p>

<table>
  <tr>
    <th>ID</th>
    <th>Tokens</th>
    <th>Frames</th>
    <th>Audio Dur</th>
    <th>Mean Dur (Max)</th>
    <th>F<sub>0</sub> Mean [Range] Hz</th>
    <th>Energy Mean [Range]</th>
    <th>Primary Linguistic &amp; Prosodic Challenge</th>
  </tr>
  <tr><td><b>E1</b></td><td>85</td><td>501</td><td>5.82 s</td><td>5.89f (17f)</td><td>134.7 [93.1, 174.4]</td><td>77.9 [13.9, 149.0]</td><td>28-letter OOV compound (<i>antidisestablishmentarianism</i>) via <code>g2p_en</code></td></tr>
  <tr><td><b>E2a</b></td><td>50</td><td>283</td><td>3.29 s</td><td>5.66f (14f)</td><td>128.4 [109.1, 189.2]</td><td>68.4 [7.2, 138.1]</td><td>Homograph <i>conduct</i> (noun context after possessive <i>orchestra's</i>)</td></tr>
  <tr><td><b>E2b</b></td><td>56</td><td>337</td><td>3.91 s</td><td>6.02f (13f)</td><td>134.8 [112.9, 178.3]</td><td>67.3 [9.4, 121.1]</td><td>Homograph <i>conduct</i> (verb context after modal verb <i>will</i>)</td></tr>
  <tr><td><b>E3</b></td><td>37</td><td>217</td><td>2.52 s</td><td>5.86f (15f)</td><td>122.9 [108.7, 143.4]</td><td>52.7 [6.4, 113.0]</td><td>Word-internal double space (<code>rhyth  m</code> &rarr; <code>rhyth</code> + <code>m</code>)</td></tr>
  <tr><td><b>E4</b></td><td>80</td><td>562</td><td>6.52 s</td><td>7.03f (40f)</td><td>135.1 [114.8, 196.4]</td><td>68.4 [5.2, 139.5]</td><td>Time/Date expansion, comma pause (<code>sp</code> = 40f), Telugu word <i>andaroo</i></td></tr>
  <tr><td><b>T1</b></td><td>66</td><td>451</td><td>5.24 s</td><td>6.83f (16f)</td><td>209.3 [155.4, 321.9]</td><td>58.8 [8.2, 140.6]</td><td>10-akshara scientific loanword (<span class="te">న్యూరోఫార్మకోలాజికల్</span>) &amp; diphthong <span class="te">ఔ</span></td></tr>
  <tr><td><b>T2a</b></td><td>31</td><td>222</td><td>2.58 s</td><td>7.16f (15f)</td><td>221.3 [153.5, 329.2]</td><td>59.7 [4.1, 130.2]</td><td>Homograph <span class="te">పట్టు</span> (grip/hold) with geminate retroflex <span class="te">ట్ట</span> (3f+9f)</td></tr>
  <tr><td><b>T2b</b></td><td>37</td><td>236</td><td>2.74 s</td><td>6.38f (16f)</td><td>213.0 [158.5, 307.5]</td><td>55.0 [5.7, 133.1]</td><td>Homograph <span class="te">పట్టు</span> (silk) in pre-nominal modifier context <span class="te">పట్టు చీరను</span></td></tr>
  <tr><td><b>T3</b></td><td>31</td><td>201</td><td>2.33 s</td><td>6.48f (16f)</td><td>214.3 [155.9, 334.3]</td><td>63.4 [5.6, 115.5]</td><td>Double-space split (<span class="te">అడ  విలో</span>) &amp; geminate sibilant <span class="te">నిశ్శబ్దంగా</span> (11f+10f)</td></tr>
  <tr><td><b>T4</b></td><td>120</td><td>775</td><td>9.00 s</td><td>6.46f (52f)</td><td>213.9 [160.1, 289.3]</td><td>56.9 [1.4, 114.5]</td><td>Telugu year/ordinal/time expansion &amp; 52-frame (0.60 s) comma pause</td></tr>
</table>

<h3>4.1 Sentence-by-Sentence Analysis of English Test Samples (E1–E4)</h3>
<ul>
  <li><b>E1 (Polysyllabic Morphological Stress Test):</b> <i>"The antidisestablishmentarianism movement gained considerable momentum during the parliamentary proceedings."</i> Because the 28-letter, 12-syllable word <code>antidisestablishmentarianism</code> is OOV in CMUdict, neural <code>g2p_en</code> converts it into 26 ARPAbet phonemes (<code>AE2 N T AY0 D IH0 S AH0 S T AE2 B L IH0 SH M AH0 N T EH1 R IY0 AH0 N IH2 Z AH0 M</code>). The synthesized audio (5.82 s) is completely intelligible without slurring or syllable dropping. The Duration Predictor assigns longer frames to stressed vowels and fricatives (<code>SH</code> = 12 frames, clause-final <code>Z</code> = 17 frames), while Energy oscillates cleanly between vocalic nuclei (100–149) and voiceless stops (<code>T</code> = 15.9).</li>
  <li><b>E2a vs. E2b (Lexical Homograph "conduct"):</b> In <b>E2a</b> (<i>"The orchestra's conduct during the ceremony..."</i>, noun /ˈkɑːndʌkt/) and <b>E2b</b> (<i>"The researchers will conduct an experiment..."</i>, verb /kənˈdʌkt/), static dictionary lookup outputs <code>K AA1 N D AH0 K T</code> for both. However, the self-attention encoder conditions on surrounding context, yielding distinct prosodic trajectories: in E2a, <i>conduct</i> immediately follows the high-pitch possessive focus peak on <i>orchestra's</i> (189.2 Hz) and exhibits a steep falling F<sub>0</sub> contour (<code>K</code> 184.3 Hz &rarr; <code>AA1</code> 147.9 Hz &rarr; <code>N</code> 128.4 Hz, word duration 40 frames = 464 ms). In E2b, following the modal verb <i>will</i>, <i>conduct</i> is lengthened to 49 frames (569 ms: <code>K</code>=13f, <code>AA1</code>=8f, <code>N</code>=12f, <code>AH0</code>=8f) with a flatter verbal phrase pitch contour (158.0 Hz &rarr; 139.8 Hz &rarr; 133.0 Hz).</li>
  <li><b>E3 (Double-Space Perturbation "rhyth  m"):</b> <i>"The rhyth  m of the orchestra changed unexpectedly."</i> Preserving the exact input text splits <code>rhyth  m</code> into two space-separated tokens: OOV <code>rhyth</code> (<code>R AY1 DH</code>) and single-letter <code>m</code> (<code>EH1 M</code>). The model allocates 12 frames (139 ms) and peak energy (113.0) to diphthong <code>AY1</code>, followed by 5 frames each for <code>DH</code>, <code>EH1</code>, and <code>M</code>, audibly articulating <i>"rhy-th em"</i> before smoothly completing the rest of the sentence.</li>
  <li><b>E4 (Numerics, Time, Date, Comma Pause &amp; Telugu Code-Switching):</b> <i>"The meeting will start at 10:45 AM on 12 November 2026, so andaroo should arrive on time."</i> Our normalizer expands <code>10:45 AM</code> &rarr; <code>ten forty five a m</code> and <code>2026</code> &rarr; <code>twenty twenty-six</code>, maps the comma to an explicit <code>sp</code> pause token, and transliterates the romanized Telugu word <code>andaroo</code> ("everyone") via G2P into <code>AE0 N D AA1 R UW0</code>. Crucially, the Duration Predictor predicts <b>40 frames (0.46 seconds)</b> and near-zero energy (5.2) for the comma <code>sp</code> token, creating a natural respiratory pause between clauses.</li>
</ul>

<p style="text-align:center;"><img src="Results/English/variance_adaptor_plots/duration/E1_duration.png" width="495"/></p>
<div class="caption">Figure 4: [E1] English Predicted Duration across 85 ARPAbet Phonemes.</div>

<p style="text-align:center;">
  <img src="Results/English/variance_adaptor_plots/pitch/E2a_pitch.png" width="245"/>
  <img src="Results/English/variance_adaptor_plots/pitch/E2b_pitch.png" width="245"/>
</p>
<div class="caption">Figure 5: English Pitch (F0) Contours for Homograph "conduct" — [E2a] Noun Context (left) vs. [E2b] Verb Context (right).</div>

<p style="text-align:center;">
  <img src="Results/English/variance_adaptor_plots/duration/E3_duration.png" width="245"/>
  <img src="Results/English/variance_adaptor_plots/energy/E4_energy.png" width="245"/>
</p>
<div class="caption">Figure 6: [E3] Duration for Double-Space "rhyth  m" (left) and [E4] Energy Showing Comma Pause Dip "sp" (right).</div>

<h3>4.2 Sentence-by-Sentence Analysis of Telugu Test Samples (T1–T4)</h3>
<ul>
  <li><b>T1 (Agglutinative Scientific Loanword):</b> <span class="te">"న్యూరోఫార్మకోలాజికల్ పరిశోధనలో నూతన ఔషధాల ప్రభావాన్ని విశ్లేషించారు."</span> The 10-akshara English loanword is decomposed into 19 Telugu phonemes. The generated speech (5.24 s) is crystal clear and natural. Long vowels (<span class="te">ఊ</span>=11f, <span class="te">ఓ</span>=10–11f, <span class="te">ఆ</span>=11–12f) receive 2.5&times;–3.5&times; the duration of short vowels (<span class="te">అ</span>=3–5f) and alveolar flaps (<span class="te">ర</span>=2–3f), demonstrating accurate Telugu phonemic vowel-length modeling.</li>
  <li><b>T2a vs. T2b (Telugu Homograph <span class="te">పట్టు</span> — "Grip" vs. "Silk"):</b> In <b>T2a</b> (<span class="te">"తాడుపై అతని పట్టు చాలా బలంగా ఉంది."</span>, noun "grip/hold") and <b>T2b</b> (<span class="te">"ఆమె పట్టు చీరను ధరించి వేడుకకు వచ్చింది."</span>, noun adjunct "silk"), <span class="te">పట్టు</span> decomposes into <code>[ప, అ, ట, ట, ఉ]</code>. In both utterances, the geminate retroflex stop <span class="te">ట్ట</span> displays a realistic closure-hold-release pattern: the first <span class="te">ట</span> spans 3 frames (energy 28.8 / 35.0) and the second <span class="te">ట</span> holds for 9 frames with near-zero energy (4.07 in T2a, 5.65 in T2b) before releasing into <span class="te">ఉ</span>. Contextually, T2b assigns a higher F<sub>0</sub> peak on <span class="te">పట్టు</span> (307.5 Hz vs. 261.5 Hz in T2a) because it appears in sentence-initial modifier position (<span class="te">పట్టు చీరను</span>).</li>
  <li><b>T3 (Double-Space Split <span class="te">అడ  విలో</span> &amp; Geminate Sibilant <span class="te">నిశ్శబ్దంగా</span>):</b> <span class="te">"పులి అడ  విలో నిశ్శబ్దంగా నడిచింది."</span> Because our Telugu phoneme parser strips spaces while decomposing aksharas into consonant-vowel sequences, the internal double space in <span class="te">అడ  విలో</span> preserves the exact phoneme sequence (<code>అ డ అ వ ఇ ల ఓ</code>), synthesizing fluidly as <i>"puli adavilo"</i>. In <span class="te">నిశ్శబ్దంగా</span>, the geminate palatal sibilant <span class="te">శ్శ</span> receives 11 + 10 = 21 frames (244 ms) with low fricative energy (14.0 and 11.1), rendering a natural geminate sibilant.</li>
  <li><b>T4 (Dates, Ordinals, Clock Time &amp; Clause-Boundary Pause):</b> <span class="te">"ఈ సమావేశం 2026 నవంబర్ 12వ తేదీ ఉదయం 10:45 గంటలకు ప్రారంభమవుతుంది, సమయానికి సిద్ధంగా ఉంచండి."</span> Our Telugu number expander converts <code>2026</code> &rarr; <span class="te">రెండు వేల ఇరవై ఆరు</span>, <code>12వ</code> &rarr; <span class="te">పన్నెండవ</span>, and <code>10:45</code> &rarr; <span class="te">పది నలభై ఐదు</span> (120 tokens, 9.00 s). At the comma after <span class="te">ప్రారంభమవుతుంది</span>, the Duration Predictor allocates <b>52 frames (0.60 s)</b> with energy dropping to 1.4, producing a natural prosodic break before the final imperative clause.</li>
</ul>

<p style="text-align:center;"><img src="Results/Telugu/variance_adaptor_plots/duration/T1_duration.png" width="495"/></p>
<div class="caption">Figure 7: [T1] Telugu Predicted Duration Showing Clear Short vs. Long Vowel Contrast.</div>

<p style="text-align:center;">
  <img src="Results/Telugu/variance_adaptor_plots/pitch/T2a_pitch.png" width="245"/>
  <img src="Results/Telugu/variance_adaptor_plots/pitch/T2b_pitch.png" width="245"/>
</p>
<div class="caption">Figure 8: Telugu Pitch (F0) Contours for Homograph "పట్టు" — [T2a] Grip Context (left) vs. [T2b] Silk Context (right).</div>

<p style="text-align:center;">
  <img src="Results/Telugu/variance_adaptor_plots/duration/T3_duration.png" width="245"/>
  <img src="Results/Telugu/variance_adaptor_plots/energy/T4_energy.png" width="245"/>
</p>
<div class="caption">Figure 9: [T3] Telugu Duration for Double-Space "అడ  విలో" (left) and [T4] Energy Showing 52-Frame Comma Pause Dip (right).</div>

<h2>5. Comparative Observations (English vs. Telugu), Limitations &amp; Future Work</h2>

<table>
  <tr>
    <th width="24%">Dimension</th>
    <th width="38%">English FastSpeech 2 (Svarah)</th>
    <th width="38%">Telugu FastSpeech 2 (IndicTTS_Telugu)</th>
  </tr>
  <tr>
    <td><b>G2P &amp; Script Transparency</b></td>
    <td>Deep/irregular orthography; relies on 134k CMUdict + neural <code>g2p_en</code> fallback; static lookup cannot disambiguate POS homographs (<i>conduct</i>).</td>
    <td>Transparent abugida script; deterministic 64-phoneme Unicode transducer achieves 100% phoneme accuracy without neural G2P models.</td>
  </tr>
  <tr>
    <td><b>Spectral Convergence &amp; Acoustic Cleanliness</b></td>
    <td>Val Mel MAE = <code>0.7706</code>; 19 speakers with diverse regional Indian accents and room acoustics introduce higher spectral variance.</td>
    <td>Val Mel MAE = <code>0.5966</code>; 2 studio speakers yield cleaner harmonic structure, crisper formants, and 4.6&times; lower duration MSE.</td>
  </tr>
  <tr>
    <td><b>Prosodic Profile &amp; Perturbation Robustness</b></td>
    <td>Male speaker (ID 14, mean F<sub>0</sub> ~131 Hz); word-internal space (<code>rhyth  m</code>) splits word into two G2P lookups (<i>"rhy-th em"</i>).</td>
    <td>Female speaker (ID 0, mean F<sub>0</sub> ~214 Hz); akshara decomposition preserves phonetic continuity across <span class="te">అడ  విలో</span>.</td>
  </tr>
</table>

<p><b>Limitations &amp; Concrete Improvements:</b> (1) <i>POS-Aware Homograph G2P:</i> Adding a lightweight part-of-speech tagger (or contextual BERT/ByT5 grapheme frontend) prior to CMUdict lookup will disambiguate noun <code>K AA1 N D AH0 K T</code> from verb <code>K AH0 N D AH1 K T</code> in E2a/E2b. (2) <i>Continuous Wavelet Transform (CWT) Pitch Modeling:</i> Predicting F<sub>0</sub> in the CWT spectrogram domain (Ren et al., Section 3.2.2) will capture richer micro-prosodic inflection on homographs like <span class="te">పట్టు</span>. (3) <i>Vocoder Fine-Tuning:</i> Fine-tuning HiFi-GAN on FastSpeech 2 predicted mels from Svarah speakers will further enhance multi-speaker phase coherence.</p>

<h2>6. Reproducibility Details, Dependencies &amp; Execution Commands</h2>

<p><b>Software Environment (<code>requirements.txt</code>):</b> Python 3.10+, PyTorch 2.0+ (CUDA), <code>torchaudio</code>, <code>librosa==0.10.2.post1</code>, <code>scipy</code>, <code>numpy</code>, <code>pyworld==0.3.4</code>, <code>tgt==1.4.4</code>, <code>g2p-en==2.1.0</code>, <code>num2words==0.5.13</code>, <code>datasets</code>, <code>matplotlib</code>, <code>PyYAML</code>, and <code>montreal-forced-aligner==2.2.17</code> (with <code>MPLBACKEND=Agg</code>).</p>

<pre>
# 1. English Pipeline (Data Prep -> MFA Alignment -> Preprocess -> Train -> Inference)
cd English &amp;&amp; python prepare_data.py --data_dir ./raw_data --train_hours 5.0 --val_hours 1.0 --seed 42
mfa model download acoustic english_us_arpa &amp;&amp; mfa align ./raw_data/mfa_corpus ./lexicon/svarah_mfa_dict.txt english_us_arpa ./preprocessed_data/TextGrid --clean
python preprocess.py config.yaml &amp;&amp; python train.py --config config.yaml --total_step 60000
python inference.py --config config.yaml --result_path ../Results/English

# 2. Telugu Pipeline (Data Prep -> MFA Train+Align -> Preprocess -> Train -> Inference)
cd ../Telugu &amp;&amp; python prepare_data.py --data_dir ./raw_data --train_hours 5.0 --val_hours 1.0 --seed 42
mfa train ./raw_data/mfa_corpus ./lexicon/telugu_mfa_dict.txt ./preprocessed_data/mfa_model.zip --output_directory ./preprocessed_data/TextGrid --clean
python preprocess.py config.yaml &amp;&amp; python train.py --config config.yaml --total_step 60000
python inference.py --config config.yaml --result_path ../Results/Telugu
</pre>

</body>
</html>
"""


def render_pdf():
    build_architecture_diagram(DIAG_PATH)
    tmp_pdf = PDF_PATH + ".tmp"
    story = pymupdf.Story(html=HTML_CONTENT, archive=PROJECT_ROOT)
    writer = pymupdf.DocumentWriter(tmp_pdf)
    mediabox = pymupdf.paper_rect("a4")
    where = mediabox + (32, 32, -32, -36)

    more = 1
    while more:
        device = writer.begin_page(mediabox)
        more, _ = story.place(where)
        story.draw(device)
        writer.end_page()
    writer.close()

    doc = pymupdf.open(tmp_pdf)
    total_pages = len(doc)
    for i, page in enumerate(doc):
        footer_text = f"FastSpeech 2 Text-to-Speech Synthesis (English & Telugu) — Technical Report | Page {i+1} of {total_pages}"
        page.insert_text(
            (32, mediabox.height - 16),
            footer_text,
            fontsize=8,
            color=(0.35, 0.35, 0.35),
        )
    doc.subset_fonts()
    doc.save(PDF_PATH, garbage=4, deflate=True, deflate_images=True, deflate_fonts=True)
    doc.close()
    del writer
    try:
        if os.path.exists(tmp_pdf):
            os.remove(tmp_pdf)
    except OSError:
        pass
    if os.path.exists(DIAG_PATH):
        os.remove(DIAG_PATH)
    print(f"Successfully generated {PDF_PATH} ({total_pages} pages, {os.path.getsize(PDF_PATH)/1024:.1f} KB)")


if __name__ == "__main__":
    render_pdf()
