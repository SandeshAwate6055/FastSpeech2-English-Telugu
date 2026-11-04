"""Build an offline audio/plot reviewer; GitHub README itself cannot execute JavaScript."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build():
    samples = []
    for language in ("English", "Telugu"):
        summary = json.loads((ROOT / "Results" / language / "variance_predictions_summary.json").read_text(encoding="utf-8"))
        for sid, row in summary.items():
            samples.append({"id": sid, "language": language, **row})
    review = json.loads((ROOT / "Results/listening_review.json").read_text(encoding="utf-8"))
    template = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>FastSpeech 2 | Audio and prosody review</title>
<style>
:root{color-scheme:light;--ink:#173048;--teal:#087d78;--line:#d7e3e8;--muted:#526976}*{box-sizing:border-box}body{margin:0;background:#f2f6f8;color:var(--ink);font:16px/1.55 system-ui,"Nirmala UI",sans-serif}header{background:#102b46;color:white;padding:36px max(24px,calc((100vw - 1140px)/2))}header p{max-width:850px;color:#d7e8ed}header small{letter-spacing:.14em}main{max-width:1140px;margin:24px auto;padding:0 20px}.toolbar{display:flex;flex-wrap:wrap;align-items:center;gap:12px;padding:18px;border:1px solid var(--line);background:white;border-radius:12px;position:sticky;top:0;z-index:2}button,select,input,textarea{font:inherit;border:1px solid #aebfc9;border-radius:7px;padding:9px;background:white;color:var(--ink)}button{cursor:pointer}button.primary{background:var(--teal);color:white;border:0}input{min-width:180px}article{margin:22px 0;padding:24px;background:white;border:1px solid var(--line);border-radius:14px;box-shadow:0 3px 14px #102b4608}h2{margin:0;font-size:25px}.meta{color:var(--muted);font-size:14px}.sentence{white-space:pre-wrap;font-size:19px;padding:18px;background:#f0f7f7;border-left:3px solid var(--teal)}audio{width:100%;margin:8px 0}.fields{display:grid;grid-template-columns:1fr 1fr;gap:14px}.fields label{display:flex;flex-direction:column;gap:5px;font-size:14px}textarea{min-height:75px;resize:vertical}.plots{display:flex;gap:8px;margin:14px 0}.plot{max-width:100%;max-height:330px;object-fit:contain;border:1px solid var(--line)}.help{font-size:14px;color:var(--muted)}a{color:var(--teal)}footer{padding:25px 0}.hidden{display:none}@media(max-width:650px){.fields{grid-template-columns:1fr}.toolbar{position:static}article{padding:16px}}
</style></head><body><header><small>FASTSPEECH 2 / ENGLISH + TELUGU</small><h1>Listen. Inspect. Record what you hear.</h1><p>Review the ten assignment outputs alongside their predicted duration, F0 and energy. This page runs locally and sends no data to a server. Notes stay in this browser until you export them.</p></header><main>
<div class="toolbar"><label>Language <select id="language"><option>All</option><option>English</option><option>Telugu</option></select></label><input id="reviewer" aria-label="Reviewer name" placeholder="Reviewer name"><label>Playback <select id="rate"><option value="1">1.0x</option><option value="0.8">0.8x</option><option value="1.2">1.2x</option></select></label><button class="primary" id="export">Export listening_review.json</button><span id="progress" aria-live="polite"></span></div>
<p class="help">Open this file after cloning/downloading the repository, or serve the repository with <code>python -m http.server 8000 --bind 127.0.0.1</code>. GitHub displays HTML source; it does not run this reviewer. Exported notes must replace <code>Results/listening_review.json</code>, then rebuild the report. A completed review can honestly describe poor speech.</p>
<div id="samples"></div><footer><a href="Technical_Report.pdf">Technical report</a> · <a href="../README.md">Project README</a> · <a href="../Results/submission_audit.json">Evidence audit</a></footer></main>
<script>
const samples=__SAMPLES__, initial=__REVIEW__;
const notes=structuredClone(initial.samples); const container=document.querySelector('#samples');
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
let storageAvailable=true;
try{document.querySelector('#reviewer').value=localStorage.getItem('fs2-reviewer')||''}catch{storageAvailable=false}
function complete(n){return ['clear','mixed','poor'].includes(n.intelligibility)&&['pronunciation','completeness','prosody'].every(k=>(n[k]||'').trim())&&document.querySelector('#reviewer').value.trim()}
function progress(){document.querySelector('#progress').textContent=`${Object.values(notes).filter(complete).length} / 10 reviewed`}
for(const s of samples){const key='fs2-'+notes[s.id].audio_sha256;
 if(storageAvailable){try{const saved=localStorage.getItem(key);if(saved)Object.assign(notes[s.id],JSON.parse(saved))}catch{}}
 const n=notes[s.id], a=document.createElement('article');a.dataset.language=s.language;
 a.innerHTML=`<h2>${esc(s.id)} <span class="meta">${esc(s.language)}</span></h2><div class="meta">Checkpoint ${s.checkpoint_step.toLocaleString()} · speaker ${s.speaker_id} · ${(s.total_frames*256/22050).toFixed(2)} s · ${s.tokens.length} phones</div><p class="sentence">${esc(s.raw_text)}</p><audio controls preload="none" src="../Results/${s.language}/test_generated_audios/${s.id}.wav"></audio><div class="plots"><button data-kind="duration">Duration</button><button data-kind="pitch">F0</button><button data-kind="energy">Energy</button></div><img class="plot" alt="${s.id} duration predictions" src="../Results/${s.language}/variance_adaptor_plots/duration/${s.id}_duration.png"><details><summary>Normalized pronunciation text</summary><p>${esc(s.normalized_text)}</p></details><div class="fields"><label>Intelligibility<select data-field="intelligibility"><option value="pending">Not reviewed</option><option value="clear">Clear</option><option value="mixed">Partly intelligible</option><option value="poor">Poor / unintelligible</option></select></label><label>Pronunciation<textarea data-field="pronunciation" placeholder="Specific words, stress, vowels or consonants"></textarea></label><label>Completeness<textarea data-field="completeness" placeholder="Omitted, repeated or complete content"></textarea></label><label>Prosody<textarea data-field="prosody" placeholder="Timing, pauses, pitch and naturalness"></textarea></label><label>Additional notes<textarea data-field="notes"></textarea></label></div>`;
 for(const e of a.querySelectorAll('[data-field]')){e.value=n[e.dataset.field]||'';e.addEventListener('input',()=>{n[e.dataset.field]=e.value;if(storageAvailable){try{localStorage.setItem(key,JSON.stringify(n))}catch{}}progress()})}
 for(const b of a.querySelectorAll('[data-kind]'))b.onclick=()=>{const kind=b.dataset.kind;a.querySelector('.plot').src=`../Results/${s.language}/variance_adaptor_plots/${kind}/${s.id}_${kind}.png`;a.querySelector('.plot').alt=`${s.id} ${kind} predictions`};container.append(a)
}
document.querySelector('#language').onchange=e=>{for(const a of container.children)a.classList.toggle('hidden',e.target.value!=='All'&&a.dataset.language!==e.target.value)};
document.querySelector('#rate').onchange=e=>{for(const a of document.querySelectorAll('audio'))a.playbackRate=Number(e.target.value)};
document.querySelector('#reviewer').oninput=e=>{if(storageAvailable){try{localStorage.setItem('fs2-reviewer',e.target.value)}catch{}}progress()};
document.querySelector('#export').onclick=()=>{const reviewer=document.querySelector('#reviewer').value.trim();if(!reviewer){alert('Enter the reviewer name before exporting.');return}for(const n of Object.values(notes)){if(n.intelligibility!=='pending'){n.reviewer=reviewer;n.reviewed_at=new Date().toISOString()}}const url=URL.createObjectURL(new Blob([JSON.stringify({schema_version:1,samples:notes},null,2)+'\n'],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download='listening_review.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)};progress();
</script></body></html>'''
    destination = ROOT / "Report/audio_review.html"
    destination.write_text(template.replace("__SAMPLES__", json.dumps(samples, ensure_ascii=False).replace("</", "<\\/")).replace("__REVIEW__", json.dumps(review, ensure_ascii=False).replace("</", "<\\/")), encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    build()
