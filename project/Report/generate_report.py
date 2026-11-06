"""Build the 20-page PDF, self-contained interactive companion, and report measurements.

Reads submitted artifacts without executing or changing the acoustic pipeline.
Run from any directory: python project/Report/generate_report.py
"""
import argparse
import base64
import csv
import hashlib
import html
import io
import json
import re
from pathlib import Path
import statistics
import sys

import numpy as np
import reportlab
import PIL
from PIL import Image as PILImage
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.platypus import Paragraph, Table, TableStyle

from report_figures import (NAVY,TEAL,ORANGE,PALE,GRID,color,waveform,architecture,
    process_diagram,draw_diagram,diagram_svg,draw_prosody,prosody_svg,audio_image,telugu_bitmap)

OUT=Path(__file__).resolve().parent
ROOT=OUT.parent
REPO=ROOT.parent
URL='https://github.com/SandeshAwate6055/FastSpeech2-English-Telugu'
RAW='https://raw.githubusercontent.com/SandeshAwate6055/FastSpeech2-English-Telugu/main'
DRIVE='https://drive.google.com/drive/folders/1WsR1NHhzpkLvk3V0g8X3Jy-xsXNlsYHf?usp=sharing'
COLAB='https://colab.research.google.com/github/SandeshAwate6055/FastSpeech2-English-Telugu/blob/main/FastSpeech2_Colab.ipynb'
CONTEXT=json.loads((OUT/'evidence_context.json').read_text(encoding='utf8'))
PAGE_W,PAGE_H=595.276,841.89
LEFT,WIDTH=46,503.276
BODY=ParagraphStyle('body',fontName='Helvetica',fontSize=9.1,leading=12.4,textColor=color(NAVY))
SMALL=ParagraphStyle('small',parent=BODY,fontSize=7.7,leading=10.2)
TINY=ParagraphStyle('tiny',parent=BODY,fontSize=7,leading=9.1)
SUB=ParagraphStyle('sub',parent=BODY,fontName='Helvetica-Bold',fontSize=11.4,leading=15,textColor=color(TEAL))
TITLE=ParagraphStyle('title',parent=BODY,fontName='Helvetica-Bold',fontSize=23,leading=28)

def esc(s):return html.escape(str(s))
def link(label,path):
    url=path if '://' in path else URL+'/blob/main/'+path
    return f'<a href="{esc(url)}" color="{TEAL}">{esc(label)}</a>'
def page_link(label,n):return f'<a href="#page-{n}" color="{TEAL}">{esc(label)}</a>'
def data_uri(blob,mime):return 'data:'+mime+';base64,'+base64.b64encode(blob).decode()
def image_uri(im):
    buf=io.BytesIO();im.save(buf,format='PNG');return data_uri(buf.getvalue(),'image/png')

FOCUS={
 'E1':('Rare long word and completeness','The long word antidisestablishmentarianism stresses lexical coverage and articulation. Token 50 (AH0) has zero predicted frames and is therefore removed by length expansion. This is a concrete token-level omission risk; it does not establish that a whole audible word is missing.'),
 'E2a':('Noun-context conduct','The saved phone sequence contains the noun-stressed CONDUCT pronunciation K AA1 N D AH0 K T. Compare E2b: the same stress sequence is used there. The two contexts therefore do not receive distinct lexical stress labels from this saved frontend; final audible stress still needs listening.'),
 'E2b':('Verb-context conduct','The verb-context sentence uses the same K AA1 N D AH0 K T sequence as E2a, rather than the alternate dictionary entry K AH0 N D AH1 K T. This is an observed frontend stress limitation. Contextual acoustic differences cannot substitute for checking the intended pronunciation.'),
 'E3':('Broken word: rhyth  m','The raw input retains the assignment double space inside rhyth  m. Its normalized form is rhyth m: repeated whitespace collapses, but the word boundary remains. There are no explicit silence tokens. The relatively narrow F0 span suggests less predicted pitch variation; it does not prove monotone audible speech.'),
 'E4':('Time, date, loanword and pauses','The saved normalization expands 10:45 AM and the date, retaining andaroo. Two sp tokens occur at indices 55 and 60. Inspection of token neighbors places index 55 at the hyphen in twenty-six and index 60 at the comma: treating the numeric hyphen as punctuation can create an unintended pause inside the year.'),
 'T1':('Long technical word and clusters','Inspect the technical word in the original Telugu input, consonant clusters and vowel length. The rule-based Unicode frontend deterministically maps characters to phones, but that mapping alone cannot establish natural technical-word pronunciation. The predicted F0 declines from the opening to the ending token window.'),
 'T2a':('Grip/hold sense of pattu','The target word has the grip/hold sense, while T2b uses the silk sense. Both require the same orthographic consonant cluster/geminate handling. Compare timing and phrase contour rather than assuming the acoustic model encodes semantic understanding.'),
 'T2b':('Silk sense of pattu','This sentence provides the silk context for the same written target word used in T2a. The saved sentence has no explicit sp tokens; phrase timing is represented by phone durations. Correct geminate realization and natural contextual prosody require Telugu-proficient listening.'),
 'T3':('Broken Telugu word','The raw input retains the double space in the assignment broken word. Normalization collapses repeated spaces without joining the pieces. No explicit sp token is present, so source spacing must not be interpreted as a measured pause. Check whether the artificial boundary changes word intelligibility.'),
 'T4':('Year, ordinal, time and clause boundary','This is the longest Telugu sample: 120 tokens and 775 mel frames. The normalized text expands the year, ordinal date and time into Telugu words. A single sp token at index 94 expands for 52 frames (0.604 s), marking the comma/clause break; assess whether the numeric grammar and pause sound appropriate.'),
}

def histories(lang):
    out={}
    for split in ['train','val']:
        path=ROOT/f'Results/{lang}/metrics/{split}_losses.csv'
        rows=[{k:float(v) if k!='step' else int(v) for k,v in r.items()} for r in csv.DictReader(path.open(encoding='utf8'))]
        assert len(rows)==(600 if split=='train' else 60)
        assert [r['step'] for r in rows]==sorted(set(r['step'] for r in rows))
        assert all(np.isfinite(v) for row in rows for v in row.values())
        out[split]=rows
    out['best']=min(out['val'],key=lambda r:r['total_loss'])
    return out

def read_samples(review):
    out={};audit=[]
    for lang in ['English','Telugu']:
        rows=json.loads((ROOT/f'Results/{lang}/variance_predictions_summary.json').read_text(encoding='utf8'))
        for sid,row in rows.items():
            path=ROOT/f'Results/{lang}/test_generated_audios/{sid}.wav';signal,metrics=waveform(path)
            fields=['tokens','durations_frames','pitch_hz','energy'];n=len(row['tokens'])
            assert all(len(row[k])==n for k in fields)
            assert all(np.isfinite(v) for k in fields[1:] for v in row[k])
            assert all(isinstance(v,int) and v>=0 for v in row['durations_frames'])
            assert sum(row['durations_frames'])==row['total_frames']
            assert metrics['samples']==row['total_frames']*256
            digest=hashlib.sha256(path.read_bytes()).hexdigest()
            prior=review['samples'].get(sid,{})
            assert prior.get('audio_sha256',digest)==digest, 'Listening record does not match this audio'
            pauses=[{'index':i,'frames':d} for i,(t,d) in enumerate(zip(row['tokens'],row['durations_frames'])) if t in ['sp','sil','']]
            measurements={**metrics,'sha256':digest,'token_count':n,'mel_frames':row['total_frames'],'zero_duration_indices':[i for i,d in enumerate(row['durations_frames']) if d==0],'pause_tokens':pauses,'f0_mean':statistics.mean(row['pitch_hz']),'f0_min':min(row['pitch_hz']),'f0_max':max(row['pitch_hz']),'energy_mean':statistics.mean(row['energy']),'energy_min':min(row['energy']),'energy_max':max(row['energy']),'duration_median':statistics.median(row['durations_frames']),'duration_max':max(row['durations_frames']),'f0_first5_mean':statistics.mean(row['pitch_hz'][:5]),'f0_last5_mean':statistics.mean(row['pitch_hz'][-5:])}
            out[sid]={'language':lang,'row':row,'signal':signal,'metrics':measurements,'path':path,'review':prior}
            audit.append({'id':sid,'language':lang,**measurements})
    for sid in ['E2a','E2b']:
        tokens=out[sid]['row']['tokens'];target='K AA1 N D AH0 K T'.split()
        assert any(tokens[i:i+len(target)]==target for i in range(len(tokens)-len(target)+1))
    assert out['E4']['row']['tokens'][55]=='sp' and out['E4']['row']['tokens'][60]=='sp'
    return out,audit

class Document:
    def __init__(self):
        self.c=canvas.Canvas(str(OUT/'Technical_Report.pdf'),pagesize=(PAGE_W,PAGE_H),pageCompression=1)
        self.c.setTitle('FastSpeech 2: English and Telugu - Technical Report')
        self.c.setAuthor('Sandesh Awate')
        self.c.setSubject('Assignment implementation, measured results and interactive evaluator guide')
        self.c.setKeywords('FastSpeech2 English Telugu Svarah IndicTTS duration pitch energy')
        self.num=0;self.sections=[];self.fragments=[];self.bounds=[]
    def page(self,title,kicker='TECHNICAL REPORT',sample=None):
        if self.num:self.end_page()
        self.num+=1;self.sample=sample;self.fragments=[];self.title=title
        c=self.c;c.bookmarkPage(f'page-{self.num}');c.addOutlineEntry(f'{self.num:02d} | {title}',f'page-{self.num}',level=0)
        c.setFillColor(color(TEAL));c.rect(0,PAGE_H-10,PAGE_W,10,fill=1,stroke=0)
        c.setFont('Helvetica-Bold',8);c.setFillColor(color(NAVY));c.drawString(LEFT,809,'FASTSPEECH 2 / ENGLISH + TELUGU')
        c.setFont('Helvetica',7);c.drawRightString(LEFT+WIDTH,809,'ASSIGNMENT EVALUATION / 06 NOV 2026')
        c.setStrokeColor(color(GRID));c.line(LEFT,799,LEFT+WIDTH,799)
        self.y=780;self.p(f'{self.num:02d} / {kicker}',SUB,after=6);self.p(title,TITLE,after=15)
    def check(self,h):
        if self.y-h<53:raise RuntimeError(f'Page {self.num} ({self.title}) overflows by {53-(self.y-h):.1f} pt')
    def p(self,text,style=BODY,after=7):
        p=Paragraph(text,style);w,h=p.wrap(WIDTH,1000);self.check(h);p.drawOn(self.c,LEFT,self.y-h);self.y-=h+after
        self.fragments.append('<p>'+text.replace(' color="'+TEAL+'"','')+'</p>')
    def sub(self,text):self.p(esc(text),SUB,after=5)
    def table(self,rows,widths=None,size=SMALL):
        data=[[Paragraph(str(x),size) for x in row] for row in rows]
        t=Table(data,colWidths=widths or [WIDTH/len(rows[0])]*len(rows[0]),repeatRows=1)
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),color(PALE)),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6),('LINEBELOW',(0,0),(-1,0),.8,color(TEAL)),('LINEBELOW',(0,1),(-1,-1),.3,color(GRID))]))
        w,h=t.wrap(WIDTH,1000);self.check(h);t.drawOn(self.c,LEFT,self.y-h);self.y-=h+9
        self.fragments.append('<div class="table-wrap"><table>'+''.join('<tr>'+''.join(f'<{"th" if i==0 else "td"}>{x}</{"th" if i==0 else "td"}>' for x in row)+'</tr>' for i,row in enumerate(rows))+'</table></div>')
    def image(self,im,h=None,caption=None):
        if isinstance(im,Path):
            source=PILImage.open(im);uri=data_uri(im.read_bytes(),'image/png')
        else:source=im;uri=image_uri(im)
        target_h=h or WIDTH*source.height/source.width
        self.check(target_h);self.c.drawImage(ImageReader(source),LEFT,self.y-target_h,width=WIDTH,height=target_h,mask='auto');self.y-=target_h+7
        self.fragments.append(f'<img alt="{esc(caption or "Report figure")}" src="{uri}" loading="lazy">')
        if caption:self.p(esc(caption),TINY,after=6)
    def diagram(self,d,caption):
        h=WIDTH*d['height']/d['width'];self.check(h);draw_diagram(self.c,d,LEFT,self.y-h,WIDTH);self.y-=h+8
        self.fragments.append('<div class="diagram">'+diagram_svg(d)+'</div>');self.p(caption,SMALL)
    def code(self,text):
        lines=text.splitlines();height=len(lines)*10+14;self.check(height)
        self.c.setFillColor(color(PALE));self.c.roundRect(LEFT,self.y-height,WIDTH,height,5,fill=1,stroke=0)
        self.c.setFillColor(color(NAVY));self.c.setFont('Courier',7.25)
        for i,line in enumerate(lines):
            if self.c.stringWidth(line,'Courier',7.25)>WIDTH-16:raise RuntimeError('Code line too long: '+line)
            self.c.drawString(LEFT+8,self.y-12-i*10,line)
        self.y-=height+8;self.fragments.append('<pre><code>'+esc(text)+'</code></pre>')
    def sample_graph(self,row):
        self.check(184);draw_prosody(self.c,row,LEFT,self.y-184,WIDTH,184);self.y-=192
        self.fragments.append(prosody_svg(row))
    def end_page(self):
        c=self.c;c.setStrokeColor(color(GRID));c.line(LEFT,43,LEFT+WIDTH,43)
        c.setFillColor(color(TEAL));c.setFont('Helvetica',7);c.drawString(LEFT,29,'CONTENTS / SAMPLE INDEX')
        c.linkRect('Return to contents','page-1',(LEFT,21,190,40),thickness=0)
        c.drawCentredString(PAGE_W/2,29,'Measured artifacts + explicit evidence limits')
        c.setFillColor(color(NAVY));c.drawRightString(LEFT+WIDTH,29,f'{self.num:02d} / 20')
        self.bounds.append({'page':self.num,'title':self.title,'lowest_content_y':round(self.y,2)})
        self.sections.append({'number':self.num,'title':self.title,'sample':self.sample,'html':'\n'.join(self.fragments)})
        c.showPage()
    def finish(self):
        self.end_page();assert self.num==20,self.num;self.c.save()

def intro_pages(d,hist):
    d.page('English and Telugu\nText-to-Speech','PROJECT / EVALUATOR GUIDE')
    d.p('Two independent acoustic models. Explicit prosody prediction. A linked, evidence-based demonstration from dataset selection to generated speech.',SUB)
    d.sub('Abstract')
    d.p('This assignment adapts ming024/FastSpeech2 for Svarah English and IndicTTS Telugu. A phoneme encoder, learned speaker embedding, duration/pitch/energy variance adaptor, length regulator and mel decoder form each acoustic model. A pretrained universal HiFi-GAN produces waveforms. This report explains the language changes, data protocol, training curves, all ten required inputs and the saved prosody predictions. It distinguishes current source behavior, historical trained artifacts and uncompleted listening judgments.')
    d.table([['Evaluation artifact','Submitted evidence'],['Separate acoustic models','English: step 8,000, speaker 14; Telugu: step 13,000, speaker 0.'],['Training record','60,000 logged steps each; selected checkpoints match the minimum logged total validation loss.'],['Outputs','10 PCM WAVs, 30 variance PNGs, 4 loss PNGs and 4 CSV histories.'],['Interactive evaluation','Companion HTML embeds all ten WAVs, hoverable plots, exact text, token tables and listening-note export.']],[125,WIDTH-125])
    d.sub('Start here: links and navigation')
    d.p(link('Main Drive: checkpoints and experiment artifacts',DRIVE)+' | '+link('Colab pipeline',COLAB)+' | '+link('Repository and detailed README',URL),SMALL)
    d.p(link('Download/open the interactive HTML companion','project/Report/Interactive_Report.html')+' | '+link('Primary assignment PDF','FastSpeech2_Assignment.pdf')+' | '+link('Report build instructions','project/Report/README.md'),SMALL)
    d.table([['Requirement / section','Pages'],['Architecture, reference reuse and language changes',page_link('2-3',2)],['Data, alignment, training settings and complete process',page_link('4',4)],['English / Telugu convergence and instability',page_link('5-6',5)+' / '+page_link('7-8',7)],['Every required sample: speech measurements, duration, F0, energy',page_link('9-13 English',9)+' / '+page_link('14-18 Telugu',14)],['Comparison, failures, limitations and improvements',page_link('19',19)],['Dependencies, configurations, commands and access',page_link('20',20)]],[355,WIDTH-355])
    d.p('<b>Evidence status:</b> the committed outputs are inspectable. Speech pronunciation, intelligibility and naturalness remain unverified by a completed listening review. Reserve waveform-hash coverage and omitted runtime helpers prevent a claim of fully reproduced training/inference. Those gaps are recorded explicitly on pages 19-20.',SMALL)

    d.page('Correct system architecture','1 / IMPLEMENTATION')
    d.diagram(architecture(),'<b>Figure 1.</b> Each language uses this topology with independent acoustic weights and metadata. Duration and pitch receive speaker-conditioned states; energy receives pitch-conditioned states. Embeddings are added before length expansion. Arrows show actual model data flow.')
    d.p(link('Acoustic forward pass','project/English/model/fastspeech2.py')+' | '+link('Variance adaptor','project/English/model/modules.py')+' | '+link('Vocoder integration','project/English/utils/model.py'),SMALL)
    d.sub('Reference-conditioned training versus free-running inference')
    d.p('Training uses aligned durations and reference pitch/energy to condition mel reconstruction, while all three heads receive prediction losses. Inference supplies predicted duration, pitch and energy instead. The model is non-autoregressive after duration expansion. Low reference-conditioned mel loss therefore does not establish free-running intelligibility.')
    d.code('z_i target = log(1 + d_i)\nd_hat_i = max(round(exp(z_i) - 1), 0)  # controls = 1\nT_mel = sum_i(d_hat_i); seconds = T_mel * 256 / 22050\nL = L1(mel, target) + L1(postnet_mel, target)\n  + MSE(z, log(1+d)) + MSE(pitch, target) + MSE(energy, target)')
    d.p('Padding masks exclude padded phones/frames from loss. Pitch is interpolated PyWorld F0 averaged per phone and normalized. Energy is the phone average of the L2 norm of linear STFT magnitudes, not dB SPL. Plots reverse normalization using the matching statistics. '+link('Loss implementation','project/English/model/loss.py'),SMALL)

    d.page('Reference reuse and language adaptations','2 / IMPLEMENTATION')
    d.p('Starting point: '+link('ming024/FastSpeech2','https://github.com/ming024/FastSpeech2/tree/d4e79eb52e8b01d24703b2dfc0385544092958f3')+'. The report records reference d4e79eb; the original training commit and exact environment are not established. Current source fixes do not retroactively retrain the submitted checkpoint/audio artifacts.',SMALL)
    d.table([['Component','English adaptation','Telugu adaptation'],['Data preparation','Svarah loader; eight-field demographic profiles; filtering, disjoint pools and reserves.','IndicTTS loader; female/male source labels; independent filtering and reserves.'],['Frontend','Lexicon lookup plus g2p_en OOV fallback; number/time/date and abbreviation expansion.','Unicode phones: consonants, inherent vowel, vowel signs, virama, anusvara and visarga; Telugu numeric expansion.'],['Symbols / speakers','ARPAbet with lexical stress; learned profile embeddings. Profiles are not proven unique people.','Language-specific phone inventory; learned embeddings for two source voices.'],['Preprocessing','Interior silence/gaps retained; invalid spans/spn rejected; finite/length checks; training-only normalization and bounds.','Same checks with consistent Telugu lexicon/frontend boundaries; historical target mismatches remain.'],['Model / controls','FFT backbone retained; phone-level variance heads; energy-control routing and device-aware length regulator.','Same design, independent acoustic weights and statistics.'],['Training / inference','CSV/figure outputs, best-validation selection, explicit checkpoint/speaker paths and prediction JSON.','Independent language results with the same audit-friendly outputs.']],[85,209,WIDTH-294],TINY)
    d.p('Source map: '+link('English preparation','project/English/prepare_data.py')+' / '+link('Telugu preparation','project/Telugu/prepare_data.py')+'; '+link('English cleaner','project/English/text/cleaners.py')+' / '+link('Telugu cleaner','project/Telugu/text/cleaners.py')+'; '+link('alignment/features','project/Telugu/preprocessor/preprocessor.py')+'.',SMALL)
    d.sub('Differences from the supplied paper v8')
    d.table([['Aspect','Paper v8','Project adaptation'],['Decoder / dropout','4 FFT blocks / 0.1','6 FFT blocks / 0.2'],['Pitch / energy','Frame CWT pitch / frame energy','Direct phone-level F0 / phone energy'],['Output / vocoder','No added PostNet / Parallel WaveGAN','Residual PostNet / universal HiFi-GAN'],['Batch / training steps','48 / 160k for paper FastSpeech 2','16 / 60k logged per language']],[100,187,WIDTH-287],TINY)
    d.p('Paper comparison: supplied arXiv v8, Section 2.3 and Appendix A-B. This is FastSpeech 2 acoustic synthesis with a separate vocoder, not the paper\'s direct-waveform FastSpeech 2s. English first-pronunciation lookup can miss contextual stress; Telugu decomposition is a phonemic approximation with allophone/loanword limits.',SMALL)

    d.page('Data-to-speech process and training setup','3 / DATA & EXPERIMENT')
    d.diagram(process_diagram(),'<b>Figure 2.</b> Complete intended lifecycle. The saved outputs are historical artifacts; reproduction requires the omitted helpers and matching bundles described on page 20.')
    d.table([['Retained data (recovered evidence)','English / Svarah','Telugu / IndicTTS'],['Training utterances / retained hours','3,359 / 5.000 h','2,164 / 5.000 h'],['Validation utterances / hours','674 / 1.001 h','421 / 1.000 h'],['Training / validation groups','115 / 114 demographic profiles','2 / 2 source voices'],['ID overlap / missing final audio hashes','0 / 382 reserve utterances','0 / 339 reserve utterances']],[215,145,WIDTH-360],TINY)
    d.p('Selection resolves a dataset revision, uses seed 42, filters 1-20 s clips, writes 22,050 Hz mono PCM/LAB files and deduplicates waveform SHA-1 values. Voice/profile-stratified pools are shuffled and split about 5:1; separate 0.75h/0.20h reserves replenish rejected train/validation clips. Svarah\'s source pool is its test split; the assignment partitions are created from that pool. Recovered manifests confirm zero ID overlap and zero overlap among known hashes; missing reserve hashes prevent a complete final waveform-level claim.',SMALL)
    d.p('MFA phone TextGrids must precede feature extraction. Compatible lexicon/acoustic phones and unchanged waveform/transcript correspondence are required. Current code preserves interior pauses and checks the full aligned timeline before slicing. PyWorld F0 is interpolated and phone-averaged; energy is phone-averaged STFT magnitude. Selected training examples alone determine current normalization and quantization bounds.',SMALL)
    d.p('<b>Configuration:</b> 4 encoder / 6 decoder FFT blocks; hidden 256; 2 heads; convolution filter 1,024, kernels 9/1; predictor channels 256, kernel 3, dropout 0.5; 256 linear pitch/energy bins; 80 mel channels, FFT/window 1,024, hop 256, 0-8 kHz. Adam betas (0.9, 0.98), epsilon 1e-9, batch 16, clipping 1.0, warmup 4k, anneal 30k/45k/55k by 0.3, total 60k. These are checked-in defaults, not a recovered environment lock.',SMALL)
    d.p(link('Dataset: Svarah','https://huggingface.co/datasets/ai4bharat/Svarah')+' | '+link('Dataset: IndicTTS Telugu','https://huggingface.co/datasets/SPRINGLab/IndicTTS_Telugu')+' | '+link('English config','project/English/config.yaml')+' / '+link('Telugu config','project/Telugu/config.yaml')+' | '+link('Recovered-evidence summary','project/Report/evidence_context.json'),TINY)

def draw_gap(d,hist):
    c=d.c;h=127;d.check(h);y=d.y-h;train={r['step']:r['total_loss'] for r in hist['train']};v=hist['val'];hi=max(max(train[r['step']],r['total_loss']) for r in v)*1.1
    c.setStrokeColor(color(GRID));c.line(LEFT+30,y+22,LEFT+WIDTH-5,y+22);c.line(LEFT+30,y+22,LEFT+30,y+110)
    for field,tone,label,lx in [('train',TEAL,'Sampled training',LEFT+275),('val',ORANGE,'Validation',LEFT+395)]:
        c.setStrokeColor(color(tone));p=c.beginPath()
        for i,r in enumerate(v):
            x=LEFT+30+r['step']/60000*(WIDTH-40);z=train[r['step']] if field=='train' else r['total_loss'];py=y+22+z/hi*88
            if i==0:p.moveTo(x,py)
            else:p.lineTo(x,py)
        c.drawPath(p);c.setFillColor(color(tone));c.setFont('Helvetica',7);c.drawString(lx,y+116,label)
    c.setFillColor(color(NAVY));c.setFont('Helvetica',7)
    for step in [0,20000,40000,60000]:c.drawCentredString(LEFT+30+step/60000*(WIDTH-40),y+8,str(step//1000)+'k')
    for val in [0,hi/2,hi]:c.drawRightString(LEFT+25,y+21+val/hi*88,f'{val:.1f}')
    d.y-=h+8
    d.fragments.append('<p>See the embedded original curves and the numerical train/validation table for the generalization gap.</p>')

def loss_pages(d,lang,hist):
    n=5 if lang=='English' else 7
    d.page(f'{lang}: training and validation curves','4 / CONVERGENCE')
    for kind in ['training','validation']:
        d.image(ROOT/f'Results/{lang}/{kind}_loss.png',caption=f'{lang} {kind}: total, mel, PostNet, duration, pitch and energy losses. Original submitted figure, original aspect ratio.')
    d.p('Training continues to decrease while validation improves mainly early and then plateaus. The selected validation minimum occurs before the final 60k step. No exploding or non-finite loss is present in the supplied CSVs; early fluctuations and the persistent generalization gap remain visible. Low losses alone cannot establish correct free-running speech.',SMALL)
    d.p(link('Training CSV',f'project/Results/{lang}/metrics/train_losses.csv')+' | '+link('Validation CSV',f'project/Results/{lang}/metrics/val_losses.csv')+' | '+page_link('Numerical analysis',n+1),SMALL)
    d.page(f'{lang}: checkpoint selection and overfitting','4 / NUMERICAL ANALYSIS')
    best=hist['best'];tmap={r['step']:r for r in hist['train']};rows=[['Record / step','Total','Mel','PostNet','Pitch','Energy','Duration']]
    chosen=[('Train first',hist['train'][0]),('Train selected',tmap[best['step']]),('Train final',hist['train'][-1]),('Val first',hist['val'][0]),('Val minimum',best),('Val final',hist['val'][-1])]
    for label,r in chosen:rows.append([f'{label}<br/>{r["step"]:,}']+[f'{r[k]:.4f}' for k in ['total_loss','mel_loss','mel_postnet_loss','pitch_loss','energy_loss','duration_loss']])
    d.table(rows,[103]+[(WIDTH-103)/6]*6,TINY)
    improvement=100*(1-best['total_loss']/hist['val'][0]['total_loss']);last=hist['val'][-1];rise=100*(last['total_loss']/best['total_loss']-1)
    d.p(f'The histories contain 600 training rows and 60 validation rows. Minimum total validation loss is <b>{best["total_loss"]:.6f} at step {best["step"]:,}</b>: {improvement:.2f}% below the first validation value. The final validation total is {last["total_loss"]:.6f}, {rise:.2f}% above that minimum. The submitted prediction JSON identifies this selected step.',BODY)
    d.sub('Generalization gap at validation checkpoints')
    draw_gap(d,hist)
    d.p('The continuing fall in training error alongside a nearly flat or rising validation curve is consistent with overfitting. Mel and PostNet validation losses no longer track the reduction in training losses. The early checkpoint is the best by the logged total-loss rule; perceptual checkpoint selection remains provisional until held-out free-running audio is reviewed.')
    d.p('Training rows are sampled minibatches, whereas validation is aggregated over held-out data. Their difference is descriptive, not a matched-sample confidence estimate. The component losses mix normalized features and mel error; absolute totals across English and Telugu cannot rank naturalness. Hardware, elapsed training time and an exact historical dependency lock are not established by these CSVs.',SMALL)
    d.p('The total loss is the unweighted sum of the five displayed terms. Validation occurs every 1,000 steps in the supplied histories. No loss smoothing or invented intermediate measurements are used in these tables.',SMALL)

def sample_page(d,sid,item):
    row=item['row'];m=item['metrics'];lang=item['language'];title,focus=FOCUS[sid]
    d.page(f'{sid}: {title}','5 / PER-SENTENCE EVALUATION',sample=sid)
    d.p(f'<b>{lang}</b> | acoustic step {row["checkpoint_step"]:,} | speaker {row["speaker_id"]} | '+link('Play / download WAV',RAW+f'/project/Results/{lang}/test_generated_audios/{sid}.wav')+' | '+link('Raw text and arrays',f'project/Results/{lang}/variance_predictions_summary.json'),SMALL,after=5)
    d.p('<b>Raw assignment input:</b>',TINY,after=3)
    if lang=='Telugu':
        d.image(telugu_bitmap(row['raw_text']))
        d.fragments[-1]='<p lang="te" style="white-space:pre-wrap;font-size:1.15em">'+esc(row['raw_text'])+'</p>'
    else:d.p(esc(row['raw_text']).replace('  ','&#160;&#160;'),SMALL,after=6)
    d.p('<b>Saved normalized model input:</b>',TINY,after=3)
    if lang=='Telugu':
        d.image(telugu_bitmap(row['normalized_text']))
        d.fragments[-1]='<p lang="te" style="white-space:pre-wrap;font-size:1.15em">'+esc(row['normalized_text'])+'</p>'
    else:d.p(esc(row['normalized_text']),TINY,after=6)
    d.table([['Measured waveform / prediction','Value'],['Tokens / frames / WAV duration',f'{m["token_count"]} / {m["mel_frames"]} / {m["seconds"]:.3f} s'],['F0 mean [range]; energy mean [range]',f'{m["f0_mean"]:.2f} [{m["f0_min"]:.2f}, {m["f0_max"]:.2f}] Hz; {m["energy_mean"]:.2f} [{m["energy_min"]:.2f}, {m["energy_max"]:.2f}]'],['Duration median / max; zero phones; pause frames',f'{m["duration_median"]:g} / {m["duration_max"]}; {len(m["zero_duration_indices"])}; {sum(p["frames"] for p in m["pause_tokens"])}']],[225,WIDTH-225],TINY)
    d.sample_graph(row)
    d.image(audio_image(item['signal'],m['sample_rate']),h=56,caption='Waveform + 0-8 kHz spectrogram; Hann 1,024 / hop 128; colors are dB relative to utterance peak.')
    d.p(f'<b>Measured audio:</b> RMS {m["rms_dbfs"]:.1f} dBFS; peak {m["peak_dbfs"]:.1f} dBFS; {m["full_scale_samples"]} full-scale samples. Nonzero energy and no full-scale saturation do not establish intelligibility.',TINY)
    d.p('<b>Sentence-specific analysis:</b> '+esc(focus),SMALL)
    d.p(f'<b>Duration:</b> median {m["duration_median"]:g} frames; longest {m["duration_max"]} frames ({m["duration_max"]*256/22050:.3f} s) at token {int(np.argmax(row["durations_frames"]))}. <b>F0:</b> peak {m["f0_max"]:.2f} Hz at token {int(np.argmax(row["pitch_hz"]))}; first/last five-token means {m["f0_first5_mean"]:.1f}/{m["f0_last5_mean"]:.1f} Hz. <b>Energy:</b> maximum {m["energy_max"]:.2f} at token {int(np.argmax(row["energy"]))}, minimum {m["energy_min"]:.2f} at token {int(np.argmin(row["energy"]))}. These positions identify timing/emphasis candidates and weak-phone/pause regions for listening, rather than proving lexical stress. Token means are unweighted; phone identity and F0 interpolation affect the curves.',TINY)
    r=item['review'];reviewed=r.get('intelligibility','pending')!='pending' and bool(r.get('reviewer'))
    if reviewed:
        d.p('<b>Listening observation (reviewer-reported):</b> '+esc('; '.join(f'{k}: {r.get(k,"")}' for k in ['reviewer','intelligibility','pronunciation','completeness','prosody','notes'])),TINY)
    else:
        d.p('<b>Pronunciation / overall quality:</b> listening review pending. Inspect the target words, omissions/repetitions, naturalness and phrase timing in the exact linked WAV; use the interactive companion to record notes. Predicted F0 is phone-level and interpolated, not a measured generated-waveform F0 track.',TINY)
    links=[link(k.title(),f'project/Results/{lang}/variance_adaptor_plots/{k}/{sid}_{k}.png') for k in ['duration','pitch','energy']]
    d.p('Original labeled plots: '+' | '.join(links)+' | '+page_link('All sample links',1),TINY)

def final_pages(d,samples):
    d.page('Comparison, limitations and improvements','6 / CROSS-LANGUAGE EVALUATION')
    d.table([['Dimension','English','Telugu'],['Source / conditioning','Accent-diverse ASR recordings; 115 demographic training profiles.','Studio TTS recordings; two source voice labels.'],['Submitted synthesis','8k checkpoint / profile 14.','13k checkpoint / female voice 0.'],['Frontend evidence','Same noun-stress conduct in E2a/E2b; broken rhyth m; numeric hyphen inserts sp.','Rule-based Unicode phones; broken word; date/time/ordinal expansion.'],['Predicted variance','E1 has one zero-duration phone; E4 has 76 pause frames in two sp tokens.','No zero-duration phones in five saved arrays; T4 has one 52-frame sp token.'],['Generalization','Validation minimum 2.702820, then plateau despite lower training loss.','Validation minimum 1.965954, then plateau despite lower training loss.'],['Quality conclusion','Pronunciation/intelligibility/naturalness pending listening.','Same; review by a Telugu-proficient listener is needed.']],[90,206,WIDTH-296],TINY)
    d.sub('What the evidence supports')
    d.p('All ten submitted WAVs are 22,050 Hz mono 16-bit PCM, nonempty and consistent with predicted frame totals. All three predictor arrays align with the saved phone list, and their duration sums match mel frames. The report includes measured RMS/peak levels and checks full-scale saturation. These checks establish structural/audio-signal properties; they do not prove correct words, natural speech or lexical stress.',SMALL)
    d.sub('Known failures and remaining evidence')
    d.p('Earlier alignment code omitted empty interior pauses while trimming retained their audio, shifting later phone/frame correspondence. Current source preserves gaps, rejects invalid spans and unknown-speech spn, and uses selected-training-only statistics. The earlier report also records 82 historical Telugu target/frontend mismatches (73 train / 9 validation). Those targets and trained weights need separate review; code corrections alone do not repair historical artifacts.',SMALL)
    d.p('Recovered manifests have zero utterance-ID overlap and zero overlap among known waveform hashes. Final waveform hashes are missing for 382 retained English reserve clips and 339 Telugu clips. Full manifests and acoustic runtime helpers are omitted from this repository snapshot. Public Drive downloads, original hardware/environment, and completed per-sample listening records are not established by this report build.',SMALL)
    d.sub('Evidence-led improvement sequence')
    d.p('1. Restore missing runtime files and verify all model/metadata/vocoder checksums. 2. Repair alignment/target disagreements and complete final split-hash evidence. 3. Review real-mel vocoder reconstructions and compare reference-conditioned versus free-running held-out audio. 4. Correct context-sensitive English stress and numeric punctuation; validate Telugu allophones, clusters and numeric grammar. 5. Use held-out speech plus losses for checkpoint selection, then consider richer pitch modeling or vocoder adaptation. More steps alone are not justified by the current plateaus.',SMALL)
    d.p('Comparison limit: corpus recording conditions, speaker characteristics and feature normalization differ. Lower Telugu losses or higher F0 values cannot establish a causal language advantage or higher naturalness. No MOS, WER, speaker-similarity or listening scores are invented.',SMALL)

    d.page('Reproducibility, commands and evaluator access','7 / REPRODUCTION & REFERENCES')
    d.p(link('Colab pipeline',COLAB)+' | '+link('Main Drive',DRIVE)+' | '+link('English config','project/English/config.yaml')+' / '+link('Telugu config','project/Telugu/config.yaml')+' | '+link('Dependencies','project/requirements.txt'),SMALL)
    d.p('<b>Current execution blockers:</b> scripts import run_support from project/tools/, which is absent. Colab also references omitted restore_artifacts.py, colab_pipeline.py, probe_audio.py, model_artifacts.json and Results/test_sentences.json, and expects root-level language paths instead of project/. Restore matching helpers and correct notebook paths before running. The commands below describe a restored environment; this report build does not claim GPU inference or retraining.',SMALL)
    d.code('git clone https://github.com/SandeshAwate6055/FastSpeech2-English-Telugu.git\ncd FastSpeech2-English-Telugu\npython -m pip install -r project/requirements.txt\n# Restore shared helpers, matching metadata, acoustic and vocoder weights.\ncd project/English\npython prepare_data.py --train_hours 5.0 --val_hours 1.0 --seed 42\n# Obtain compatible MFA phone TextGrids before preprocessing.\npython preprocess.py --config config.yaml\n# Validate features/provenance and held-out speech before full training.\npython train.py --config config.yaml --total_step 60000\npython inference.py --config config.yaml \\\n  --ckpt_file checkpoints/8000.pth.tar --speaker_id 14 \\\n  --result_path ../Results/reproduced/English\n# Repeat independent preparation, alignment and training from ../Telugu.\ncd ../Telugu\npython inference.py --config config.yaml \\\n  --ckpt_file checkpoints/13000.pth.tar --speaker_id 0 \\\n  --result_path ../Results/reproduced/Telugu')
    d.p('Inference omits --text to synthesize all five built-in assignment sentences and saves WAVs, three plots per sample and prediction JSON. Use fresh result paths to preserve supplied artifacts. MFA is a separate prerequisite; prepare_data.py alone does not produce TextGrids. New aligner fitting must exclude validation audio. Svarah may require Hugging Face approval/HF_TOKEN. Install NLTK tagger resources and cmudict as described in the README.',TINY)
    d.p('<b>Model restoration:</b> '+link('English 8k download','https://drive.google.com/uc?export=download&id=1asQbbLU4y9qPAHd8Jvu2T1G-5VES6X7P')+' / '+link('Telugu 13k download','https://drive.google.com/uc?export=download&id=1loNuZ9gRVLTobXHzzwZGsDe8KkB_FB-p')+'. Keep original speakers.json, stats.json, symbol inventory/lexicon and universal HiFi-GAN weights coupled to each checkpoint. '+link('English hashes/metadata','project/English/checkpoints/README.md')+' / '+link('Telugu hashes/metadata','project/Telugu/checkpoints/README.md')+'. The companion includes exact checkpoint hashes. External downloads were not revalidated here.',TINY)
    d.p('<b>Software:</b> acoustic requirements use lower bounds, not a version lock (PyTorch/torchaudio, NumPy/SciPy/librosa, PyWorld, tgt, g2p_en, datasets, PyYAML, etc.). Record tested Python, CUDA/PyTorch, MFA and dataset revisions for reproduction. Report-only build uses reportlab, Pillow and NumPy; rendering checks use pypdf/pypdfium2. '+link('Report build and HTML usage','project/Report/README.md'),TINY)
    d.p('<b>References:</b> [1] '+link('Assignment Sections 1-4','FastSpeech2_Assignment.pdf')+'. [2] '+link('FastSpeech 2 paper, supplied v8','https://arxiv.org/abs/2006.04558')+', Section 2.3 and Appendix A-B. [3] '+link('ming024 reference','https://github.com/ming024/FastSpeech2')+'. [4] '+link('Svarah','https://huggingface.co/datasets/ai4bharat/Svarah')+' and '+link('IndicTTS Telugu','https://huggingface.co/datasets/SPRINGLab/IndicTTS_Telugu')+'. [5] '+link('Measured report data','project/Report/report_measurements.json')+'; current source, CSVs, WAVs and prediction JSON linked throughout.',TINY)

def interactive_html(d,samples,audit,hist):
    css='''*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:#f2f5f8;color:#142c44;font:16px/1.65 system-ui,"Nirmala UI",sans-serif}header{background:#142c44;color:white;padding:48px max(24px,calc((100vw - 1100px)/2))}header h1{font-size:clamp(30px,5vw,54px);line-height:1.12;margin:12px 0}header p{max-width:850px;color:#dbe8ed}a{color:#087f8c}header a{color:#b6f2e8}nav{position:sticky;top:0;z-index:4;background:white;border-bottom:1px solid #d5e0e6;padding:12px 20px;display:flex;gap:14px;flex-wrap:wrap;align-items:center}main{max-width:1140px;margin:auto;padding:24px}section{scroll-margin-top:110px;background:white;border:1px solid #d5e0e6;border-radius:14px;padding:28px;margin-bottom:24px;box-shadow:0 3px 15px #142c4407}h2{font-size:27px;line-height:1.25}h3{color:#087f8c}.table-wrap{overflow:auto}table{border-collapse:collapse;width:100%;font-size:.91em;margin:14px 0}td,th{padding:11px;border-bottom:1px solid #d5e0e6;text-align:left;vertical-align:top}th{background:#eef5f7}img,svg{max-width:100%;height:auto}img{display:block;margin:18px auto}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#eef5f7;padding:16px;border-radius:8px;font-size:13px}button,select,input,textarea{font:inherit}button{border:1px solid #087f8c;border-radius:7px;background:white;color:#087f8c;cursor:pointer;padding:7px 12px}button:hover{background:#e5f7f3}audio{width:100%;margin:12px 0}summary{cursor:pointer;font-size:1.2em;font-weight:700;padding:6px 0}details{margin:14px 0}.tag{display:inline-block;font-size:13px;border-radius:30px;padding:4px 12px;background:#eef5f7}.notice{padding:16px;border-left:4px solid #b85e32;background:#fff6ef}.review{display:grid;grid-template-columns:1fr 1fr;gap:12px}.review label{display:block;font-size:14px}.review input,.review select,.review textarea{display:block;width:100%;border:1px solid #becdd7;border-radius:6px;padding:8px;background:white}.review textarea{min-height:75px}.review .wide{grid-column:1/-1}.sample[hidden]{display:none}.token-table{max-height:320px;overflow:auto}.small{font-size:13px;color:#4b6070}.diagram{background:#fff;border:1px solid #edf2f5;padding:12px;border-radius:10px}.prosody{border:1px solid #d5e0e6;border-radius:8px;padding:12px}svg g:focus{outline:2px solid #b85e32}.toolbar{display:flex;gap:10px;flex-wrap:wrap}.review-status{font-weight:700;color:#b85e32}@media(max-width:650px){main{padding:10px}section{padding:18px}.review{grid-template-columns:1fr}nav{position:static}}@media print{nav,button,audio,.review{display:none}section{break-before:page;box-shadow:none;border:0}details>*{display:block!important}body{background:white}}'''
    navigation='<nav aria-label="Report navigation"><a href="#page-1">Overview</a><a href="#page-2">Architecture</a><a href="#page-4">Process/data</a><a href="#page-5">Losses</a><a href="#samples">10 samples</a><a href="#page-19">Comparison</a><a href="#page-20">Reproduce</a><button id="export">Export listening notes</button><button id="print">Print</button></nav>'
    out=['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>FastSpeech 2 - Interactive Technical Report</title><style>'+css+'</style><header><span class="tag" style="color:#142c44">ENGLISH + TELUGU / TECHNICAL REPORT</span><h1>From text to speech,<br>with the evidence in view.</h1><p>Two separate acoustic models. Ten embedded audio samples. Explorable duration, pitch and energy predictions. This self-contained report works after download without loading third-party scripts.</p><p><a href="'+URL+'/blob/main/project/Report/Technical_Report.pdf">PDF report</a> · <a href="'+COLAB+'">Colab</a> · <a href="'+DRIVE+'">Main Drive</a> · <a href="'+URL+'">Repository</a></p></header>',navigation,'<main>']
    review_seed={sid:item['review'] for sid,item in samples.items()}
    sample_index=False
    for sec in d.sections:
        sid=sec['sample'];num=sec['number']
        if sid and not sample_index:
            out.append('<section id="samples"><h2>Listen and inspect all ten samples</h2><p>Audio is embedded from the committed WAV bytes. Listen before recording pronunciation, completeness, prosody and quality. Hover or keyboard-focus plot points for phone values; use the token tables for exact arrays. Notes stay in this browser unless you export them.</p><div class="toolbar"><button data-filter="all">All 10</button><button data-filter="English">English</button><button data-filter="Telugu">Telugu</button><button id="expand">Expand samples</button><button id="collapse">Collapse samples</button></div><p>'+ ' · '.join(f'<a href="#page-{i+9}">{s}</a>' for i,s in enumerate(samples))+'</p><p class="notice">Perceptual listening review is pending in the published evidence. The form below starts pending; waveform and predictor statistics do not automatically populate human judgments.</p></section>');sample_index=True
        attrs=f' class="sample" data-language="{samples[sid]["language"]}"' if sid else ''
        out.append(f'<section id="page-{num}"{attrs}>')
        if sid:
            item=samples[sid];row=item['row'];out.append(f'<details class="sample-detail"><summary>{esc(sec["title"])} <span class="tag">{item["metrics"]["seconds"]:.3f} s</span></summary>')
            out.append(f'<audio controls preload="none" aria-label="Play {sid} {item["language"]} speech"><source type="audio/wav" src="{data_uri(item["path"].read_bytes(),"audio/wav")}">Your browser cannot play this WAV. Use the direct WAV link.</audio>')
        else:out.append(f'<h2>{num:02d} / {esc(sec["title"])}</h2>')
        out.append(sec['html'])
        if sid:
            row=item['row'];out.append('<details><summary>Full token-level predictions</summary><div class="token-table"><table><tr><th>Index</th><th>Phone</th><th>Frames</th><th>F0 / Hz</th><th>Energy</th></tr>')
            out.extend(f'<tr><td>{i}</td><td lang="{ "te" if item["language"]=="Telugu" else "en"}">{esc(t)}</td><td>{row["durations_frames"][i]}</td><td>{row["pitch_hz"][i]:.2f}</td><td>{row["energy"][i]:.2f}</td></tr>' for i,t in enumerate(row['tokens']))
            out.append('</table></div></details><h3>Listening review: '+sid+'</h3><p class="review-status" id="status-'+sid+'">Pending</p><div class="review" data-sample="'+sid+'">')
            out.append('<label>Reviewer<input data-field="reviewer" placeholder="Your name or reviewer label"></label><label>Intelligibility<select data-field="intelligibility"><option value="pending">Pending review</option><option>Mostly clear</option><option>Partly intelligible</option><option>Unintelligible</option></select></label>')
            for field,label in [('pronunciation','Pronunciation: target words and errors'),('completeness','Completeness: missing or repeated content'),('prosody','Prosody: timing, pitch, pauses and stress'),('notes','Overall speech quality / other observations')]:
                out.append(f'<label>{label}<textarea data-field="{field}" placeholder="Record what you actually hear"></textarea></label>')
            out.append('</div><p class="small">Linked to audio SHA-256 '+item['metrics']['sha256']+'. Exported notes are not published automatically. Rebuild with --review-json only after the reviewer identifies the exact audio and supplies observations.</p></details>')
        out.append('<p class="small"><a href="#page-1">Back to report contents</a></p></section>')
    out.append('<section id="bundles"><h2>Model bundles and full integrity details</h2><p>External links are owner-supplied; public downloads were not revalidated by this report build. Restore matching statistics, speaker IDs and vocoder weights with the acoustic checkpoint.</p><table><tr><th>Language</th><th>Checkpoint SHA-256</th><th>Size / step</th></tr>')
    for lang in ['English','Telugu']:
        m=CONTEXT['models'][lang];out.append(f'<tr><td><a href="{m["download_url"]}">{lang}</a></td><td><code style="overflow-wrap:anywhere">{m["sha256"]}</code></td><td>{m["size_bytes"]:,} bytes / {m["checkpoint_step"]:,}</td></tr>')
    out.append('</table><h3>Report-only rebuild</h3><pre>python -m pip install -r project/Report/requirements.txt\npython project/Report/generate_report.py\n# Optional: incorporate completed, exact-audio reviewer records\npython project/Report/generate_report.py --review-json listening_review.json</pre><p>Input artifacts are read without running or changing acoustic-model code. The report measurements JSON records audio hashes and structural checks. Windows uses Nirmala UI through GDI for shaped Telugu in the PDF; non-Windows builds require Pillow libraqm and TELUGU_FONT. In the HTML, UTF-8 Telugu remains selectable text.</p></section>')
    state={'schema_version':1,'samples':review_seed}
    script=r'''const seed=SEED;const key='fastspeech2-listening-report-v1';let saved={};try{saved=JSON.parse(localStorage.getItem(key)||'{}')}catch(e){}const state={schema_version:1,samples:{}};
for(const [sid,row] of Object.entries(seed.samples)){state.samples[sid]={...row,...(saved.samples?.[sid]?.audio_sha256===row.audio_sha256?saved.samples[sid]:{})}}
function persist(){try{localStorage.setItem(key,JSON.stringify(state))}catch(e){}}
for(const group of document.querySelectorAll('.review')){const sid=group.dataset.sample;const row=state.samples[sid];for(const input of group.querySelectorAll('[data-field]')){input.value=row[input.dataset.field]||'';input.addEventListener('input',()=>{row[input.dataset.field]=input.value;row.reviewed_at=new Date().toISOString();persist();status(sid)})}status(sid)}
function status(sid){const row=state.samples[sid];document.getElementById('status-'+sid).textContent=row.reviewer&&row.intelligibility&&row.intelligibility!=='pending'?'Reviewer notes saved locally':'Pending: identify reviewer and assess the audio'}
document.getElementById('export').onclick=()=>{const b=new Blob([JSON.stringify(state,null,2)],{type:'application/json'});const a=document.createElement('a');a.href=URL.createObjectURL(b);a.download='listening_review.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)};
document.getElementById('print').onclick=()=>window.print();
document.querySelectorAll('[data-filter]').forEach(b=>b.onclick=()=>document.querySelectorAll('.sample').forEach(s=>s.hidden=b.dataset.filter!=='all'&&s.dataset.language!==b.dataset.filter));
document.getElementById('expand').onclick=()=>document.querySelectorAll('.sample-detail').forEach(d=>d.open=true);
document.getElementById('collapse').onclick=()=>document.querySelectorAll('.sample-detail').forEach(d=>d.open=false);
document.querySelectorAll('a[href^="#page-"]').forEach(a=>a.addEventListener('click',()=>{const target=document.querySelector(a.getAttribute('href'));if(target?.classList.contains('sample')){target.hidden=false;target.querySelector('details').open=true}}));
document.querySelectorAll('audio').forEach(a=>a.addEventListener('play',()=>document.querySelectorAll('audio').forEach(other=>{if(other!==a)other.pause()})));
'''.replace('SEED',json.dumps(state,ensure_ascii=False).replace('</',r'<\/'))
    out.append('</main><script>'+script+'</script></html>')
    (OUT/'Interactive_Report.html').write_text('\n'.join(out),encoding='utf8')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--review-json',type=Path);args=parser.parse_args()
    review=json.loads(args.review_json.read_text(encoding='utf8')) if args.review_json else CONTEXT['listening_review']
    samples,audit=read_samples(review);hist={l:histories(l) for l in ['English','Telugu']}
    d=Document();intro_pages(d,hist)
    loss_pages(d,'English',hist['English']);loss_pages(d,'Telugu',hist['Telugu'])
    for sid,item in samples.items():sample_page(d,sid,item)
    final_pages(d,samples);d.finish();interactive_html(d,samples,audit,hist)
    data={'schema_version':1,'source_commit':CONTEXT['source_commit'],'report_build_environment':{'python':sys.version.split()[0],'reportlab':reportlab.Version,'Pillow':PIL.__version__,'numpy':np.__version__,'platform':sys.platform},'method':'Read submitted PCM WAVs, prediction JSON and CSVs; no acoustic inference or training was executed. F0/energy statistics describe saved predictions, not tracks extracted from generated speech.','samples':audit,'losses':{l:{'best_validation':hist[l]['best'],'first_validation':hist[l]['val'][0],'last_validation':hist[l]['val'][-1],'last_training':hist[l]['train'][-1]} for l in hist},'layout':d.bounds,'listening_review_complete':{sid:bool(s['review'].get('reviewer')) and s['review'].get('intelligibility','pending')!='pending' for sid,s in samples.items()}}
    (OUT/'report_measurements.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({'pdf_pages':d.num,'html_audio_players':len(samples),'pdf_bytes':(OUT/'Technical_Report.pdf').stat().st_size,'html_bytes':(OUT/'Interactive_Report.html').stat().st_size,'min_content_y':min(x['lowest_content_y'] for x in d.bounds)},indent=2))

if __name__=='__main__':main()
