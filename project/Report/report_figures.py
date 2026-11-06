"""Report-only diagrams, waveform measurements, and Telugu text shaping."""
import ctypes
import html
import io
import math
import os
import struct
import wave
from pathlib import Path

import numpy as np
from PIL import Image as PILImage, ImageChops, ImageDraw, ImageFont, features
from reportlab.lib import colors
from reportlab.lib.utils import ImageReader

NAVY='#142c44'
TEAL='#087f8c'
ORANGE='#b85e32'
PALE='#eef5f7'
GRID='#d5e0e6'

def color(s):
    return colors.HexColor(s)

def waveform(path):
    with wave.open(str(path),'rb') as w:
        assert (w.getnchannels(),w.getsampwidth(),w.getframerate())==(1,2,22050)
        a=np.frombuffer(w.readframes(w.getnframes()),dtype='<i2').astype(float)/32768
        rate=w.getframerate()
    return a,{'seconds':len(a)/rate,'sample_rate':rate,'channels':1,'sample_width':2,'samples':len(a),'rms_dbfs':float(20*np.log10(max(np.sqrt(np.mean(a*a)),1e-12))),'peak_dbfs':float(20*np.log10(max(np.max(np.abs(a)),1e-12))),'full_scale_samples':int(np.sum((a<=-1)|(a>=32767/32768)))}

def architecture():
    boxes=[]; edges=[]
    def box(x,y,w,h,title,detail='',tone=TEAL): boxes.append((x,y,w,h,title,detail,tone))
    def edge(points): edges.append(points)
    for y,title,detail in [(0,'Phone IDs','Embedding + positional encoding'),(60,'FFT encoder','4 blocks / hidden 256 / 2 heads'),(120,'Speaker-conditioned states','Add learned speaker embedding'),(200,'Pitch-conditioned states','Add reference or predicted pitch embedding'),(280,'Energy-conditioned states','Add reference or predicted energy embedding'),(360,'Length regulator','Expand using reference or predicted durations'),(420,'FFT decoder','6 blocks / hidden 256 / 2 heads'),(480,'80-channel mel + residual PostNet','Refined mel = linear output + PostNet(linear output)'),(540,'Pretrained universal HiFi-GAN','22,050 Hz mono waveform')]:
        box(250,y,500,42,title,detail)
    for a,b in [(42,60),(102,120),(162,200),(242,280),(322,360),(402,420),(462,480),(522,540)]: edge([(500,a),(500,b)])
    box(10,95,185,48,'Speaker ID','Learned embedding',NAVY)
    edge([(195,119),(218,119),(218,141),(250,141)])
    box(10,190,185,48,'Duration predictor','Predict log(1 + d)',ORANGE)
    box(10,278,185,55,'Duration selection','Reference / prediction',ORANGE)
    edge([(300,162),(215,162),(215,214),(195,214)])
    edge([(102,238),(102,278)])
    edge([(102,333),(102,381),(250,381)])
    box(805,118,185,46,'Pitch predictor','Phone-level F0')
    box(805,188,185,46,'Pitch embedding','256 linear bins')
    edge([(750,141),(805,141)]);edge([(897,164),(897,188)])
    edge([(805,211),(778,211),(778,221),(750,221)])
    box(805,267,185,46,'Energy predictor','Receives pitch states')
    box(805,338,185,46,'Energy embedding','256 linear bins')
    edge([(750,230),(790,230),(790,250),(897,250),(897,267)])
    edge([(897,313),(897,338)])
    edge([(805,361),(778,361),(778,301),(750,301)])
    return {'width':1000,'height':592,'boxes':boxes,'edges':edges}

def process_diagram():
    labels=[('1. Select and partition','Resolve dataset revision; seed 42; filter/deduplicate; disjoint train/validation + reserves'),('2. Normalize and align','22,050 Hz WAV/LAB; language lexicon; compatible MFA phone TextGrids'),('3. Extract and validate features','80-bin mel, durations, interpolated phone F0 and energy; check timeline and token lengths'),('4. Train independent acoustic models','Training-only statistics; speaker embeddings; masked mel and predictor losses'),('5. Select and synthesize','Held-out validation plus listening; selected 8k English / 13k Telugu; pretrained HiFi-GAN'),('6. Evaluate and package','Ten exact raw inputs; ten WAVs; thirty variance plots; four loss figures; report')]
    boxes=[(15,i*76,970,56,title,detail,TEAL) for i,(title,detail) in enumerate(labels)]
    edges=[[(500,i*76+56),(500,(i+1)*76)] for i in range(5)]
    return {'width':1000,'height':444,'boxes':boxes,'edges':edges}

def draw_diagram(c,d,x,y,width):
    scale=width/d['width'];c.saveState();c.translate(x,y);c.scale(scale,scale)
    H=d['height']
    c.setLineWidth(2)
    for points in d['edges']:
        c.setStrokeColor(color(TEAL)); p=c.beginPath(); p.moveTo(points[0][0],H-points[0][1])
        for a,b in points[1:]:p.lineTo(a,H-b)
        c.drawPath(p)
        a,b=points[-2],points[-1]; angle=math.atan2(b[1]-a[1],b[0]-a[0]);end=(b[0],H-b[1])
        for delta in [-0.5,0.5]:
            c.line(*end,end[0]-8*math.cos(angle+delta),end[1]+8*math.sin(angle+delta))
    for bx,by,bw,bh,title,detail,tone in d['boxes']:
        c.setFillColor(color(PALE));c.setStrokeColor(color(tone));c.roundRect(bx,H-by-bh,bw,bh,8,stroke=1,fill=1)
        c.setFillColor(color(NAVY));c.setFont('Helvetica-Bold',15 if bw<200 else 18);c.drawCentredString(bx+bw/2,H-by-19,title)
        c.setFont('Helvetica',12 if bw<200 else 15);c.drawCentredString(bx+bw/2,H-by-bh+10,detail)
    c.restoreState()
    return d['height']*scale

def diagram_svg(d):
    marker='arrow-'+str(len(d['boxes']))
    out=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {d["width"]} {d["height"]}" role="img" aria-label="Project architecture or process diagram"><defs><marker id="{marker}" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7" fill="{TEAL}"/></marker></defs>']
    for points in d['edges']:
        out.append(f'<polyline points="{" ".join(f"{x},{y}" for x,y in points)}" fill="none" stroke="{TEAL}" stroke-width="2" marker-end="url(#{marker})"/>')
    for x,y,w,h,title,detail,tone in d['boxes']:
        out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{PALE}" stroke="{tone}" stroke-width="2"/>')
        size=15 if w<200 else 18;sub=12 if w<200 else 15
        out.append(f'<text x="{x+w/2}" y="{y+19}" text-anchor="middle" font-family="Arial,sans-serif" font-size="{size}" font-weight="700" fill="{NAVY}">{html.escape(title)}</text>')
        out.append(f'<text x="{x+w/2}" y="{y+h-10}" text-anchor="middle" font-family="Arial,sans-serif" font-size="{sub}" fill="{NAVY}">{html.escape(detail)}</text>')
    return ''.join(out)+'</svg>'

def draw_prosody(c,row,x,y,w,h=185):
    c.saveState();c.translate(x,y)
    for k,(field,title,tone) in enumerate([('durations_frames','Duration / mel frames',TEAL),('pitch_hz','Predicted phone F0 / Hz',ORANGE),('energy','Predicted phone energy / STFT units',NAVY)]):
        vals=row[field];n=len(vals);segment=h/3;by=h-(k+1)*segment+13;gh=segment-30
        low=0 if field!='pitch_hz' else max(0,min(vals)*0.9);high=max(vals)*1.08 or 1
        c.setFillColor(color(NAVY));c.setFont('Helvetica-Bold',7.8);c.drawString(0,by+gh+9,title)
        c.setFont('Helvetica',6.7);c.drawRightString(29,by+gh-2,f'{high:.0f}');c.drawRightString(29,by,f'{low:.0f}')
        c.setStrokeColor(color(GRID));c.line(35,by,w-5,by)
        c.setStrokeColor(color(tone));path=c.beginPath()
        for i,value in enumerate(vals):
            px=36+i*(w-46)/max(n-1,1);py=by+(value-low)/(high-low)*gh
            if field=='durations_frames':c.line(px,by,px,py)
            elif i==0:path.moveTo(px,py)
            else:path.lineTo(px,py)
        if field!='durations_frames':c.drawPath(path)
        c.setFillColor(color(NAVY));c.setFont('Helvetica',6.4)
        for q in [0,0.25,0.5,0.75,1]:c.drawCentredString(36+q*(w-46),by-9,str(round(q*(n-1))))
    c.restoreState()

def prosody_svg(row):
    out=['<svg viewBox="0 0 1000 480" xmlns="http://www.w3.org/2000/svg" class="prosody" aria-label="Token duration, pitch and energy. Hover or focus a point for values.">']
    for k,(field,title,tone) in enumerate([('durations_frames','Duration (mel frames)',TEAL),('pitch_hz','Predicted F0 (Hz)',ORANGE),('energy','Predicted energy (STFT magnitude units)',NAVY)]):
        v=row[field];n=len(v);top=k*158;lo=0 if k!=1 else min(v)*0.9;hi=max(v)*1.08 or 1
        out.append(f'<text x="5" y="{top+18}" font-size="18" font-family="Arial" fill="{NAVY}">{title}</text><path d="M55,{top+32}V{top+126}H985" fill="none" stroke="{GRID}"/>')
        pts=[]
        for i,z in enumerate(v):
            x=58+i*920/max(n-1,1);y=top+123-(z-lo)/(hi-lo)*85;pts.append(f'{x:.2f},{y:.2f}')
            tip=html.escape(f'Token {i}: {row["tokens"][i]} | {z:.2f}')
            shape=f'<line x1="{x:.2f}" y1="{top+123}" x2="{x:.2f}" y2="{y:.2f}" stroke="{tone}" stroke-width="5"/>' if k==0 else f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.3" fill="{tone}"/>'
            out.append(f'<g tabindex="0" aria-label="{tip}"><title>{tip}</title>{shape}</g>')
        if k:out.append(f'<polyline points="{" ".join(pts)}" stroke="{tone}" fill="none" stroke-width="2"/>')
        for q in [0,.25,.5,.75,1]:out.append(f'<text x="{58+q*920}" y="{top+145}" font-size="13" text-anchor="middle" fill="{NAVY}">{round(q*(n-1))}</text>')
        out.append(f'<text x="45" y="{top+42}" text-anchor="end" font-size="13">{hi:.0f}</text><text x="45" y="{top+125}" text-anchor="end" font-size="13">{lo:.0f}</text>')
    return ''.join(out)+'</svg>'

def audio_image(signal,rate,width=1200):
    image=PILImage.new('RGB',(width,165),'white');draw=ImageDraw.Draw(image)
    chunks=np.array_split(signal,width)
    for i,a in enumerate(chunks):draw.line((i,32-int(max(a)*27),i,32-int(min(a)*27)),fill=TEAL)
    frames=np.lib.stride_tricks.sliding_window_view(signal,1024)[::128]*np.hanning(1024)
    spec=np.abs(np.fft.rfft(frames,axis=1)).T[:int(8000*1024/rate)+1]
    db=np.clip(20*np.log10(np.maximum(spec,1e-9)/max(spec.max(),1e-9)),-70,0);q=(db+70)/70
    rgb=np.stack([18+215*q,38+152*q,61+65*q],axis=-1).astype(np.uint8)
    image.paste(PILImage.fromarray(rgb[::-1]).resize((width,100)),(0,65))
    return image

def telugu_bitmap(text):
    """Shape Telugu through Windows GDI, or Pillow libraqm on other platforms."""
    width,height=1900,360
    if os.name=='nt':
        from ctypes import wintypes
        class Rect(ctypes.Structure):
            _fields_=[('left',ctypes.c_long),('top',ctypes.c_long),('right',ctypes.c_long),('bottom',ctypes.c_long)]
        class LogFont(ctypes.Structure):
            _fields_=[(n,ctypes.c_long) for n in ['height','width','escapement','orientation','weight']]+[(n,ctypes.c_ubyte) for n in ['italic','underline','strike','charset','out','clip','quality','pitch']]+[('face',ctypes.c_wchar*32)]
        gdi=ctypes.WinDLL('gdi32');user=ctypes.WinDLL('user32')
        gdi.CreateCompatibleDC.argtypes=[wintypes.HDC];gdi.CreateCompatibleDC.restype=wintypes.HDC
        gdi.CreateDIBSection.argtypes=[wintypes.HDC,ctypes.c_void_p,ctypes.c_uint,ctypes.POINTER(ctypes.c_void_p),wintypes.HANDLE,ctypes.c_ulong];gdi.CreateDIBSection.restype=wintypes.HBITMAP
        gdi.SelectObject.argtypes=[wintypes.HDC,wintypes.HGDIOBJ];gdi.SelectObject.restype=wintypes.HGDIOBJ
        gdi.CreateFontIndirectW.argtypes=[ctypes.POINTER(LogFont)];gdi.CreateFontIndirectW.restype=wintypes.HFONT
        gdi.DeleteObject.argtypes=[wintypes.HGDIOBJ];gdi.DeleteDC.argtypes=[wintypes.HDC]
        user.DrawTextW.argtypes=[wintypes.HDC,ctypes.c_wchar_p,ctypes.c_int,ctypes.POINTER(Rect),ctypes.c_uint]
        dc=gdi.CreateCompatibleDC(None);bits=ctypes.c_void_p();header=ctypes.create_string_buffer(struct.pack('<IiiHHIIiiII',40,width,-height,1,32,0,width*height*4,0,0,0,0))
        bmp=gdi.CreateDIBSection(dc,header,0,ctypes.byref(bits),None,0)
        if not bmp or not bits.value:raise RuntimeError('Telugu bitmap initialization failed')
        prevbmp=gdi.SelectObject(dc,bmp);f=LogFont();f.height=-43;f.weight=400;f.charset=1;f.quality=4;f.face='Nirmala UI'
        font=gdi.CreateFontIndirectW(ctypes.byref(f));prevfont=gdi.SelectObject(dc,font);ctypes.memset(bits,255,width*height*4)
        rect=Rect(4,4,width-4,height-4);user.DrawTextW(dc,text,len(text),ctypes.byref(rect),0x10|0x800)
        image=PILImage.frombytes('RGB',(width,height),ctypes.string_at(bits,width*height*4),'raw','BGRX')
        gdi.SelectObject(dc,prevfont);gdi.SelectObject(dc,prevbmp);gdi.DeleteObject(font);gdi.DeleteObject(bmp);gdi.DeleteDC(dc)
    else:
        if not features.check('raqm'):raise RuntimeError('Telugu rendering needs Pillow libraqm and TELUGU_FONT')
        font=ImageFont.truetype(os.environ.get('TELUGU_FONT','/usr/share/fonts/truetype/noto/NotoSansTelugu-Regular.ttf'),43)
        image=PILImage.new('RGB',(width,height),'white');d=ImageDraw.Draw(image);lines=[];line=''
        for word in text.split(' '):
            candidate=line+(' ' if line else '')+word
            if d.textlength(candidate,font=font)>width-10:lines.append(line);line=word
            else:line=candidate
        lines.append(line);d.multiline_text((4,4),'\n'.join(lines),font=font,fill='black',spacing=15)
    bounds=ImageChops.difference(image,PILImage.new('RGB',image.size,'white')).getbbox()
    if not bounds or bounds[3]>=height-3:raise RuntimeError('Telugu text is empty or clipped')
    return image.crop((0,0,width,bounds[3]+9))
