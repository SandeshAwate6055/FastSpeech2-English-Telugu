"""Vector architecture and measured waveform figures used by the report."""
from pathlib import Path
import wave

import numpy as np
from PIL import Image
from reportlab.lib import colors
from reportlab.lib.utils import ImageReader
from reportlab.platypus import Flowable

NAVY = colors.HexColor("#102b46")
TEAL = colors.HexColor("#087d78")
PALE = colors.HexColor("#edf4f7")


class Architecture(Flowable):
    def __init__(self):
        super().__init__()
        self.width, self.height = 499, 265

    def draw(self):
        c = self.canv
        def box(x, y, w, h, text, detail=""):
            c.setFillColor(PALE); c.setStrokeColor(TEAL)
            c.roundRect(x, y, w, h, 5, fill=1)
            c.setFillColor(NAVY); c.setFont("Helvetica-Bold", 9)
            c.drawCentredString(x+w/2, y+h-15, text)
            if detail:
                c.setFont("Helvetica", 7.5); c.drawCentredString(x+w/2, y+9, detail)
        def arrow(x1, y1, x2, y2):
            c.setStrokeColor(TEAL); c.line(x1, y1, x2, y2)
            if y1 != y2:
                c.line(x2, y2, x2-3, y2+4);c.line(x2, y2, x2+3, y2+4)
            else:
                back = -4 if x2 > x1 else 4
                c.line(x2,y2,x2+back,y2-3);c.line(x2,y2,x2+back,y2+3)
        box(0,218,160,42,"Text frontend","Normalization -> phone IDs")
        box(190,218,160,42,"FFT encoder","4 blocks / hidden size 256")
        arrow(160,239,190,239)
        box(380,218,119,42,"Speaker embedding","Learned profile / voice")
        arrow(380,239,350,239)
        for x, title, detail in [(30,"Duration head","log(1+d), MSE"),(184,"Pitch + embedding","Phone F0, MSE"),(338,"Energy + embedding","Pitch-conditioned states")]:
            box(x,137,132,44,title,detail)
            c.line(270,218,270,200);c.line(96,200,250,200)
            if x != 338:
                arrow(x+66,200,x+66,181)
        arrow(316,159,338,159)
        box(45,65,190,43,"Length regulator","Repeat by integer duration")
        box(265,65,210,43,"FFT decoder + PostNet","6 blocks -> 80-bin mel -> residual")
        arrow(96,137,96,108)
        c.line(250,137,250,122);c.line(404,137,404,122);c.line(250,122,140,122);c.line(404,122,140,122);arrow(140,122,140,108)
        arrow(235,86,265,86)
        box(265,0,210,40,"Universal HiFi-GAN","22,050 Hz mono speech")
        arrow(370,65,370,40)
        c.setFont("Helvetica",7.5);c.setFillColor(NAVY)
        c.drawString(0,29,"Training: reference duration / pitch / energy")
        c.drawString(0,14,"Inference: all three heads supply predictions")


def waveform_metrics(path):
    with wave.open(str(path), "rb") as source:
        rate = source.getframerate()
        signal = np.frombuffer(source.readframes(source.getnframes()), dtype="<i2").astype(np.float64) / 32768
    return signal, {"seconds": len(signal)/rate, "rms_dbfs": float(20*np.log10(max(np.sqrt(np.mean(signal**2)),1e-12))), "peak_dbfs": float(20*np.log10(max(np.max(np.abs(signal)),1e-12))), "saturated_samples": int(np.sum(np.abs(signal)>=32767/32768)), "sample_rate": rate}


class AudioStrip(Flowable):
    width = 499
    height = 82

    def __init__(self, path):
        super().__init__()
        self.width, self.height = 499, 82
        self.signal, self.metrics = waveform_metrics(path)

    def draw(self):
        c = self.canv
        signal = self.signal
        frames = np.lib.stride_tricks.sliding_window_view(signal,1024)[::128] * np.hanning(1024)
        magnitude = np.abs(np.fft.rfft(frames,axis=1)).T
        magnitude = magnitude[:int(8000*1024/self.metrics["sample_rate"])+1]
        db = np.clip(20*np.log10(np.maximum(magnitude,1e-8)/max(magnitude.max(),1e-8)),-70,0)
        level=(db+70)/70
        rgb=np.stack([18+215*level,38+152*level,61+65*level],axis=-1).astype(np.uint8)
        image=Image.fromarray(rgb[::-1]).resize((950,100))
        c.drawImage(ImageReader(image),x=30,y=4,width=469,height=43)
        c.setFillColor(NAVY);c.setFont("Helvetica",7)
        c.drawString(0,40,"8 kHz");c.drawString(0,4,"0")
        c.drawString(30,51,"Measured waveform and spectrogram (Hann 1024 / hop 128; dB relative to utterance peak)")
        c.setStrokeColor(TEAL)
        chunks=np.array_split(signal, min(499,len(signal)))
        for i,chunk in enumerate(chunks):
            c.line(30+i*469/499,68-8*float(np.max(chunk)),30+i*469/499,68-8*float(np.min(chunk)))


class LossTrend(Flowable):
    def __init__(self, train, validation):
        super().__init__()
        self.width, self.height = 499, 160
        training = {int(row["step"]): float(row["total_loss"]) for row in train}
        self.series = [(int(row["step"]), training[int(row["step"])], float(row["total_loss"])) for row in validation]

    def draw(self):
        c = self.canv
        c.setFillColor(NAVY);c.setFont("Helvetica-Bold",8)
        c.drawString(0,148,"Generalization gap at validation checkpoints / total loss")
        high = max(max(row[1:]) for row in self.series)*1.1
        right = max(row[0] for row in self.series)
        c.setStrokeColor(colors.HexColor("#c5d2da"))
        c.line(30,25,490,25);c.line(30,25,30,130)
        c.setFont("Helvetica",7)
        for value in (0,high/2,high):
            c.drawString(2,25+value/high*105,f"{value:.1f}")
        for step in (0,20000,40000,60000):
            c.drawString(30+step/right*450,10,f"{step//1000}k")
        for column,tone,label,x in [(1,TEAL,"Sampled train",300),(2,colors.HexColor("#b35a35"),"Validation",400)]:
            c.setStrokeColor(tone);path=c.beginPath()
            for i,row in enumerate(self.series):
                point=(30+row[0]/right*450,25+row[column]/high*105)
                (path.moveTo if i==0 else path.lineTo)(*point)
            c.drawPath(path);c.setFillColor(tone);c.drawString(x,136,label)
