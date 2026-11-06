"""Independent report checks; optional page renders for visual review."""
import argparse
import base64
import hashlib
import html
import json
import re
import subprocess
import tempfile
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlparse

from pypdf import PdfReader
import pypdfium2 as pdfium
from PIL import Image, ImageDraw

OUT=Path(__file__).resolve().parent
ROOT=OUT.parent
REPO=ROOT.parent

class InspectHTML(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True);self.ids=[];self.links=[];self.audios=[];self.images=[];self.players=0;self.script=False;self.scripts=[];self.record=[]
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if 'id' in a:self.ids.append(a['id'])
        if tag=='a' and 'href' in a:self.links.append(a['href'])
        if tag=='source' and a.get('type')=='audio/wav':self.audios.append(a['src'])
        if tag=='audio':self.players+=1
        if tag=='img':self.images.append(a['src'])
        if tag=='script':
            assert 'src' not in a,'Companion must not load third-party scripts'
            self.script=True;self.record=[]
    def handle_endtag(self,tag):
        if tag=='script':self.scripts.append(''.join(self.record));self.script=False
    def handle_data(self,data):
        if self.script:self.record.append(data)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--render-dir',type=Path);args=parser.parse_args()
    reader=PdfReader(OUT/'Technical_Report.pdf');assert len(reader.pages)==20
    assert len(reader.outline)==20
    text=[p.extract_text() or '' for p in reader.pages]
    for i,sid in enumerate(['E1','E2a','E2b','E3','E4','T1','T2a','T2b','T3','T4']):
        assert sid+':' in text[i+8],f'Wrong sample page: {sid}'
    assert 'run_support' in text[19] and '60,000' in ''.join(text)
    links=[]
    for page in reader.pages:
        for ref in page.get('/Annots',[]):
            a=ref.get_object()
            if a.get('/A',{}).get('/URI'):links.append(str(a['/A']['/URI']))
    assert all(x.startswith('https://') for x in links)
    doc=pdfium.PdfDocument(OUT/'Technical_Report.pdf')
    for i,page in enumerate(doc):
        tp=page.get_textpage()
        for j in range(tp.count_chars()):
            x0,y0,x1,y1=tp.get_charbox(j)
            assert min(x0,y0)>=-1 and x1<=page.get_width()+1 and y1<=page.get_height()+1,(i,j,(x0,y0,x1,y1))
    source=(OUT/'Interactive_Report.html').read_text(encoding='utf8');dom=InspectHTML();dom.feed(source)
    assert dom.players==len(dom.audios)==10
    assert not [k for k,v in Counter(dom.ids).items() if v>1], 'Duplicate HTML IDs'
    ids=set(dom.ids)
    for href in dom.links:
        if href.startswith('#'):assert href[1:] in ids,href
        elif 'github.com/SandeshAwate6055/FastSpeech2-English-Telugu/blob/main/' in href:
            path=unquote(href.split('/blob/main/',1)[1].split('#')[0]);assert (REPO/path).exists(),path
    records=json.loads((OUT/'report_measurements.json').read_text(encoding='utf8'))
    for blob,row in zip(dom.audios,records['samples']):
        data=base64.b64decode(blob.split(',',1)[1]);assert hashlib.sha256(data).hexdigest()==row['sha256']
        assert len(data)>44 and data[:4]==b'RIFF'
    assert all(im.startswith('data:image/png;base64,') for im in dom.images)
    for lang in ['English','Telugu']:
        for sid,row in json.loads((ROOT/f'Results/{lang}/variance_predictions_summary.json').read_text(encoding='utf8')).items():
            escaped=html.escape(row['raw_text'])
            assert escaped in source or escaped.replace('  ','&#160;&#160;') in source
    with tempfile.TemporaryDirectory(prefix='report-js-') as tmp:
        js=Path(tmp)/'companion.js';js.write_text('\n'.join(dom.scripts),encoding='utf8')
        subprocess.run(['node','--check',str(js)],check=True)
    if args.render_dir:
        args.render_dir.mkdir(parents=True,exist_ok=True)
        for i,page in enumerate(doc):page.render(scale=1.5).to_pil().save(args.render_dir/f'page-{i+1:02d}.png')
        for start in [0,10]:
            sheet=Image.new('RGB',(1200,760),'#cdd5db');draw=ImageDraw.Draw(sheet)
            for j in range(10):
                im=Image.open(args.render_dir/f'page-{start+j+1:02d}.png');im.thumbnail((232,340));x=(j%5)*240+4;y=(j//5)*380+22
                sheet.paste(im,(x,y));draw.text((x,y-17),f'PAGE {start+j+1}',fill='#142c44')
            sheet.save(args.render_dir/f'contact-{start+1:02d}.png')
    print(json.dumps({'pdf_pages':20,'bookmarks':20,'pdf_external_links':len(links),'html_audio_players':10,'embedded_audio_hashes':'all match submitted WAVs','html_unique_ids':len(ids),'internal_links':'all resolve','javascript':'syntax passes','pdf_text_bounds':'all inside page','rendered_to':str(args.render_dir) if args.render_dir else None},indent=2))

if __name__=='__main__':main()
