#!/usr/bin/env python3
"""Camera in any browser, no installs: serve ROS image topics as MJPEG (2026-10-17; several topics 2026-10-20).

    python3 compliant_insertion_studio/scripts/mjpeg_view.py [--topic T ...] [--fps 5] [--port 8080]
    then open  http://<this PC's Tailscale IP>:8080/   (e.g. from the Mac)

Default page (2026-10-20, operator): the annotated stream (localizer overlay + the robot +-x/+-y compass),
the depth height map (depth_view.py: what the depth checks see, height above the table in colour) and the
raw colour camera, side by side. /stream/<i> is one topic alone.

--transcript PATH (2026-10-23, operator): a Claude Code session transcript (.jsonl) live on the left - the operator's
messages, the replies and each tool call as one line - and the topics stacked on the right; the page polls /log.

Topics ending in /compressed are passed through as JPEG (no re-encode); raw sensor_msgs/Image topics are
JPEG-encoded here, at most --fps per topic (the OptiPlex is CPU-bound; the localizer comes first).
Foxglove needs its app/layout set up and the web app could not open ws:// from https; this is a plain page.
"""
import argparse
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import rclpy
from sensor_msgs.msg import CompressedImage, Image

ap = argparse.ArgumentParser()
ap.add_argument('--topic', action='append', default=None,
                help='repeatable; default: /annotated_stream, /depth_view, /camera/camera/color/image_raw/compressed')
ap.add_argument('--fps', type=float, default=5.0)
ap.add_argument('--port', type=int, default=8080)
ap.add_argument('--bind', default='0.0.0.0')
ap.add_argument('--transcript', default=None, help='a Claude Code session .jsonl to show live beside the views')
a = ap.parse_args()
TOPICS = a.topic or ['/annotated_stream', '/depth_view', '/camera/camera/color/image_raw/compressed']

latest = [{'jpg': None, 't': 0.0, 'raw': None} for _ in TOPICS]
rclpy.init(); node = rclpy.create_node('mjpeg_view')


def _raw_cb(i):
    def cb(m):
        latest[i]['raw'] = m          # encoded lazily, only while someone is watching
    return cb


for i, t in enumerate(TOPICS):
    if t.endswith('/compressed'):
        node.create_subscription(CompressedImage, t, lambda m, i=i: latest[i].update(jpg=bytes(m.data), t=time.time()), 2)
    else:
        node.create_subscription(Image, t, _raw_cb(i), 2)
threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()


def frame(i):
    L = latest[i]
    if L['raw'] is not None:
        import cv2
        import numpy as np
        m, L['raw'] = L['raw'], None
        img = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, -1)
        if m.encoding == 'rgb8':
            img = img[:, :, ::-1]
        if img.shape[1] > 960:
            img = cv2.resize(img, (960, int(img.shape[0] * 960 / img.shape[1])))
        ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if ok:
            L['jpg'] = buf.tobytes(); L['t'] = time.time()
    return L['jpg']


def _local(iso):
    import datetime
    try:
        return datetime.datetime.fromisoformat(iso.replace('Z', '+00:00')).astimezone().strftime('%H:%M:%S')
    except Exception:
        return ''


def log_entries(since):
    """Transcript records from byte offset `since` (the last ~12 MB when negative: tool output and images fill it) as compact entries, and the new
    offset. Only complete lines; thinking, tool results and bookkeeping records are left out."""
    import json
    import os
    out = []
    size = os.path.getsize(a.transcript)
    start = max(0, size - 12_000_000) if since < 0 else min(since, size)
    with open(a.transcript, 'rb') as f:
        f.seek(start)
        data = f.read(size - start)
    end = data.rfind(b'\n') + 1
    lines = data[:end].split(b'\n')
    if since < 0 and start > 0:
        lines = lines[1:]                                     # the first one is cut
    for ln in lines:
        try:
            d = json.loads(ln)
        except Exception:
            continue
        m, ts = d.get('message') or {}, _local(d.get('timestamp'))
        c = m.get('content')
        if d.get('type') == 'user' and isinstance(c, str):
            if '<task-notification>' in c:
                ev = c.split('<event>')[1].split('</event>')[0] if '<event>' in c else 'background event'
                out.append({'k': 'event', 't': ts, 'x': ev[:400]})
            elif not c.lstrip().startswith('<'):
                out.append({'k': 'user', 't': ts, 'x': c})
        elif d.get('type') == 'user' and isinstance(c, list):
            for b in c:
                if b.get('type') == 'text' and not b.get('text', '').lstrip().startswith('<'):
                    out.append({'k': 'user', 't': ts, 'x': b['text']})
        elif d.get('type') == 'assistant' and isinstance(c, list):
            for b in c:
                if b.get('type') == 'text' and b.get('text', '').strip():
                    out.append({'k': 'claude', 't': ts, 'x': b['text']})
                elif b.get('type') == 'tool_use':
                    i = b.get('input') or {}
                    what = i.get('description') or i.get('file_path') or i.get('pattern') or i.get('prompt', '')[:120] \
                        or (i.get('command') or '').split('\n')[0][:160]
                    out.append({'k': 'tool', 't': ts, 'x': f"{b.get('name')}: {what}"})
    return out, start + end


PAGE_SPLIT = """<html><head><title>FMB1 cell</title><meta name='viewport' content='width=device-width'>
<style>
:root{--bg:#f5f4ed;--chat:#faf9f5;--line:#e5e2d6;--text:#2d2b27;--muted:#8a8578;--accent:#d97757;--me:#ece9df;--metext:#2d2b27;
  --tool:#8a8578;--event:#9a5a3a;--eventbg:#f6ebe3;--code:#efece3;--pre:#f1eee5;--views:#e9e6dc}
@media (prefers-color-scheme:dark){:root{--bg:#1f1e1d;--chat:#262624;--line:#3a3935;--text:#ecebe6;--muted:#9a978e;--accent:#d97757;
  --me:#3a3936;--metext:#ecebe6;--tool:#8f8c83;--event:#e3a98c;--eventbg:#33281f;--code:#33322f;--pre:#1c1b1a;--views:#1a1918}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);font:15px/1.6 ui-sans-serif,-apple-system,'Segoe UI',sans-serif;display:flex;height:100vh;overflow:hidden}
#chat{flex:1 1 45%;display:flex;flex-direction:column;background:var(--chat);border-right:1px solid var(--line);min-width:0}
#chat header{padding:12px 20px;border-bottom:1px solid var(--line);display:flex;align-items:center;gap:10px;font-weight:600}
#chat header .mark{width:22px;height:22px;color:var(--accent)}
#chat header small{color:var(--muted);font-weight:400}
#log{flex:1;overflow-y:auto;padding:18px 22px;display:flex;flex-direction:column;gap:12px}
.row{display:flex;flex-direction:column;max-width:92%}
.row.user{align-self:flex-end;align-items:flex-end;max-width:80%}
.row.claude{align-self:flex-start;align-items:flex-start}
.user .bubble{background:var(--me);color:var(--metext);padding:9px 14px;border-radius:16px}
.claude .bubble{padding:0 2px;font-family:ui-serif,Georgia,'Times New Roman',serif;font-size:16px;line-height:1.65}
.bubble{word-wrap:break-word;overflow-wrap:anywhere}
.ts{color:var(--muted);font-size:11px;margin:3px 4px 0;font-variant-numeric:tabular-nums}
details.tools{align-self:flex-start;color:var(--tool);font:12px/1.45 ui-monospace,Menlo,monospace;max-width:92%}
details.tools summary{cursor:pointer;list-style:none}
details.tools summary::before{content:'▸ ';color:var(--accent)}details.tools[open] summary::before{content:'▾ '}
details.tools div{padding:1px 0 1px 16px;white-space:pre-wrap;overflow-wrap:anywhere}
.event{align-self:flex-start;color:var(--event);background:var(--eventbg);border-radius:8px;padding:3px 10px;
  font:12px/1.45 ui-monospace,Menlo,monospace;max-width:92%;overflow-wrap:anywhere;white-space:pre-wrap}
.event .ts{margin-left:6px}
.md>:first-child{margin-top:0}.md>:last-child{margin-bottom:0}
.md p{margin:.45em 0}.md ul,.md ol{margin:.45em 0;padding-left:1.5em}.md li{margin:.2em 0}
.md h1,.md h2,.md h3{font-size:16px;margin:.7em 0 .3em}
.md code{font:13px ui-monospace,Menlo,monospace;background:var(--code);padding:1px 5px;border-radius:4px}
.md pre{background:var(--pre);border:1px solid var(--line);padding:10px;border-radius:8px;overflow-x:auto}
.md pre code{background:none;padding:0}
.md table{border-collapse:collapse;margin:.5em 0;font:13px ui-sans-serif,-apple-system,sans-serif}
.md th,.md td{border:1px solid var(--line);padding:4px 9px;text-align:left}
.md a{color:var(--accent)}.md blockquote{margin:.5em 0;padding-left:12px;border-left:3px solid var(--line);color:var(--muted)}
.md strong{font-weight:650}
#views{flex:1 1 55%;display:flex;flex-direction:column;gap:8px;padding:8px;min-width:0;background:var(--views)}
#views figure{margin:0;flex:1 1 50%;min-height:0;display:flex;flex-direction:column}
#views img{width:100%;height:100%;object-fit:contain;background:#000;min-height:0;border-radius:8px}
figcaption{color:var(--muted);font-size:12px;padding:3px 4px}
@media (max-width:900px){body{flex-direction:column;height:auto;overflow:auto}#chat{height:60vh}}
</style></head><body>
<div id='chat'><header><svg class='mark' viewBox='0 0 24 24' fill='currentColor'><path d='M12 2l1.6 6.9L20 6l-4.6 5.1L22 12l-6.6.9L20 18l-6.4-2.9L12 22l-1.6-6.9L4 18l4.6-5.1L2 12l6.6-.9L4 6l6.4 2.9z'/></svg>Claude <small>FMB1 cell session</small></header><div id='log'></div></div>
<div id='views'>__FIGS__</div>
<script src='https://cdn.jsdelivr.net/npm/marked@12.0.2/marked.min.js'></script>
<script src='https://cdn.jsdelivr.net/npm/dompurify@3.1.6/dist/purify.min.js'></script>
<script>
let since=-1;const log=document.getElementById('log');
function esc(s){return s.replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))}
function md(s){try{return DOMPurify.sanitize(marked.parse(s,{gfm:true,breaks:false}))}catch(e){return '<div style=white-space:pre-wrap>'+esc(s)+'</div>'}}
function add(e){
  if(e.k==='tool'){
    let g=log.lastElementChild;
    if(!g||g.tagName!=='DETAILS'){g=document.createElement('details');g.className='tools';g.innerHTML='<summary></summary>';g.n=0;log.appendChild(g)}
    const d=document.createElement('div');d.textContent=e.x;g.appendChild(d);g.n++;
    g.querySelector('summary').textContent=e.x.slice(0,90)+(g.n>1?'  (+'+(g.n-1)+' more)':'');return}
  if(e.k==='event'){const d=document.createElement('div');d.className='event';
    d.innerHTML=esc(e.x)+"<span class='ts'>"+e.t+"</span>";log.appendChild(d);return}
  const r=document.createElement('div');r.className='row '+e.k;
  r.innerHTML="<div class='bubble md'>"+md(e.x)+"</div><div class='ts'>"+e.t+"</div>";log.appendChild(r)}
async function poll(){
  try{
    const r=await fetch('log?since='+since);const j=await r.json();
    const atBottom=log.scrollHeight-log.scrollTop-log.clientHeight<80;
    j.entries.forEach(add);
    while(log.children.length>500)log.removeChild(log.firstChild);
    since=j.next;if(atBottom||j.first)log.scrollTop=log.scrollHeight;
  }catch(err){}
  setTimeout(poll,2000)}
poll();
</script></body></html>"""


class H(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        if self.path.startswith('/log') and a.transcript:
            import json
            from urllib.parse import parse_qs, urlparse
            since = int((parse_qs(urlparse(self.path).query).get('since') or ['-1'])[0])
            ent, nxt = log_entries(since)
            body = json.dumps({'entries': ent, 'next': nxt, 'first': since < 0}).encode()
            self.send_response(200); self.send_header('Content-Type', 'application/json'); self.end_headers()
            self.wfile.write(body)
            return
        if self.path.startswith('/stream'):
            try:
                i = int(self.path.split('/')[2]) if self.path.count('/') >= 2 and self.path.split('/')[2] else 0
            except ValueError:
                i = 0
            i = min(max(i, 0), len(TOPICS) - 1)
            self.send_response(200)
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
            self.end_headers()
            try:
                while True:
                    jpg = frame(i)
                    if jpg:
                        self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: '
                                         + str(len(jpg)).encode() + b'\r\n\r\n' + jpg + b'\r\n')
                    time.sleep(1.0 / a.fps)
            except (BrokenPipeError, ConnectionResetError):
                return
        if a.transcript:
            figs = ''.join(f"<figure><img src='stream/{i}'><figcaption>{t}</figcaption></figure>" for i, t in enumerate(TOPICS))
            page = PAGE_SPLIT.replace('__FIGS__', figs).encode()
            self.send_response(200); self.send_header('Content-Type', 'text/html; charset=utf-8'); self.end_headers()
            self.wfile.write(page)
            return
        cells = ''.join(
            f"<div style='flex:1 1 30%;min-width:300px'><img src='/stream/{i}' style='width:100%;height:auto'>"
            f"<div style='color:#aaa;font:12px sans-serif;padding:4px'>{t} - <a style='color:#8af' href='/stream/{i}'>alone</a></div></div>"
            for i, t in enumerate(TOPICS))
        page = (f"<html><head><title>Cell cameras</title><meta name='viewport' content='width=device-width'></head>"
                f"<body style='margin:0;background:#111'><div style='display:flex;flex-wrap:wrap;gap:4px'>{cells}</div>"
                f"<div style='color:#777;font:12px sans-serif;padding:4px'>@ {a.fps:g} fps per stream</div></body></html>").encode()
        self.send_response(200); self.send_header('Content-Type', 'text/html'); self.end_headers()
        self.wfile.write(page)


import os as _os  # noqa: E402
import signal as _signal  # noqa: E402
for _sg in (_signal.SIGTERM, _signal.SIGINT):
    _signal.signal(_sg, lambda *_: _os._exit(0))   # open streams and the rclpy thread kept SIGTERM from ending it
ThreadingHTTPServer((a.bind, a.port), H).serve_forever()
