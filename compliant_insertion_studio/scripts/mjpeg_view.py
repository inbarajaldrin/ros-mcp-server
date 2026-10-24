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
ap.add_argument('--timeline', default=None,
                help="a background job's timeline.jsonl (~/.claude/jobs/<id>/): what the session is doing right now, as a live "
                     "status line under the chat (default: the job whose id starts the transcript's session id)")
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


_IMG_CACHE = {}


def _img_src(src):
    """A transcript image (base64) as a small JPEG data URL (<= 900 px wide) for the page; cached."""
    import base64
    data = (src or {}).get('data') or ''
    if not data:
        return None
    key = (len(data), data[:64])
    if key in _IMG_CACHE:
        return _IMG_CACHE[key]
    url = None
    try:
        import cv2
        import numpy as np
        img = cv2.imdecode(np.frombuffer(base64.b64decode(data), np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            if img.shape[1] > 900:
                img = cv2.resize(img, (900, int(img.shape[0] * 900 / img.shape[1])), interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                url = 'data:image/jpeg;base64,' + base64.b64encode(buf.tobytes()).decode()
    except Exception:
        url = None
    if url is None and len(data) < 1_500_000:
        url = f"data:{(src or {}).get('media_type', 'image/png')};base64,{data}"
    if len(_IMG_CACHE) > 200:
        _IMG_CACHE.clear()
    _IMG_CACHE[key] = url
    return url


def log_entries(since):
    """Transcript records from byte offset `since` (the last ~12 MB when negative: tool output and images fill it) as compact entries, and the new
    offset. Only complete lines; replies, thinking, tool calls with a line of each result, images; bookkeeping left out."""
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
        at = d.get('attachment') if isinstance(d.get('attachment'), dict) else {}
        if d.get('type') == 'attachment' and at.get('type') == 'queued_command' and \
                (at.get('origin') or {}).get('kind') == 'human' and at.get('prompt'):
            # a message typed while the session was busy: recorded as a queued command, not a user record (operator 10-23)
            out.append({'k': 'user', 't': ts, 'x': at['prompt']})
            continue
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
                elif b.get('type') == 'image':                       # an image the operator attached
                    u = _img_src(b.get('source'))
                    if u:
                        out.append({'k': 'img', 'who': 'user', 't': ts, 'x': u})
                elif b.get('type') == 'tool_result':
                    # the result under its call, one short line (prompt2assembly's watch_agent shows it the same way)
                    bc = b.get('content')
                    txt = bc if isinstance(bc, str) else ' '.join(x.get('text', '') for x in (bc or [])
                                                                  if isinstance(x, dict) and x.get('type') == 'text')
                    txt = ' '.join(txt.split())
                    if txt:
                        out.append({'k': 'result', 't': ts, 'x': ('error: ' if b.get('is_error') else '') + txt[:220]})
                    for x in (bc if isinstance(bc, list) else []):  # an image Claude read (Read on a PNG / JPG)
                        if isinstance(x, dict) and x.get('type') == 'image':
                            u = _img_src(x.get('source'))
                            if u:
                                out.append({'k': 'img', 'who': 'claude', 't': ts, 'x': u})
        elif d.get('type') == 'assistant' and isinstance(c, list):
            for b in c:
                if b.get('type') == 'text' and b.get('text', '').strip():
                    out.append({'k': 'claude', 't': ts, 'x': b['text']})
                elif b.get('type') == 'thinking' and b.get('thinking', '').strip():
                    # the terminal shows these; the page left them out, so what the operator read there never came here
                    # (10-24: 'Run 81 halted before assembly ...' was a thinking block)
                    out.append({'k': 'thinking', 't': ts, 'x': b['thinking']})
                elif b.get('type') == 'tool_use':
                    i = b.get('input') or {}
                    what = i.get('description') or i.get('file_path') or i.get('pattern') or i.get('prompt', '')[:120] \
                        or (i.get('command') or '').split('\n')[0][:160]
                    out.append({'k': 'tool', 't': ts, 'x': f"{b.get('name')}: {what}"})
    return out, start + end


def _timeline_path():
    import glob
    import os
    if a.timeline:
        return a.timeline
    sid = os.path.basename(a.transcript or '').split('.')[0]
    hits = [d for d in glob.glob(os.path.expanduser('~/.claude/jobs/*/timeline.jsonl')) if sid.startswith(os.path.basename(os.path.dirname(d)))]
    return hits[0] if hits else None


def status():
    """What the session is doing now: the job timeline's last record (state, detail, since when). Claude Code writes the
    transcript one finished block at a time and no partial tokens anywhere, so this line is what moves between blocks."""
    import json
    import os
    path = _timeline_path()
    if not path or not os.path.exists(path):
        return {}
    with open(path, 'rb') as f:
        f.seek(max(0, os.path.getsize(path) - 20000))
        lines = f.read().split(b'\n')
    for ln in reversed(lines):
        try:
            d = json.loads(ln)
        except Exception:
            continue
        return {'state': d.get('state', ''), 'detail': (d.get('detail') or '')[:200], 'at': d.get('at', ''),
                't': _local(d.get('at'))}
    return {}


PAGE_SPLIT = """<html><head><title>FMB1 cell</title><meta name='viewport' content='width=device-width'>
<meta property='og:title' content='FMB1 cell - live'><meta property='og:description' content='Claude at work on the UR5e: chat, annotated camera, depth'>
<meta property='og:image' content='snap/0'>
<style>
:root{--bg:#f5f4ed;--chat:#faf9f5;--line:#e5e2d6;--text:#2d2b27;--muted:#8a8578;--accent:#d97757;--me:#ece9df;--metext:#2d2b27;
  --tool:#8a8578;--event:#9a5a3a;--eventbg:#f6ebe3;--code:#efece3;--pre:#f1eee5;--views:#e9e6dc}
@media (prefers-color-scheme:dark){:root{--bg:#1f1e1d;--chat:#262624;--line:#3a3935;--text:#ecebe6;--muted:#9a978e;--accent:#d97757;
  --me:#3a3936;--metext:#ecebe6;--tool:#8f8c83;--event:#e3a98c;--eventbg:#33281f;--code:#33322f;--pre:#1c1b1a;--views:#1a1918}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);font:15px/1.6 ui-sans-serif,-apple-system,'Segoe UI',sans-serif;display:flex;height:100vh;overflow:hidden}
#chat{flex:1 1 auto;display:flex;flex-direction:column;background:var(--chat);border-right:1px solid var(--line);min-width:0}
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
img.shot{max-width:100%;max-height:420px;border-radius:10px;border:1px solid var(--line);display:block;cursor:zoom-in}
.lightbox{position:fixed;inset:0;background:rgba(0,0,0,.85);display:flex;align-items:center;justify-content:center;z-index:10;cursor:zoom-out}
.lightbox img{max-width:96vw;max-height:94vh;border-radius:6px}
details.tools{align-self:flex-start;color:var(--tool);font:12px/1.45 ui-monospace,Menlo,monospace;max-width:92%}
details.tools summary{cursor:pointer;list-style:none}
details.tools summary::before{content:'▸ ';color:var(--accent)}details.tools[open] summary::before{content:'▾ '}
details.tools div{padding:1px 0 1px 16px;white-space:pre-wrap;overflow-wrap:anywhere}
.event{align-self:flex-start;color:var(--event);background:var(--eventbg);border-radius:8px;padding:3px 10px;
  font:12px/1.45 ui-monospace,Menlo,monospace;max-width:92%;overflow-wrap:anywhere;white-space:pre-wrap}
.event .ts{margin-left:6px}
.row.thinking{align-self:flex-start;max-width:92%}
.thinking .bubble{color:var(--muted);font-style:italic;font-size:14px;line-height:1.55;border-left:2px solid var(--line);padding:0 0 0 10px}
details.tools div.res{color:var(--muted);opacity:.85}
#status{padding:7px 20px;border-top:1px solid var(--line);color:var(--muted);font:12px/1.4 ui-monospace,Menlo,monospace;
  display:flex;gap:8px;align-items:center;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
#status .dot{width:8px;height:8px;border-radius:50%;background:var(--muted);flex:0 0 auto}
#status.working .dot{background:var(--accent);animation:pulse 1.2s ease-in-out infinite}
@keyframes pulse{50%{opacity:.25}}
#status span.d{overflow:hidden;text-overflow:ellipsis}
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
/* the views column is as wide as two 16:9 frames stacked in the window's height allow (no letterbox), the chat takes the rest */
#views{flex:0 0 auto;width:min(62vw,calc((100vh - 76px) / 2 * 16 / 9 + 16px));display:flex;flex-direction:column;justify-content:center;
  gap:8px;padding:8px;background:var(--views);overflow:hidden}
#views figure{margin:0;display:flex;flex-direction:column}
#views img{width:100%;height:auto;aspect-ratio:16/9;object-fit:contain;background:#000;border-radius:8px}
figcaption{color:var(--muted);font-size:12px;padding:3px 4px}
@media (max-width:900px){body{flex-direction:column;height:auto;overflow:auto}#chat{height:60vh}#views{width:auto}}
</style></head><body>
<div id='chat'><header><svg class='mark' viewBox='0 0 24 24' fill='currentColor'><path d='M12 2l1.6 6.9L20 6l-4.6 5.1L22 12l-6.6.9L20 18l-6.4-2.9L12 22l-1.6-6.9L4 18l4.6-5.1L2 12l6.6-.9L4 6l6.4 2.9z'/></svg>Claude <small>FMB1 cell session</small></header><div id='log'></div><div id='status'><span class='dot'></span><span class='d'></span></div></div>
<div id='views'>__FIGS__</div>
<script src='https://cdn.jsdelivr.net/npm/marked@12.0.2/marked.min.js'></script>
<script src='https://cdn.jsdelivr.net/npm/dompurify@3.1.6/dist/purify.min.js'></script>
<script>
let since=-1;const log=document.getElementById('log');const recentUser=[];   // a message typed while busy is recorded twice
const q=new URLSearchParams(location.search).get('fps');
// an MJPEG stream through the proxy can stall or be cut, and an <img> never reconnects by itself (operator 10-23: the
// views froze until a reload): reconnect each stream every 40 s, on an error, and when the tab comes back into view
const imgs=[...document.querySelectorAll('#views img')];imgs.forEach(im=>im.dataset.base=im.getAttribute('src'));
function connect(im){im.src=im.dataset.base+'?'+(q?'fps='+encodeURIComponent(q)+'&':'')+'t='+Date.now()}
imgs.forEach(im=>{im.addEventListener('error',()=>setTimeout(()=>connect(im),2000));connect(im)});
setInterval(()=>imgs.forEach(connect),40000);
document.addEventListener('visibilitychange',()=>{if(!document.hidden)imgs.forEach(connect)});
function esc(s){return s.replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))}
function md(s){try{return DOMPurify.sanitize(marked.parse(s,{gfm:true,breaks:false}))}catch(e){return '<div style=white-space:pre-wrap>'+esc(s)+'</div>'}}
function add(e){
  if(e.k==='result'){const g=log.lastElementChild;if(!g||g.tagName!=='DETAILS')return;
    const d=document.createElement('div');d.className='res';d.textContent='  ← '+e.x;g.appendChild(d);return}
  if(e.k==='tool'){
    let g=log.lastElementChild;
    if(!g||g.tagName!=='DETAILS'){g=document.createElement('details');g.className='tools';g.innerHTML='<summary></summary>';g.n=0;log.appendChild(g)}
    const d=document.createElement('div');d.textContent=e.x;g.appendChild(d);g.n++;
    g.querySelector('summary').textContent=e.x.slice(0,90)+(g.n>1?'  (+'+(g.n-1)+' more)':'');return}
  if(e.k==='img'){const r=document.createElement('div');r.className='row '+(e.who==='user'?'user':'claude');
    const im=document.createElement('img');im.className='shot';im.src=e.x;
    im.onclick=()=>{const o=document.createElement('div');o.className='lightbox';const b=document.createElement('img');b.src=e.x;
      o.appendChild(b);o.onclick=()=>o.remove();document.body.appendChild(o)};
    r.appendChild(im);const t=document.createElement('div');t.className='ts';t.textContent=e.t;r.appendChild(t);
    log.appendChild(r);return}
  if(e.k==='event'){const d=document.createElement('div');d.className='event';
    d.innerHTML=esc(e.x)+"<span class='ts'>"+e.t+"</span>";log.appendChild(d);return}
  if(e.k==='user'){if(recentUser.includes(e.x))return;recentUser.push(e.x);if(recentUser.length>4)recentUser.shift()}
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
  setTimeout(poll,document.hidden?3000:500)}
poll();
// what the session is doing right now (the job timeline), with how long it has been at it
const st=document.getElementById('status');let stAt=null,stDetail='';
async function pollStatus(){
  try{const j=await (await fetch('status')).json();
    if(j.at){stAt=Date.parse(j.at);stDetail=j.detail||j.state;st.className=j.state==='working'?'working':''}}catch(err){}
  setTimeout(pollStatus,document.hidden?5000:1000)}
setInterval(()=>{if(stAt===null)return;const s=Math.max(0,Math.round((Date.now()-stAt)/1000));
  st.querySelector('.d').textContent=(st.className==='working'?'working · ':'')+stDetail+' · '+(s<120?s+'s':Math.round(s/60)+' min')+' ago'},500);
pollStatus();
</script></body></html>"""


class H(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        if self.path.startswith('/status'):
            import json
            body = json.dumps(status()).encode()
            self.send_response(200); self.send_header('Content-Type', 'application/json'); self.end_headers()
            self.wfile.write(body)
            return
        if self.path.startswith('/log') and a.transcript:
            import json
            from urllib.parse import parse_qs, urlparse
            since = int((parse_qs(urlparse(self.path).query).get('since') or ['-1'])[0])
            ent, nxt = log_entries(since)
            body = json.dumps({'entries': ent, 'next': nxt, 'first': since < 0}).encode()
            self.send_response(200); self.send_header('Content-Type', 'application/json'); self.end_headers()
            self.wfile.write(body)
            return
        if self.path.startswith('/snap'):
            # one JPEG: link previews (WhatsApp) take a page's image as the thumbnail and an MJPEG stream never ends
            try:
                i = int(self.path.split('/')[2])
            except (IndexError, ValueError):
                i = 0
            jpg = frame(min(max(i, 0), len(TOPICS) - 1)) or b''
            self.send_response(200 if jpg else 503); self.send_header('Content-Type', 'image/jpeg')
            self.send_header('Content-Length', str(len(jpg))); self.end_headers(); self.wfile.write(jpg)
            return
        if self.path.startswith('/stream'):
            try:
                pth = self.path.split('?')[0]
                i = int(pth.split('/')[2]) if pth.count('/') >= 2 and pth.split('/')[2] else 0
            except ValueError:
                i = 0
            i = min(max(i, 0), len(TOPICS) - 1)
            from urllib.parse import parse_qs, urlparse
            try:                                            # ?fps= per viewer, at most --fps (the cell's own screen: 1)
                fps = min(a.fps, max(0.2, float((parse_qs(urlparse(self.path).query).get('fps') or [a.fps])[0])))
            except ValueError:
                fps = a.fps
            self.send_response(200)
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
            self.end_headers()
            t_end = time.time() + 90                        # the page reconnects every 40 s; a dropped viewer frees its thread
            try:
                while time.time() < t_end:
                    jpg = frame(i)
                    if jpg:
                        self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: '
                                         + str(len(jpg)).encode() + b'\r\n\r\n' + jpg + b'\r\n')
                    time.sleep(1.0 / fps)
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
