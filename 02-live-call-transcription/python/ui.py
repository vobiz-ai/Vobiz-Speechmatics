"""Live transcript web UI for the proxy.

The proxy serves this page at /ui/<token> and pushes every call event and
transcript to it over a WebSocket at /events/<token>. Open the page in a
browser before (or during) a call to watch both speakers transcribed live.
"""

import asyncio
import base64
import json
from pathlib import Path
from html import escape

CLIENTS: set = set()


def publish(event: dict) -> None:
    """Send one event to every open browser tab. Safe to call from sync code."""
    if not CLIENTS:
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    message = json.dumps(event)
    for ws in list(CLIENTS):
        loop.create_task(_send(ws, message))


async def _send(ws, message: str) -> None:
    try:
        await ws.send(message)
    except Exception:  # noqa: BLE001 - a closed tab just drops out
        CLIENTS.discard(ws)


async def serve_events(ws) -> None:
    """Hold one browser connection open until the tab closes."""
    CLIENTS.add(ws)
    try:
        await ws.send(json.dumps({"type": "hello"}))
        async for _ in ws:
            pass
    except Exception:  # noqa: BLE001
        pass
    finally:
        CLIENTS.discard(ws)


def _logo(name: str) -> str:
    """The logo as a data URI, so the page needs no extra route."""
    path = Path(__file__).resolve().parents[2] / "logo" / name
    try:
        return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()
    except OSError:
        return ""


def page(events_path: str, vobiz_number: str, forward_to: str, language: str, lines: list | None = None) -> bytes:
    return (
        PAGE.replace("__EVENTS_PATH__", events_path)
        .replace("__LINES__", json.dumps(lines or []))
        .replace("__LOGO_VZ__", _logo("vobiz-light.png"))
        .replace("__LOGO_SM__", _logo("speechmatics-black.png"))
        .replace("__VOBIZ_NUMBER__", escape(vobiz_number or ""))
        .replace("__FORWARD_TO__", escape(forward_to or ""))
        .replace("__LANGUAGE__", escape(language or "en"))
        .encode()
    )


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Vobiz × Speechmatics · Live call transcription</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=DM+Sans:opsz,wght@9..40,400;9..40,500;9..40,600&family=Inter+Tight:wght@500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root{
  --bg:#f7f8f8; --panel:#fff; --ink:#160c08; --muted:#6b6f72; --line:#e6e8e9;
  --vobiz:#e8541e; --sm:#1e3d49; --a:#e8541e; --a-soft:#fff3ec; --b:#1e3d49; --b-soft:#eef4f5;
  --ok:#29a383; --warn:#e8541e;
  --display:"Inter Tight","DM Sans",system-ui,sans-serif; --body:"DM Sans",system-ui,sans-serif;
  --mono:"IBM Plex Mono",ui-monospace,Menlo,monospace;
}
*{box-sizing:border-box}
html,body{height:100%}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 var(--body);display:flex;flex-direction:column}
header{display:flex;align-items:center;gap:16px;flex-wrap:wrap;padding:14px max(16px,3vw);border-bottom:1px solid var(--line);background:var(--panel)}
.brand{font-family:var(--display);font-weight:700;font-size:20px;letter-spacing:-.01em}
.brand .x{color:var(--muted);font-weight:600;margin:0 6px}
.brand{display:flex;align-items:center;gap:14px}.brand img.vz{height:28px}.brand img.sm{height:17px}.brand .title{margin-left:8px;padding-left:16px;border-left:1px solid var(--line);font-family:var(--display);font-size:16px;font-weight:600;letter-spacing:-.01em}
.status{display:inline-flex;align-items:center;gap:8px;font-size:13px;font-weight:600;padding:5px 12px;border-radius:999px;border:1px solid var(--line)}
.dot{width:9px;height:9px;border-radius:50%;background:var(--muted)}
.status.live{background:var(--sm);color:#fff;border-color:var(--sm)}
.status.live .dot{background:#3ccf8e;box-shadow:0 0 0 0 var(--ok);animation:pulse 1.6s infinite}
.status.waiting .dot{background:var(--warn)}
@keyframes pulse{0%{box-shadow:0 0 0 0 color-mix(in srgb,var(--ok) 60%,transparent)}100%{box-shadow:0 0 0 10px transparent}}
.meta{margin-left:auto;display:flex;gap:18px;flex-wrap:wrap;font-size:13px;color:var(--muted)}
.meta b{color:var(--ink);font-family:var(--mono);font-weight:500}
main{flex:1;display:grid;grid-template-columns:minmax(0,1fr) 280px;gap:0;min-height:0}
@media (max-width:860px){main{grid-template-columns:1fr}aside{display:none}}
#feed{overflow-y:auto;padding:24px max(16px,3vw) 40px;display:flex;flex-direction:column;gap:14px}
.empty{margin:auto;text-align:center;color:var(--muted);max-width:44ch}
.empty h2{font-family:var(--display);color:var(--ink);font-size:22px;margin:0 0 6px}
.turn{display:flex;flex-direction:column;max-width:70%;gap:4px}
.turn.inbound{align-self:flex-start}
.turn.outbound{align-self:flex-end;align-items:flex-end}
.who{font-size:11.5px;font-weight:600;letter-spacing:.06em;text-transform:uppercase}
.inbound .who{color:var(--a)} .outbound .who{color:var(--b)}
.bubble{padding:12px 16px;border-radius:16px;font-size:17px;line-height:1.5}
.inbound .bubble{background:var(--a-soft);border:1px solid #f7c9b2;border-top-left-radius:3px}
.outbound .bubble{background:var(--b-soft);border:1px solid #cfdde1;border-top-right-radius:3px}
.partial{color:var(--muted);font-style:italic}
.live{position:sticky;bottom:0;margin-top:auto;display:flex;flex-direction:column;gap:6px;padding-top:10px;background:linear-gradient(transparent,var(--bg) 30%)}
.live div{font-size:14px;color:var(--muted);font-style:italic;min-height:0}
.live div:empty{display:none}
.live b{font-style:normal;font-weight:600;font-size:11.5px;letter-spacing:.06em;text-transform:uppercase;margin-right:8px}
.live .inbound b{color:var(--a)} .live .outbound b{color:var(--b)}
.greet{align-self:center;font-size:13px;color:var(--ink);padding:6px 14px;border-radius:999px;background:color-mix(in srgb,var(--vobiz) 14%,transparent);border:1px solid color-mix(in srgb,var(--vobiz) 40%,transparent)}
.greet b{color:var(--vobiz);margin-right:6px}
.sys{align-self:center;font-size:12.5px;color:var(--muted);font-family:var(--mono);padding:4px 10px;border:1px dashed var(--line);border-radius:999px}
aside{border-left:1px solid var(--line);background:var(--panel);color:var(--ink);padding:22px;overflow-y:auto;display:flex;flex-direction:column;gap:22px}
aside h3{font-size:12px;font-weight:600;letter-spacing:.08em;text-transform:uppercase;color:var(--sm);margin:0 0 10px}
.legend{display:flex;flex-direction:column;gap:10px;font-size:14px}
.legend div{display:flex;align-items:center;gap:10px}
.sw{width:12px;height:12px;border-radius:4px}
.stats{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.stat{border:1px solid var(--line);background:#f7f8f8;border-radius:10px;padding:12px}
.stat .v{font-family:var(--mono);font-size:22px;font-weight:500;font-variant-numeric:tabular-nums}
.stat .l{font-size:12px;color:var(--muted)}
.pipeline{font-size:13.5px;color:var(--muted);display:flex;flex-direction:column;gap:6px}
.pipeline span{font-family:var(--mono);font-size:12.5px;color:var(--ink)}
button{font:inherit;font-size:13.5px;padding:8px 14px;border-radius:6px;border:1px solid var(--line);background:#fff;color:var(--ink);cursor:pointer}
button:hover{border-color:var(--sm);color:var(--sm)}
main{grid-template-columns:minmax(0,1fr) 320px}
.gear{display:inline-flex;align-items:center;gap:8px;height:36px;font-weight:500}
.scrim{position:fixed;inset:0;background:rgba(22,12,8,.28);opacity:0;pointer-events:none;transition:opacity .2s;z-index:10}
.drawer{position:fixed;top:0;right:0;bottom:0;width:min(440px,100vw);background:#fff;border-left:1px solid var(--line);box-shadow:-12px 0 40px rgba(0,0,0,.08);transform:translateX(100%);transition:transform .22s ease;z-index:11;display:flex;flex-direction:column}
body.open .scrim{opacity:1;pointer-events:auto} body.open .drawer{transform:none}
.dh{display:flex;align-items:center;justify-content:space-between;height:64px;padding:0 24px;border-bottom:1px solid var(--line);font-family:var(--display);font-weight:600;font-size:17px}
.dh button{border:0;font-size:24px;line-height:1;color:var(--muted);padding:4px 8px}
.db{flex:1;overflow-y:auto;padding:22px 24px}
.db h4{margin:0 0 4px;font-size:12px;font-weight:600;letter-spacing:.08em;text-transform:uppercase;color:var(--sm)}
.db p{margin:0 0 16px;font-size:13.5px;color:var(--muted)}
.fld{display:flex;flex-direction:column;gap:6px;margin-bottom:14px}
.fld label{font-size:13px;font-weight:500}
.fld label i{font-style:normal;display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:6px}
.fld input,.fld select{height:42px;padding:0 12px;border:1px solid #d9dddf;border-radius:8px;font:15px var(--body);color:var(--ink);background:#fff}
.fld input{font-family:var(--mono);font-size:16px}
.fld input:focus,.fld select:focus{outline:2px solid #cddcdf;border-color:var(--sm)}
.flow{display:flex;align-items:center;gap:8px;font-size:12.5px;color:var(--muted);margin:2px 0 16px;font-family:var(--mono)}
.go{width:100%;height:44px;border:0;border-radius:8px;background:var(--vobiz);color:#fff;font-weight:600;font-size:15px}
.go:hover{color:#fff;opacity:.92} .go:disabled{opacity:.55;cursor:default}
.msgline{margin-top:10px;font-size:13.5px;min-height:20px} .msgline.ok{color:var(--ok)} .msgline.err{color:#c2410c}
button:focus-visible{outline:2px solid var(--sm);outline-offset:2px}
</style>
</head>
<body>
<header>
  <div class="brand"><img class="vz" src="__LOGO_VZ__" alt="Vobiz"><span class="x">×</span><img class="sm" src="__LOGO_SM__" alt="Speechmatics"><span class="title">Live call transcription</span></div>
  <div id="status" class="status waiting"><span class="dot"></span><span id="statusText">Waiting for a call</span></div>
  <div class="meta">
    <span>Vobiz number <b id="mFrom">__VOBIZ_NUMBER__</b></span>
    <span>Bridged to <b id="mTo">__FORWARD_TO__</b></span>
    <span>Language <b>__LANGUAGE__</b></span>
    <span>Duration <b id="timer">0:00</b></span>
  </div>
  <button class="gear" id="gear" type="button"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.7 1.7 0 0 0 .3 1.8l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.7 1.7 0 0 0-1.8-.3 1.7 1.7 0 0 0-1 1.5V21a2 2 0 1 1-4 0v-.1a1.7 1.7 0 0 0-1.1-1.5 1.7 1.7 0 0 0-1.8.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1a1.7 1.7 0 0 0 .3-1.8 1.7 1.7 0 0 0-1.5-1H3a2 2 0 1 1 0-4h.1a1.7 1.7 0 0 0 1.5-1.1 1.7 1.7 0 0 0-.3-1.8l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.7 1.7 0 0 0 1.8.3H9a1.7 1.7 0 0 0 1-1.5V3a2 2 0 1 1 4 0v.1a1.7 1.7 0 0 0 1 1.5 1.7 1.7 0 0 0 1.8-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.7 1.7 0 0 0-.3 1.8V9a1.7 1.7 0 0 0 1.5 1H21a2 2 0 1 1 0 4h-.1a1.7 1.7 0 0 0-1.5 1z"/></svg>Settings</button>
</header>
<main>
  <section id="feed" aria-live="polite">
    <div class="empty" id="empty">
      <h2>Ready to transcribe</h2>
      Call the Vobiz number. Both sides of the conversation appear here live,
      transcribed by Speechmatics Realtime as separate speakers.
    </div>
    <div class="live" id="live"><div class="inbound" id="liveIn"></div><div class="outbound" id="liveOut"></div></div>
  </section>
  <aside>
    <div>
      <h3>Speakers</h3>
      <div class="legend">
        <div><span class="sw" style="background:var(--a)"></span>Caller <small style="color:var(--muted)">· inbound track</small></div>
        <div><span class="sw" style="background:#29a383"></span>Bridged party <small style="color:var(--muted)">· outbound track</small></div>
      </div>
    </div>
    <div>
      <h3>This call</h3>
      <div class="stats">
        <div class="stat"><div class="v" id="nIn">0</div><div class="l">caller words</div></div>
        <div class="stat"><div class="v" id="nOut">0</div><div class="l">bridged words</div></div>
        <div class="stat"><div class="v" id="nTurns">0</div><div class="l">turns</div></div>
        <div class="stat"><div class="v" id="fmt">–</div><div class="l">audio format</div></div>
      </div>
    </div>
    <div>
      <h3>Pipeline</h3>
      <div class="pipeline">
        <div>Vobiz <span>&lt;Stream audioTrack="both"&gt;</span></div>
        <div>→ proxy, no transcoding</div>
        <div>→ Speechmatics RT <span>diarization="channel"</span></div>
      </div>
    </div>
    <div><button id="clear" type="button">Clear transcript</button></div>
  </aside>
</main>
<div class="scrim" id="scrim"></div>
<aside class="drawer" id="drawer" aria-label="Settings">
  <div class="dh">Settings<button id="close" type="button" aria-label="Close">×</button></div>
  <div class="db">
    <h4>Connect two people</h4>
    <p>Vobiz rings party A. When A answers, it rings party B and joins them. Both sides are transcribed here live.</p>
    <div class="flow">Vobiz → A → B</div>
    <div class="fld"><label for="from">Call from (Vobiz number)</label><select id="from"></select></div>
    <div class="fld"><label for="pa"><i style="background:var(--a)"></i>Party A (rung first)</label><input id="pa" type="tel" placeholder="+44 7700 900123"></div>
    <div class="fld"><label for="pb"><i style="background:#29a383"></i>Party B (rung when A answers)</label><input id="pb" type="tel" placeholder="+91 91482 27303"></div>
    <button class="go" id="go" type="button">Call A, then connect B</button>
    <div class="msgline" id="gom"></div>
  </div>
</aside>
<script>
const LINES = __LINES__;
const fmtN = (n) => { let m; if ((m = n.match(/^\+44(\d{4})(\d{6})$/))) return `+44 ${m[1]} ${m[2]}`; if ((m = n.match(/^\+91(\d{5})(\d{5})$/))) return `+91 ${m[1]} ${m[2]}`; return n; };
const $ = (id) => document.getElementById(id);
LINES.forEach(n => { const o = document.createElement('option'); o.value = n; o.textContent = fmtN(n); $('from').appendChild(o); });
const clean = (v) => v.replace(/[\s().-]/g, '');
$('pa').oninput = () => { const a = clean($('pa').value); const l = LINES.find(n => a.length > 3 && n.slice(0, 3) === a.slice(0, 3)); if (l) $('from').value = l; };
const openD = () => { document.body.classList.add('open'); setTimeout(() => $('pa').focus(), 250); };
const closeD = () => document.body.classList.remove('open');
$('gear').onclick = openD; $('close').onclick = closeD; $('scrim').onclick = closeD;
document.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeD(); });
function say(t, ok){ $('gom').textContent = t; $('gom').className = 'msgline ' + (ok === undefined ? '' : ok ? 'ok' : 'err'); }
$('go').onclick = async () => {
  const a = clean($('pa').value), b = clean($('pb').value);
  if (!a || !b) { say('Enter both numbers with their country code.', false); return; }
  $('go').disabled = true; say('Placing the call…');
  const q = new URLSearchParams({a, b, from: $('from').value});
  let ok = false, d = {};
  try { const r = await fetch(location.pathname.replace('/ui/', '/bridge/') + '?' + q); ok = r.ok; d = await r.json(); } catch (_) {}
  say(ok ? `Ringing ${fmtN(a)}. When A answers, ${fmtN(b)} is rung and joined.` : (d.error || 'The call could not be placed.'), ok);
  setTimeout(() => $('go').disabled = false, ok ? 10000 : 500);
  if (ok) setTimeout(closeD, 1800);
};
const feed = document.getElementById('feed');
const empty = document.getElementById('empty');
const statusEl = document.getElementById('status');
const statusText = document.getElementById('statusText');
const timerEl = document.getElementById('timer');
const LABEL = {inbound: 'Caller', outbound: 'Bridged party'};
let last = null, startedAt = null, timer = null;
const count = {inbound: 0, outbound: 0, turns: 0};

function setStatus(cls, text){ statusEl.className = 'status ' + cls; statusText.textContent = text; }
function scroll(){ feed.scrollTop = feed.scrollHeight; }
const liveBox = document.getElementById('live');
const liveEl = {inbound: document.getElementById('liveIn'), outbound: document.getElementById('liveOut')};
function add(node){ feed.insertBefore(node, liveBox); }
function sys(text){ Object.keys(open).forEach(k => delete open[k]); empty.hidden = true; const d = document.createElement('div'); d.className = 'sys'; d.textContent = text; add(d); last = null; scroll(); }
function greet(text){ empty.hidden = true; const d = document.createElement('div'); d.className = 'greet'; d.innerHTML = '<b>Vobiz played</b>'; d.appendChild(document.createTextNode(text)); add(d); last = null; scroll(); }
function setLive(ch, text){ const el = liveEl[ch]; if (!el) return; el.innerHTML = ''; if (text){ const b = document.createElement('b'); b.textContent = LABEL[ch] || ch; el.appendChild(b); el.appendChild(document.createTextNode(text + ' …')); } }
const GREETING = /this call is being transcribed\s*\.?/i;
let greeted = false;
// One open bubble per speaker: words keep joining it until that speaker pauses for PAUSE_MS,
// even if the other side speaks in between. A pause starts a new bubble.
const PAUSE_MS = 1200;
const open = {};
function turnFor(ch){
  const o = open[ch];
  if (o && Date.now() - o.at < PAUSE_MS) { o.at = Date.now(); return o; }
  empty.hidden = true;
  const t = document.createElement('div'); t.className = 'turn ' + ch;
  t.innerHTML = '<div class="who"></div><div class="bubble"><span class="final"></span></div>';
  t.querySelector('.who').textContent = LABEL[ch] || ch;
  add(t);
  count.turns++; document.getElementById('nTurns').textContent = count.turns;
  last = {ch, final: t.querySelector('.final'), at: Date.now()};
  open[ch] = last;
  return last;
}
function words(s){ return (s.match(/\S+/g) || []).length; }
function tick(){ const s = Math.floor((Date.now() - startedAt) / 1000); timerEl.textContent = Math.floor(s/60) + ':' + String(s%60).padStart(2,'0'); }

function onEvent(e){
  if (e.type === 'bridge'){ $('mFrom').textContent = e.from; $('mTo').textContent = e.b; LABEL.inbound = 'Party A'; LABEL.outbound = 'Party B'; return; }
  if (e.type === 'call_started'){
    setStatus('live', 'Live call');
    document.getElementById('fmt').textContent = (e.format || '').replace('audio/x-','').replace('/', ' ');
    sys('call started · ' + new Date().toLocaleTimeString()); greeted = false;
    startedAt = Date.now(); clearInterval(timer); timer = setInterval(tick, 500);
  } else if (e.type === 'call_ended'){
    setStatus('', 'Call ended'); clearInterval(timer);
    setLive('inbound', ''); setLive('outbound', '');
    sys('call ended · ' + new Date().toLocaleTimeString());
  } else if (e.type === 'partial'){
    setLive(e.channel, (e.text || '').replace(GREETING, '').trim()); scroll();
  } else if (e.type === 'final'){
    if (!e.text) return;
    setLive(e.channel, '');
    let text = e.text;
    if (!greeted && e.channel === 'outbound' && GREETING.test(text)){
      greeted = true; greet('This call is being transcribed.');
      text = text.replace(GREETING, '').replace(/^\s+/, '');
      if (!text) return;
    }
    if (!/[A-Za-z0-9]/.test(text)) { const o = open[e.channel]; if (o) { o.final.textContent += text.trim(); o.at = Date.now(); } return; }
    const t = turnFor(e.channel); const cur = t.final.textContent;
    t.final.textContent = cur ? (cur + ' ' + text.trim()).replace(/\s+([.,?!])/g, '$1') : text.trim();
    count[e.channel] = (count[e.channel] || 0) + words(e.text);
    document.getElementById('nIn').textContent = count.inbound;
    document.getElementById('nOut').textContent = count.outbound;
    scroll();
  }
}

function connect(){
  const ws = new WebSocket((location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host + '__EVENTS_PATH__');
  ws.onmessage = m => onEvent(JSON.parse(m.data));
  ws.onclose = () => { if (!startedAt) setStatus('waiting', 'Reconnecting…'); setTimeout(connect, 1500); };
  ws.onopen = () => { if (!startedAt) setStatus('waiting', 'Waiting for a call'); };
}
document.getElementById('clear').onclick = () => {
  feed.querySelectorAll('.turn,.sys,.greet').forEach(n => n.remove()); setLive('inbound',''); setLive('outbound',''); last = null; empty.hidden = false; Object.keys(open).forEach(k => delete open[k]);
  count.inbound = count.outbound = count.turns = 0;
  ['nIn','nOut','nTurns'].forEach(id => document.getElementById(id).textContent = '0');
};
connect();
</script>
</body>
</html>
"""
