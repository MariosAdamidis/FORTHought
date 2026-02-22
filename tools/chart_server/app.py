# chart_server/app.py -- Chart.js server-side rendering service
# Author:  Marios Adamidis (FORTHought Lab)

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.requests import Request
import base64, json, html as htmlmod

app = FastAPI(title="OWUI Chart Server")

# Serve static both at root and with /charts prefix (for path-based proxy)
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/charts/static", StaticFiles(directory="static"), name="static_charts")

# Allow embedding by your domain (adjust domain if needed)
@app.middleware("http")
async def add_frame_headers(request: Request, call_next):
    resp = await call_next(request)
    # embed from same origin (when proxied) and explicit domain
    resp.headers["Content-Security-Policy"] = "frame-ancestors 'self' *"
    return resp

HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <title>__TITLE__</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <!-- NOTE: RELATIVE path so it works at /chart_b64 AND /charts/chart_b64 -->
  <script src="static/chart.umd.min.js"></script>
  <style>
    html, body { height: 100%; margin: 0; padding: 0; overflow: hidden; }
    :root {
      --bg: #0b0f19;            /* dark defaults */
      --fg: #e6edf3;
      --grid: rgba(255,255,255,0.15);
    }
    body.light {
      --bg: #ffffff;            /* light overrides */
      --fg: #0b0f19;
      --grid: rgba(0,0,0,0.25);  /* stronger so grid is clearly visible */
    }
    body {
      font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
      background: var(--bg); color: var(--fg);
    }
    .wrap { display:flex; flex-direction:column; height:100%; box-sizing:border-box; padding:8px 12px 12px; }
    .toolbar { flex:0 0 auto; display:flex; gap:8px; align-items:center; margin-bottom:8px; }
    .btn {
      border:1px solid rgba(127,127,127,.4); background: transparent; color: var(--fg);
      padding:6px 10px; border-radius:8px; cursor:pointer;
    }
    .frame { flex:1 1 auto; position:relative; width:100%; background:transparent; border-radius:12px; overflow:hidden; }
    canvas { width:100% !important; height:100% !important; display:block; }
    .err { position:absolute; inset:0; display:flex; align-items:center; justify-content:center; color:#fff;
           background:#b00020; font-weight:600; padding:12px; text-align:center; }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="toolbar">
      <strong style="flex:1">__TITLE__</strong>
      <button id="themeBtn" class="btn" title="Toggle theme">Toggle Theme</button>
      <button id="dlBtn" class="btn" title="Download PNG">Download PNG</button>
    </div>
    <div class="frame">
      <canvas id="owuiChart"></canvas>
      <div id="err" class="err" style="display:none"></div>
    </div>
  </div>

  <!-- Safe config embedding -->
  <script id="cfg" type="application/json">__CONFIG__</script>

  <script>
    function showErr(msg){
      var el = document.getElementById('err');
      el.textContent = msg; el.style.display = 'flex';
      console.error(msg);
    }

    // --- Palette: neon but subtle ---
    var PALETTE_BASE = [
      "#00E5FF", "#FF3B81", "#7C4DFF", "#10B981", "#F59E0B",
      "#3B82F6", "#F43F5E", "#14B8A6", "#EAB308", "#A855F7"
    ];

    function hexToRgb(hex) {
      var h = hex.replace('#','');
      var b = parseInt(h.length===3 ? h.split('').map(function(x){return x+x;}).join('') : h, 16);
      return [(b>>16)&255, (b>>8)&255, b&255];
    }
    function rgba(hex, a) {
      var c = hexToRgb(hex); return "rgba(" + c[0] + "," + c[1] + "," + c[2] + "," + a + ")";
    }
    function deepClone(obj){ return JSON.parse(JSON.stringify(obj || {})); }

    function buildConfig(base, isLight){
      var cfg = deepClone(base);
      var cs  = getComputedStyle(document.body);
      var fg  = cs.getPropertyValue('--fg').trim();
      var grid= cs.getPropertyValue('--grid').trim();

      // Slightly stronger on dark; lighter on light
      var fillAlpha = isLight ? 0.18 : 0.26;

      // per-type defaults
      if (cfg.type === 'line') {
        var dsLine = (cfg.data && cfg.data.datasets) ? cfg.data.datasets : [];
        dsLine.forEach(function(d){
          if (d.tension === undefined) d.tension = 0.30;
          if (d.borderWidth === undefined) d.borderWidth = 2.5;
        });
      }
      if (cfg.type === 'bar') {
        var dsBar = (cfg.data && cfg.data.datasets) ? cfg.data.datasets : [];
        dsBar.forEach(function(d){
          if (d.borderWidth === undefined) d.borderWidth = 1.5;
          if (d.borderRadius === undefined) d.borderRadius = 4;
        });
      }

      // dataset colors (respect manual colors)
      if (cfg.type === 'pie' || cfg.type === 'doughnut' || cfg.type === 'polarArea') {
        var d0 = (cfg.data && cfg.data.datasets && cfg.data.datasets[0]) ? cfg.data.datasets[0] : null;
        if (d0 && !(d0.borderColor || d0.backgroundColor)) {
          var N = (d0.data || []).length;
          d0.borderColor     = Array.from({length:N}, function(_,i){ return PALETTE_BASE[i % PALETTE_BASE.length]; });
          d0.backgroundColor = Array.from({length:N}, function(_,i){ return rgba(PALETTE_BASE[i % PALETTE_BASE.length], fillAlpha); });
          d0.hoverBackgroundColor = Array.from({length:N}, function(_,i){
            return rgba(PALETTE_BASE[i % PALETTE_BASE.length], isLight ? 0.28 : 0.35);
          });
        }
      } else {
        var ds = (cfg.data && cfg.data.datasets) ? cfg.data.datasets : [];
        ds.forEach(function(d,i){
          if (!(d.borderColor || d.backgroundColor)) {
            var base = PALETTE_BASE[i % PALETTE_BASE.length];
            d.borderColor     = base;
            d.backgroundColor = rgba(base, fillAlpha);
          }
        });
      }

      // options: axes/legend/title/tooltip + animations + elements
      cfg.options = cfg.options || {};
      cfg.options.elements = Object.assign({}, cfg.options.elements || {}, {
        arc:   { borderWidth: 1, borderColor: grid },
        line:  { borderWidth: 2.5 },
        point: { radius: 3, hoverRadius: 5 }
      });
      if (!cfg.options.animation) cfg.options.animation = { duration: 600, easing: 'easeOutQuart' };
      cfg.options.animations = Object.assign({}, cfg.options.animations || {}, {
        colors:  { type: 'color', duration: 500 },
        numbers: { duration: 300 }
      });

      cfg.options.plugins = cfg.options.plugins || {};
      var legend = cfg.options.plugins.legend || {};
      legend.labels = Object.assign({}, legend.labels || {}, { color: fg });
      cfg.options.plugins.legend = legend;

      var title = cfg.options.plugins.title || {};
      title.color = fg;
      cfg.options.plugins.title = title;

      // Tooltip: light theme uses dark bg with WHITE text; dark theme uses translucent light bg with light text
      var tooltip = cfg.options.plugins.tooltip || {};
      tooltip.backgroundColor = isLight ? "rgba(0,0,0,0.85)" : "rgba(255,255,255,0.10)";
      tooltip.titleColor      = isLight ? "#ffffff"          : fg;
      tooltip.bodyColor       = isLight ? "#ffffff"          : fg;
      tooltip.borderColor     = grid;
      tooltip.borderWidth     = 1;
      cfg.options.plugins.tooltip = tooltip;

      if (cfg.options.scales) {
        for (var k in cfg.options.scales) {
          if (!cfg.options.scales.hasOwnProperty(k)) continue;
          var s = cfg.options.scales[k] || {};
          s.ticks = Object.assign({}, s.ticks || {}, { color: fg });
          s.grid  = Object.assign({}, s.grid  || {}, { color: grid });
          cfg.options.scales[k] = s;
        }
      }
      return cfg;
    }

    (function(){
      function init(){
        try {
          if (typeof window.Chart === 'undefined') return showErr("Chart.js failed to load");
          var cfgEl = document.getElementById('cfg');
          var txt = cfgEl ? cfgEl.textContent : "{}";
          var base;
          try { base = JSON.parse(txt || "{}"); } catch(e){ return showErr("Bad JSON: " + e.message); }
          if (!base || !base.type) return showErr("Invalid chart config");

          var canvas = document.getElementById('owuiChart');
          if (!canvas) return showErr("Canvas not found");
          var ctx = canvas.getContext('2d');

          // start dark (no .light)
          var isLight = false;
          var chart = new Chart(ctx, buildConfig(base, isLight));

          // 🔔 Handshake so the wrapper knows scripts are running
          try {
            var title = (chart.options && chart.options.plugins && chart.options.plugins.title && chart.options.plugins.title.text) || "chart";
            if (window.parent && window.parent !== window) {
              window.parent.postMessage({type:"owui-chart-ready", title: title}, "*");
            }
          } catch(e) {}

          // PNG
          var dl = document.getElementById('dlBtn');
          if (dl) dl.addEventListener('click', function(){
            var url = chart.toBase64Image('image/png', 1.0);
            var a = document.createElement('a');
            var name = (chart.options && chart.options.plugins && chart.options.plugins.title && chart.options.plugins.title.text) || 'chart';
            a.href = url; a.download = name + '.png';
            a.target = '_blank'; a.rel='noopener'; a.click();
          });

          // Theme toggle -> rebuild
          var tb = document.getElementById('themeBtn');
          if (tb) tb.addEventListener('click', function(){
            isLight = !isLight;
            document.body.classList.toggle('light', isLight);
            chart.destroy();
            chart = new Chart(ctx, buildConfig(base, isLight));
          });

        } catch (e) {
          console.error(e);
          showErr("Chart error: " + e.message);
        }
      }
      if (document.readyState === 'complete') {
        init();
      } else {
        window.addEventListener('load', init, { once: true });
      }
    })();
  </script>
</body>
</html>
"""

def _render_html(config: dict, title: str) -> HTMLResponse:
    cfg_json = json.dumps(config, ensure_ascii=False).replace("</script", "<\\/script")
    html_out = (
        HTML_TEMPLATE
        .replace("__TITLE__", htmlmod.escape(title))
        .replace("__CONFIG__", cfg_json)
    )
    return HTMLResponse(content=html_out, media_type="text/html")

# Routes (both root and /charts prefix)
@app.get("/chart_b64", response_class=HTMLResponse)
def chart_b64(cfg: str = Query(...), title: str = Query("Chart"), height: int = Query(440)):
    try:
        data = base64.urlsafe_b64decode(cfg.encode()).decode()
        config = json.loads(data)
    except Exception as e:
        return HTMLResponse(f"<b>Bad cfg</b>: {htmlmod.escape(str(e))}", status_code=400)
    return _render_html(config, title)

@app.get("/charts/chart_b64", response_class=HTMLResponse)
def chart_b64_prefixed(cfg: str = Query(...), title: str = Query("Chart"), height: int = Query(440)):
    return chart_b64(cfg=cfg, title=title, height=height)

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/charts/health")
def health_prefixed():
    return health()
