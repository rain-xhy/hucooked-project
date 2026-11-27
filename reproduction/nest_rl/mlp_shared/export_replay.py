"""Export an interactive HTML replay + animated GIF + JSON trace for SharedKitchen."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch

from .env import (
    HEIGHT, LAYOUT, WIDTH, SharedKitchenConfig, TorchSharedKitchen,
)
from .fast_partners import GpuFastPartners
from .gpu_partners import GpuSharedPartners
from .history import OnlineHistory
from .policy import V3Policy, partner_action_ids

ACT_NAMES = ("N", "S", "E", "W", "STAY", "INT")
DIR_NAMES = ("N", "S", "E", "W")
ITEM_NAMES = ("", "RM", "RS", "CS", "CM", "PL", "MEAL")
KIND_NAMES = ("normal", "joint")
STAT_NAMES = ("queued", "active", "started", "served")


def collect_episode(mode="script", ckpt=None, seed=0, horizon=300, device="cuda"):
    cfg = SharedKitchenConfig(horizon=horizon, shuffle_players=False)
    env = TorchSharedKitchen(cfg, num_envs=1, device=device, use_cuda_graph=False)
    obs, _ = env.reset(seed=seed)
    partner_cls = GpuFastPartners if mode == "fast_script" else GpuSharedPartners
    partners = partner_cls(
        env, roles=torch.zeros(1, dtype=torch.long, device=device),
        learner_mask=torch.zeros(1, dtype=torch.bool, device=device), use_cuda_graph=False)
    partners.reset()

    model = None
    hs = None
    if mode == "all_policy":
        model = V3Policy(3).to(device).eval()
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False)["model"])
        hs = [OnlineHistory(1, device, stagger=False) for _ in range(3)]
        for h in hs:
            h.refresh(model)

    def snapshot(tick, actions=None, reward=0.0):
        s = env.state
        ck_item = int(s["cooker_item"][0])
        ck_slot = int(s["cooker_slot"][0])
        ck_rem = int(s["cooker_rem"][0])
        ck_kind = KIND_NAMES[int(s["order_kind"][0, max(0, min(7, ck_slot))])] if ck_item != 0 else ""
        orders = []
        for sl in range(8):
            st = int(s["order_status"][0, sl])
            if st in (1, 2):
                orders.append(dict(
                    slot=sl, id=int(s["order_id"][0, sl]),
                    kind=KIND_NAMES[int(s["order_kind"][0, sl])],
                    status=STAT_NAMES[st],
                ))
        return dict(
            tick=tick,
            actions=[ACT_NAMES[a] for a in actions] if actions is not None else ["", "", ""],
            reward=float(reward),
            score=float(s["score"][0]),
            normal_completed=int(s["normal_completed"][0]),
            joint_completed=int(s["joint_completed"][0]),
            joint_wait_steps=int(s["joint_ready_steps"][0]),
            unmatched_steps=int(s["unmatched_attempt_steps"][0]),
            chefs=[dict(
                id=c,
                x=int(s["pos"][0, c]) % WIDTH,
                y=int(s["pos"][0, c]) // WIDTH,
                facing=DIR_NAMES[int(s["facing"][0, c])],
                held=ITEM_NAMES[int(s["held"][0, c])],
            ) for c in range(3)],
            cooker=dict(item=ITEM_NAMES[ck_item], rem=ck_rem, kind=ck_kind),
            cutting=dict(item=ITEM_NAMES[int(s["cutting_item"][0])], prog=int(s["cutting_prog"][0])),
            asm=dict(
                main=ITEM_NAMES[int(s["asm_main"][0])],
                side=ITEM_NAMES[int(s["asm_side"][0])],
                plate=ITEM_NAMES[int(s["asm_plate"][0])],
            ),
            buf=ITEM_NAMES[int(s["buf_item"][0])],
            orders=orders,
        )

    frames = [snapshot(0)]
    for t in range(horizon):
        prev = obs.clone()
        if mode in ("script", "fast_script"):
            joint = partners.actions()
        else:
            joint = torch.stack([hs[c].forward(model, prev[:, c])[0].argmax(-1) for c in range(3)], dim=1)
        act_list = joint[0].tolist()
        obs, rew, _, _, _ = env.step(joint)
        if hs is not None:
            pids = partner_action_ids(joint)
            for c in range(3):
                hs[c].append(prev[:, c], pids[:, c], model)
        frames.append(snapshot(t + 1, act_list, float(rew[0])))

    return dict(
        mode=mode,
        layout=list(LAYOUT),
        width=WIDTH,
        height=HEIGHT,
        horizon=horizon,
        metrics=env.episode_metrics(),
        frames=frames,
    )


def render_gif(data, out_gif: Path, fps=8):
    from PIL import Image, ImageDraw, ImageFont

    cell = 76
    pad = 16
    top_h = 74
    W = data["width"] * cell + 2 * pad
    H = data["height"] * cell + 2 * pad + top_h
    try:
        font_lg = ImageFont.truetype("arial.ttf", 18)
        font_md = ImageFont.truetype("arial.ttf", 14)
        font_sm = ImageFont.truetype("arial.ttf", 12)
    except Exception:
        font_lg = font_md = font_sm = ImageFont.load_default()

    tile_colors = {
        "X": (52, 58, 64), " ": (241, 243, 245),
        "M": (231, 111, 81), "L": (106, 153, 78),
        "G": (214, 40, 40), "K": (233, 196, 106),
        "S": (42, 157, 143), "A": (173, 181, 189),
        "B": (141, 153, 174), "D": (72, 149, 239),
    }
    chef_colors = [(33, 118, 255), (247, 127, 0), (56, 176, 0)]
    chef_tags = ("S", "M", "D")
    dir_delta = {"N": (0, -24), "S": (0, 24), "E": (24, 0), "W": (-24, 0)}

    imgs = []
    for f in data["frames"]:
        im = Image.new("RGB", (W, H), (25, 28, 32))
        dr = ImageDraw.Draw(im)
        # Header
        meals = f["normal_completed"] + f["joint_completed"]
        hdr = f"Tick {f['tick']:3d}/{data['horizon']}  |  Score: {int(f['score']):3d}  |  Meals: {meals:2d} (N:{f['normal_completed']} J:{f['joint_completed']})"
        dr.text((pad, 10), hdr, fill=(248, 249, 250), font=font_lg)
        ck = f["cooker"]
        jw_color = (255, 107, 107) if (ck["item"] == "CM" and ck["rem"] == 0 and ck["kind"] == "joint") else (173, 181, 189)
        sub = f"Joint Wait: {f['joint_wait_steps']:3d}  |  Unmatched: {f['unmatched_steps']}  |  Acts: S={f['actions'][0]:4s} M={f['actions'][1]:4s} D={f['actions'][2]:4s}"
        dr.text((pad, 40), sub, fill=jw_color, font=font_md)

        # Grid
        for y, row in enumerate(data["layout"]):
            for x, ch in enumerate(row):
                x0, y0 = pad + x * cell, top_h + pad + y * cell
                x1, y1 = x0 + cell - 2, y0 + cell - 2
                col = tile_colors.get(ch, (120, 120, 120))
                if ch == "G" and ck["item"] == "CM" and ck["rem"] == 0 and ck["kind"] == "joint":
                    col = (255, 59, 48)
                dr.rectangle([x0, y0, x1, y1], fill=col, outline=(33, 37, 41), width=1)
                if ch not in ("X", " "):
                    dr.text((x0 + 5, y0 + 3), ch, fill=(255, 255, 255), font=font_sm)
                # Station contents
                if ch == "G" and ck["item"]:
                    tag = f"{ck['item']}({ck['rem']})" if ck["rem"] > 0 else f"{ck['item']}!"
                    if ck["kind"] == "joint":
                        tag += " [J]"
                    dr.text((x0 + 5, y0 + 26), tag, fill=(255, 255, 255), font=font_sm)
                elif ch == "K" and f["cutting"]["item"]:
                    dr.text((x0 + 5, y0 + 26), f"{f['cutting']['item']}({f['cutting']['prog']}/2)", fill=(20, 20, 20), font=font_sm)
                elif ch == "A":
                    a = f["asm"]
                    lines = [f"M:{a['main'] or '-'}", f"S:{a['side'] or '-'}", f"P:{a['plate'] or '-'}"]
                    dr.text((x0 + 5, y0 + 20), "\n".join(lines), fill=(20, 20, 20), font=font_sm)
                elif ch == "B" and f["buf"]:
                    dr.text((x0 + 5, y0 + 28), f["buf"], fill=(255, 255, 255), font=font_sm)

        # Chefs
        for c in f["chefs"]:
            cx = pad + c["x"] * cell + cell // 2
            cy = top_h + pad + c["y"] * cell + cell // 2
            r = 22
            col = chef_colors[c["id"]]
            dr.ellipse([cx - r, cy - r, cx + r, cy + r], fill=col, outline=(255, 255, 255), width=2)
            dx, dy = dir_delta[c["facing"]]
            dr.line([cx, cy, cx + dx, cy + dy], fill=(255, 255, 255), width=3)
            label = f"{chef_tags[c['id']]}" + (f":{c['held']}" if c["held"] else "")
            dr.text((cx - 18, cy - 8), label, fill=(255, 255, 255), font=font_sm)

        imgs.append(im)

    imgs[0].save(
        out_gif, save_all=True, append_images=imgs[1:],
        duration=int(1000 / fps), loop=0, optimize=False,
    )


def write_html(data, out_html: Path):
    payload = json.dumps(data, ensure_ascii=False)
    html = f"""<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8"/>
<title>SharedKitchen Replay ({data['mode']})</title>
<style>
  body {{ font-family: system-ui, sans-serif; background: #14171c; color: #f1f3f5; margin: 20px; display: flex; flex-direction: column; align-items: center; }}
  .panel {{ background: #1e222a; border-radius: 10px; padding: 16px 20px; box-shadow: 0 4px 20px rgba(0,0,0,.4); max-width: 720px; width: 100%; }}
  .controls {{ display: flex; gap: 10px; align-items: center; margin: 12px 0; flex-wrap: wrap; }}
  button {{ background: #339af0; color: #fff; border: none; border-radius: 6px; padding: 6px 14px; cursor: pointer; font-weight: 600; }}
  button:hover {{ background: #228be6; }}
  input[type=range] {{ flex: 1; min-width: 220px; }}
  canvas {{ display: block; margin: 0 auto; border-radius: 6px; }}
  .stats {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; margin-bottom: 10px; font-size: 14px; }}
  .stat {{ background: #262b36; padding: 8px 10px; border-radius: 6px; text-align: center; }}
  .stat b {{ font-size: 18px; display: block; color: #74c0fc; }}
  .warn b {{ color: #ff6b6b; }}
  .legend {{ font-size: 12px; color: #adb5bd; margin-top: 10px; line-height: 1.6; }}
</style>
</head>
<body>
<div class="panel">
  <h3 style="margin:0 0 10px 0;">SharedKitchen 回放 — {data['mode']} (300 步 / 得分 {int(data['metrics']['team_score'])})</h3>
  <div class="stats">
    <div class="stat">Tick<b id="sTick">0</b></div>
    <div class="stat">得分 / 出餐<b id="sScore">0 / 0</b></div>
    <div class="stat warn">联合等待步数<b id="sWait">0</b></div>
    <div class="stat">动作 (S/M/D)<b id="sAct">-</b></div>
  </div>
  <canvas id="cv" width="588" height="588"></canvas>
  <div class="controls">
    <button id="btnPlay">播放 / 暂停</button>
    <button id="btnPrev">← 上一步</button>
    <button id="btnNext">下一步 →</button>
    <input id="slider" type="range" min="0" max="{len(data['frames'])-1}" value="0"/>
    <label>FPS <input id="fps" type="number" value="6" min="1" max="30" style="width:46px;"/></label>
  </div>
  <div class="legend">
    <b>工位</b>: M=生主菜, G=锅(4步, [J]为双人联合单), L=生配菜, K=案板(切2步), D=盘子, A=拼盘台, B=缓冲台, S=出餐口<br/>
    <b>厨师</b>: <span style="color:#4dabf7">蓝=S(Server)</span> | <span style="color:#ffa94d">橙=M(Main)</span> | <span style="color:#69db7c">绿=D(Side)</span> &nbsp; (键盘 ←/→ 单步，空格 播放/暂停)
  </div>
</div>
<script>
const DATA = {payload};
const cv = document.getElementById('cv'), ctx = cv.getContext('2d');
const slider = document.getElementById('slider'), fpsIn = document.getElementById('fps');
const TILE = {{X:'#343a40',' ':'#f1f3f5',M:'#e76f51',L:'#6a994e',G:'#d62828',K:'#e9c46a',S:'#2a9d8f',A:'#adb5bd',B:'#8d99ae',D:'#4895ef'}};
const CHEF_COL = ['#2176ff', '#f77f00', '#38b000'];
const CHEF_TAG = ['S', 'M', 'D'];
const DIR = {{N:[0,-26],S:[0,26],E:[26,0],W:[-26,0]}};
let idx = 0, playing = false, timer = null;

function draw() {{
  const f = DATA.frames[idx], cell = 84;
  ctx.clearRect(0, 0, cv.width, cv.height);
  const ck = f.cooker;
  const waitingJoint = (ck.item === 'CM' && ck.rem === 0 && ck.kind === 'joint');
  for (let y = 0; y < DATA.height; y++) {{
    for (let x = 0; x < DATA.width; x++) {{
      const ch = DATA.layout[y][x];
      const x0 = x * cell, y0 = y * cell;
      ctx.fillStyle = (ch === 'G' && waitingJoint) ? '#ff2a2a' : (TILE[ch] || '#888');
      ctx.fillRect(x0 + 1, y0 + 1, cell - 2, cell - 2);
      if (ch !== 'X' && ch !== ' ') {{
        ctx.fillStyle = (ch === 'K' || ch === 'A') ? '#111' : '#fff';
        ctx.font = 'bold 13px sans-serif';
        ctx.fillText(ch, x0 + 6, y0 + 16);
      }}
      ctx.font = 'bold 12px monospace';
      if (ch === 'G' && ck.item) {{
        ctx.fillStyle = '#fff';
        const t = (ck.rem > 0 ? `${{ck.item}}(${{ck.rem}})` : `${{ck.item}}!`) + (ck.kind === 'joint' ? ' [J]' : '');
        ctx.fillText(t, x0 + 6, y0 + 44);
        if (waitingJoint) ctx.fillText('WAIT!', x0 + 6, y0 + 62);
      }} else if (ch === 'K' && f.cutting.item) {{
        ctx.fillStyle = '#111';
        ctx.fillText(`${{f.cutting.item}}(${{f.cutting.prog}}/2)`, x0 + 6, y0 + 44);
      }} else if (ch === 'A') {{
        ctx.fillStyle = '#111';
        ctx.fillText(`M:${{f.asm.main||'-'}}`, x0 + 6, y0 + 34);
        ctx.fillText(`S:${{f.asm.side||'-'}}`, x0 + 6, y0 + 50);
        ctx.fillText(`P:${{f.asm.plate||'-'}}`, x0 + 6, y0 + 66);
      }} else if (ch === 'B' && f.buf) {{
        ctx.fillStyle = '#fff';
        ctx.fillText(f.buf, x0 + 6, y0 + 44);
      }}
    }}
  }}
  for (const c of f.chefs) {{
    const cx = c.x * cell + cell / 2, cy = c.y * cell + cell / 2;
    ctx.beginPath(); ctx.arc(cx, cy, 24, 0, Math.PI * 2);
    ctx.fillStyle = CHEF_COL[c.id]; ctx.fill();
    ctx.lineWidth = 2.5; ctx.strokeStyle = '#fff'; ctx.stroke();
    const [dx, dy] = DIR[c.facing];
    ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(cx + dx, cy + dy);
    ctx.lineWidth = 3.5; ctx.strokeStyle = '#fff'; ctx.stroke();
    ctx.fillStyle = '#fff'; ctx.font = 'bold 13px monospace'; ctx.textAlign = 'center';
    ctx.fillText(CHEF_TAG[c.id] + (c.held ? ':' + c.held : ''), cx, cy + 5);
    ctx.textAlign = 'left';
  }}
  document.getElementById('sTick').textContent = `${{f.tick}} / ${{DATA.horizon}}`;
  document.getElementById('sScore').textContent = `${{f.score}} / ${{f.normal_completed + f.joint_completed}}`;
  document.getElementById('sWait').textContent = f.joint_wait_steps;
  document.getElementById('sAct').textContent = f.actions.join(' / ');
  slider.value = idx;
}}
function toggle() {{
  playing = !playing;
  if (playing) {{
    timer = setInterval(() => {{ idx = (idx + 1) % DATA.frames.length; draw(); }}, 1000 / Math.max(1, +fpsIn.value || 6));
  }} else {{ clearInterval(timer); }}
}}
document.getElementById('btnPlay').onclick = toggle;
document.getElementById('btnPrev').onclick = () => {{ if (idx > 0) {{ idx--; draw(); }} }};
document.getElementById('btnNext').onclick = () => {{ if (idx < DATA.frames.length - 1) {{ idx++; draw(); }} }};
slider.oninput = () => {{ idx = +slider.value; draw(); }};
window.addEventListener('keydown', e => {{
  if (e.code === 'Space') {{ e.preventDefault(); toggle(); }}
  else if (e.code === 'ArrowRight' && idx < DATA.frames.length - 1) {{ idx++; draw(); }}
  else if (e.code === 'ArrowLeft' && idx > 0) {{ idx--; draw(); }}
}});
draw();
</script>
</body>
</html>"""
    out_html.write_text(html, encoding="utf-8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("script", "fast_script", "all_policy"), default="script")
    p.add_argument("--ckpt", type=Path, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "replays")
    p.add_argument("--name", default="bc_rule_19meals")
    p.add_argument("--fps", type=int, default=8)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    data = collect_episode(mode=args.mode, ckpt=args.ckpt, seed=args.seed)
    json_path = args.out_dir / f"{args.name}.json"
    gif_path = args.out_dir / f"{args.name}.gif"
    html_path = args.out_dir / f"{args.name}.html"
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    render_gif(data, gif_path, fps=args.fps)
    write_html(data, html_path)
    print(f"Exported: {html_path} | {gif_path} | {json_path}")
    print("Metrics:", data["metrics"])


if __name__ == "__main__":
    main()
