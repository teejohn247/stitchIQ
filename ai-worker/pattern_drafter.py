"""
pattern_drafter.py — Standard Metric Pattern Block Drafting
Based on Winnifred Aldrich "Metric Pattern Cutting" method.

All coordinates are in centimetres. Rendered to anti-aliased PNG via
matplotlib and returned as a base64 data-URL.
"""

import io
import base64
import math
import numpy as np
import logging

logger = logging.getLogger("stitchiq-worker")

# ── Default measurements: UK/EU size 12 ─────────────────────────────
DEFAULTS = {
    "bust":          88,
    "waist":         68,
    "hip":           92,
    "back_length":   40,   # nape → waist (cm)
    "front_length":  42,
    "skirt_length":  60,
    "sleeve_length": 58,
    "shoulder":      12,   # shoulder seam length
    "upper_arm":     30,
    "neck":          36,   # neck circumference
    "ease_bust":      8,
    "ease_waist":     4,
    "ease_hip":       6,
    "sa":             1.5, # seam allowance
}

# ── Colour palette ───────────────────────────────────────────────────
BG   = "#0a1a0f"
LINE = "#ffffff"
DIM  = "#999999"
GOLD = "#D4A843"
GRID = "#ffffff"


# ── Geometry helpers ─────────────────────────────────────────────────

def _c3(p0, p1, p2, p3, n=60):
    """Points along a cubic Bezier."""
    t  = np.linspace(0, 1, n)
    m  = 1 - t
    x  = m**3*p0[0] + 3*m**2*t*p1[0] + 3*m*t**2*p2[0] + t**3*p3[0]
    y  = m**3*p0[1] + 3*m**2*t*p1[1] + 3*m*t**2*p2[1] + t**3*p3[1]
    return list(zip(x, y))

def _c2(p0, p1, p2, n=40):
    """Points along a quadratic Bezier."""
    t  = np.linspace(0, 1, n)
    m  = 1 - t
    x  = m**2*p0[0] + 2*m*t*p1[0] + t**2*p2[0]
    y  = m**2*p0[1] + 2*m*t*p1[1] + t**2*p2[1]
    return list(zip(x, y))

def _line(p0, p1, n=2):
    """Straight segment as point list."""
    xs = np.linspace(p0[0], p1[0], n)
    ys = np.linspace(p0[1], p1[1], n)
    return list(zip(xs, ys))

def _inset(pts, ratio=0.93):
    """
    Scale all points toward the centroid by `ratio`.
    Used to draw the seam-allowance dashed line.
    """
    if not pts:
        return pts
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    return [(cx + (p[0]-cx)*ratio, cy + (p[1]-cy)*ratio) for p in pts]


# ── matplotlib drawing helpers ───────────────────────────────────────

def _setup(w_cm, h_cm, margin=2.5, dpi=96):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig_w = (w_cm + 2*margin) / 2.54
    fig_h = (h_cm + 2*margin) / 2.54
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(-margin, w_cm + margin)
    ax.set_ylim(-margin, h_cm + margin)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_position([0, 0, 1, 1])

    # Subtle 5cm grid
    for x in np.arange(int(-margin), w_cm + margin + 1, 5):
        ax.axvline(x, color=GRID, alpha=0.05, lw=0.35, zorder=0)
    for y in np.arange(int(-margin), h_cm + margin + 1, 5):
        ax.axhline(y, color=GRID, alpha=0.05, lw=0.35, zorder=0)

    return fig, ax


def _draw_piece(ax, pts, sa_ratio=0.93):
    """Draw cut line + seam-allowance dash."""
    if not pts:
        return
    xs = [p[0] for p in pts] + [pts[0][0]]
    ys = [p[1] for p in pts] + [pts[0][1]]
    ax.fill(xs[:-1], ys[:-1], color="#ffffff", alpha=0.03, zorder=1)
    ax.plot(xs, ys, color=LINE, lw=1.6, solid_capstyle='round',
            solid_joinstyle='round', zorder=3)

    inner = _inset(pts, sa_ratio)
    ix = [p[0] for p in inner] + [inner[0][0]]
    iy = [p[1] for p in inner] + [inner[0][1]]
    ax.plot(ix, iy, color=LINE, lw=0.65, ls=(0, (4, 3)),
            alpha=0.45, zorder=2)


def _grain(ax, x, y1, y2, bias=False):
    """Double-headed grain-line arrow."""
    kw = dict(arrowstyle='<->', color=LINE, lw=0.9,
              mutation_scale=7, shrinkA=0, shrinkB=0)
    if bias:
        ax.annotate('', xy=(x+2.5, y2), xytext=(x-2.5, y1), arrowprops=kw)
        ax.text(x+3.2, (y1+y2)/2, 'BIAS', color=DIM, fontsize=5.5,
                fontfamily='monospace', va='center', alpha=0.7)
    else:
        ax.annotate('', xy=(x, y2), xytext=(x, y1), arrowprops=kw)


def _notch(ax, x, y, angle=90):
    """Small filled triangle notch."""
    a   = math.radians(angle)
    sz  = 0.5
    dx, dy  = math.cos(a)*sz, math.sin(a)*sz
    nx, ny  = -math.sin(a)*sz*0.6, math.cos(a)*sz*0.6
    tri = plt.Polygon([(x+nx, y+ny), (x-nx, y-ny), (x+dx, y+dy)],
                      closed=True, fc=LINE, ec='none', zorder=5)
    ax.add_patch(tri)


def _dart(ax, apex, l1, l2):
    """V-shaped dart lines."""
    ax.plot([l1[0], apex[0], l2[0]], [l1[1], apex[1], l2[1]],
            color=LINE, lw=0.85, zorder=4)


def _dim_h(ax, x1, x2, y, text):
    kw = dict(arrowstyle='<->', color=DIM, lw=0.5,
              mutation_scale=5, shrinkA=0, shrinkB=0, alpha=0.6)
    ax.annotate('', xy=(x2, y), xytext=(x1, y), arrowprops=kw)
    ax.text((x1+x2)/2, y - 0.6, text, color=DIM, fontsize=5.5,
            ha='center', fontfamily='monospace', alpha=0.7)


def _dim_v(ax, x, y1, y2, text):
    kw = dict(arrowstyle='<->', color=DIM, lw=0.5,
              mutation_scale=5, shrinkA=0, shrinkB=0, alpha=0.6)
    ax.annotate('', xy=(x, y2), xytext=(x, y1), arrowprops=kw)
    ax.text(x + 0.7, (y1+y2)/2, text, color=DIM, fontsize=5.5,
            ha='left', va='center', fontfamily='monospace', alpha=0.7,
            rotation=90)


def _label_piece(ax, cx, cy, label, cut_text):
    d = label if len(label) <= 22 else label[:20] + '…'
    ax.text(cx, cy + 1, d, color=LINE, fontsize=8.5, fontweight='bold',
            ha='center', va='center', fontfamily='monospace', zorder=6)
    ax.text(cx, cy - 1, cut_text, color=GOLD, fontsize=6.5,
            ha='center', va='center', fontfamily='monospace',
            alpha=0.9, zorder=6)


def _finish(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=96,
                facecolor=BG, edgecolor='none', bbox_inches=None)
    import matplotlib.pyplot as plt
    plt.close(fig)
    buf.seek(0)
    return 'data:image/png;base64,' + base64.b64encode(buf.read()).decode()


# ── Pattern piece builders ───────────────────────────────────────────

def _piece_front_bodice(label, m):
    """
    Fitted front bodice block.
    Origin: bottom-left = (waist, CF). Y increases toward shoulder.
    """
    B  = m['bust']  + m['ease_bust']
    W  = m['waist'] + m['ease_waist']
    h  = m['front_length']
    sh = m['shoulder']

    # Key measurements
    w         = B / 4
    arm_depth = B / 10 + 9       # armhole depth from top shoulder line
    neck_w    = B / 16 + 1.5
    neck_d    = B / 10 + 1.5     # front neck drop from shoulder line
    sh_slope  = 2.0
    dart_x    = B / 10 + 1       # bust dart position from CF
    dart_dep  = arm_depth - 2.5  # dart apex height from waist
    dart_wid  = (B - W) / 6 * 0.7

    # Shoulder end point
    sh_end = (neck_w + sh * math.cos(math.radians(12)),
              h - sh_slope)
    underarm = (w, h - arm_depth + 2)

    # Build outline (counter-clockwise, y-up)
    cf   = _line((0, 0), (0, h - neck_d))
    neck = _c2((0, h - neck_d), (neck_w * 0.3, h + 0.4), (neck_w, h))
    shld = _line((neck_w, h), sh_end)
    arm  = _c3(sh_end,
               (sh_end[0] + 1.2, sh_end[1] - 3.5),
               (w + 0.6,         underarm[1] + 4),
               underarm)
    side = _c3(underarm,
               (w + 0.4, underarm[1] - 3),
               (w - 0.2, 3),
               (w, 0))
    hem  = _line((w, 0), (0, 0))

    outline = (cf + neck[1:] + shld[1:] + arm[1:] + side[1:] + hem[1:])

    canvas_w = w + 3.5
    canvas_h = h + 2.5
    fig, ax = _setup(canvas_w, canvas_h)

    _draw_piece(ax, outline)
    _dart(ax, (dart_x, dart_dep), (dart_x - dart_wid/2, 0), (dart_x + dart_wid/2, 0))
    _grain(ax, w * 0.35, 4, h - 7)
    _notch(ax, underarm[0], underarm[1], angle=0)
    _notch(ax, w * 0.5, h * 0.95, angle=90)  # shoulder balance mark

    _dim_h(ax, 0, w, -1.5, f'{w:.0f} cm')
    _dim_v(ax, w + 1.8, 0, h, f'{h:.0f} cm')
    ax.text(0.3, h * 0.55, 'CF', color=DIM, fontsize=5.5,
            fontfamily='monospace', alpha=0.5, rotation=90)

    _label_piece(ax, w * 0.5, h * 0.35, label, 'CUT 1 PAIR')
    return _finish(fig)


def _piece_back_bodice(label, m):
    """Fitted back bodice block."""
    B  = m['bust']  + m['ease_bust']
    W  = m['waist'] + m['ease_waist']
    h  = m['back_length']
    sh = m['shoulder']

    w         = B / 4 - 1
    arm_depth = B / 10 + 9
    neck_w    = B / 16 + 2
    neck_d    = 2.0
    sh_slope  = 4.0
    dart_x    = w * 0.45
    dart_wid  = (B - W) / 8
    dart_dep  = h * 0.55

    sh_end   = (neck_w + sh * math.cos(math.radians(14)),
                h - sh_slope)
    underarm = (w, h - arm_depth + 2)

    cf   = _line((0, 0), (0, h - neck_d))
    neck = _c2((0, h - neck_d), (neck_w * 0.5, h + 0.2), (neck_w, h))
    shld = _line((neck_w, h), sh_end)
    arm  = _c3(sh_end,
               (sh_end[0] + 1.5, sh_end[1] - 4),
               (w + 0.6,         underarm[1] + 5),
               underarm)
    side = _c3(underarm,
               (w + 0.4, underarm[1] - 4),
               (w - 0.2, 3),
               (w, 0))
    hem  = _line((w, 0), (0, 0))
    outline = cf + neck[1:] + shld[1:] + arm[1:] + side[1:] + hem[1:]

    fig, ax = _setup(w + 3.5, h + 2.5)
    _draw_piece(ax, outline)
    _dart(ax, (dart_x, dart_dep),
          (dart_x - dart_wid/2, 0), (dart_x + dart_wid/2, 0))
    _grain(ax, w * 0.35, 4, h - 7)
    _notch(ax, underarm[0], underarm[1], angle=0)

    _dim_h(ax, 0, w, -1.5, f'{w:.0f} cm')
    _dim_v(ax, w + 1.8, 0, h, f'{h:.0f} cm')
    ax.text(0.3, h * 0.5, 'CB', color=DIM, fontsize=5.5,
            fontfamily='monospace', alpha=0.5, rotation=90)

    _label_piece(ax, w * 0.5, h * 0.35, label, 'CUT 1 PAIR')
    return _finish(fig)


def _piece_skirt(label, m, is_front=True, flare=1.0):
    """
    Skirt panel — front or back.
    Y-up: y=0 at hem, y=L at waist.
    """
    H  = m['hip']   + m['ease_hip']
    W  = m['waist'] + m['ease_waist']
    L  = m['skirt_length']

    waist_w  = W / 4
    hip_w    = H / 4
    hem_w    = hip_w * (1 + flare * 0.22)
    hip_lvl  = L - 20           # hip line is 20cm below waist
    dart_w   = (hip_w - waist_w) * 0.55
    dart_x   = waist_w * 0.48
    dart_dep = 10 if is_front else 14

    # Outline: waist at top (y=L), hem at bottom (y=0)
    cf   = _line((0, L), (0, 0))
    hem  = _line((0, 0), (hem_w, 0))
    side = _c3((hem_w, 0),
               (hip_w + 0.4, hip_lvl * 0.35),
               (hip_w,       hip_lvl * 0.8),
               (waist_w,     L))
    top  = _line((waist_w, L), (0, L))
    outline = cf + hem[1:] + list(reversed(side[:-1])) + top[1:]

    fig, ax = _setup(hem_w + 3.5, L + 2.5)

    _draw_piece(ax, outline)
    _dart(ax, (dart_x, L - dart_dep),
          (dart_x - dart_w/2, L), (dart_x + dart_w/2, L))
    _grain(ax, hem_w * 0.45, 4, L - 5)

    # Hip line
    ax.axhline(hip_lvl, color=DIM, lw=0.5, ls='--', alpha=0.35)
    ax.text(0.3, hip_lvl + 0.7, 'HIP LINE', color=DIM, fontsize=5,
            fontfamily='monospace', alpha=0.5)

    _notch(ax, hip_w, hip_lvl, angle=0)
    _dim_h(ax, 0, waist_w, L + 1.2, f'W {waist_w:.0f}cm')
    _dim_h(ax, 0, hem_w,  -1.5,     f'H {hem_w:.0f}cm')
    _dim_v(ax, hem_w + 1.8, 0, L,   f'{L:.0f} cm')

    ax.text(0.3, L * 0.5, 'CF' if is_front else 'CB', color=DIM,
            fontsize=5.5, fontfamily='monospace', alpha=0.5, rotation=90)

    _label_piece(ax, hem_w * 0.45, L * 0.5, label, 'CUT 1 PAIR')
    return _finish(fig)


def _piece_sleeve(label, m):
    """
    One-piece sleeve block.
    Y-up: y=0 at cuff hem, y increases toward cap.
    """
    UA = m['upper_arm'] + 6
    L  = m['sleeve_length']

    w      = UA / 2
    cap    = UA / 3          # sleeve cap height
    cuff_w = w * 0.68

    # Cap (top of sleeve): bell curve centred at x=w/2, peak at y=L+cap
    cap_l = _c3((0, L), (0, L + cap * 0.5), (w*0.2, L + cap * 1.05), (w/2, L + cap))
    cap_r = _c3((w/2, L + cap), (w*0.8, L + cap * 1.05), (w, L + cap * 0.5), (w, L))

    # Underseams tapered to cuff
    seam_l = _c3((0, L), (w*0.1, L*0.6), (w/2 - cuff_w/2, L*0.2), (w/2 - cuff_w/2, 0))
    seam_r = _c3((w, L), (w*0.9, L*0.6), (w/2 + cuff_w/2, L*0.2), (w/2 + cuff_w/2, 0))
    cuff   = _line((w/2 - cuff_w/2, 0), (w/2 + cuff_w/2, 0))

    outline = (list(cap_l) + list(cap_r[1:]) +
               list(seam_r[1:]) + list(reversed(cuff[1:])) +
               list(reversed(seam_l[:-1])))

    canvas_h = L + cap + 2.5
    fig, ax  = _setup(w + 3.5, canvas_h)

    _draw_piece(ax, outline)
    _grain(ax, w/2, 4, L - 4)
    _notch(ax, w * 0.25, L, angle=90)
    _notch(ax, w * 0.75, L, angle=90)
    _notch(ax, w / 2, L + cap, angle=270)  # cap notch

    _dim_h(ax, 0, w, -1.5, f'{w*2:.0f} cm')
    _dim_v(ax, w + 1.8, 0, L, f'{L:.0f} cm')
    ax.text(w/2, L + cap * 0.45, '↑ CAP', color=DIM, fontsize=5.5,
            ha='center', fontfamily='monospace', alpha=0.5)

    _label_piece(ax, w/2, L * 0.45, label, 'CUT 1 PAIR')
    return _finish(fig)


def _piece_collar(label, m):
    """Flat collar — crescent shape."""
    nk   = m['neck'] / 2 + 1   # half neck circumference
    h    = 8.0                  # collar fall + stand depth

    outer = _c3((0, h/2), (nk*0.25, -h*0.3), (nk*0.75, -h*0.3), (nk, h/2))
    inner = _c3((1.0, h/2), (nk*0.25, h*0.1), (nk*0.75, h*0.1), (nk-1, h/2))

    outline = (list(outer) +
               _line((nk, h/2), (nk-1, h/2))[1:] +
               list(reversed(inner))[1:] +
               _line((1, h/2), (0, h/2))[1:])

    fig, ax = _setup(nk + 3, h + 3)
    _draw_piece(ax, outline)
    _grain(ax, nk/2, -h*0.35, -h*0.05)
    _dim_h(ax, 0, nk, -h*0.6, f'{nk:.0f} cm')
    _dim_v(ax, nk + 1.8, -h*0.35, h*0.5, f'{h:.0f} cm')

    _label_piece(ax, nk/2, h*0.1, label, 'CUT 2')
    return _finish(fig)


def _piece_facing(label, m):
    """Front neck/armhole facing strip."""
    B  = m['bust'] + m['ease_bust']
    h  = m['front_length']
    w  = B / 4

    neck_w = B / 16 + 1.5
    neck_d = B / 10 + 1.5
    sh     = m['shoulder']
    depth  = 7.0   # facing depth

    cf    = _line((0, 0), (0, h - neck_d))
    neck  = _c2((0, h - neck_d), (neck_w*0.3, h + 0.4), (neck_w, h))
    sh_pt = (neck_w + sh * 0.4, h - 2)
    outer = cf + neck[1:] + [sh_pt]

    # Inner edge: scale toward piece interior
    inner = _inset(outer, 0.75)

    outline = outer + list(reversed(inner))

    # Bounding box for canvas
    all_x = [p[0] for p in outline]
    all_y = [p[1] for p in outline]
    bw = max(all_x) - min(all_x) + 1
    bh = max(all_y) - min(all_y) + 1

    fig, ax = _setup(bw + 4, bh + 2)
    _draw_piece(ax, outline)
    cx = (max(all_x) + min(all_x)) / 2
    cy = (max(all_y) + min(all_y)) / 2
    _grain(ax, cx, cy - depth*0.3, cy + depth*0.3)
    _dim_v(ax, max(all_x) + 2, min(all_y), max(all_y), f'{bh:.0f} cm')

    _label_piece(ax, cx, cy, label, 'CUT 1')
    return _finish(fig)


def _piece_band(label, m):
    """Waistband, cuff, strap, loop or tie."""
    W  = m['waist'] + m['ease_waist']
    bw = W / 2
    bh = 4.0   # standard waistband height

    outline = [(0, 0), (bw, 0), (bw, bh), (0, bh)]

    # Button extension
    ext = 2.5
    outline_ext = [(0, 0), (bw + ext, 0), (bw + ext, bh), (0, bh)]

    fig, ax = _setup(bw + ext + 3, bh + 3)
    _draw_piece(ax, outline_ext)

    # Fold line
    ax.axvline(bw, color=DIM, lw=0.7, ls='--', alpha=0.5)
    ax.text(bw + 0.2, bh * 0.55, 'FOLD', color=DIM, fontsize=5,
            fontfamily='monospace', alpha=0.5, rotation=90)

    # CF marking
    ax.axvline(0, color=DIM, lw=0.5, ls=':', alpha=0.35)

    _grain(ax, bw * 0.5, 1, 3)
    _dim_h(ax, 0, bw + ext, -1.2, f'{bw + ext:.0f} cm')
    _dim_v(ax, bw + ext + 1.8, 0, bh, f'{bh:.0f} cm')

    _label_piece(ax, (bw + ext) * 0.45, bh / 2, label, 'CUT 2 ON FOLD')
    return _finish(fig)


def _piece_lining(label, m):
    """Full lining — mirrors front bodice slightly smaller."""
    B  = m['bust'] + m['ease_bust'] * 0.6   # less ease for lining
    h  = m['front_length'] - 1
    w  = B / 4

    neck_w = B / 16 + 1.2
    neck_d = B / 10 + 1.3
    sh     = m['shoulder'] - 0.5
    sh_end = (neck_w + sh * math.cos(math.radians(12)), h - 2.5)
    under  = (w, h - (B/10 + 9) + 2)

    cf    = _line((0, 0), (0, h - neck_d))
    neck  = _c2((0, h - neck_d), (neck_w*0.3, h + 0.3), (neck_w, h))
    shld  = _line((neck_w, h), sh_end)
    arm   = _c3(sh_end, (sh_end[0]+1, sh_end[1]-3), (w+0.5, under[1]+4), under)
    side  = _c3(under, (w+0.3, under[1]-2), (w-0.2, 3), (w, 0))
    hem   = _line((w, 0), (0, 0))
    outline = cf + neck[1:] + shld[1:] + arm[1:] + side[1:] + hem[1:]

    fig, ax = _setup(w + 3.5, h + 2.5)
    _draw_piece(ax, outline)
    _grain(ax, w * 0.35, 4, h - 7)
    _dim_h(ax, 0, w, -1.5, f'{w:.0f} cm')
    _dim_v(ax, w + 1.8, 0, h, f'{h:.0f} cm')
    _label_piece(ax, w * 0.5, h * 0.38, label, 'CUT 1 PAIR')
    return _finish(fig)


def _piece_generic(label, m, is_bias=False):
    """Generic rectangular panel with slight taper."""
    H  = m['hip'] + m['ease_hip']
    L  = m['back_length'] + m['skirt_length'] * 0.45
    w  = H / 4

    top_w = w * 0.82
    outline = (
        _line((0, 0), (top_w, 0)) +
        _c3((top_w, 0), (w+0.4, L*0.3), (w-0.2, L*0.7), (w, L))[1:] +
        _line((w, L), (0, L))[1:] +
        _c3((0, L), (-0.3, L*0.7), (0.2, L*0.3), (0, 0))[1:]
    )

    fig, ax = _setup(w + 3.5, L + 2.5)
    _draw_piece(ax, outline)
    _grain(ax, w * 0.48, 5, L - 5, bias=is_bias)
    _dim_h(ax, 0, top_w, -1.5, f'{top_w:.0f} cm')
    _dim_v(ax, w + 1.8, 0, L, f'{L:.0f} cm')
    _label_piece(ax, w * 0.48, L * 0.5, label,
                 'CUT ON BIAS' if is_bias else 'CUT 1 PAIR')
    return _finish(fig)


# ── Public API ────────────────────────────────────────────────────────

def draft_piece(label: str, measurements: dict | None = None) -> str:
    """
    Generate a metric pattern block image for the given piece label.
    Returns a base64 PNG data-URL (data:image/png;base64,...).

    Falls back to a generic panel on any error.
    """
    m   = {**DEFAULTS, **(measurements or {})}
    lbl = label.upper()

    try:
        # Bodice
        if any(k in lbl for k in ("FRONT BODICE", "BODICE FRONT")):
            return _piece_front_bodice(label, m)
        if any(k in lbl for k in ("BACK BODICE", "BODICE BACK")):
            return _piece_back_bodice(label, m)

        # Skirt panels
        if any(k in lbl for k in ("FRONT SKIRT", "SKIRT FRONT")):
            return _piece_skirt(label, m, is_front=True)
        if any(k in lbl for k in ("BACK SKIRT", "SKIRT BACK")):
            return _piece_skirt(label, m, is_front=False)

        # Mermaid / trumpet gets extra flare
        if "SKIRT" in lbl or "PANEL" in lbl:
            flare = 1.5 if any(k in lbl for k in ("MERMAID", "TRUMPET", "FLARE", "GORE")) else 1.0
            is_front = "BACK" not in lbl
            return _piece_skirt(label, m, is_front=is_front, flare=flare)

        # Sleeve
        if "SLEEVE" in lbl:
            return _piece_sleeve(label, m)

        # Collar / neckline
        if any(k in lbl for k in ("COLLAR", "NECKLINE", "NECK PIECE")):
            return _piece_collar(label, m)

        # Facing / underlining
        if any(k in lbl for k in ("FACING", "UNDERLINING")):
            return _piece_facing(label, m)

        # Lining
        if "LINING" in lbl:
            return _piece_lining(label, m)

        # Bands, straps, loops
        if any(k in lbl for k in ("BAND", "STRAP", "LOOP", "WAISTBAND", "CUFF", "TIE")):
            return _piece_band(label, m)

        # Bias detection for generic
        is_bias = any(k in lbl for k in ("BIAS", "DIAGONAL"))
        return _piece_generic(label, m, is_bias=is_bias)

    except Exception as e:
        logger.warning(f"PatternDrafter error for '{label}': {e}", exc_info=True)
        try:
            return _piece_generic(label, m)
        except Exception:
            return ""
