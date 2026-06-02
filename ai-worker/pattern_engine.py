"""
StitchIQ Pattern Engine
=======================
Generates real technical sewing pattern pieces from body measurements
and Gemini garment analysis output.

Outputs:
  - SVG at 1:1 scale (printable, tiled A4)
  - PDF pattern sheet with annotations
  - JSON geometry (for frontend canvas rendering)

Usage:
  from pattern_engine import PatternEngine
  engine = PatternEngine(measurements)
  result = engine.generate(gemini_specs)
"""

import math
import json
import io
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import numpy as np
import svgwrite
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm, mm
from reportlab.pdfgen import canvas as rl_canvas
from reportlab.lib import colors

# ── Constants ────────────────────────────────────────────────────────────────
SA = 1.5          # seam allowance cm (standard)
SA_HEM = 4.0      # hem allowance cm
EASE_BUST = 4.0   # ease added to bust
EASE_WAIST = 2.0  # ease added to waist
EASE_HIP = 4.0    # ease added to hip

PX_PER_CM = 37.795275591  # SVG px per cm at 96dpi


# ── Data classes ──────────────────────────────────────────────────────────────
@dataclass
class Measurements:
    bust: float         # cm
    waist: float        # cm
    hip: float          # cm
    back_length: float  # nape to waist
    skirt_length: float # waist to hem
    shoulder: float     # shoulder width
    sleeve_length: float = 60.0
    neck_circ: float = 36.0
    size_label: str = "Custom"

    @classmethod
    def from_uk_size(cls, size: int) -> "Measurements":
        table = {
            6:  (80, 60, 85, 39, 60, 37),
            8:  (82, 62, 87, 39, 60, 37),
            10: (86, 66, 91, 40, 61, 38),
            12: (90, 70, 95, 40, 61, 38),
            14: (94, 74, 99, 41, 62, 39),
            16: (98, 78, 103, 41, 62, 39),
            18: (102, 82, 107, 42, 63, 40),
            20: (106, 86, 111, 42, 63, 40),
        }
        b, w, h, bl, sl, sh = table.get(size, table[12])
        return cls(bust=b, waist=w, hip=h, back_length=bl,
                   skirt_length=sl, shoulder=sh, size_label=f"UK {size}")


@dataclass
class Point:
    x: float
    y: float

    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)

    def offset(self, dx=0, dy=0):
        return Point(self.x + dx, self.y + dy)

    def as_tuple(self):
        return (self.x, self.y)


@dataclass
class PatternPiece:
    id: str
    label: str
    note: str            # cut instruction
    seam_note: str       # seam allowance note
    points: List[Point]  # outer cut line (with SA)
    inner_points: List[Point]  # stitch line (without SA)
    grain_line: Tuple[Point, Point] = None
    notches: List[Point] = field(default_factory=list)
    fold_line: Optional[Tuple[Point, Point]] = None
    is_bias: bool = False
    dimensions: List[dict] = field(default_factory=list)  # annotation arrows


# ── Geometry helpers ──────────────────────────────────────────────────────────
def offset_polygon(points: List[Point], amount: float) -> List[Point]:
    """Inset a polygon by `amount` cm using normal-based offsetting."""
    n = len(points)
    result = []
    for i in range(n):
        p0 = points[(i - 1) % n]
        p1 = points[i]
        p2 = points[(i + 1) % n]

        d1 = np.array([p1.x - p0.x, p1.y - p0.y])
        d2 = np.array([p2.x - p1.x, p2.y - p1.y])

        n1 = np.array([-d1[1], d1[0]])
        n2 = np.array([-d2[1], d2[0]])

        ln1 = np.linalg.norm(n1)
        ln2 = np.linalg.norm(n2)
        if ln1 < 1e-9 or ln2 < 1e-9:
            result.append(p1)
            continue

        n1 = n1 / ln1
        n2 = n2 / ln2
        bisector = n1 + n2
        lb = np.linalg.norm(bisector)
        if lb < 1e-9:
            result.append(p1)
            continue

        bisector = bisector / lb
        dot = np.dot(n1, bisector)
        if abs(dot) < 1e-9:
            result.append(p1)
            continue

        offset = bisector * (amount / dot)
        result.append(Point(p1.x + offset[0], p1.y + offset[1]))
    return result


def curve_points(p1: Point, p2: Point, ctrl: Point, steps=12) -> List[Point]:
    """Quadratic bezier curve between p1 and p2 through ctrl."""
    pts = []
    for i in range(steps + 1):
        t = i / steps
        x = (1-t)**2 * p1.x + 2*(1-t)*t * ctrl.x + t**2 * p2.x
        y = (1-t)**2 * p1.y + 2*(1-t)*t * ctrl.y + t**2 * p2.y
        pts.append(Point(x, y))
    return pts


# ── Pattern generators ────────────────────────────────────────────────────────
class BodiceDrafter:
    """Drafts a basic fitted bodice block (front and back)."""

    def __init__(self, m: Measurements):
        self.m = m
        self.hb = (m.bust + EASE_BUST) / 4
        self.hw = (m.waist + EASE_WAIST) / 4
        self.bl = m.back_length

    def front(self) -> PatternPiece:
        m = self.m
        hb, hw, bl = self.hb, self.hw, self.bl

        neck_w = m.neck_circ / 5 + 0.3
        neck_d = m.neck_circ / 10 + 0.5
        sh_slope = 1.5

        A = Point(0, 0)
        B = Point(neck_w, 0)
        C = Point(neck_w, neck_d)
        D = Point(0, sh_slope)
        E = Point(-hb, sh_slope + 3)
        F = Point(-hb, bl - 2)
        G = Point(0, bl)

        arm_ctrl = Point(-hb * 0.6, sh_slope + 1.2)

        inner = (
            [A, B]
            + curve_points(B, C, Point(neck_w + neck_d * 0.3, neck_d * 0.6), 8)
            + [G, F]
            + curve_points(F, E, Point(-hb - 1, bl - (bl - sh_slope) * 0.3), 8)
            + curve_points(E, D, arm_ctrl, 8)
            + [A]
        )

        outer = offset_polygon(inner, -SA)

        grain_x = -hb / 2
        grain = (Point(grain_x, sh_slope + 5), Point(grain_x, bl - 5))

        dims = [
            {"x1": 0, "y1": -SA - 0.5, "x2": -hb, "y2": -SA - 0.5,
             "label": f"{hb:.1f}cm (¼ bust)", "horizontal": True},
            {"x1": SA + 0.5, "y1": 0, "x2": SA + 0.5, "y2": bl,
             "label": f"{bl:.0f}cm", "horizontal": False},
        ]

        return PatternPiece(
            id="FB", label="FRONT BODICE", note="Cut 1 on fold",
            seam_note="1.5cm all edges; neckline 1cm",
            points=outer, inner_points=inner,
            grain_line=grain,
            notches=[Point(-hb, sh_slope + (bl - sh_slope) / 2)],
            fold_line=(Point(0, -SA), Point(0, bl + SA)),
            dimensions=dims
        )

    def back(self) -> PatternPiece:
        m = self.m
        hb, hw, bl = self.hb, self.hw, self.bl

        neck_w = m.neck_circ / 6
        neck_d = 2.0
        sh_slope = 1.0

        A = Point(0, 0)
        B = Point(neck_w, 0)
        C = Point(neck_w, neck_d)
        D = Point(0, sh_slope)
        E = Point(-hb, sh_slope + 2.5)
        F = Point(-hb, bl - 1.5)
        G = Point(0, bl)

        arm_ctrl = Point(-hb * 0.55, sh_slope + 1)

        inner = (
            [A, B]
            + curve_points(B, C, Point(neck_w + neck_d * 0.2, neck_d * 0.5), 8)
            + [G, F]
            + curve_points(F, E, Point(-hb - 0.8, bl - (bl - sh_slope) * 0.35), 8)
            + curve_points(E, D, arm_ctrl, 8)
            + [A]
        )

        outer = offset_polygon(inner, -SA)
        grain_x = -hb / 2
        grain = (Point(grain_x, sh_slope + 5), Point(grain_x, bl - 5))

        dims = [
            {"x1": 0, "y1": -SA - 0.5, "x2": -hb, "y2": -SA - 0.5,
             "label": f"{hb:.1f}cm (¼ back)", "horizontal": True},
            {"x1": SA + 0.5, "y1": 0, "x2": SA + 0.5, "y2": bl,
             "label": f"{bl:.0f}cm", "horizontal": False},
        ]

        return PatternPiece(
            id="BB", label="BACK BODICE", note="Cut 2 (mirror pair)",
            seam_note="1.5cm all edges; CB zip allowance 2cm",
            points=outer, inner_points=inner,
            grain_line=grain,
            notches=[Point(-hb, sh_slope + (bl - sh_slope) / 2),
                     Point(-hb * 0.5, sh_slope)],
            dimensions=dims
        )


class SkirtDrafter:
    """Drafts skirt panels (front + back)."""

    def __init__(self, m: Measurements, flare: float = 0.0):
        self.m = m
        self.flare = flare
        self.hw = (m.waist + EASE_WAIST) / 4
        self.hh = (m.hip + EASE_HIP) / 4
        self.sl = m.skirt_length
        self.hip_depth = 20.0

    def _panel(self, label, id_, note, fold) -> PatternPiece:
        hw, hh, sl = self.hw, self.hh, self.sl
        flare_add = self.flare * 8

        TL = Point(0, 0)
        TR = Point(-hw, 0)
        MR = Point(-hh - self.flare * 2, self.hip_depth)
        BR = Point(-hh - flare_add, sl)
        BL = Point(0, sl)

        inner = [TL, TR, MR, BR, BL, TL]
        outer = offset_polygon(inner, -SA)
        for i, p in enumerate(outer):
            if abs(p.y - (sl + SA)) < 1:
                outer[i] = Point(p.x, sl + SA_HEM)

        grain_x = -hw / 2
        grain = (Point(grain_x, 5), Point(grain_x, sl - 5))

        dims = [
            {"x1": 0, "y1": -SA - 0.5, "x2": -hw, "y2": -SA - 0.5,
             "label": f"{hw:.1f}cm (¼ waist)", "horizontal": True},
            {"x1": SA + 0.5, "y1": 0, "x2": SA + 0.5, "y2": sl,
             "label": f"{sl:.0f}cm", "horizontal": False},
            {"x1": 0, "y1": self.hip_depth, "x2": -hh, "y2": self.hip_depth,
             "label": f"{hh:.1f}cm (¼ hip)", "horizontal": True},
        ]

        result = PatternPiece(
            id=id_, label=label, note=note,
            seam_note=f"1.5cm sides; {SA_HEM}cm hem",
            points=outer, inner_points=inner,
            grain_line=grain,
            notches=[Point(-hw, 0), Point(-hh, self.hip_depth)],
            dimensions=dims
        )
        if fold:
            result.fold_line = (Point(0, -SA), Point(0, sl + SA_HEM))
        return result

    def front(self) -> PatternPiece:
        return self._panel("FRONT SKIRT", "FS", "Cut 1 on fold", fold=True)

    def back(self) -> PatternPiece:
        return self._panel("BACK SKIRT", "BS", "Cut 2 (mirror pair)", fold=False)


class SleeveDrafter:
    """Drafts a basic sleeve block."""

    def __init__(self, m: Measurements, cap_height: float = 14.0):
        self.m = m
        self.cap_height = cap_height
        self.sw = m.shoulder + 4
        self.sl = m.sleeve_length

    def sleeve(self) -> PatternPiece:
        sw, sl, ch = self.sw, self.sl, self.cap_height

        centre_top = Point(0, 0)
        left_base = Point(-sw / 2, ch)
        right_base = Point(sw / 2, ch)
        left_wrist = Point(-sw / 4, sl)
        right_wrist = Point(sw / 4, sl)

        cap_left = curve_points(left_base, centre_top,
                                Point(-sw / 4, ch * 0.3), 10)
        cap_right = curve_points(centre_top, right_base,
                                 Point(sw / 4, ch * 0.3), 10)

        inner = cap_left + cap_right + [right_wrist, left_wrist, left_base]
        outer = offset_polygon(inner, -SA)

        grain = (Point(0, ch + 5), Point(0, sl - 5))

        dims = [
            {"x1": -sw/2, "y1": ch + 0.5, "x2": sw/2, "y2": ch + 0.5,
             "label": f"{sw:.0f}cm (sleeve width)", "horizontal": True},
            {"x1": SA + 1, "y1": ch, "x2": SA + 1, "y2": sl,
             "label": f"{sl - ch:.0f}cm (sleeve length)", "horizontal": False},
        ]

        return PatternPiece(
            id="SL", label="SLEEVE", note="Cut 2",
            seam_note="1.5cm seam; 3cm hem",
            points=outer, inner_points=inner,
            grain_line=grain,
            notches=[Point(0, 0), Point(-sw / 2, ch), Point(sw / 2, ch)],
            dimensions=dims
        )


# ── Main engine ───────────────────────────────────────────────────────────────
class PatternEngine:

    SILHOUETTE_MAP = {
        "column":  ("bodice", "straight_skirt"),
        "mermaid": ("bodice", "mermaid_skirt"),
        "trumpet": ("bodice", "trumpet_skirt"),
        "a-line":  ("bodice", "aline_skirt"),
        "a line":  ("bodice", "aline_skirt"),
        "full":    ("bodice", "full_skirt"),
        "shift":   ("bodice", "straight_skirt"),
        "wrap":    ("bodice", "aline_skirt"),
        "pencil":  ("bodice", "straight_skirt"),
        "ball":    ("bodice", "full_skirt"),
        "empire":  ("bodice", "aline_skirt"),
        "fit":     ("bodice", "mermaid_skirt"),
    }

    FLARE_MAP = {
        "straight_skirt": 0.0,
        "aline_skirt":    1.0,
        "trumpet_skirt":  1.5,
        "mermaid_skirt":  0.5,
        "full_skirt":     2.5,
    }

    def __init__(self, measurements: Measurements):
        self.m = measurements

    def generate(self, gemini_specs: dict) -> dict:
        silhouette  = gemini_specs.get("silhouette", "a-line").lower()
        has_sleeves = "sleeveless" not in gemini_specs.get("sleeves", "sleeveless").lower()

        _, skirt_type = self._resolve_silhouette(silhouette)
        flare = self.FLARE_MAP.get(skirt_type, 1.0)

        pieces = []
        bodice = BodiceDrafter(self.m)
        pieces.append(bodice.front())
        pieces.append(bodice.back())

        skirt = SkirtDrafter(self.m, flare=flare)
        pieces.append(skirt.front())
        pieces.append(skirt.back())

        if has_sleeves:
            sleeve = SleeveDrafter(self.m)
            pieces.append(sleeve.sleeve())

        svg_str   = self._render_svg(pieces, gemini_specs)
        pdf_bytes = self._render_pdf(pieces, gemini_specs)
        json_geo  = self._to_json(pieces, gemini_specs)

        return {
            "svg":          svg_str,
            "pdf_bytes":    pdf_bytes,
            "json":         json_geo,
            "piece_count":  len(pieces),
            "garment_label": gemini_specs.get("style_name", "Custom garment"),
        }

    def _resolve_silhouette(self, silhouette: str):
        for key, val in self.SILHOUETTE_MAP.items():
            if key in silhouette:
                return val
        return ("bodice", "aline_skirt")

    # ── SVG renderer ─────────────────────────────────────────────────────────
    def _render_svg(self, pieces: List[PatternPiece], specs: dict) -> str:
        MARGIN  = 3.0
        PIECE_W = 25.0
        PIECE_H = 70.0
        cols    = 3

        page_w = cols * (PIECE_W + MARGIN) + MARGIN
        page_h = math.ceil(len(pieces) / cols) * (PIECE_H + MARGIN * 3) + MARGIN * 4

        W = page_w * PX_PER_CM
        H = page_h * PX_PER_CM
        scale = PX_PER_CM

        dwg = svgwrite.Drawing(size=(f"{W:.0f}px", f"{H:.0f}px"),
                               viewBox=f"0 0 {W:.0f} {H:.0f}")

        dwg.add(dwg.rect(insert=(0, 0), size=("100%", "100%"), fill="white"))

        # Grid
        for x in range(int(page_w) + 1):
            xp  = x * scale
            clr = "#d0d0d0" if x % 5 == 0 else "#eeeeee"
            dwg.add(dwg.line((xp, 0), (xp, H), stroke=clr, stroke_width=0.6))
        for y in range(int(page_h) + 1):
            yp  = y * scale
            clr = "#d0d0d0" if y % 5 == 0 else "#eeeeee"
            dwg.add(dwg.line((0, yp), (W, yp), stroke=clr, stroke_width=0.6))

        # Ruler labels every 5cm
        for x in range(0, int(page_w) + 1, 5):
            dwg.add(dwg.text(f"{x}", insert=(x * scale + 2, 11),
                             font_size="8px", fill="#aaa",
                             font_family="monospace"))
        for y in range(0, int(page_h) + 1, 5):
            dwg.add(dwg.text(f"{y}", insert=(2, y * scale - 2),
                             font_size="8px", fill="#aaa",
                             font_family="monospace"))

        # Title block
        title = (f"{specs.get('style_name','Garment')}  ·  "
                 f"{self.m.size_label}  ·  "
                 f"Bust {self.m.bust}cm  Waist {self.m.waist}cm  Hip {self.m.hip}cm  ·  "
                 f"Scale 1:1  ·  SA 1.5cm INCLUDED")
        dwg.add(dwg.text(title,
                         insert=(MARGIN * scale, MARGIN * 0.65 * scale),
                         font_size="11px", fill="#333",
                         font_family="sans-serif", font_weight="600"))

        for idx, piece in enumerate(pieces):
            col = idx % cols
            row = idx // cols
            ox  = (MARGIN + col * (PIECE_W + MARGIN)) * scale
            oy  = (MARGIN * 2.5 + row * (PIECE_H + MARGIN * 3)) * scale
            self._draw_piece_svg(dwg, piece, ox, oy, scale)

        # Legend
        ly = H - 1.8 * scale
        items = [
            ("Cut line (outer)",   "#1D9E75", "solid"),
            ("Stitch line (inner)", "#378ADD", "dashed"),
            ("Fold line",          "#9F77DD", "fold"),
        ]
        lx = MARGIN * scale
        for lbl, clr, style in items:
            if style == "solid":
                dwg.add(dwg.line((lx, ly), (lx + 24, ly),
                                 stroke=clr, stroke_width=2))
            elif style == "dashed":
                dwg.add(dwg.line((lx, ly), (lx + 24, ly),
                                 stroke=clr, stroke_width=1, stroke_dasharray="4,3"))
            elif style == "fold":
                dwg.add(dwg.line((lx, ly), (lx + 24, ly),
                                 stroke=clr, stroke_width=1.2, stroke_dasharray="6,4"))
            dwg.add(dwg.text(lbl, insert=(lx + 28, ly + 4),
                             font_size="10px", fill="#555",
                             font_family="sans-serif"))
            lx += 140

        return dwg.tostring()

    def _draw_piece_svg(self, dwg, piece: PatternPiece, ox, oy, scale):
        xs = [p.x for p in piece.points]
        ys = [p.y for p in piece.points]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        cx = ox - min_x * scale + (22 * scale - (max_x - min_x) * scale) / 2
        cy = oy - min_y * scale + 2 * scale

        def to_px(pts):
            return [(cx + p.x * scale, cy + p.y * scale) for p in pts]

        g = dwg.g()

        # Stitch line (dashed blue)
        inner_px = to_px(piece.inner_points)
        g.add(dwg.polygon(inner_px,
                          fill="#1D9E75", fill_opacity=0.05,
                          stroke="#378ADD", stroke_width=0.8,
                          stroke_dasharray="5,3"))

        # Cut line (solid green)
        outer_px = to_px(piece.points)
        g.add(dwg.polygon(outer_px,
                          fill="none",
                          stroke="#1D9E75", stroke_width=2.2))

        # Fold line
        if piece.fold_line:
            p1, p2 = piece.fold_line
            fx1 = cx + p1.x * scale
            fy1 = cy + p1.y * scale
            fx2 = cx + p2.x * scale
            fy2 = cy + p2.y * scale
            g.add(dwg.line((fx1, fy1), (fx2, fy2),
                           stroke="#9F77DD", stroke_width=1.2,
                           stroke_dasharray="6,4"))
            mid_fy = (fy1 + fy2) / 2
            g.add(dwg.text("FOLD",
                           insert=(fx1 + 5, mid_fy),
                           font_size="8px", fill="#9F77DD",
                           font_family="sans-serif",
                           transform=f"rotate(-90,{fx1+5},{mid_fy})"))

        # Grain line
        if piece.grain_line:
            gp1, gp2 = piece.grain_line
            gx1, gy1 = cx + gp1.x * scale, cy + gp1.y * scale
            gx2, gy2 = cx + gp2.x * scale, cy + gp2.y * scale
            g.add(dwg.line((gx1, gy1), (gx2, gy2),
                           stroke="#1D9E75", stroke_width=1.4))
            g.add(dwg.polygon([(gx1, gy1), (gx1-4, gy1+10), (gx1+4, gy1+10)],
                              fill="#1D9E75"))
            g.add(dwg.polygon([(gx2, gy2), (gx2-4, gy2-10), (gx2+4, gy2-10)],
                              fill="#1D9E75"))

        # Notches (orange circles)
        for n in piece.notches:
            nx = cx + n.x * scale
            ny = cy + n.y * scale
            g.add(dwg.circle((nx, ny), 3.5, fill="#D85A30", stroke="none"))

        # Dimension lines
        for dim in piece.dimensions:
            self._draw_dim_svg(g, dwg, dim, cx, cy, scale)

        # Labels below piece
        lx = cx + (min_x + max_x) / 2 * scale
        ly = cy + max_y * scale + 16
        g.add(dwg.text(piece.label,
                       insert=(lx, ly),
                       text_anchor="middle",
                       font_size="10.5px", font_weight="600",
                       fill="#111", font_family="sans-serif"))
        g.add(dwg.text(piece.note,
                       insert=(lx, ly + 14),
                       text_anchor="middle",
                       font_size="8.5px", fill="#666",
                       font_family="sans-serif"))
        g.add(dwg.text(f"SA: {piece.seam_note}",
                       insert=(lx, ly + 26),
                       text_anchor="middle",
                       font_size="8px", fill="#999",
                       font_family="sans-serif"))
        dwg.add(g)

    def _draw_dim_svg(self, g, dwg, dim, cx, cy, scale):
        x1 = cx + dim["x1"] * scale
        y1 = cy + dim["y1"] * scale
        x2 = cx + dim["x2"] * scale
        y2 = cy + dim["y2"] * scale
        clr = "#378ADD"

        g.add(dwg.line((x1, y1), (x2, y2), stroke=clr, stroke_width=0.8))
        if dim["horizontal"]:
            g.add(dwg.line((x1, y1-4), (x1, y1+4), stroke=clr, stroke_width=0.8))
            g.add(dwg.line((x2, y2-4), (x2, y2+4), stroke=clr, stroke_width=0.8))
            g.add(dwg.text(dim["label"],
                           insert=((x1+x2)/2, y1 - 6),
                           text_anchor="middle",
                           font_size="8px", fill=clr,
                           font_family="sans-serif"))
        else:
            g.add(dwg.line((x1-4, y1), (x1+4, y1), stroke=clr, stroke_width=0.8))
            g.add(dwg.line((x2-4, y2), (x2+4, y2), stroke=clr, stroke_width=0.8))
            g.add(dwg.text(dim["label"],
                           insert=(x1 - 6, (y1+y2)/2),
                           text_anchor="end",
                           font_size="8px", fill=clr,
                           font_family="sans-serif"))

    # ── PDF renderer ─────────────────────────────────────────────────────────
    def _render_pdf(self, pieces: List[PatternPiece], specs: dict) -> bytes:
        buf = io.BytesIO()
        c   = rl_canvas.Canvas(buf, pagesize=A4)
        AW, AH = A4

        MARGIN_PT = 1.5 * cm
        SCALE     = cm

        pieces_per_page = 2
        for page_start in range(0, len(pieces), pieces_per_page):
            c.setFont("Helvetica-Bold", 9)
            title = (f"{specs.get('style_name','Garment')}  ·  {self.m.size_label}  ·  "
                     f"Bust {self.m.bust}cm  Waist {self.m.waist}cm  ·  "
                     f"Scale 1:1  (print at 100%)")
            c.drawString(MARGIN_PT, AH - MARGIN_PT, title)

            page_pieces = pieces[page_start:page_start + pieces_per_page]
            col_w = (AW - MARGIN_PT * 3) / 2

            for ci, piece in enumerate(page_pieces):
                ox = MARGIN_PT + ci * (col_w + MARGIN_PT)
                oy = MARGIN_PT + 1 * cm

                xs = [p.x for p in piece.inner_points]
                ys = [p.y for p in piece.inner_points]
                min_x, min_y = min(xs), min(ys)

                def ptx(p): return ox + (p.x - min_x) * SCALE
                def pty(p): return AH - oy - (p.y - min_y) * SCALE

                # Stitch line
                c.setStrokeColor(colors.HexColor("#378ADD"))
                c.setLineWidth(0.5)
                c.setDash(4, 3)
                pts = piece.inner_points
                path = c.beginPath()
                path.moveTo(ptx(pts[0]), pty(pts[0]))
                for p in pts[1:]:
                    path.lineTo(ptx(p), pty(p))
                path.close()
                c.drawPath(path, stroke=1, fill=0)

                # Cut line
                c.setStrokeColor(colors.HexColor("#1D9E75"))
                c.setLineWidth(1.5)
                c.setDash()
                pts = piece.points
                path = c.beginPath()
                path.moveTo(ptx(pts[0]), pty(pts[0]))
                for p in pts[1:]:
                    path.lineTo(ptx(p), pty(p))
                path.close()
                c.drawPath(path, stroke=1, fill=0)

                # Grain line
                if piece.grain_line:
                    gp1, gp2 = piece.grain_line
                    c.setStrokeColor(colors.HexColor("#1D9E75"))
                    c.setLineWidth(1)
                    c.line(ptx(gp1), pty(gp1), ptx(gp2), pty(gp2))

                # Notches
                c.setFillColor(colors.HexColor("#D85A30"))
                for n in piece.notches:
                    c.circle(ptx(n), pty(n), 3, fill=1, stroke=0)

                # Labels
                mx = ox + (max(xs) - min_x) * SCALE / 2
                my = AH - oy - (max(ys) - min_y) * SCALE - 0.8 * cm
                c.setFillColor(colors.black)
                c.setFont("Helvetica-Bold", 9)
                c.drawCentredString(mx, my, piece.label)
                c.setFont("Helvetica", 8)
                c.drawCentredString(mx, my - 11, piece.note)
                c.setFont("Helvetica", 7)
                c.setFillColor(colors.HexColor("#777777"))
                c.drawCentredString(mx, my - 21, piece.seam_note)

            c.showPage()

        c.save()
        return buf.getvalue()

    # ── JSON geometry ────────────────────────────────────────────────────────
    def _to_json(self, pieces: List[PatternPiece], specs: dict) -> dict:
        def pts_to_list(pts):
            return [{"x": round(p.x, 2), "y": round(p.y, 2)} for p in pts]

        return {
            "garment":  specs.get("style_name", "Garment"),
            "size":     self.m.size_label,
            "measurements": {
                "bust": self.m.bust, "waist": self.m.waist,
                "hip":  self.m.hip,  "back_length": self.m.back_length,
                "skirt_length": self.m.skirt_length,
            },
            "pieces": [
                {
                    "id":         p.id,
                    "label":      p.label,
                    "note":       p.note,
                    "seam_note":  p.seam_note,
                    "cut_line":   pts_to_list(p.points),
                    "stitch_line": pts_to_list(p.inner_points),
                    "grain_line": [pts_to_list([p.grain_line[0], p.grain_line[1]])]
                                  if p.grain_line else [],
                    "notches":    pts_to_list(p.notches),
                    "fold_line":  pts_to_list([p.fold_line[0], p.fold_line[1]])
                                  if p.fold_line else [],
                    "dimensions": p.dimensions,
                    "is_bias":    p.is_bias,
                }
                for p in pieces
            ]
        }


# ── FastAPI endpoint helper ───────────────────────────────────────────────────
def run_pattern_engine(gemini_specs: dict, size: int = None,
                       custom_measurements: dict = None) -> dict:
    """
    Drop-in helper for main.py worker endpoint.

    Returns dict with svg, pdf_b64, json, piece_count, garment_label.
    """
    import base64

    if custom_measurements:
        m = Measurements(**custom_measurements)
    elif size:
        m = Measurements.from_uk_size(size)
    else:
        m = Measurements.from_uk_size(12)

    engine = PatternEngine(m)
    result = engine.generate(gemini_specs)
    result["pdf_b64"] = base64.b64encode(result.pop("pdf_bytes")).decode()
    return result


# ── CLI smoke test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, os

    test_specs = {
        "style_name": "Black Column Gown with Chiffon Panel",
        "silhouette": "Column / Mermaid",
        "sleeves":    "Sleeveless",
        "fabric":     "Matte black bodycon blend",
    }

    size = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    m    = Measurements.from_uk_size(size)
    print(f"Generating — {m.size_label}  Bust:{m.bust} Waist:{m.waist} Hip:{m.hip}")

    engine = PatternEngine(m)
    result = engine.generate(test_specs)

    out = os.path.join(os.path.dirname(__file__), "pattern_output")
    os.makedirs(out, exist_ok=True)

    with open(f"{out}/pattern_size{size}.svg", "w") as f:
        f.write(result["svg"])
    with open(f"{out}/pattern_size{size}.pdf", "wb") as f:
        f.write(result["pdf_bytes"])
    with open(f"{out}/pattern_size{size}.json", "w") as f:
        json.dump(result["json"], f, indent=2)

    print(f"  Pieces: {result['piece_count']}")
    print(f"  Output: {out}/")
