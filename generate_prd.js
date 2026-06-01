const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  Header, Footer, AlignmentType, HeadingLevel, BorderStyle, WidthType,
  ShadingType, VerticalAlign, SimpleField, LevelFormat, TabStopType,
  TabStopPosition
} = require('docx');
const fs = require('fs');

// ── Colour palette ──────────────────────────────────────────────
const BRAND  = "1A1A2E";
const ACCENT = "E94560";
const LIGHT  = "F5F5F5";
const MID    = "D0D0D0";
const WHITE  = "FFFFFF";
const TEXT   = "1A1A2E";
const MUTED  = "6B7280";
const TEAL   = "0F766E";
const PURPLE = "6D28D9";
const AMBER  = "B45309";
const GREEN  = "15803D";
const ORANGE = "C2410C";

// ── Border helpers ──────────────────────────────────────────────
const bdr   = (c = MID) => ({ style: BorderStyle.SINGLE, size: 1, color: c });
const bdrs  = (c = MID) => ({ top: bdr(c), bottom: bdr(c), left: bdr(c), right: bdr(c) });
const noBdr = { style: BorderStyle.NONE, size: 0, color: WHITE };
const noBdrs= { top: noBdr, bottom: noBdr, left: noBdr, right: noBdr };

// ── Cell helpers ────────────────────────────────────────────────
function cell(children, { fill=WHITE, bold=false, w=null, color=TEXT, vAlign=VerticalAlign.CENTER }={}) {
  return new TableCell({
    borders: bdrs(),
    width: w ? { size: w, type: WidthType.DXA } : undefined,
    shading: { fill, type: ShadingType.CLEAR },
    margins: { top: 80, bottom: 80, left: 140, right: 140 },
    verticalAlign: vAlign,
    children: Array.isArray(children) ? children : [
      new Paragraph({ children: [new TextRun({ text: String(children), bold, size: 20, font: "Arial", color })] })
    ]
  });
}
function hCell(text, w=null) {
  return cell(text, { fill: BRAND, bold: true, w, color: WHITE });
}

// ── Text helpers ────────────────────────────────────────────────
function h1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    spacing: { before: 360, after: 160 },
    children: [new TextRun({ text, bold: true, size: 36, font: "Arial", color: BRAND })]
  });
}
function h2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    spacing: { before: 280, after: 120 },
    border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: ACCENT, space: 1 } },
    children: [new TextRun({ text, bold: true, size: 28, font: "Arial", color: BRAND })]
  });
}
function h3(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_3,
    spacing: { before: 200, after: 80 },
    children: [new TextRun({ text, bold: true, size: 24, font: "Arial", color: ACCENT })]
  });
}
function h4(text) {
  return new Paragraph({
    spacing: { before: 160, after: 60 },
    children: [new TextRun({ text, bold: true, size: 22, font: "Arial", color: TEAL })]
  });
}
function para(text, { size=22, color=TEXT, italic=false, spacing={ before:60, after:80 } }={}) {
  return new Paragraph({ spacing, children: [new TextRun({ text, size, font:"Arial", color, italic })] });
}
function bullet(text, level=0) {
  return new Paragraph({
    numbering: { reference: "bullets", level },
    spacing: { before: 40, after: 40 },
    children: [new TextRun({ text, size: 20, font: "Arial", color: TEXT })]
  });
}
function numbered(text, level=0) {
  return new Paragraph({
    numbering: { reference: "numbers", level },
    spacing: { before: 40, after: 40 },
    children: [new TextRun({ text, size: 20, font: "Arial", color: TEXT })]
  });
}
function spacer(n=1) {
  return Array.from({ length: n }, () => new Paragraph({ children: [new TextRun("")] }));
}
function divider() {
  return new Paragraph({
    spacing: { before: 120, after: 120 },
    border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: LIGHT, space: 1 } },
    children: [new TextRun("")]
  });
}
function code(text) {
  return new Paragraph({
    spacing: { before: 60, after: 60 },
    shading: { fill: "1E1E2E", type: ShadingType.CLEAR },
    children: [new TextRun({ text, font: "Courier New", size: 18, color: "A6E3A1" })]
  });
}
function infoBox(text, fill="E0F2FE", color="075985") {
  return new Table({
    width: { size: 9360, type: WidthType.DXA },
    columnWidths: [9360],
    rows: [new TableRow({ children: [
      new TableCell({
        borders: { left: { style: BorderStyle.SINGLE, size: 12, color: TEAL }, top: bdr(TEAL), bottom: bdr(TEAL), right: bdr(TEAL) },
        shading: { fill, type: ShadingType.CLEAR },
        margins: { top: 100, bottom: 100, left: 200, right: 200 },
        children: [new Paragraph({ children: [new TextRun({ text, size: 20, font: "Arial", color })] })]
      })
    ]})]
  });
}
function changeBox(text) {
  return new Table({
    width: { size: 9360, type: WidthType.DXA },
    columnWidths: [9360],
    rows: [new TableRow({ children: [
      new TableCell({
        borders: { left: { style: BorderStyle.SINGLE, size: 12, color: GREEN }, top: bdr(GREEN), bottom: bdr(GREEN), right: bdr(GREEN) },
        shading: { fill: "F0FDF4", type: ShadingType.CLEAR },
        margins: { top: 100, bottom: 100, left: 200, right: 200 },
        children: [new Paragraph({ children: [new TextRun({ text: `✦ v2.0 change: ${text}`, size: 20, font: "Arial", color: GREEN, bold: true })] })]
      })
    ]})]
  });
}

// ── Table builders ──────────────────────────────────────────────
function twoCol(rows) {
  return new Table({
    width: { size: 9360, type: WidthType.DXA },
    columnWidths: [3120, 6240],
    rows: rows.map(([a,b], i) => new TableRow({ children: [
      cell(a, { fill: LIGHT, bold: true, w: 3120 }),
      cell(b, { w: 6240 })
    ]}))
  });
}

function featureTable(rows) {
  const cols = [2000, 2800, 2400, 1160, 1000];
  const hdr = new TableRow({ children: [
    hCell("Feature", cols[0]), hCell("Description", cols[1]),
    hCell("AI model", cols[2]), hCell("Host", cols[3]), hCell("Priority", cols[4])
  ]});
  return new Table({
    width: { size: 9360, type: WidthType.DXA }, columnWidths: cols,
    rows: [hdr, ...rows.map(([feat,desc,ai,host,pri], i) => {
      const priColor = pri==="P0" ? ACCENT : pri==="P1" ? AMBER : TEAL;
      const hostColor = host==="GPU server" ? PURPLE : host==="Browser" ? TEAL : host==="Claude API" ? ORANGE : MUTED;
      return new TableRow({ children: [
        cell(feat,  { fill: i%2===0?LIGHT:WHITE, bold:true, w:cols[0] }),
        cell(desc,  { fill: i%2===0?LIGHT:WHITE, w:cols[1] }),
        cell(ai,    { fill: i%2===0?LIGHT:WHITE, w:cols[2], color:PURPLE }),
        cell(host,  { fill: i%2===0?LIGHT:WHITE, w:cols[3], color:hostColor }),
        cell([new Paragraph({ children:[new TextRun({text:pri,bold:true,size:20,font:"Arial",color:priColor})] })],
             { fill: i%2===0?LIGHT:WHITE, w:cols[4] })
      ]});
    })]
  });
}

function stackTable(rows) {
  const cols = [2200, 2600, 2400, 2160];
  return new Table({
    width: { size: 9360, type: WidthType.DXA }, columnWidths: cols,
    rows: [
      new TableRow({ children: [hCell("Layer",cols[0]),hCell("Technology",cols[1]),hCell("Open source?",cols[2]),hCell("Purpose",cols[3])] }),
      ...rows.map(([l,t,os,p], i) => {
        const osColor = os==="Yes — free" ? GREEN : os==="Partial (cheap)" ? AMBER : ORANGE;
        return new TableRow({ children: [
          cell(l,  { fill:i%2===0?LIGHT:WHITE, bold:true, w:cols[0] }),
          cell(t,  { fill:i%2===0?LIGHT:WHITE, w:cols[1], color:PURPLE }),
          cell([new Paragraph({children:[new TextRun({text:os,bold:true,size:20,font:"Arial",color:osColor})]})],
               { fill:i%2===0?LIGHT:WHITE, w:cols[2] }),
          cell(p,  { fill:i%2===0?LIGHT:WHITE, w:cols[3] }),
        ]});
      })
    ]
  });
}

function costTable(rows) {
  const cols = [3000, 2000, 2000, 2360];
  return new Table({
    width: { size: 9360, type: WidthType.DXA }, columnWidths: cols,
    rows: [
      new TableRow({ children: [hCell("Item",cols[0]),hCell("v1.0 cost/mo",cols[1]),hCell("v2.0 cost/mo",cols[2]),hCell("Saving",cols[3])] }),
      ...rows.map(([item,v1,v2,save], i) => new TableRow({ children: [
        cell(item, { fill:i%2===0?LIGHT:WHITE, bold:true, w:cols[0] }),
        cell(v1,   { fill:i%2===0?LIGHT:WHITE, w:cols[1], color:ACCENT }),
        cell(v2,   { fill:i%2===0?LIGHT:WHITE, w:cols[2], color:GREEN }),
        cell(save, { fill:i%2===0?LIGHT:WHITE, bold:true, w:cols[3], color:TEAL }),
      ]}))
    ]
  });
}

function riskTable(rows) {
  const cols = [2500,1800,1800,3260];
  return new Table({
    width:{size:9360,type:WidthType.DXA}, columnWidths:cols,
    rows:[
      new TableRow({children:[hCell("Risk",cols[0]),hCell("Likelihood",cols[1]),hCell("Impact",cols[2]),hCell("Mitigation",cols[3])]}),
      ...rows.map(([r,l,im,m],i)=>{
        const lc=l==="High"?ACCENT:l==="Medium"?AMBER:TEAL;
        const ic=im==="High"?ACCENT:im==="Medium"?AMBER:TEAL;
        return new TableRow({children:[
          cell(r,  {fill:i%2===0?LIGHT:WHITE,bold:true,w:cols[0]}),
          cell([new Paragraph({children:[new TextRun({text:l, bold:true,size:20,font:"Arial",color:lc})]})],{fill:i%2===0?LIGHT:WHITE,w:cols[1]}),
          cell([new Paragraph({children:[new TextRun({text:im,bold:true,size:20,font:"Arial",color:ic})]})],{fill:i%2===0?LIGHT:WHITE,w:cols[2]}),
          cell(m,  {fill:i%2===0?LIGHT:WHITE,w:cols[3]}),
        ]});
      })
    ]
  });
}

function schemaTable(rows) {
  const cols=[2200,2000,2400,2760];
  return new Table({
    width:{size:9360,type:WidthType.DXA},columnWidths:cols,
    rows:[
      new TableRow({children:[hCell("Field",cols[0]),hCell("Type",cols[1]),hCell("Ref",cols[2]),hCell("Description",cols[3])]}),
      ...rows.map(([f,t,r,d],i)=>new TableRow({children:[
        cell(f,{fill:i%2===0?LIGHT:WHITE,bold:true,w:cols[0],color:TEAL}),
        cell(t,{fill:i%2===0?LIGHT:WHITE,w:cols[1],color:PURPLE}),
        cell(r,{fill:i%2===0?LIGHT:WHITE,w:cols[2],color:MUTED}),
        cell(d,{fill:i%2===0?LIGHT:WHITE,w:cols[3]}),
      ]}))
    ]
  });
}

// ════════════════════════════════════════════════════════════════
//  DOCUMENT
// ════════════════════════════════════════════════════════════════
const doc = new Document({
  styles:{
    default:{ document:{ run:{ font:"Arial", size:22 } } },
    paragraphStyles:[
      { id:"Heading1", name:"Heading 1", basedOn:"Normal", next:"Normal", quickFormat:true,
        run:{ size:36, bold:true, font:"Arial", color:BRAND },
        paragraph:{ spacing:{ before:360, after:160 }, outlineLevel:0 } },
      { id:"Heading2", name:"Heading 2", basedOn:"Normal", next:"Normal", quickFormat:true,
        run:{ size:28, bold:true, font:"Arial", color:BRAND },
        paragraph:{ spacing:{ before:280, after:120 }, outlineLevel:1 } },
      { id:"Heading3", name:"Heading 3", basedOn:"Normal", next:"Normal", quickFormat:true,
        run:{ size:24, bold:true, font:"Arial", color:ACCENT },
        paragraph:{ spacing:{ before:200, after:80 }, outlineLevel:2 } },
    ]
  },
  numbering:{
    config:[
      { reference:"bullets",
        levels:[
          { level:0, format:LevelFormat.BULLET, text:"\u2022", alignment:AlignmentType.LEFT,
            style:{ paragraph:{ indent:{ left:720, hanging:360 } } } },
          { level:1, format:LevelFormat.BULLET, text:"\u25E6", alignment:AlignmentType.LEFT,
            style:{ paragraph:{ indent:{ left:1080, hanging:360 } } } },
        ]
      },
      { reference:"numbers",
        levels:[
          { level:0, format:LevelFormat.DECIMAL, text:"%1.", alignment:AlignmentType.LEFT,
            style:{ paragraph:{ indent:{ left:720, hanging:360 } } } }
        ]
      }
    ]
  },
  sections:[{
    properties:{
      page:{
        size:{ width:12240, height:15840 },
        margin:{ top:1080, right:1080, bottom:1080, left:1080 }
      }
    },
    headers:{
      default: new Header({ children:[
        new Paragraph({
          spacing:{ after:0 },
          border:{ bottom:{ style:BorderStyle.SINGLE, size:6, color:ACCENT, space:4 } },
          tabStops:[{ type:TabStopType.RIGHT, position:9360 }],
          children:[
            new TextRun({ text:"STITCHIQ — Product Requirements Document", bold:true, size:18, font:"Arial", color:BRAND }),
            new TextRun({ text:"\t\tv2.0  |  Open-Source AI Edition  |  May 2026", size:18, font:"Arial", color:MUTED }),
          ]
        })
      ]})
    },
    footers:{
      default: new Footer({ children:[
        new Paragraph({
          spacing:{ before:0 },
          border:{ top:{ style:BorderStyle.SINGLE, size:4, color:LIGHT, space:4 } },
          tabStops:[{ type:TabStopType.RIGHT, position:9360 }],
          children:[
            new TextRun({ text:"© 2026 StitchIQ — Confidential", size:16, font:"Arial", color:MUTED }),
            new TextRun({ text:"\t\tOpen-Source AI Stack  |  Google Colab Pro Edition", size:16, font:"Arial", color:MUTED }),
          ]
        })
      ]})
    },
    children:[

      // ── COVER ──────────────────────────────────────────────────
      new Paragraph({
        spacing:{ before:600, after:120 },
        children:[new TextRun({ text:"STITCHIQ", bold:true, size:72, font:"Arial", color:BRAND })]
      }),
      new Paragraph({
        spacing:{ before:0, after:80 },
        children:[new TextRun({ text:"AI-Powered Fashion Platform", bold:true, size:40, font:"Arial", color:ACCENT })]
      }),
      new Paragraph({
        spacing:{ before:0, after:60 },
        children:[new TextRun({ text:"Product Requirements Document  ·  Version 2.0  ·  May 2026", size:22, font:"Arial", color:MUTED, italic:true })]
      }),
      new Paragraph({
        spacing:{ before:0, after:400 },
        children:[new TextRun({ text:"Open-Source AI Stack  ·  Google Colab Pro Edition", size:22, font:"Arial", color:TEAL, bold:true })]
      }),

      twoCol([
        ["Document status",  "Updated — Open-Source AI Revision"],
        ["Version",          "2.0 (supersedes v1.0)"],
        ["Previous version", "v1.0 — May 2026"],
        ["Key change",       "Full AI stack migrated to open-source models; Google Colab Pro for GPU"],
        ["Date",             "May 2026"],
        ["Platform",         "Web App (React + Node.js + MongoDB + Python AI Worker)"],
        ["Monthly AI cost",  "~$95/month (down from ~$800/month in v1.0)"],
      ]),

      ...spacer(2),
      changeBox("This document supersedes PRD v1.0. All paid AI API services (Fashn.ai, Ideogram, Adobe Firefly) have been replaced with self-hosted open-source models running on Google Colab Pro. Claude API is retained only for low-cost text reasoning."),
      ...spacer(1),
      divider(),

      // ── 1. EXECUTIVE SUMMARY ───────────────────────────────────
      h1("1. Executive Summary"),
      para("StitchIQ is a web-based AI-powered fashion platform designed for the African market, bridging the gap between fashion inspiration and bespoke tailoring. Version 2.0 of this PRD updates the entire AI infrastructure to use open-source models, dramatically reducing operating costs while maintaining feature parity with the original specification."),
      para("The core architectural change is the introduction of a Python AI Worker service — a separate FastAPI server running open-source models on Google Colab Pro GPU instances. This worker handles all computationally expensive AI tasks (try-on, image generation, segmentation) via a Bull job queue, while the Node.js API handles business logic and Claude API handles lightweight text reasoning."),
      ...spacer(1),
      twoCol([
        ["Product name",       "StitchIQ"],
        ["Target markets",     "Nigeria, Ghana, Kenya (Phase 1); Africa & diaspora (Phase 2)"],
        ["Core users",         "Fashion customers (B2C) and professional tailors / designers (B2B)"],
        ["Monetisation",       "SaaS subscriptions (tailors) + marketplace commission + premium features"],
        ["AI infrastructure",  "Open-source models (SAM 2, IDM-VTON, SDXL, CLIP, LLaVA) on Colab Pro"],
        ["GPU platform",       "Google Colab Pro+ ($50/month) — Phase 1; Vast.ai GPU on demand — Phase 2"],
        ["Monthly cost",       "~$95/month total (AI + hosting + infra)"],
        ["Launch target",      "Q4 2026 (MVP); Q2 2027 (full platform)"],
      ]),
      ...spacer(2),
      divider(),

      // ── 2. PROBLEM STATEMENT ───────────────────────────────────
      h1("2. Problem Statement"),
      h2("2.1  Customer pain points"),
      bullet("No easy way to translate a style photo or inspiration into an actual garment without visiting multiple shops and tailors."),
      bullet("Measurements are lost between visits — orders get wrong repeatedly."),
      bullet("Customers cannot visualise how a style will look on their body before committing to expensive fabric and labour."),
      bullet("Sourcing quality fabric is time-consuming; prices are opaque and comparison is hard."),
      bullet("No single platform combines styling, fabric shopping, and tailor booking."),
      ...spacer(1),
      h2("2.2  Tailor pain points"),
      bullet("Client measurement records kept in handwritten notebooks — easily lost, not searchable."),
      bullet("Pattern drafting for each new client is time-consuming and error-prone."),
      bullet("Fabric yardage calculations done by guesswork, leading to waste or shortfalls."),
      bullet("No digital tools to bridge the communication gap between tailor and client."),
      bullet("Tailors have no online presence beyond word-of-mouth."),
      ...spacer(1),
      h2("2.3  Market opportunity"),
      para("Africa's fashion industry is valued at over $31 billion and growing at 6% CAGR. The bespoke tailoring sector alone employs millions of micro-entrepreneurs. No incumbent platform addresses both sides of the market with AI-powered tooling built specifically for African fashion contexts (Ankara, Aso-oke, Kente, lace, adire etc.)."),
      ...spacer(2),
      divider(),

      // ── 3. GOALS & METRICS ────────────────────────────────────
      h1("3. Goals & Success Metrics"),
      h2("3.1  Business goals"),
      bullet("Acquire 50,000 registered users within 12 months of launch."),
      bullet("On-board 2,000 verified tailors in Nigeria, Ghana, and Kenya by end of Year 1."),
      bullet("Process ₦500M in fabric marketplace GMV within 18 months."),
      bullet("Achieve 60% monthly active user retention at Month 6."),
      ...spacer(1),
      h2("3.2  Cost goals (new in v2.0)"),
      bullet("Keep total monthly infrastructure cost below $150/month through MVP phase."),
      bullet("Keep AI cost per user action below $0.01 for 95% of requests."),
      bullet("Migrate off Google Colab Pro to dedicated GPU server only when monthly AI requests exceed 50,000."),
      ...spacer(1),
      h2("3.3  Key metrics (OKRs)"),
      ...spacer(1),
      new Table({
        width:{size:9360,type:WidthType.DXA}, columnWidths:[2000,3680,3680],
        rows:[
          new TableRow({children:[hCell("Objective",2000),hCell("Key result",3680),hCell("Target",3680)]}),
          ...([
            ["Grow user base",        "Registered users (Month 12)",                    "50,000"],
            ["Engage customers",      "Sessions per active user per month",              ">= 4"],
            ["Monetise tailors",      "Paid tailor subscriptions (Month 12)",            "1,500"],
            ["Drive marketplace GMV", "Monthly fabric marketplace transactions",         "N40M by Month 12"],
            ["AI quality",            "Pattern analysis satisfaction (5-star survey)",   ">= 4.2 / 5"],
            ["Cost efficiency",       "Monthly AI infra cost at 10k active users",       "< $150"],
            ["Retention",             "D30 retention rate",                              ">= 40%"],
          ]).map(([o,k,t],i)=>new TableRow({children:[
            cell(o,{fill:i%2===0?LIGHT:WHITE,bold:true,w:2000}),
            cell(k,{fill:i%2===0?LIGHT:WHITE,w:3680}),
            cell(t,{fill:i%2===0?LIGHT:WHITE,bold:true,w:3680,color:TEAL}),
          ]}))
        ]
      }),
      ...spacer(2),
      divider(),

      // ── 4. USER PERSONAS ──────────────────────────────────────
      h1("4. User Personas"),
      h2("4.1  Adaeze — The Style-Conscious Customer"),
      twoCol([
        ["Age / location",  "28, Lagos Nigeria"],
        ["Occupation",      "Marketing executive"],
        ["Tech comfort",    "High — heavy Instagram and TikTok user"],
        ["Goals",           "Look great at events; get custom clothes without the stress"],
        ["Frustrations",    "Tailors ruin her designs; fabric shopping is exhausting; can't visualise styles before sewing"],
        ["Key features",    "Virtual try-on, occasion stylist, wardrobe vault, fabric marketplace"],
      ]),
      ...spacer(1),
      h2("4.2  Emeka — The Professional Tailor"),
      twoCol([
        ["Age / location",  "35, Aba Nigeria"],
        ["Occupation",      "Master tailor, 12 years experience, 40+ clients"],
        ["Tech comfort",    "Medium — uses WhatsApp heavily"],
        ["Goals",           "Grow client base; reduce pattern errors; look professional"],
        ["Frustrations",    "Loses measurement notebooks; spends hours on pattern drafting; no business tools"],
        ["Key features",    "Client management, auto pattern generator, yardage calculator, tailor profile page"],
      ]),
      ...spacer(1),
      h2("4.3  Fatima — The Fabric Vendor"),
      twoCol([
        ["Age / location",  "42, Kano Nigeria"],
        ["Occupation",      "Textile wholesaler, 20 years in trade"],
        ["Tech comfort",    "Low-medium — uses WhatsApp for orders"],
        ["Goals",           "Reach more customers digitally; reduce time spent on manual orders"],
        ["Frustrations",    "No digital storefront; customers can't find her online; inventory tracking is manual"],
        ["Key features",    "Vendor listings, inventory dashboard, order management, AI auto-tagging"],
      ]),
      ...spacer(2),
      divider(),

      // ── 5. OPEN-SOURCE AI STACK ───────────────────────────────
      h1("5. Open-Source AI Stack (v2.0)"),
      changeBox("Entire AI infrastructure replaced with open-source models. Paid API costs reduced from ~$800/month to ~$10/month (Claude API only)."),
      ...spacer(1),

      h2("5.1  Model selection"),
      new Table({
        width:{size:9360,type:WidthType.DXA}, columnWidths:[2200,2600,2000,2560],
        rows:[
          new TableRow({children:[hCell("Feature",2200),hCell("Model",2600),hCell("Replaces",2000),hCell("Where it runs",2560)]}),
          ...([
            ["Garment segmentation",  "Meta SAM 2 (open source)",              "Paid SAM 2 API",       "Google Colab Pro GPU"],
            ["Garment body parsing",  "SCHP — Self-Correction Human Parsing",  "Paid SCHP API",        "Google Colab Pro GPU"],
            ["Body pose / measure",   "MediaPipe Pose (Google, open source)",  "Avaturn API",          "Browser (free, zero cost)"],
            ["Virtual try-on",        "IDM-VTON (2024 SOTA open source)",      "Fashn.ai API ($0.10+/render)", "Google Colab Pro GPU"],
            ["Style image gen",       "Stable Diffusion XL (SDXL)",            "Ideogram API",         "Google Colab Pro GPU"],
            ["Alterations inpaint",   "SDXL Inpainting model",                 "Adobe Firefly API",    "Google Colab Pro GPU"],
            ["Fabric image search",   "CLIP (OpenAI, open source)",            "Google Vision API",    "Google Colab Pro GPU"],
            ["Vision understanding",  "LLaVA 1.6 / Llama 3.2 Vision",         "Claude Vision (heavy use)", "Google Colab Pro GPU"],
            ["Text reasoning",        "Claude API (Sonnet — retained)",         "—",                   "Anthropic (API)"],
            ["Fabric search",         "Typesense (self-hosted)",                "Algolia ($50+/mo)",    "Your VPS"],
          ]).map(([f,m,r,w],i)=>new TableRow({children:[
            cell(f,{fill:i%2===0?LIGHT:WHITE,bold:true,w:2200}),
            cell(m,{fill:i%2===0?LIGHT:WHITE,w:2600,color:PURPLE}),
            cell(r,{fill:i%2===0?LIGHT:WHITE,w:2000,color:MUTED}),
            cell(w,{fill:i%2===0?LIGHT:WHITE,w:2560,color:TEAL}),
          ]}))
        ]
      }),
      ...spacer(1),

      h2("5.2  Why Claude API is retained for text reasoning"),
      infoBox("At MVP scale (under 10,000 users), Claude API costs approximately $5–15/month for text reasoning tasks (pattern descriptions, occasion stylist JSON, budget trade-off explanations, yardage analysis). This is cheaper than hosting a 70B Llama model on a GPU server. Self-hosted Llama 3.1 becomes cost-effective only above ~50,000 monthly AI text requests. The recommendation is to keep Claude API through Phase 1 and Phase 2, then evaluate self-hosting in Phase 3."),
      ...spacer(1),

      h2("5.3  Google Colab Pro — Phase 1 GPU platform"),
      para("Google Colab Pro+ ($50/month) provides access to A100 and V100 GPUs on demand. For Phase 1 (MVP), all open-source AI models are deployed as persistent notebooks on Colab Pro, exposed as internal HTTP endpoints consumed by the Node.js API via the Bull job queue."),
      ...spacer(1),
      twoCol([
        ["Plan",              "Google Colab Pro+ ($50/month)"],
        ["GPU access",        "A100 40GB (priority), V100 16GB (standard)"],
        ["Session length",    "Up to 24 hours per session; auto-reconnect scripts keep alive"],
        ["Models loaded",     "SAM 2, IDM-VTON, SDXL, CLIP, LLaVA — all in one persistent notebook"],
        ["API exposure",      "FastAPI + ngrok tunnel → internal URL stored in Node.js env vars"],
        ["Concurrent jobs",   "1 GPU job at a time; Bull queue serialises requests"],
        ["Cold start",        "~3–5 minutes to load all models on a fresh session"],
        ["Migration trigger", "Move to Vast.ai dedicated GPU when monthly requests exceed 15,000"],
      ]),
      ...spacer(1),
      infoBox("Colab Pro limitation: Sessions can be interrupted by Google. The AI Worker includes auto-reconnect logic and a health-check endpoint. The Node.js API retries failed jobs up to 3 times before returning an error to the user. Users see a 'processing' spinner with real-time status."),
      ...spacer(2),
      divider(),

      // ── 6. SYSTEM ARCHITECTURE ────────────────────────────────
      h1("6. System Architecture"),
      changeBox("New component added: Python AI Worker (FastAPI) running on Google Colab Pro. All AI inference routes through this worker via Bull queue."),
      ...spacer(1),

      h2("6.1  Architecture overview"),
      para("StitchIQ v2.0 has four distinct service layers:"),
      bullet("React Frontend — web UI; talks only to Node.js API."),
      bullet("Node.js API (Express) — business logic, auth, MongoDB, job dispatch. Hosted on Railway or Render."),
      bullet("Python AI Worker (FastAPI on Colab Pro) — all open-source model inference. Exposed via ngrok tunnel."),
      bullet("Claude API — text reasoning calls made directly from Node.js API (no GPU needed)."),
      ...spacer(1),

      h2("6.2  Request flow for AI features"),
      numbered("User triggers an AI feature (e.g. virtual try-on) from React frontend."),
      numbered("React sends request to Node.js API (/api/v1/ai/try-on)."),
      numbered("Node.js API validates request, uploads image to Cloudinary, enqueues a Bull job."),
      numbered("Bull worker picks up the job and sends HTTP POST to Python AI Worker (Colab Pro via ngrok)."),
      numbered("Python AI Worker runs IDM-VTON inference on Colab GPU (~8–15 seconds)."),
      numbered("Result image is returned to Bull worker, uploaded to Cloudinary, saved to MongoDB."),
      numbered("Node.js API emits WebSocket event to the user's browser with the result URL."),
      numbered("React frontend displays result. Total user-facing wait: 10–20 seconds."),
      ...spacer(1),

      h2("6.3  Technology stack (updated)"),
      stackTable([
        ["Frontend",          "React 18 + Vite",              "Yes — free",      "Single-page web app"],
        ["Styling",           "Tailwind CSS",                 "Yes — free",      "Utility-first responsive design"],
        ["State management",  "Zustand",                      "Yes — free",      "Global state: auth, cart, vault"],
        ["Backend API",       "Node.js + Express",            "Yes — free",      "REST API; job dispatch; auth"],
        ["Database",          "MongoDB Atlas",                "Yes — free",      "All app data (free tier to start)"],
        ["Image storage",     "Cloudinary",                   "Yes — free",      "25GB free; image CDN"],
        ["Search",            "Typesense (self-hosted)",      "Yes — free",      "Sub-500ms fabric search"],
        ["Auth",              "JWT + bcrypt",                 "Yes — free",      "Stateless auth; httpOnly cookies"],
        ["AI text reasoning", "Claude API (Sonnet)",          "Partial (cheap)", "$5–15/month at MVP scale"],
        ["AI segmentation",   "Meta SAM 2 (self-hosted)",     "Yes — free",      "Garment region detection"],
        ["AI body parsing",   "SCHP (self-hosted)",           "Yes — free",      "Body part segmentation"],
        ["AI pose",           "MediaPipe Pose",               "Yes — free",      "Runs in browser — zero server cost"],
        ["AI try-on",         "IDM-VTON (self-hosted)",       "Yes — free",      "Virtual garment draping"],
        ["AI image gen",      "SDXL + fashion LoRA",          "Yes — free",      "Occasion mockups; alterations"],
        ["AI fabric search",  "CLIP (self-hosted)",           "Yes — free",      "Visual fabric similarity search"],
        ["AI vision (alt)",   "LLaVA 1.6 (self-hosted)",     "Yes — free",      "Image understanding backup"],
        ["GPU platform",      "Google Colab Pro+",            "Partial (cheap)", "$50/month — Phase 1"],
        ["AI worker server",  "Python FastAPI",               "Yes — free",      "Serves all open-source models"],
        ["Job queue",         "Bull + Redis",                 "Yes — free",      "Serialises GPU jobs; retries"],
        ["Payments",          "Paystack + Flutterwave",       "Partial (cheap)", "1.5% + N100/transaction"],
        ["Real-time",         "Socket.io",                    "Yes — free",      "Live job status; messaging"],
        ["Hosting (API)",     "Railway / Render",             "Partial (cheap)", "$5–20/month"],
        ["CDN",               "Cloudflare",                   "Yes — free",      "Free tier covers MVP traffic"],
        ["Monitoring",        "Sentry (free tier)",           "Yes — free",      "Error tracking"],
      ]),
      ...spacer(2),
      divider(),

      // ── 7. COST BREAKDOWN ────────────────────────────────────
      h1("7. Cost Breakdown (v1.0 vs v2.0)"),
      changeBox("Total monthly cost reduced from ~$800/month (v1.0 paid APIs) to ~$95/month (v2.0 open source). Saving: ~$700/month = ~$8,400/year."),
      ...spacer(1),
      costTable([
        ["Fashn.ai try-on API (1k renders/mo)", "$100–150",  "$0 (IDM-VTON self-hosted)",  "~$125/mo saved"],
        ["Ideogram image generation API",        "$80–120",   "$0 (SDXL self-hosted)",      "~$100/mo saved"],
        ["Adobe Firefly inpainting API",         "$60–100",   "$0 (SDXL Inpaint hosted)",   "~$80/mo saved"],
        ["Google Vision API (fabric search)",    "$30–60",    "$0 (CLIP self-hosted)",       "~$45/mo saved"],
        ["Algolia search",                       "$50",       "$0 (Typesense self-hosted)",  "$50/mo saved"],
        ["SAM 2 hosted API",                     "$40–80",    "$0 (SAM 2 self-hosted)",      "~$60/mo saved"],
        ["Claude API (text reasoning)",          "$10–20",    "$5–15 (retained, minimal)",   "Minimal change"],
        ["GPU compute",                          "$0 (via APIs)", "$50 (Colab Pro+)",         "-$50 (new cost)"],
        ["Node.js hosting (Railway)",            "$10–20",    "$10–20",                      "No change"],
        ["MongoDB Atlas",                        "$0 (free)", "$0 (free tier)",              "No change"],
        ["Cloudinary",                           "$0 (free)", "$0 (free tier)",              "No change"],
        ["Domain + misc",                        "$15",       "$15",                         "No change"],
      ]),
      ...spacer(1),
      new Table({
        width:{size:9360,type:WidthType.DXA}, columnWidths:[4680,2340,2340],
        rows:[
          new TableRow({children:[hCell("",4680), hCell("v1.0 total/month",2340), hCell("v2.0 total/month",2340)]}),
          new TableRow({children:[
            cell("Estimated monthly running cost", {fill:LIGHT, bold:true, w:4680}),
            cell("~$400–$600", {fill:LIGHT, bold:true, w:2340, color:ACCENT}),
            cell("~$80–$100", {fill:LIGHT, bold:true, w:2340, color:GREEN}),
          ]})
        ]
      }),
      ...spacer(2),
      divider(),

      // ── 8. FEATURE REQUIREMENTS ──────────────────────────────
      h1("8. Feature Requirements"),
      para("All feature descriptions remain the same as v1.0. The table below shows the updated AI model and hosting for each feature."),
      ...spacer(1),

      h2("8.1  Authentication & user management"),
      featureTable([
        ["Register / login",      "Email + password; JWT; httpOnly cookies",        "—",                     "VPS",         "P0"],
        ["Role selection",        "Customer, Tailor, or Vendor",                    "—",                     "VPS",         "P0"],
        ["Measurement profile",   "Manual entry: chest, waist, hips, height",       "—",                     "MongoDB",     "P0"],
        ["Body pose estimation",  "Live camera pose for measurement hints",          "MediaPipe Pose",        "Browser",     "P1"],
        ["Google OAuth",          "Sign in with Google",                            "—",                     "VPS",         "P1"],
        ["Tailor verification",   "Business info upload; admin approval",           "—",                     "MongoDB",     "P0"],
      ]),
      ...spacer(1),

      h2("8.2  Pattern cut & description"),
      featureTable([
        ["Upload style photo",    "Drag-and-drop or camera; stored in Cloudinary",  "—",                     "VPS",         "P0"],
        ["Garment segmentation",  "Detect collar, sleeve, bodice, hem regions",     "Meta SAM 2",            "GPU server",  "P0"],
        ["Body parsing",          "Full garment part isolation",                    "SCHP",                  "GPU server",  "P0"],
        ["Cut description",       "Structured text: cut type, seam notes",          "Claude API",            "Claude API",  "P0"],
        ["Annotated image",       "Overlay segment labels on image; downloadable",  "SAM 2 + Canvas",        "GPU server",  "P0"],
        ["Save to vault",         "Save analysis to wardrobe vault",                "—",                     "MongoDB",     "P0"],
      ]),
      ...spacer(1),

      h2("8.3  Virtual try-on"),
      featureTable([
        ["Measurement input",     "Pull from profile or enter ad hoc",              "—",                     "MongoDB",     "P0"],
        ["Style selection",       "From vault, occasion results, or upload",        "—",                     "VPS",         "P0"],
        ["Try-on render",         "Drape garment on body from measurements",        "IDM-VTON",              "GPU server",  "P0"],
        ["Async job status",      "WebSocket progress indicator while GPU renders", "Bull + Socket.io",      "VPS",         "P0"],
        ["Fabric colour swap",    "Swap fabric colour; re-render",                  "IDM-VTON + SDXL",       "GPU server",  "P1"],
        ["Style alternatives",    "3 similar style suggestions",                    "Claude API",            "Claude API",  "P1"],
      ]),
      ...spacer(1),

      h2("8.4  Occasion stylist"),
      featureTable([
        ["Event input",           "Wedding, owambe, interview, casual etc.",        "—",                     "VPS",         "P0"],
        ["Style recommendation",  "JSON: silhouette, neckline, fabric, palette",    "Claude API",            "Claude API",  "P0"],
        ["Visual mockup",         "Rendered style image per suggestion",            "SDXL + fashion LoRA",   "GPU server",  "P0"],
        ["Saved looks",           "Save favourite suggestions to vault",            "—",                     "MongoDB",     "P0"],
      ]),
      ...spacer(1),

      h2("8.5  Fabric recommender"),
      featureTable([
        ["Style analysis",        "Reads style photo for drape, structure cues",    "Claude API",            "Claude API",  "P0"],
        ["Recommendation output", "Fabric type, weight, texture, colour palette",   "Claude API",            "Claude API",  "P0"],
        ["Marketplace link",      "Link recommendations to live inventory",         "—",                     "MongoDB",     "P0"],
      ]),
      ...spacer(1),

      h2("8.6  Budget filter"),
      featureTable([
        ["Budget range input",    "Min/max in local currency",                      "—",                     "VPS",         "P0"],
        ["Cost calculation",      "Fabric cost + estimated tailor fee",             "—",                     "MongoDB",     "P0"],
        ["Trade-off explanation", "Plain-language money-saving alternatives",       "Claude API",            "Claude API",  "P1"],
      ]),
      ...spacer(1),

      h2("8.7  Alterations assistant"),
      featureTable([
        ["Upload garment photo",  "Photo of existing garment to Cloudinary",        "—",                     "VPS",         "P0"],
        ["Describe alteration",   "Free text: 'shorten sleeves, add peplum'",       "—",                     "VPS",         "P0"],
        ["Region interpretation", "Identify which regions to change",               "Claude API",            "Claude API",  "P0"],
        ["Altered image gen",     "Generate altered garment image",                 "SDXL Inpainting",       "GPU server",  "P0"],
        ["Tailor instructions",   "Step-by-step instructions for tailor",           "Claude API",            "Claude API",  "P0"],
      ]),
      ...spacer(1),

      h2("8.8  Wardrobe vault"),
      featureTable([
        ["Save items",            "Save any try-on, analysis, or occasion look",    "—",                     "MongoDB",     "P0"],
        ["Tagging",               "Tag by event, season, status",                   "—",                     "VPS",         "P0"],
        ["AI recommendations",    "Suggest styles based on vault history",          "Claude API",            "Claude API",  "P1"],
      ]),
      ...spacer(1),

      h2("8.9  Client management (tailor)"),
      featureTable([
        ["Client profiles",       "Name, contact, measurements, style notes",       "—",                     "MongoDB",     "P0"],
        ["Order tracking",        "Style photo, fabric, status, due date",          "—",                     "MongoDB",     "P0"],
        ["AI insights",           "Repeat preferences; suggested upsells",          "Claude API",            "Claude API",  "P1"],
        ["In-app messaging",      "Tailor-client direct messaging",                 "Socket.io",             "VPS",         "P1"],
      ]),
      ...spacer(1),

      h2("8.10  Auto pattern generator (tailor)"),
      featureTable([
        ["Photo to pattern pieces","Identify pattern pieces from style photo",      "SAM 2 + Claude API",    "GPU server",  "P0"],
        ["Pattern calculation",   "Piece dimensions with seam allowances",          "Custom engine",         "VPS",         "P0"],
        ["PDF export",            "Print-ready PDF with labels, grain lines",       "PDFKit",                "VPS",         "P0"],
        ["Pattern library",       "Save generated patterns for reuse",             "—",                     "MongoDB",     "P1"],
      ]),
      ...spacer(1),

      h2("8.11  Fabric yardage calculator (tailor)"),
      featureTable([
        ["Style complexity",      "Detect lining, pleats, ruffles from photo",      "Claude API",            "Claude API",  "P0"],
        ["Yardage output",        "Qty per fabric type: main, lining, interfacing", "Custom engine",         "VPS",         "P0"],
        ["Add to cart",           "Send yardage to marketplace cart",              "—",                     "MongoDB",     "P1"],
      ]),
      ...spacer(1),

      h2("8.12  Fabric marketplace"),
      featureTable([
        ["Vendor onboarding",     "Upload photos, name, price, width, stock",       "—",                     "Cloudinary",  "P0"],
        ["AI auto-tagging",       "Auto-tag fabric type from photo",                "CLIP + LLaVA",          "GPU server",  "P0"],
        ["Full-text search",      "Fast search by name, type, colour, price",       "Typesense",             "VPS",         "P0"],
        ["Image search",          "Upload photo — find similar fabrics",            "CLIP",                  "GPU server",  "P1"],
        ["Natural lang search",   "Interpret 'flowy green under N3k/yard'",         "Claude API",            "Claude API",  "P1"],
        ["Checkout",              "Paystack or Flutterwave payment",                "Paystack",              "Paystack",    "P0"],
        ["Vendor dashboard",      "Sales analytics, stock alerts, revenue",        "—",                     "MongoDB",     "P0"],
      ]),
      ...spacer(2),
      divider(),

      // ── 9. MONGODB SCHEMA ────────────────────────────────────
      h1("9. MongoDB Schema"),
      h2("9.1  users"),
      schemaTable([
        ["_id",             "ObjectId",  "—",            "Auto-generated primary key"],
        ["email",           "String",    "—",            "Unique; indexed"],
        ["passwordHash",    "String",    "—",            "bcrypt hash"],
        ["role",            "String[]",  "—",            "['customer','tailor','vendor']"],
        ["profile",         "Object",    "—",            "name, bio, avatar, location, phone"],
        ["measurements",    "Object",    "—",            "chest, waist, hips, height, shoulder (cm)"],
        ["isVerified",      "Boolean",   "—",            "Email verification flag"],
        ["tailorVerified",  "Boolean",   "—",            "Admin-approved tailor flag"],
        ["createdAt",       "Date",      "—",            "Account creation timestamp"],
      ]),
      ...spacer(1),
      h2("9.2  ai_jobs"),
      para("New collection in v2.0 — tracks async GPU job status.", { color:MUTED, italic:true }),
      schemaTable([
        ["_id",             "ObjectId",  "—",            "Primary key"],
        ["userId",          "ObjectId",  "users._id",    "User who triggered the job"],
        ["jobType",         "String",    "—",            "tryon | pattern | sdxl | clip | sam2"],
        ["bullJobId",       "String",    "—",            "Bull queue job reference"],
        ["status",          "String",    "—",            "queued | processing | done | failed"],
        ["inputData",       "Object",    "—",            "Input params (image URLs, measurements etc.)"],
        ["resultUrl",       "String",    "—",            "Cloudinary URL of output image or PDF"],
        ["errorMessage",    "String",    "—",            "Error details if status = failed"],
        ["retryCount",      "Number",    "—",            "Auto-retry counter (max 3)"],
        ["createdAt",       "Date",      "—",            "Job creation timestamp"],
        ["completedAt",     "Date",      "—",            "Job completion timestamp"],
      ]),
      ...spacer(1),
      h2("9.3  styles (wardrobe vault)"),
      schemaTable([
        ["_id",             "ObjectId",  "—",            "Primary key"],
        ["userId",          "ObjectId",  "users._id",    "Owner"],
        ["type",            "String",    "—",            "analysis | tryon | occasion | alteration"],
        ["imageUrl",        "String",    "—",            "Cloudinary URL"],
        ["aiDescription",   "String",    "—",            "Claude-generated description"],
        ["patternNotes",    "String",    "—",            "Cut and construction notes"],
        ["tags",            "String[]",  "—",            "User-applied tags"],
        ["status",          "String",    "—",            "wishlist | in-progress | sewn"],
        ["aiJobId",         "ObjectId",  "ai_jobs._id",  "Reference to the GPU job that created this"],
        ["createdAt",       "Date",      "—",            "Timestamp"],
      ]),
      ...spacer(1),
      h2("9.4  clients, orders, fabrics, marketplace_orders"),
      para("These collections are unchanged from v1.0. See section 9 of PRD v1.0 for full schema definitions.", { color:MUTED, italic:true }),
      ...spacer(2),
      divider(),

      // ── 10. API ENDPOINTS ────────────────────────────────────
      h1("10. API Design (REST)"),
      h2("10.1  AI job endpoints (updated in v2.0)"),
      new Table({
        width:{size:9360,type:WidthType.DXA}, columnWidths:[1200,2800,5360],
        rows:[
          new TableRow({children:[hCell("Method",1200),hCell("Route",2800),hCell("Description",5360)]}),
          ...([
            ["POST","/ai/pattern-analysis",  "Upload style photo → SAM 2 + Claude → returns job ID; poll for result"],
            ["POST","  /ai/try-on",           "Measurements + style image → IDM-VTON → returns job ID"],
            ["POST","  /ai/occasion-stylist", "Event type → Claude JSON + SDXL mockup → returns job ID"],
            ["POST","  /ai/fabric-recommend", "Style photo + occasion → Claude → returns recommendation JSON directly"],
            ["POST","  /ai/alterations",      "Garment photo + description → Claude + SDXL inpaint → returns job ID"],
            ["POST","  /ai/pattern-generate", "Style photo + measurements → SAM 2 + PDFKit → returns job ID"],
            ["POST","  /ai/yardage",          "Style photo + fabric width → Claude → returns yardage JSON directly"],
            ["GET", "  /ai/jobs/:jobId",      "Poll job status; returns { status, resultUrl, errorMessage }"],
            ["POST","  /ai/clip-search",      "Upload fabric image → CLIP → returns similar fabric IDs from MongoDB"],
          ]).map(([m,r,d],i)=>new TableRow({children:[
            cell(m,{fill:i%2===0?LIGHT:WHITE,bold:true,w:1200,color:m==="GET"?TEAL:ACCENT}),
            cell(r,{fill:i%2===0?LIGHT:WHITE,w:2800,color:PURPLE}),
            cell(d,{fill:i%2===0?LIGHT:WHITE,w:5360}),
          ]}))
        ]
      }),
      ...spacer(1),
      h2("10.2  Internal Python AI Worker endpoints"),
      para("These endpoints are NOT public. They are called only by the Node.js Bull worker via the ngrok tunnel URL.", { color:MUTED, italic:true }),
      new Table({
        width:{size:9360,type:WidthType.DXA}, columnWidths:[1200,2800,5360],
        rows:[
          new TableRow({children:[hCell("Method",1200),hCell("Route",2800),hCell("Description",5360)]}),
          ...([
            ["POST","  /worker/health",       "Health check; returns GPU availability and loaded models"],
            ["POST","  /worker/sam2",         "Run SAM 2 segmentation; returns mask JSON + annotated image"],
            ["POST","  /worker/idmvton",      "Run IDM-VTON try-on; returns rendered image base64"],
            ["POST","  /worker/sdxl",         "Run SDXL generation; returns generated image base64"],
            ["POST","  /worker/sdxl-inpaint", "Run SDXL inpainting; returns edited image base64"],
            ["POST","  /worker/clip-embed",   "Compute CLIP embedding for image; returns vector"],
            ["POST","  /worker/llava",        "Run LLaVA vision understanding; returns text response"],
          ]).map(([m,r,d],i)=>new TableRow({children:[
            cell(m,{fill:i%2===0?LIGHT:WHITE,bold:true,w:1200,color:ACCENT}),
            cell(r,{fill:i%2===0?LIGHT:WHITE,w:2800,color:PURPLE}),
            cell(d,{fill:i%2===0?LIGHT:WHITE,w:5360}),
          ]}))
        ]
      }),
      ...spacer(2),
      divider(),

      // ── 11. NON-FUNCTIONAL REQUIREMENTS ──────────────────────
      h1("11. Non-Functional Requirements"),
      h2("11.1  Performance"),
      bullet("API response (non-AI endpoints): < 200ms p95."),
      bullet("Claude API text responses: < 3 seconds p95."),
      bullet("GPU job queue wait (Colab Pro): < 20 seconds p95 for try-on; < 60 seconds for pattern PDF."),
      bullet("CLIP fabric search: < 800ms including embedding computation."),
      bullet("Fabric text search (Typesense): < 500ms p99."),
      bullet("Page load (LCP): < 2.5 seconds on 4G mobile."),
      ...spacer(1),
      h2("11.2  Reliability (Colab Pro specifics)"),
      bullet("Colab Pro sessions can be interrupted by Google. The system must handle this gracefully."),
      bullet("Node.js API checks AI Worker health every 60 seconds via /worker/health endpoint."),
      bullet("If AI Worker is unreachable, jobs remain in Bull queue with 'queued' status — no data loss."),
      bullet("Auto-reconnect: Colab notebook runs a keep-alive script; if session drops, it restarts in < 5 minutes."),
      bullet("User-facing message: 'Your request is in the queue — we will notify you when it is ready.'"),
      bullet("Maximum 3 automatic retries per failed job before marking as failed and notifying the user."),
      ...spacer(1),
      h2("11.3  Security"),
      bullet("AI Worker ngrok URL is a secret stored in Node.js environment variables — never exposed to frontend."),
      bullet("All AI Worker endpoints require a shared secret header (X-Worker-Token) to prevent abuse."),
      bullet("Input images validated for type and size (max 10MB) before queuing."),
      bullet("Payments handled entirely through Paystack — no card data touches StitchIQ servers."),
      bullet("Rate limiting: 10 GPU job requests per user per hour."),
      bullet("MongoDB Atlas: IP allowlisting; encryption at rest and in transit."),
      ...spacer(1),
      h2("11.4  Privacy"),
      bullet("Body measurement data encrypted at rest in MongoDB using field-level encryption."),
      bullet("User-uploaded photos deleted from Cloudinary after 24 hours unless saved to vault."),
      bullet("No image data is retained by open-source models after inference (stateless worker)."),
      bullet("Users can delete account and all associated data at any time."),
      ...spacer(1),
      h2("11.5  Scalability path"),
      bullet("Phase 1 (0–5k users): Google Colab Pro+ ($50/month) — single GPU, serialised queue."),
      bullet("Phase 2 (5k–20k users): Migrate to Vast.ai RTX 3090 ($60–80/month) — dedicated, always-on."),
      bullet("Phase 3 (20k+ users): Multiple Vast.ai pods or RunPod cluster behind a load balancer."),
      bullet("Text reasoning: Keep Claude API through Phase 2; evaluate self-hosted Llama 3.1 70B in Phase 3."),
      ...spacer(2),
      divider(),

      // ── 12. BUILD ROADMAP ─────────────────────────────────────
      h1("12. Build Roadmap"),
      h2("Phase 1 — MVP (Months 1–4)"),
      para("Goal: Working product on Colab Pro with core AI features end-to-end.", { color:MUTED, italic:true }),
      bullet("Set up Python AI Worker on Google Colab Pro (SAM 2, SDXL, CLIP, LLaVA)"),
      bullet("Auth (register, login, JWT, roles, measurement profile)"),
      bullet("Pattern cut & description (SAM 2 + Claude API)"),
      bullet("Occasion stylist (Claude API + SDXL)"),
      bullet("Fabric recommender (Claude API)"),
      bullet("Fabric yardage calculator (Claude API + custom engine)"),
      bullet("Client management for tailors"),
      bullet("Fabric marketplace (vendor onboarding, Typesense search, Paystack checkout)"),
      bullet("Wardrobe vault (save items, basic tags)"),
      bullet("Bull queue + WebSocket job status system"),
      ...spacer(1),
      h2("Phase 2 — Intelligence layer (Months 5–8)"),
      para("Goal: Full AI feature set; migrate GPU to Vast.ai.", { color:MUTED, italic:true }),
      bullet("Virtual try-on with IDM-VTON"),
      bullet("Alterations assistant (SDXL Inpainting + Claude)"),
      bullet("Auto pattern generator (SAM 2 + PDFKit)"),
      bullet("Budget filter with Claude trade-off explanations"),
      bullet("CLIP image search for fabric marketplace"),
      bullet("Natural language fabric search (Claude API)"),
      bullet("AI client insights for tailors"),
      bullet("In-app tailor-client messaging (Socket.io)"),
      bullet("Migrate GPU from Colab Pro to Vast.ai dedicated instance"),
      ...spacer(1),
      h2("Phase 3 — Scale & monetise (Months 9–12)"),
      para("Goal: Retention, monetisation, and multi-GPU scale.", { color:MUTED, italic:true }),
      bullet("Wardrobe vault timeline view + AI style recommendations"),
      bullet("Tailor public profile page + booking flow"),
      bullet("Tailor SaaS subscription billing"),
      bullet("Fabric vendor reviews and ratings"),
      bullet("Evaluate self-hosted Llama 3.1 70B to replace Claude API"),
      bullet("Fine-tune SDXL LoRA on African fashion dataset for better image quality"),
      bullet("Multilingual support (Yoruba, Igbo, Hausa)"),
      bullet("Mobile PWA with offline measurement access"),
      ...spacer(2),
      divider(),

      // ── 13. RISKS ────────────────────────────────────────────
      h1("13. Risks & Mitigations"),
      riskTable([
        ["Colab Pro session drops mid-job",           "High",   "Medium", "Bull queue retains job; auto-reconnect notebook; user sees 'processing' status; max 5min recovery"],
        ["IDM-VTON quality below user expectations",  "Medium", "High",   "Set expectations in UI; A/B test with SDXL try-on alternative; show example outputs before first use"],
        ["GPU queue backlog at peak usage",           "Medium", "Medium", "Rate-limit to 10 GPU jobs/user/hour; show estimated wait time; email notification on completion"],
        ["ngrok tunnel URL changes on reconnect",     "High",   "Medium", "Node.js API fetches current ngrok URL from a Redis key; Colab notebook updates Redis on startup"],
        ["SDXL image quality for African fashion",    "Medium", "High",   "Fine-tune a LoRA on Ankara/Kente/Aso-oke images early; use fashion-specific base models from HuggingFace"],
        ["Claude API cost spike at scale",            "Low",    "Medium", "Cache identical prompts in Redis (TTL 1hr); evaluate LLaVA replacement at 50k monthly requests"],
        ["Tailor resistance to digital tools",        "High",   "High",   "WhatsApp-integrated onboarding; in-person training; referral incentives; simple tailor-focused UI"],
        ["Paystack payment failures",                 "Medium", "High",   "Flutterwave fallback; retry logic; clear error messages; support chat widget"],
      ]),
      ...spacer(2),
      divider(),

      // ── 14. GLOSSARY ─────────────────────────────────────────
      h1("14. Glossary"),
      twoCol([
        ["IDM-VTON",     "Improving Diffusion Models for Virtual Try-ON — 2024 open-source SOTA try-on model"],
        ["SAM 2",        "Meta Segment Anything Model 2 — open-source image segmentation model"],
        ["SCHP",         "Self-Correction Human Parsing — open-source body part segmentation model"],
        ["SDXL",         "Stable Diffusion XL — open-source image generation model"],
        ["CLIP",         "Contrastive Language-Image Pre-training — OpenAI open-source vision-text model"],
        ["LLaVA",        "Large Language and Vision Assistant — open-source multimodal vision model"],
        ["LoRA",         "Low-Rank Adaptation — technique for fine-tuning AI models cheaply"],
        ["Bull queue",   "Redis-backed Node.js job queue for background processing"],
        ["ngrok",        "Tool that exposes a local server to the internet via a tunnel URL"],
        ["Colab Pro+",   "Google Colab Pro+ — $50/month GPU access (A100/V100) for AI model hosting"],
        ["FastAPI",      "Python web framework used for the AI Worker server"],
        ["MediaPipe",    "Google open-source ML framework; Pose model runs in browser — zero server cost"],
        ["Ankara",       "Brightly coloured African wax-print cotton fabric"],
        ["Aso-oke",      "Hand-loomed Nigerian fabric used for ceremonial wear"],
        ["Owambe",       "Nigerian term for a lavish party or celebration"],
        ["GMV",          "Gross Merchandise Value — total value of goods transacted on the marketplace"],
        ["P0 / P1 / P2", "Priority tiers: P0 = MVP must-have; P1 = Phase 2; P2 = future roadmap"],
      ]),

      ...spacer(2),
      divider(),

      new Paragraph({
        spacing:{ before:120, after:60 },
        alignment: AlignmentType.CENTER,
        children:[new TextRun({ text:"End of Document — StitchIQ PRD v2.0 — Open-Source AI Edition — Confidential", size:18, font:"Arial", color:MUTED, italic:true })]
      })

    ]
  }]
});

Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync("/Users/fullcircle_dev/Downloads/tailor app/StitchIQ_PRD_v2.0.docx", buf);
  console.log("PRD Done");
});
