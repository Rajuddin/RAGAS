const pptxgen = require("pptxgenjs");

const NAVY = "1E2761";
const ICE = "CADCFC";
const WHITE = "FFFFFF";
const INK = "1B1F3B";
const MUTED = "5B6394";
const GREEN = "1F8A5F";
const GREEN_BG = "E4F4EC";
const RED = "C0392B";
const RED_BG = "FBEAE8";
const CARD_BG = "F4F7FD";

function freshShadow(opts) {
  return Object.assign({ type: "outer", color: "1E2761", blur: 10, offset: 3, angle: 90, opacity: 0.18 }, opts || {});
}

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.3 x 7.5
const W = 13.333, H = 7.5;

function baseSlide(bg) {
  const s = pres.addSlide();
  s.background = { color: bg || WHITE };
  return s;
}

function pageNum(s, n, dark) {
  s.addText(String(n).padStart(2, "0"), {
    x: W - 0.9, y: H - 0.55, w: 0.6, h: 0.35, fontFace: "Calibri",
    fontSize: 10, color: dark ? "8B93C9" : "A7AEDB", align: "right", margin: 0,
  });
}

function kicker(s, text, color) {
  s.addText(text.toUpperCase(), {
    x: 0.7, y: 0.55, w: 8, h: 0.4, fontFace: "Calibri", bold: true,
    fontSize: 13, color: color || MUTED, charSpacing: 2, margin: 0,
  });
}

function title(s, text, opts) {
  s.addText(text, Object.assign({
    x: 0.7, y: 0.92, w: 11.5, h: 1.0, fontFace: "Cambria", bold: true,
    fontSize: 32, color: INK, margin: 0,
  }, opts || {}));
}

function iconCircle(s, x, y, d, bg, glyph, glyphColor, glyphSize) {
  s.addShape("ellipse", { x, y, w: d, h: d, fill: { color: bg }, line: { type: "none" } });
  s.addText(glyph, {
    x, y: y - 0.02, w: d, h: d, align: "center", valign: "middle",
    fontFace: "Calibri", bold: true, fontSize: glyphSize || 20, color: glyphColor, margin: 0,
  });
}

function scoreTile(s, x, y, w, h, label, score, band, good) {
  const bg = good ? GREEN_BG : RED_BG;
  const fg = good ? GREEN : RED;
  s.addShape("roundRect", { x, y, w, h, rectRadius: 0.1, fill: { color: bg }, line: { type: "none" }, shadow: freshShadow() });
  s.addText(label, { x: x + 0.15, y: y + 0.12, w: w - 0.3, h: 0.35, fontFace: "Calibri", bold: true, fontSize: 11, color: INK, margin: 0 });
  s.addText(score.toFixed(2), { x: x + 0.15, y: y + 0.42, w: w - 0.3, h: 0.55, fontFace: "Cambria", bold: true, fontSize: 26, color: fg, margin: 0 });
  s.addText(band, { x: x + 0.15, y: y + h - 0.4, w: w - 0.3, h: 0.3, fontFace: "Calibri", bold: true, fontSize: 10, color: fg, charSpacing: 1, margin: 0 });
}

// ---------- Slide 1: Title ----------
{
  const s = baseSlide(NAVY);
  s.addShape("ellipse", { x: 9.6, y: -2.2, w: 7, h: 7, fill: { color: "263480" }, line: { type: "none" } });
  s.addShape("ellipse", { x: 11.2, y: 4.6, w: 4.4, h: 4.4, fill: { color: "263480" }, line: { type: "none" } });
  s.addText("RAG EVALUATION PLATFORM  ·  BUILT ON RAGAS", {
    x: 0.9, y: 2.0, w: 9, h: 0.4, fontFace: "Calibri", bold: true, fontSize: 13,
    color: "8FA3E8", charSpacing: 2, margin: 0,
  });
  s.addText("The Answer That\nSounds Right", {
    x: 0.85, y: 2.5, w: 10.8, h: 2.3, fontFace: "Cambria", bold: true, fontSize: 50,
    color: WHITE, margin: 0, lineSpacingMultiple: 1.05,
  });
  s.addText("A story about trusting what your AI tells customers — and catching it when it's wrong.", {
    x: 0.9, y: 4.85, w: 8.8, h: 0.7, fontFace: "Calibri", fontSize: 17, color: ICE, margin: 0,
  });
  s.addShape("line", { x: 0.9, y: 5.7, w: 1.0, h: 0, line: { color: "5A6BC0", width: 2 } });
  s.addText("A Demo Walkthrough", {
    x: 0.9, y: 5.85, w: 6, h: 0.4, fontFace: "Calibri", fontSize: 13, color: "8FA3E8", margin: 0,
  });
}

// ---------- Slide 2: The Problem ----------
{
  const s = baseSlide(WHITE);
  kicker(s, "Act One — The Problem", MUTED);
  title(s, "It never says “I don’t know.”");
  s.addText(
    "Every AI chatbot answers fast, and answers confidently. That's exactly what makes a wrong answer dangerous — it doesn't look wrong. It reads clean, on-topic, well-formatted. Exactly like a right answer.",
    { x: 0.7, y: 2.05, w: 6.3, h: 2.0, fontFace: "Calibri", fontSize: 15.5, color: INK, margin: 0, lineSpacingMultiple: 1.3 }
  );
  s.addShape("roundRect", { x: 0.7, y: 4.3, w: 6.3, h: 1.9, rectRadius: 0.12, fill: { color: NAVY }, line: { type: "none" }, shadow: freshShadow() });
  s.addText("“Confidently wrong looks exactly like\nconfidently right.”", {
    x: 1.0, y: 4.55, w: 5.7, h: 1.4, fontFace: "Cambria", italic: true, fontSize: 19, color: WHITE, margin: 0, lineSpacingMultiple: 1.2,
  });

  // right column: two chat bubbles contrast
  const cardX = 7.5, cardW = 5.1;
  s.addShape("roundRect", { x: cardX, y: 2.05, w: cardW, h: 1.9, rectRadius: 0.12, fill: { color: GREEN_BG }, line: { type: "none" } });
  iconCircle(s, cardX + 0.25, 2.3, 0.5, GREEN, "✓", WHITE, 20);
  s.addText("“Standard items can be returned within 30 days...”", {
    x: cardX + 0.95, y: 2.28, w: cardW - 1.2, h: 0.9, fontFace: "Calibri", fontSize: 12.5, color: INK, margin: 0,
  });
  s.addText("Reads confident. Is correct.", { x: cardX + 0.25, y: 3.45, w: cardW - 0.5, h: 0.35, fontFace: "Calibri", bold: true, fontSize: 11, color: GREEN, margin: 0 });

  s.addShape("roundRect", { x: cardX, y: 4.15, w: cardW, h: 1.9, rectRadius: 0.12, fill: { color: RED_BG }, line: { type: "none" } });
  iconCircle(s, cardX + 0.25, 4.4, 0.5, RED, "✕", WHITE, 18);
  s.addText("“Clearance items can be returned within 30 days too...”", {
    x: cardX + 0.95, y: 4.38, w: cardW - 1.2, h: 0.9, fontFace: "Calibri", fontSize: 12.5, color: INK, margin: 0,
  });
  s.addText("Reads just as confident. Is false.", { x: cardX + 0.25, y: 5.55, w: cardW - 0.5, h: 0.35, fontFace: "Calibri", bold: true, fontSize: 11, color: RED, margin: 0 });
  pageNum(s, 2);
}

// ---------- Slide 3: How teams check today ----------
{
  const s = baseSlide(WHITE);
  kicker(s, "Act One — The Old Way", MUTED);
  title(s, "“Someone asks it 30 questions and says it looks right.”");

  const items = [
    ["○", "20–30 questions, read by hand", "No scale past a handful of test cases before a release."],
    ["○", "Same 2–3 tired reviewers", "Subjective, inconsistent, and skimmed at the end of a long day."],
    ["○", "Never repeated after launch", "Quality regressions go unnoticed — until someone complains."],
  ];
  let x = 0.7;
  const cw = 3.85, gap = 0.25;
  items.forEach(([g, h, b]) => {
    s.addShape("roundRect", { x, y: 2.2, w: cw, h: 2.5, rectRadius: 0.12, fill: { color: CARD_BG }, line: { type: "none" }, shadow: freshShadow() });
    iconCircle(s, x + 0.3, 2.5, 0.6, NAVY, "!", WHITE, 22);
    s.addText(h, { x: x + 0.3, y: 3.3, w: cw - 0.6, h: 0.7, fontFace: "Calibri", bold: true, fontSize: 14.5, color: INK, margin: 0, lineSpacingMultiple: 1.1 });
    s.addText(b, { x: x + 0.3, y: 3.95, w: cw - 0.6, h: 0.7, fontFace: "Calibri", fontSize: 11.5, color: MUTED, margin: 0, lineSpacingMultiple: 1.2 });
    x += cw + gap;
  });

  s.addText("The first person to catch a bad answer is usually the customer.", {
    x: 0.7, y: 5.15, w: 11.5, h: 0.6, fontFace: "Cambria", italic: true, fontSize: 19, color: NAVY, margin: 0,
  });
  pageNum(s, 3);
}

// ---------- Slide 4: Introducing the platform ----------
{
  const s = baseSlide(NAVY);
  kicker(s, "Act Two — Meet the Platform", "8FA3E8");
  s.addText("Every answer, scored\nbefore it reaches a customer.", {
    x: 0.7, y: 1.5, w: 8.2, h: 1.9, fontFace: "Cambria", bold: true, fontSize: 33, color: WHITE, margin: 0, lineSpacingMultiple: 1.08,
  });
  s.addText(
    "Instead of a person reading an answer and guessing, this platform scores every answer against four industry-standard metrics — automatically, in seconds, from one question to a full regression suite.",
    { x: 0.7, y: 3.35, w: 7.6, h: 1.5, fontFace: "Calibri", fontSize: 15, color: ICE, margin: 0, lineSpacingMultiple: 1.3 }
  );
  s.addShape("roundRect", { x: 0.7, y: 5.05, w: 5.6, h: 0.85, rectRadius: 0.1, fill: { color: "263480" }, line: { type: "none" } });
  s.addText("Built on RAGAS — a peer-reviewed, open-source standard. Not a black box.", {
    x: 0.95, y: 5.05, w: 5.1, h: 0.85, fontFace: "Calibri", bold: true, fontSize: 12.5, color: ICE, valign: "middle", margin: 0,
  });

  // right: mini flow
  const fx = 9.0, fy = 1.6, fw = 3.3;
  const steps = ["Connect", "Configure", "Evaluate"];
  const descs = ["Point at your RAG API, or paste answers manually", "Pick a provider, save credentials once", "Run one question or a full batch — get a scored report"];
  let fyy = fy;
  steps.forEach((st, i) => {
    s.addShape("roundRect", { x: fx, y: fyy, w: fw, h: 1.5, rectRadius: 0.1, fill: { color: "263480" }, line: { type: "none" } });
    iconCircle(s, fx + 0.25, fyy + 0.25, 0.5, ICE, String(i + 1), NAVY, 18);
    s.addText(st, { x: fx + 0.95, y: fyy + 0.18, w: fw - 1.15, h: 0.4, fontFace: "Calibri", bold: true, fontSize: 15, color: WHITE, margin: 0 });
    s.addText(descs[i], { x: fx + 0.95, y: fyy + 0.58, w: fw - 1.15, h: 0.85, fontFace: "Calibri", fontSize: 10.5, color: ICE, margin: 0, lineSpacingMultiple: 1.15 });
    fyy += 1.7;
  });
  pageNum(s, 4, true);
}

// ---------- Slide 5: Four metrics ----------
{
  const s = baseSlide(WHITE);
  kicker(s, "Act Two — The Scorecard", MUTED);
  title(s, "Four metrics. Plain-language questions.");

  const metrics = [
    ["Response\nRelevancy", "Did it actually answer what was asked?"],
    ["Faithfulness", "Is it grounded in real source data — or invented?"],
    ["Context\nPrecision", "Is it retrieving the right material, without noise?"],
    ["Context\nRecall", "Is it finding everything it needs to answer fully?"],
  ];
  const gx = 0.7, gy = 2.15, gw = 5.75, gh = 2.25, gap = 0.3;
  metrics.forEach(([h, b], i) => {
    const col = i % 2, row = Math.floor(i / 2);
    const x = gx + col * (gw + gap);
    const y = gy + row * (gh + gap);
    s.addShape("roundRect", { x, y, w: gw, h: gh, rectRadius: 0.12, fill: { color: CARD_BG }, line: { type: "none" }, shadow: freshShadow() });
    iconCircle(s, x + 0.35, y + 0.35, 0.7, NAVY, String(i + 1), WHITE, 24);
    s.addText(h, { x: x + 1.3, y: y + 0.28, w: gw - 1.6, h: 0.85, fontFace: "Cambria", bold: true, fontSize: 17, color: INK, margin: 0, lineSpacingMultiple: 1.05 });
    s.addText(b, { x: x + 0.35, y: y + 1.3, w: gw - 0.7, h: 0.8, fontFace: "Calibri", fontSize: 13, color: MUTED, margin: 0, lineSpacingMultiple: 1.25 });
  });
  pageNum(s, 5);
}

// ---------- Slide 6: The clean win ----------
{
  const s = baseSlide(WHITE);
  kicker(s, "Act Three — The Clean Win", MUTED);
  title(s, "When it's right, you'll know exactly why.");

  s.addShape("roundRect", { x: 0.7, y: 2.1, w: 5.9, h: 2.6, rectRadius: 0.12, fill: { color: CARD_BG }, line: { type: "none" } });
  s.addText("QUERY", { x: 1.0, y: 2.3, w: 5.3, h: 0.3, fontFace: "Calibri", bold: true, fontSize: 10, color: MUTED, charSpacing: 1, margin: 0 });
  s.addText("“What's your return window for a standard unopened item?”", { x: 1.0, y: 2.6, w: 5.3, h: 0.7, fontFace: "Cambria", italic: true, fontSize: 13.5, color: INK, margin: 0, lineSpacingMultiple: 1.2 });
  s.addText("BOT'S ANSWER", { x: 1.0, y: 3.35, w: 5.3, h: 0.3, fontFace: "Calibri", bold: true, fontSize: 10, color: MUTED, charSpacing: 1, margin: 0 });
  s.addText("“You can return a standard, unused item within 30 days of delivery for a full refund, as long as it's in its original packaging.”", { x: 1.0, y: 3.65, w: 5.3, h: 0.95, fontFace: "Calibri", fontSize: 12.5, color: INK, margin: 0, lineSpacingMultiple: 1.25 });

  const tiles = [["Response\nRelevancy", 0.96, "EXCELLENT"], ["Faithfulness", 0.94, "EXCELLENT"], ["Context\nPrecision", 0.91, "EXCELLENT"], ["Context\nRecall", 0.88, "GOOD"]];
  let tx = 7.0;
  tiles.forEach(([l, sc, b]) => { scoreTile(s, tx, 2.1, 1.42, 1.9, l, sc, b, true); tx += 1.55; });

  s.addText("Green across the board. Good — but not the interesting part yet.", {
    x: 7.0, y: 4.25, w: 5.6, h: 0.5, fontFace: "Calibri", italic: true, fontSize: 13, color: MUTED, margin: 0,
  });
  pageNum(s, 6);
}

// ---------- Slide 7: The catch ----------
{
  const s = baseSlide(WHITE);
  kicker(s, "Act Four — The Catch", RED);
  title(s, "Confidently wrong. And now, caught.");

  s.addShape("roundRect", { x: 0.7, y: 2.1, w: 5.9, h: 2.6, rectRadius: 0.12, fill: { color: RED_BG }, line: { type: "none" } });
  s.addText("QUERY", { x: 1.0, y: 2.3, w: 5.3, h: 0.3, fontFace: "Calibri", bold: true, fontSize: 10, color: RED, charSpacing: 1, margin: 0 });
  s.addText("“I bought a clearance jacket — can I return it within 30 days like everything else?”", { x: 1.0, y: 2.6, w: 5.3, h: 0.7, fontFace: "Cambria", italic: true, fontSize: 13.5, color: INK, margin: 0, lineSpacingMultiple: 1.2 });
  s.addText("BOT'S ANSWER", { x: 1.0, y: 3.35, w: 5.3, h: 0.3, fontFace: "Calibri", bold: true, fontSize: 10, color: RED, charSpacing: 1, margin: 0 });
  s.addText("“Yes, clearance items can be returned within 30 days for a full refund, just like standard items.”  ―  Actually false: clearance is final sale.", { x: 1.0, y: 3.65, w: 5.3, h: 1.0, fontFace: "Calibri", fontSize: 12.5, color: INK, margin: 0, lineSpacingMultiple: 1.25 });

  const tiles = [["Response\nRelevancy", 0.91, "EXCELLENT", true], ["Faithfulness", 0.32, "POOR", false], ["Context\nPrecision", 0.88, "GOOD", true], ["Context\nRecall", 0.83, "GOOD", true]];
  let tx = 7.0;
  tiles.forEach(([l, sc, b, good]) => { scoreTile(s, tx, 2.1, 1.42, 1.9, l, sc, b, good); tx += 1.55; });

  s.addText("Relevancy and retrieval are fine. Faithfulness collapses — the answer contradicts its own source.", {
    x: 7.0, y: 4.25, w: 5.6, h: 0.6, fontFace: "Calibri", italic: true, fontSize: 12.5, color: MUTED, margin: 0, lineSpacingMultiple: 1.2,
  });
  pageNum(s, 7);
}

// ---------- Slide 8: Diagnosis ----------
{
  const s = baseSlide(NAVY);
  kicker(s, "Act Four — Not Just a Red Flag", "8FA3E8");
  title(s, "It tells you where to look next.", { color: WHITE });
  s.addText(
    "Retrieval was healthy — the right paragraph was there. But the generated answer wasn't grounded in it. So the diagnosis narrows to two places.",
    { x: 0.7, y: 1.95, w: 11.5, h: 0.7, fontFace: "Calibri", fontSize: 14.5, color: ICE, margin: 0, lineSpacingMultiple: 1.25 }
  );

  const stages = [
    ["Retrieval", "Healthy", GREEN, "Pulled the right paragraph, no noise."],
    ["Augmentation", "Check here", "F2C744", "How the retrieved text was handed into the prompt."],
    ["Generation", "Check here", "F2C744", "How the model used it once it got there."],
  ];
  let x = 0.7;
  stages.forEach(([h, tag, color, d]) => {
    s.addShape("roundRect", { x, y: 2.95, w: 3.85, h: 3.0, rectRadius: 0.12, fill: { color: "263480" }, line: { type: "none" } });
    s.addShape("roundRect", { x: x + 0.3, y: 3.25, w: 1.7, h: 0.4, rectRadius: 0.2, fill: { color }, line: { type: "none" } });
    s.addText(tag, { x: x + 0.3, y: 3.25, w: 1.7, h: 0.4, align: "center", valign: "middle", fontFace: "Calibri", bold: true, fontSize: 10.5, color: NAVY, margin: 0 });
    s.addText(h, { x: x + 0.3, y: 3.85, w: 3.25, h: 0.55, fontFace: "Cambria", bold: true, fontSize: 20, color: WHITE, margin: 0 });
    s.addText(d, { x: x + 0.3, y: 4.5, w: 3.25, h: 1.2, fontFace: "Calibri", fontSize: 12.5, color: ICE, margin: 0, lineSpacingMultiple: 1.3 });
    x += 4.1;
  });

  s.addText("A hover tooltip shows the judge model's own reasoning for the score — every time.", {
    x: 0.7, y: 6.15, w: 11.5, h: 0.5, fontFace: "Cambria", italic: true, fontSize: 15, color: "8FA3E8", margin: 0,
  });
  pageNum(s, 8, true);
}

// ---------- Slide 9: Scaling to batch ----------
{
  const s = baseSlide(WHITE);
  kicker(s, "Act Five — From One Question to All of Them", MUTED);
  title(s, "One click. Unattended. At scale.");

  // stat comparison
  s.addShape("roundRect", { x: 0.7, y: 2.1, w: 3.5, h: 1.75, rectRadius: 0.12, fill: { color: RED_BG }, line: { type: "none" } });
  s.addText("BY HAND", { x: 0.95, y: 2.28, w: 3, h: 0.3, fontFace: "Calibri", bold: true, fontSize: 10, color: RED, charSpacing: 1, margin: 0 });
  s.addText("An afternoon", { x: 0.95, y: 2.6, w: 3, h: 0.6, fontFace: "Cambria", bold: true, fontSize: 24, color: INK, margin: 0 });
  s.addText("to read 24 cases carefully", { x: 0.95, y: 3.2, w: 3, h: 0.5, fontFace: "Calibri", fontSize: 11.5, color: MUTED, margin: 0 });

  s.addShape("roundRect", { x: 4.4, y: 2.1, w: 3.5, h: 1.75, rectRadius: 0.12, fill: { color: GREEN_BG }, line: { type: "none" } });
  s.addText("WITH THE PLATFORM", { x: 4.65, y: 2.28, w: 3, h: 0.3, fontFace: "Calibri", bold: true, fontSize: 10, color: GREEN, charSpacing: 1, margin: 0 });
  s.addText("One click", { x: 4.65, y: 2.6, w: 3, h: 0.6, fontFace: "Cambria", bold: true, fontSize: 24, color: INK, margin: 0 });
  s.addText("runs unattended in the background", { x: 4.65, y: 3.2, w: 3, h: 0.5, fontFace: "Calibri", fontSize: 11.5, color: MUTED, margin: 0 });

  // batch summary tiles
  const tiles = [["Response Relevancy", 0.93, "EXCELLENT", true], ["Faithfulness", 0.81, "GOOD", true], ["Context Precision", 0.88, "GOOD", true], ["Context Recall", 0.79, "GOOD", true]];
  s.addText("BATCH AVERAGE — 24 TEST CASES", { x: 8.2, y: 2.1, w: 4.4, h: 0.3, fontFace: "Calibri", bold: true, fontSize: 10, color: MUTED, charSpacing: 1, margin: 0 });
  let ty = 2.5;
  tiles.forEach(([l, sc, b]) => {
    s.addShape("roundRect", { x: 8.2, y: ty, w: 4.4, h: 0.62, rectRadius: 0.08, fill: { color: CARD_BG }, line: { type: "none" } });
    s.addText(l, { x: 8.4, y: ty, w: 2.4, h: 0.62, valign: "middle", fontFace: "Calibri", fontSize: 11.5, color: INK, margin: 0 });
    s.addText(sc.toFixed(2) + "  " + b, { x: 10.7, y: ty, w: 1.8, h: 0.62, valign: "middle", align: "right", fontFace: "Calibri", bold: true, fontSize: 11, color: GREEN, margin: 0 });
    ty += 0.72;
  });

  s.addShape("roundRect", { x: 0.7, y: 4.35, w: 7.3, h: 1.85, rectRadius: 0.12, fill: { color: NAVY }, line: { type: "none" } });
  s.addText("SUGGESTED FOCUS ACROSS THIS BATCH", { x: 1.0, y: 4.55, w: 6.7, h: 0.3, fontFace: "Calibri", bold: true, fontSize: 10.5, color: "8FA3E8", charSpacing: 1, margin: 0 });
  s.addText("Augmentation + Generation — 3 of 24 cases share the exact fingerprint of the clearance question. Retrieval — 2 of 24 point at knowledge-base gaps instead.", {
    x: 1.0, y: 4.9, w: 6.7, h: 1.2, fontFace: "Calibri", fontSize: 13.5, color: WHITE, margin: 0, lineSpacingMultiple: 1.3,
  });
  pageNum(s, 9);
}

// ---------- Slide 10: The paper trail ----------
{
  const s = baseSlide(WHITE);
  kicker(s, "Act Six — The Paper Trail", MUTED);
  title(s, "Evidence, not a screenshot that gets lost.");

  const items = [
    ["Shareable", "Every case — single and batch — in one report: input, answer, contexts, all four scores, timestamped."],
    ["Sign-off ready", "Hand it to whoever approves a release. It stays around as a record."],
    ["Accumulates", "UI runs and CI test-suite runs land in the same place — trend quality across releases, not just once."],
  ];
  let x = 0.7;
  items.forEach(([h, b]) => {
    s.addShape("roundRect", { x, y: 2.2, w: 3.85, h: 3.4, rectRadius: 0.12, fill: { color: CARD_BG }, line: { type: "none" }, shadow: freshShadow() });
    iconCircle(s, x + 0.3, 2.5, 0.6, NAVY, "✓", WHITE, 20);
    s.addText(h, { x: x + 0.3, y: 3.3, w: 3.25, h: 0.5, fontFace: "Cambria", bold: true, fontSize: 18, color: INK, margin: 0 });
    s.addText(b, { x: x + 0.3, y: 3.85, w: 3.25, h: 1.6, fontFace: "Calibri", fontSize: 12.5, color: MUTED, margin: 0, lineSpacingMultiple: 1.3 });
    x += 3.95 + 0.15;
  });
  pageNum(s, 10);
}

// ---------- Slide 11: What's next ----------
{
  const s = baseSlide(NAVY);
  kicker(s, "Act Seven — What's Next", "8FA3E8");
  s.addText("On the roadmap:\nQuery Generator", { x: 0.7, y: 1.6, w: 7.5, h: 1.7, fontFace: "Cambria", bold: true, fontSize: 32, color: WHITE, margin: 0, lineSpacingMultiple: 1.08 });
  s.addText(
    "Today, someone still has to hand-write the test questions and correct answers before evaluation can run. Query Generator reads a knowledge base directly and generates that test set automatically — so even that manual step goes away.",
    { x: 0.7, y: 3.35, w: 7.3, h: 1.7, fontFace: "Calibri", fontSize: 15, color: ICE, margin: 0, lineSpacingMultiple: 1.35 }
  );

  s.addShape("roundRect", { x: 8.6, y: 1.6, w: 3.9, h: 4.3, rectRadius: 0.12, fill: { color: "263480" }, line: { type: "none" } });
  s.addText("TODAY", { x: 8.9, y: 1.85, w: 3.3, h: 0.3, fontFace: "Calibri", bold: true, fontSize: 10, color: "8FA3E8", charSpacing: 1, margin: 0 });
  s.addText("Someone writes\ntest cases by hand", { x: 8.9, y: 2.15, w: 3.3, h: 0.9, fontFace: "Cambria", bold: true, fontSize: 16, color: WHITE, margin: 0, lineSpacingMultiple: 1.15 });
  s.addShape("line", { x: 10.55, y: 3.2, w: 0, h: 0.55, line: { color: "8FA3E8", width: 2, dashType: "dash" } });
  s.addText("NEXT", { x: 8.9, y: 3.85, w: 3.3, h: 0.3, fontFace: "Calibri", bold: true, fontSize: 10, color: "F2C744", charSpacing: 1, margin: 0 });
  s.addText("Platform generates\nthem from your content", { x: 8.9, y: 4.15, w: 3.3, h: 0.9, fontFace: "Cambria", bold: true, fontSize: 16, color: WHITE, margin: 0, lineSpacingMultiple: 1.15 });
  pageNum(s, 11, true);
}

// ---------- Slide 12: The Close ----------
{
  const s = baseSlide(WHITE);
  kicker(s, "Act Eight — The Close", MUTED);
  title(s, "Find it in twenty seconds. Before launch.");

  const stats = [["4", "metrics scored automatically"], ["1", "open standard — RAGAS, not a black box"], ["Minutes", "not afternoons, to check a full batch"]];
  let x = 0.7;
  stats.forEach(([n, l]) => {
    s.addShape("roundRect", { x, y: 2.15, w: 3.85, h: 1.7, rectRadius: 0.12, fill: { color: CARD_BG }, line: { type: "none" } });
    s.addText(n, { x: x + 0.25, y: 2.25, w: 3.35, h: 0.9, fontFace: "Cambria", bold: true, fontSize: 40, color: NAVY, margin: 0 });
    s.addText(l, { x: x + 0.25, y: 3.15, w: 3.35, h: 0.6, fontFace: "Calibri", fontSize: 12, color: MUTED, margin: 0, lineSpacingMultiple: 1.2 });
    x += 3.95 + 0.15;
  });

  s.addShape("roundRect", { x: 0.7, y: 4.15, w: 11.9, h: 2.05, rectRadius: 0.12, fill: { color: NAVY }, line: { type: "none" }, shadow: freshShadow() });
  s.addText(
    "The question isn't whether this works — it's whether a team wants to find its next “clearance jacket” problem in twenty seconds, before launch, or the way most teams still do: after a customer already has.",
    { x: 1.05, y: 4.4, w: 11.2, h: 1.55, fontFace: "Cambria", italic: true, fontSize: 19, color: WHITE, margin: 0, lineSpacingMultiple: 1.3, valign: "middle" }
  );
  pageNum(s, 12);
}

// ---------- Slide 13: Thank you ----------
{
  const s = baseSlide(NAVY);
  s.addShape("ellipse", { x: -2.5, y: 4.2, w: 6, h: 6, fill: { color: "263480" }, line: { type: "none" } });
  s.addText("Thank You", { x: 0.9, y: 2.7, w: 8, h: 1.2, fontFace: "Cambria", bold: true, fontSize: 46, color: WHITE, margin: 0 });
  s.addText("Questions & discussion", { x: 0.9, y: 3.75, w: 8, h: 0.6, fontFace: "Calibri", fontSize: 18, color: ICE, margin: 0 });
  s.addText("RAG Evaluation Platform  ·  Built on RAGAS", { x: 0.9, y: 6.6, w: 8, h: 0.4, fontFace: "Calibri", fontSize: 12, color: "8FA3E8", margin: 0 });
}

pres.writeFile({ fileName: "/sessions/clever-relaxed-cannon/mnt/RAGAS/Demo/RAGAS_Demo_Story_Slides.pptx" }).then(() => {
  console.log("done");
});
