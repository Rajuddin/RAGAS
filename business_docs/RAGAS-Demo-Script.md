# Demo Script — "Catching the Answer That Sounds Right But Isn't"

**Cast:** You (presenter) · **Maya Chen**, VP of Customer Experience at
**Northwind Outfitters** (fictional client — swap in the real company/product
before you present)

**Setup, before the call:**
- `config.yaml` already has Maya's (or your demo) Azure OpenAI credentials saved,
  so you never type a key on screen.
- The RAG API is pointed at Northwind's staging support-bot endpoint (or swap
  in manual mode — see the aside after Scene 4 if you don't have a live API
  to hook up).
- A batch file, `northwind_tickets.json`, is sitting in `test_data/` with 24
  real (anonymized) support questions and their correct answers.
- Have `allure-results/` cleared beforehand so the report you generate live is
  clean.

Runtime: ~12 minutes.

---

## Scene 1 — The problem, in Maya's own words

*(Screen: nothing yet — just talking.)*

**You:** Before I open anything — you told me last week that you're about to
launch Ari, your support chatbot, to all customers, but the team's still doing
QA by hand. Walk me through that process real quick.

**Maya:** Basically... someone on the team asks it twenty or thirty questions,
reads the answers, and says "yeah, that looks right." We do that before every
release.

**You:** And how confident are you that "looks right" catches everything?

**Maya:** *(pause)* Honestly? Not very. It's the same three people, they get
tired, they skim. We've definitely shipped answers that were subtly wrong
before.

**You:** That's exactly the gap this closes. It's not a replacement for your
team's judgment — it's a way to make "looks right" objective, and to run it on
every question, every release, automatically. Let me show you on your own
policy docs.

---

## Scene 2 — The dashboard

*(Screen: open the app. The Dashboard loads.)*

**You:** This is the landing page. One card per evaluation tool. Today there's
one live tool — RAG Evaluation — and one on the roadmap.

*(Point at the greyed "Query Generator" card.)*

**Maya:** What's that one?

**You:** Good eye — I'll come back to it at the end, it actually solves the
next problem you're going to have. Let's open RAG Evaluation.

*(Click "Open →".)*

---

## Scene 3 — The setup nobody has to think about twice

*(Screen: the evaluation page loads, sidebar visible.)*

**You:** Left side is all setup. Provider's already Azure OpenAI, your key and
deployment are already saved — you'd only touch this once, the first time you
set it up. Same for the metrics list here —

*(Point at the Metrics multiselect in the sidebar.)*

**You:** — four scores, all switched on: Response Relevancy, Faithfulness,
Context Precision, Context Recall. I'll explain each as we hit it, not before —
they'll make more sense with a real answer in front of us.

---

## Scene 4 — The clean win

*(Screen: Evaluation Mode set to "Single Case". Type into Query and Ground Truth.)*

**You:** Let's start with something easy — a question a customer actually
asked Ari last month.

- **Query:** *"What's your return window for a standard unopened item?"*
- **Ground Truth:** *"Standard, unused items can be returned within 30 days of
  delivery for a full refund, provided they're in original packaging."*

*(Click "Run Evaluation." The RAG API Response panel expands to show what Ari
actually said.)*

**Ari's answer:** *"You can return a standard, unused item within 30 days of
delivery for a full refund, as long as it's in its original packaging."*

*(Scores render: four tiles, all green.)*

| Metric | Score | Label |
|---|---|---|
| Response Relevancy | 0.96 | EXCELLENT |
| Faithfulness | 0.94 | EXCELLENT |
| Context Precision | 0.91 | EXCELLENT |
| Context Recall | 0.88 | GOOD |

**You:** Green across the board. Response Relevancy — did it actually answer
what was asked. Faithfulness — is it grounded in your real return-policy doc,
not invented. Context Precision and Recall — did the system pull the right
paragraph, and all of it. This is what "looks right" *and actually is right*
looks like. Good, but not the interesting part yet.

> **No live RAG API for the demo?** Uncheck "Fetch answer & contexts from RAG
> API" and paste Ari's answer and the policy paragraph into Generated Answer /
> Contexts by hand — same scores, same story, you just play the role of the
> API for thirty seconds.

---

## Scene 5 — The catch

*(Screen: still on Single Case. Clear the fields, type a new query.)*

**You:** Here's the one that matters. Your policy doc has a carve-out —
clearance items are final sale. Let's ask Ari about that specifically.

- **Query:** *"I bought a clearance jacket — can I return it within 30 days
  like everything else?"*
- **Ground Truth:** *"No — items marked Clearance are final sale and are not
  eligible for return or exchange."*

*(Click "Run Evaluation.")*

**Ari's answer:** *"Yes, clearance items can be returned within 30 days for a
full refund, just like standard items."*

**Maya:** *(reading the answer)* ...That's just wrong. That's confidently,
completely wrong.

**You:** And it *reads* fine — confident, on-topic, well-formatted. This is
exactly the answer a tired reviewer skims past at 4pm on a Friday. Watch the
scores.

*(Scores render — three green, one red.)*

| Metric | Score | Label |
|---|---|---|
| Response Relevancy | 0.91 | EXCELLENT |
| Faithfulness | 0.32 | POOR |
| Context Precision | 0.88 | GOOD |
| Context Recall | 0.83 | GOOD |

**You:** Look at this pattern. Relevancy's fine — it did answer the question
asked. Context Precision and Recall are fine — the retrieval system pulled the
clearance paragraph, the right material was right there. But Faithfulness
cratered, because the generated answer directly contradicts the context it was
given.

*(Scroll to "Suggested Focus: Augmentation + Generation".)*

**You:** And it doesn't just flag it — it tells you *where* to look. Since
retrieval was healthy but the final answer wasn't grounded in it, the tool
narrows this to two places: how the retrieved text got handed into the prompt,
or how the model used it once it got there. Not "something's wrong somewhere"
— a specific, narrowed starting point.

*(Hover the Faithfulness tile — tooltip shows the judge model's own reasoning
for the low score.)*

**Maya:** So it's not just a number, it's telling you *why*.

**You:** Every time. That's the difference between "the bot failed QA" and
"the bot failed QA, and here's the sentence that proves it, and here's the two
places to go check first."

---

## Scene 6 — Scaling it up

**You:** Two questions is a demo. Here's what it looks like on your actual
ticket history.

*(Switch Evaluation Mode to "Batch (JSON file)". Upload `northwind_tickets.json`.)*

**You:** 24 real questions from your support queue, with the correct answer
for each already attached. One click.

*(Click "Run Batch Evaluation." Progress bar climbs live: "Evaluated 9/24 test
cases..." with a ✅/❌/🔄 status line per case.)*

**Maya:** How long does this normally take your team by hand?

**You:** For 24 cases, read carefully? The better part of an afternoon. This
runs unattended — kick it off before a release and go do something else.

*(Batch finishes. Summary tiles render.)*

| Metric | Average | Label |
|---|---|---|
| Response Relevancy | 0.93 | EXCELLENT |
| Faithfulness | 0.81 | GOOD |
| Context Precision | 0.88 | GOOD |
| Context Recall | 0.79 | GOOD |

**You:** Faithfulness pulled the average down a bit — let's see why.

*(Point at "Suggested focus areas across this batch: Augmentation + Generation
(3/24 test cases), Retrieval (2/24 test cases)".)*

**You:** Three of your 24 cases have the exact same fingerprint as the
clearance question we just saw live. Two more point at retrieval gaps instead
— worth checking whether your knowledge base is missing something for those.
Check "Show full details" and you get every one of those 24, individually, with
the same breakdown we just walked through by hand.

---

## Scene 7 — The paper trail

*(Sidebar: click "Generate & Open Report," then "Open Allure Report ↗.")*

**You:** And this is what makes it real for sign-off, not just for us in this
call. Every case we just ran — the single ones and the batch — is now in a
shareable report: input, answer, contexts, all four scores, timestamped. You
can hand this to whoever needs to approve a release, and it stays around as a
record, not a Slack screenshot that gets lost in a week.

**Maya:** Does it keep history across releases, or just this run?

**You:** It accumulates — every run from the UI, and every run from an
automated test suite if your engineers wire this into CI, lands in the same
place. So you can watch Faithfulness trend over four releases, not just check
it once before this one.

---

## Scene 8 — The close

*(Back to the Dashboard. Point at "Query Generator" again.)*

**You:** Remember this? Today, someone still has to write the 24 questions and
correct answers in that JSON file by hand. This tool — on the roadmap — reads
your own help center content and generates that test set for you. So the
"someone has to sit down and write test cases" step goes away too.

**Maya:** When's that ready?

**You:** *(your actual roadmap answer here — don't overpromise on a date you
haven't committed to)*

**You:** But everything you saw today is real, running, and works against your
data right now. The question in front of you isn't "will this work" — it's
whether you want your team finding the clearance-jacket problem the way we
just did, in twenty seconds, before launch — or finding it the way you found
the last one.

---

## If Maya asks... (quick answers, not scripted lines)

- **"Does it work with [some other LLM], not just Azure?"** — Yes at the
  configuration layer; OpenAI and Azure AI Foundry are both already built in,
  today's sidebar just has Azure switched on. Turning on a second option is a
  small change, not new development.
- **"Can this run in our CI pipeline, not just this UI?"** — Yes — the same
  four metrics run from an automated pytest suite, writing into the same
  report you just saw. The UI is for ad-hoc spot checks; the pytest path is
  for "run this on every merge."
- **"What if our support API doesn't return JSON in this exact shape?"** — The
  tool auto-detects most common shapes, and there's a manual field-mapping
  panel (Answer Field / Contexts Field, with dotted-path support) for anything
  nested or unusual — that's a five-minute setup, not an integration project.
- **"What does POOR vs MODERATE actually mean?"** — A fixed scale, not a
  vibe: EXCELLENT ≥ 0.85, GOOD ≥ 0.70, MODERATE ≥ 0.50, POOR below that, the
  same four bands on every score, every run.
