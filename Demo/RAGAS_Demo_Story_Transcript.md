# Demo Transcript — "The Answer That Sounds Right"
### A storytelling walkthrough of the RAG Evaluation Platform

Runtime: ~10–12 minutes · Delivered as a narrated monologue with live product demo · No scripted client dialogue required — pause for questions wherever marked **[PAUSE]**

---

## Act 1 — The Problem

*(Screen: nothing yet. Just talking.)*

Every team that ships an AI chatbot or knowledge assistant eventually runs into
the same quiet problem.

The bot answers fast. It answers confidently. It's never once said "I don't
know." And that's exactly what makes it dangerous — because a wrong answer
from this kind of system doesn't look wrong. It's grammatically perfect,
on-topic, well-formatted. It reads exactly like a right answer.

So how do most teams check it before launch? Someone sits down, asks the bot
twenty or thirty questions, reads the answers, and says "yeah, that looks
right." That's the QA process at most companies shipping this technology
today. Not because people are careless — because there hasn't been a better
option.

Here's the problem with "looks right": it's the same two or three tired people
doing it, it doesn't scale past a few dozen questions, and it never gets
repeated after launch. So the first time anyone finds out an answer was wrong
is when a customer does.

**[PAUSE]** *That's the gap this tool closes.*

---

## Act 2 — Meet the Platform

*(Screen: open the app. Dashboard loads.)*

This is the landing page — one card per tool. Today, one is live: **RAG
Evaluation**. The second, **Query Generator**, is on the roadmap — I'll come
back to it at the end, because it solves the next problem you're going to
have, right after this one.

*(Click "Open →".)*

Here's the idea in one sentence: instead of a person reading an answer and
guessing whether it's right, this platform scores every answer against four
industry-standard metrics — automatically, in seconds, at any scale from one
question to a thousand.

It's built on **RAGAS**, a peer-reviewed, open-source evaluation framework —
not a black box, not a proprietary scoring trick. The same standard used
across the industry to measure retrieval-augmented AI systems.

*(Point at sidebar.)*

Setup happens once. Pick a provider — OpenAI, Azure OpenAI, or Azure AI
Foundry — save your credentials, and you never touch this again. Same with the
four metrics: **Response Relevancy, Faithfulness, Context Precision, Context
Recall.** I won't explain those yet — they'll make a lot more sense once
you've seen a real answer scored.

---

## Act 3 — The Clean Win

*(Evaluation Mode: Single Case. Type into Query and Ground Truth.)*

Let's start easy — a real kind of question a customer might ask a support
bot.

- **Query:** *"What's your return window for a standard unopened item?"*
- **Ground Truth:** *"Standard, unused items can be returned within 30 days of
  delivery for a full refund, provided they're in original packaging."*

*(Click "Run Evaluation." The RAG API Response panel expands.)*

**Bot's answer:** *"You can return a standard, unused item within 30 days of
delivery for a full refund, as long as it's in its original packaging."*

*(Scores render — four tiles, all green.)*

| Metric | Score | Label |
|---|---|---|
| Response Relevancy | 0.96 | EXCELLENT |
| Faithfulness | 0.94 | EXCELLENT |
| Context Precision | 0.91 | EXCELLENT |
| Context Recall | 0.88 | GOOD |

Green across the board. And now the metrics mean something concrete:

- **Response Relevancy** — did it actually answer what was asked?
- **Faithfulness** — is it grounded in the real source document, or invented?
- **Context Precision** — did the system pull the *right* material, without noise?
- **Context Recall** — did it pull *everything* it needed to answer fully?

This is what "looks right" looks like *when it's also actually right.* Good.
But not the interesting part yet.

---

## Act 4 — The Catch

*(Still Single Case. Clear fields, type a new query.)*

Here's the one that matters. Say the policy has a carve-out — clearance items
are final sale. Let's ask about that specifically.

- **Query:** *"I bought a clearance jacket — can I return it within 30 days
  like everything else?"*
- **Ground Truth:** *"No — items marked Clearance are final sale and are not
  eligible for return or exchange."*

*(Click "Run Evaluation.")*

**Bot's answer:** *"Yes, clearance items can be returned within 30 days for a
full refund, just like standard items."*

That's wrong. Confidently, completely wrong. And it *reads* fine — same
confident tone, same clean formatting as the correct answer a moment ago. This
is exactly the sentence a tired reviewer skims past at the end of a long
review session. Watch the scores instead.

*(Scores render — three green, one red.)*

| Metric | Score | Label |
|---|---|---|
| Response Relevancy | 0.91 | EXCELLENT |
| Faithfulness | 0.32 | POOR |
| Context Precision | 0.88 | GOOD |
| Context Recall | 0.83 | GOOD |

Look at the pattern, not just the color. Relevancy's fine — it answered the
question asked. Context Precision and Recall are fine — retrieval did its job,
the clearance paragraph was right there. But Faithfulness collapsed, because
the generated answer directly contradicts the material it was given.

*(Scroll to "Suggested Focus.")*

And it doesn't stop at the red tile — it tells you *where to look next*. Since
retrieval was healthy but the final answer wasn't grounded in it, the tool
narrows the problem to two places: how the retrieved text was handed into the
prompt, or how the model used it once it got there. Not "something's wrong
somewhere" — a specific, narrowed starting point, with a hover tooltip
explaining the judge model's own reasoning for the score.

**[PAUSE]** *This is the whole pitch, in one moment: not just a number — a
diagnosis.*

---

## Act 5 — From One Question to All of Them

Two questions is a demo. Here's what happens on a real ticket history.

*(Switch to Batch. Upload a JSON file of real test cases.)*

Every row needs just a question and a correct answer — the platform fetches
the bot's actual response and scores it. One click.

*(Click "Run Batch Evaluation." Progress bar climbs live, per-case status.)*

For a batch like this, read carefully by hand, that's the better part of an
afternoon for a person. This runs unattended — kick it off before a release
and go do something else.

*(Batch finishes. Summary tiles render.)*

| Metric | Average | Label |
|---|---|---|
| Response Relevancy | 0.93 | EXCELLENT |
| Faithfulness | 0.81 | GOOD |
| Context Precision | 0.88 | GOOD |
| Context Recall | 0.79 | GOOD |

Faithfulness pulled the average down slightly — and the tool already knows
why: *"Suggested focus areas across this batch: Augmentation + Generation (3
of N cases), Retrieval (2 of N cases)."*

Three cases share the exact fingerprint of the clearance question we just saw
live. Two more point at gaps in retrieval instead — worth checking whether the
knowledge base is missing something there. Every one of those cases expands
into the same full breakdown we just walked through by hand.

---

## Act 6 — The Paper Trail

*(Sidebar: "Generate & Open Report" → "Open Allure Report ↗".)*

And this is what makes it real beyond the demo. Every case just run — single
and batch — lands in a shareable report: input, answer, contexts, all four
scores, timestamped. Something that can be handed to whoever signs off on a
release, and that stays around as a record — not a screenshot that gets lost
in a week.

It accumulates. Every UI run and every automated test-suite run, if this gets
wired into CI, lands in the same place. So quality can be tracked as a trend
across releases — not just checked once, right before launch, and forgotten.

---

## Act 7 — What's Next

*(Back to Dashboard, point at "Query Generator".)*

Today, someone still has to sit down and hand-write the test questions and
correct answers before any of this can run. That's the next bottleneck — and
it's already on the roadmap. Query Generator reads a knowledge base directly
and generates that test set automatically, so even that manual step goes
away.

---

## Act 8 — The Close

Here's the story in one line: teams shipping AI chatbots have no reliable way
to know whether an answer is right, grounded, and complete — until now. Four
scores, plain-language quality bands, a diagnosis instead of just a red flag,
and a report that holds up as evidence, not a Slack screenshot.

Everything shown today is real, running, and works against real data right
now. The only question left is whether a team wants to find its next
"clearance jacket" problem in twenty seconds, before launch — or find it the
way most teams still do: after a customer already has.

**[PAUSE]** *Questions & discussion.*

---

## If they ask... (quick, honest answers — not scripted lines)

- **"Does it work with other LLM providers?"** — Yes, at the configuration
  layer. OpenAI and Azure AI Foundry are already built in; enabling a second
  provider in the sidebar is a small config change, not new development.
- **"Can this run in a CI pipeline, not just this UI?"** — Yes — the same four
  metrics run from an automated test suite, writing into the same report. The
  UI is for ad-hoc spot checks; the automated path is for "run this on every
  merge."
- **"What if our API doesn't return this exact response shape?"** — The tool
  auto-detects the common shapes, and there's a manual field-mapping panel
  with dotted-path support for anything nested or unusual — a five-minute
  setup, not an integration project.
- **"What does POOR vs MODERATE actually mean?"** — A fixed scale, not a
  vibe: EXCELLENT ≥ 0.85, GOOD ≥ 0.70, MODERATE ≥ 0.50, POOR below that — the
  same four bands on every score, every run.

---

## Delivery notes

- Keep this a monologue with pauses, not a rehearsed back-and-forth — let real
  questions land wherever **[PAUSE]** is marked.
- Swap the "clearance jacket" example for a real edge case from the actual
  audience's domain if one is available beforehand — the story lands harder
  on their own content.
- If there's no live RAG API to hook up, uncheck "Fetch answer & contexts from
  RAG API" and paste in the answer and source paragraph by hand — same
  scores, same story, just narrated instead of fetched.
