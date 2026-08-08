# Learning-science methods for Depth Dive output types

> Research note resolving Wayfinder ticket #216. Feeds the downstream
> "Output-type taxonomy & routing" ticket.
>
> Survey question: which proven learning-science / cognitive-science methods
> apply to comprehension of dense technical (AI/ML) material, what
> **interactive output artifact types** do they imply for a *stateless*
> Depth Dive request, and which methods are out because they fundamentally
> require per-learner state?

## 0. Scope and locked constraints

Per the map (not re-litigated here):

- **Stateless interactive artifacts.** Interactivity lives entirely in the
  output payload of one Depth Dive request. No per-passage memory, no learner
  model, no quizzing, no scheduling. A method that fundamentally requires
  per-learner state is OUT as a Depth Dive output type — and saying so
  explicitly is itself a finding of this survey.
- **Non-text passages are first-class.** A table is not markdown; an image is
  not its caption. Depth Dive must handle captured figures/tables as
  first-class inputs and may emit first-class non-text outputs.
- **Interactive animation is the accepted flagship** output candidate. This
  survey's job is to find what *else* the evidence supports.
- **One self-contained payload.** Artifacts must be producible by the
  Depth Dive harness (an LLM-driven harness, in the Harness A/B sense) in a
  single response payload that the client renders without further server
  calls.

Domain terms used as defined in `CONTEXT.md`: **Depth Dive** (the synthesis
system producing richer, non-plain-text outputs; stateless; no quizzing, no
scheduling), **Captured Passage** (the user-selected paragraph/figure/excerpt
that grounds a Depth Dive request), **dual coding** (Depth Dive's MVP output
format is already defined as a dual-coding explanation of a captured passage —
text paired with a diagram, carousel, or coding example).

### Method note

Claims were traced to primary sources: original peer-reviewed studies,
monographs, and meta-analyses, verified via DOI resolution, PubMed abstracts,
and Crossref records. Where a full text is paywalled (most journal articles),
the abstract/landing page was verified and is what is cited; this is flagged
per source in §5. No secondary blog write-ups were used as evidence.

## 1. Findings at a glance

Evidence levels: **well-replicated** = robust multi-decade literature and/or
high utility rating in Dunlosky et al.'s monograph [1]; **moderate** =
Dunlosky "moderate utility" or real but heterogeneous/conditional effects;
**mixed** = genuine effects with fragile boundary conditions; **contested** =
popular claim out ahead of the evidence.

| Method | Evidence level | Implied stateless artifact type(s) | Statefulness verdict |
|---|---|---|---|
| Dual coding (Paivio) [10][11][12] | Well-replicated for presented visuals; contested for learner-generated imagery [1] | Dual-coding explanation (text + adjacent diagram/figure); annotated-figure walkthrough | **Stateless — in** |
| Mayer's multimedia principles [13][14][15][16][17] | Well-replicated (with boundary conditions) | Not an artifact type — design constraints on all audiovisual artifacts: segmented carousel, concept primer, signaling overlays | **Stateless — in (as constraints)** |
| Worked examples (Sweller) [18][19][20][21][22] | Well-replicated for novices; expertise reversal is a boundary condition [22] | Step-through worked-example player with hidden next steps; faded/completion walkthrough; worked code derivation | **Stateless — in** |
| Self-explanation (Chi) [23][24][1] | Moderate (Dunlosky: moderate utility) [1] | Model self-explanations attached to derivation steps; self-explanation prompt cards with reveal | **Stateless — in** (feedback loop excluded) |
| Elaborative interrogation [25][26][27][1] | Moderate (Dunlosky: moderate utility; prior-knowledge dependent) [1] | "Why is this true?" prompt cards with expert-answer reveal | **Stateless — in** |
| Generation effect (Slamecka & Graf) [28] | Well-replicated in memory tasks; indirect for text comprehension | Predict-the-next-step / fill-in-with-reveal moments; predict-the-output for code | **Stateless — in** |
| Prediction / pretesting (Brod; Richland; Kornell) [29][30][31][32][33] | Well-replicated and growing (newer literature) | Prediction-reveal pauses inside animations/derivations; misconception → prediction → resolution structure | **Stateless — in** |
| Concrete examples & analogy (Gick & Holyoak; Gentner) [34][35][36] | Mixed — transfer is fragile without explicit mapping; multiple compared examples help | Analogy-mapping artifact (source↔target correspondences); multi-example schema-induction carousel | **Stateless — in** |
| Concept/knowledge maps (Nesbit & Adesope) [37] | Moderate — small-to-large heterogeneous effects | Interactive concept map (node-link, click-to-expand) | **Stateless — in** |
| Desirable difficulties (Bjork) [38][39][40] | Well-replicated for specific instantiations; umbrella framing has boundary conditions [40] | No standalone artifact; within-session instantiations (generation, prediction, example variation) are components of the artifacts above | **Split: within-session forms in; canonical forms (spacing, interleaving, repeated testing) fundamentally stateful — out** |
| Retrieval practice / testing effect (Roediger & Karpicke) [1][41][42][43] | Well-replicated; Dunlosky **high utility** [1] | (Would imply: recall prompt with reveal, embedded quiz card) | **OUT** — no-quizzing lock; full-strength spaced retrieval fundamentally requires scheduling state |
| Interleaving (Rohrer) [45][1] | Well-replicated in math/categorization; Dunlosky moderate utility [1] | (Would imply: mixed-topic practice scheduler) | **OUT — fundamentally stateful** (sequencing across sessions) |
| Distributed practice / spacing (Cepeda) [46][1][47] | Well-replicated; Dunlosky **high utility** [1] | (Would imply: scheduled resurfacing of captured passages) | **OUT — fundamentally stateful** |

## 2. Method-by-method findings

### 2.1 Dual coding

**Mechanism.** Paivio's dual coding theory holds that cognition operates
through two partially independent channels — a verbal system for language and
a nonverbal (imagery) system for objects and events. Material encoded in both
channels has two potential retrieval routes rather than one, so paired
text-plus-image representations are more likely to be retained and connected
than text alone [10][11].

**Evidence level: well-replicated for presented visuals.** The picture
superiority / concreteness advantage is one of the oldest robust findings in
memory research (e.g., concrete items and pictures recalled better than
abstract words; Paivio & Csapo 1969 [12]). Mayer's multimedia-learning
research program — hundreds of experimental comparisons — is in large part an
instructional elaboration of dual coding: people learn more deeply from words
and pictures together than from words alone [13][14]. **Important caveat:**
Dunlosky et al. 2013 rated *imagery use for text learning* (instructing
learners to generate their own mental images while reading) **low utility** —
the evidence supports *presenting* visuals alongside text, not telling
learners to imagine [1]. Depth Dive should render visuals, not prescribe
imagining.

**Candidate artifact types.**

- **Dual-coding explanation** — text paired with an adjacent,
  passage-specific diagram or figure. This is already Depth Dive's MVP output
  format per `CONTEXT.md` (text paired with a diagram, carousel, or coding
  example); the survey confirms the evidence base for it.
- **Annotated-figure walkthrough** — for a captured figure (non-text passages
  are first-class), an overlay of callouts linking figure regions to
  explanatory text, honoring Mayer's spatial-contiguity and signaling
  principles (§2.2).

**Statefulness verdict: stateless — in.** A text+diagram payload requires no
learner state. Note for the taxonomy ticket: the diagram itself must be a
first-class payload element (e.g., a structured diagram spec the client
renders), consistent with the lock that a table is not markdown and an image
is not its caption.

### 2.2 Mayer's multimedia-learning principles

**Mechanism.** Mayer's cognitive theory of multimedia learning assumes limited
working-memory capacity in separate visual/verbal channels, and that meaningful
learning requires selecting, organizing, and integrating material across both
channels. The resulting design principles manage three kinds of cognitive
processing: reduce *extraneous* processing (coherence, signaling, redundancy,
spatial/temporal contiguity), manage *essential* processing (segmenting,
pre-training, modality), and foster *generative* processing (personalization,
voice, image) [13][14][15].

**Evidence level: well-replicated, with boundary conditions.** Mayer's second
edition reports double the experimental comparisons of the first and
reorganizes 12 principles into the three processing categories above, with
explicit boundary conditions per principle [13]. The supporting cognitive-load
account (Mayer & Moreno's nine ways to reduce cognitive load) is standard [15].
Three findings matter especially for Depth Dive:

- **Segmenting principle** — learners do better when a continuous lesson is
  broken into user-paced segments rather than delivered as one continuous
  unit. Directly relevant: one-pass reading of dense material fails partly
  because pacing is not under the reader's control.
- **Pre-training principle** — learners do better when they know the names and
  characteristics of key concepts before the main explanation.
- **Coherence principle / seductive details** — interesting-but-irrelevant
  material (text, images, sounds) *harms* retention and transfer (Harp & Mayer
  1998 [16]). Any interactivity Depth Dive adds must be load-bearing, not
  decorative.

**Candidate artifact types.** Mayer's principles are **design constraints, not
an artifact type** — but two artifact shapes exist mainly to satisfy them:

- **Segmented carousel** — one idea per card, user advances; each card a
  self-contained dual-coded unit (segmenting + spatial contiguity).
- **Concept primer card** — a short glossary of the passage's key terms and
  their roles, presented before the main explanation (pre-training).

**Statefulness verdict: stateless — in (as constraints).** All of the above
are payload-internal. One principle points beyond MVP: the **modality
principle** (animation + narration can beat animation + on-screen text under
the right conditions [13]) would require audio/TTS in the payload — a
post-MVP consideration for the flagship animation, not a blocker.

### 2.3 Worked examples and example-based learning

**Mechanism.** Sweller's cognitive load theory argues that novice problem
solving is dominated by means-ends search that overloads working memory
without building the schemas learners need; studying a fully worked solution
frees capacity for schema acquisition instead [18][19]. Sweller & Cooper (1985)
showed worked examples outperforming equivalent problem-solving practice for
learning algebra [19]. Renkl's instructionally oriented theory of example-based
learning systematizes the design rules: use meaningful, typical examples;
compare multiple examples; prompt learners to self-explain the steps; and
**fade** worked steps progressively toward independent solving [20].

**Evidence level: well-replicated for novices, with a sharp boundary
condition.** The worked-example effect is among the most robust findings in
instructional research [18][19][20]. The boundary condition is the **expertise
reversal effect**: guidance techniques that help novices (worked examples,
extra explanations) become ineffective or even harmful as learners gain
expertise, because the guidance becomes redundant with their existing schemas
(Kalyuga et al. 2003 [22]). Depth Dive cannot know the reader's expertise
(stateless, no learner model) — so worked-example artifacts must expose
**user-controlled detail levels** (e.g., "show/hide intermediate steps")
rather than adapt automatically.

**Candidate artifact types.**

- **Step-through worked-example player** — a derivation, proof, or computation
  from the captured passage unfolded one step at a time; the next step is
  hidden until the user advances; each step can carry an optional model
  self-explanation (§2.4). This is the single artifact most directly implied
  by the evidence for dense AI/ML material (derivations of loss functions,
  attention computations, gradient steps).
- **Faded/completion walkthrough** — the same player, but later steps are
  blanked with a reveal control ("what comes next? → reveal"), blending the
  worked-example effect with the generation effect (§2.6).
- **Worked code example with step annotations** — for documentation-style
  captured passages: a runnable-looking code sample with per-line/per-block
  annotations revealed progressively.

**Statefulness verdict: stateless — in.** All steps, reveals, and detail
levels live in the payload; expertise adaptation is replaced by user control.

### 2.4 Self-explanation

**Mechanism.** Chi et al. (1989) found that students who spontaneously
generate self-explanations while studying worked examples — explaining to
themselves why each step follows, connecting steps to principles, and
monitoring their own comprehension — learn substantially more than students
who do not, and that good problem solvers self-explain more [23]. Chi's later
dual-process account identifies two mechanisms: generating inferences that the
material leaves implicit, and repairing/monitoring the reader's mental model
when it breaks down [24].

**Evidence level: moderate.** Dunlosky et al. 2013 assessed self-explanation
as **moderate utility**: benefits generalize across some materials and learner
groups, but the technique has not been adequately evaluated in representative
educational contexts, and effect sizes depend on implementation (prompts vs.
spontaneous, with vs. without feedback) [1].

**Candidate artifact types.**

- **Model self-explanations attached to steps** — the worked-example player
  (§2.3) carries, per step, a hidden "why this step" panel written by the
  harness. This delivers the content of a good self-explainer's inferences
  without requiring the learner to produce them — the stateless-compatible
  half of the evidence.
- **Self-explanation prompt cards with reveal** — a prompt ("In your own
  words: why does the gradient point uphill here?") followed by a reveal of an
  expert self-explanation the reader compares against their own answer.
  Covert generation + comparison needs no state (§2.6).

**Statefulness verdict: stateless — in, with a caveat.** The *full*
self-explanation loop — learner produces an explanation, system gives feedback
on its quality — requires capturing and evaluating learner output across turns
and is out under the stateless lock. What survives is model self-explanations
plus prompts with expert-answer reveals.

### 2.5 Elaborative interrogation

**Mechanism.** Elaborative interrogation asks learners to generate an answer
to a "why is this true?" / "why does this make sense?" question about the
material, forcing integration of new facts with prior knowledge. It descends
from the Pressley lineage of elaboration strategies and was studied both
standalone and combined with analogy (McDaniel & Donnelly 1996 [25]).

**Evidence level: moderate, prior-knowledge dependent.** Dunlosky et al. 2013
rated elaborative interrogation **moderate utility** [1]. A recurring boundary
condition is prior knowledge: generating a plausible "why" answer requires
enough existing knowledge to draw on (Woloshyn, Wood & Willoughby 1994 [26]);
benefits for low-knowledge learners improve when the question is scaffolded,
e.g., with an analogy [25]. For dense AI/ML passages — where the reader
captured the passage precisely because they lack understanding — the scaffolded
(analogy-backed, expert-answer-revealed) variant is the appropriate design.

**Candidate artifact types.**

- **Elaborative prompt card with reveal** — attached to a key claim in the
  passage: "Why is dropout applied at training time rather than inference? →
  reveal." The reveal contains the harness-generated expert answer plus, where
  useful, the analogy that scaffolds it (§2.8).

**Statefulness verdict: stateless — in.** Prompt + reveal is payload-internal.
As with self-explanation, the version with feedback on the learner's own
generated answer is out.

### 2.6 Generation effect

**Mechanism.** Slamecka & Graf (1978) showed that information a learner
*generates* (e.g., completing a cue to produce a target) is remembered better
than the same information simply read — the generation effect [28]. Generation
forces deeper, more distinctive processing of the relationship between cue and
target.

**Evidence level: well-replicated in memory tasks; indirect for comprehension
of dense text.** The effect itself is one of the most replicated in
experimental memory research [28], but the canonical paradigms are word-list
and paired-associate memory, and generalization to *comprehension* of dense
technical prose is extrapolation rather than direct evidence. Treat the effect
as strong for retention of key facts/definitions inside a Depth Dive, and
plausible-but-less-proven for deep comprehension. (Flagged for the taxonomy
ticket in §3.3.)

**Candidate artifact types.**

- **Predict-the-next-step with reveal** — inside a derivation or algorithm
  walkthrough: "What is the next line? → reveal."
- **Predict-the-output for code** — a code snippet from the captured passage,
  a "what does this print?" pause, then a reveal (pairs naturally with the
  worked code example, §2.3).
- **Fill-in-the-blank key term with reveal** — for definitions the passage
  introduces.

These are **interaction moments, not a standalone artifact type**: they can be
layered onto the worked-example player, the animation flagship, or a
derivation carousel. That composability is the design point — the generation
effect is cheap to embed anywhere a reveal is possible.

**Statefulness verdict: stateless — in.** Generation happens covertly at
reading time; the reveal provides the feedback. No state required.

### 2.7 Prediction and pretesting

**Mechanism.** Making an explicit prediction before seeing the outcome
enhances learning of the outcome, especially when the outcome violates the
prediction: prediction sharpens attention and the resulting prediction error
boosts encoding (Brod et al. 2022 [29]; mechanistic review in Shing, Brod &
Greve 2023 [33]). A closely related family is the **pretesting effect**: even
*unsuccessful* retrieval attempts before studying (guessing an answer one
cannot yet know) enhance later learning of the correct answer relative to
studying first (Richland, Kornell & Kao 2009 [30]; Kornell, Hays & Bjork 2009
[31]). Prediction and surprise can also drive misconception revision
(Theobald & Brod 2021 [32]).

**Evidence level: well-replicated and growing.** The pretesting effect has
been replicated across materials and formats [30][31], and the prediction
literature is newer but converging, with a plausible neurocognitive mechanism
(prediction-error-driven hippocampal encoding) [29][33]. Boundary conditions
exist (effects are strongest for expectancy-violating information and for
material the learner has some basis to predict about [29]) — which fits Depth
Dive well: a captured passage is usually adjacent to knowledge the reader
already has.

**Candidate artifact types.**

- **Prediction-reveal pauses** — the same interaction primitive as §2.6, but
  framed as prediction ("Before reading on: what do you expect happens to the
  loss when the learning rate is doubled?"). Can be embedded in the flagship
  animation (predict the curve shape before the parameter moves), in
  derivations, and in code.
- **Misconception → prediction → resolution structure** — an artifact-level
  narrative shape: surface a plausible wrong intuition, ask the reader to
  commit to a prediction, then resolve with the correct account and an
  explicit contrast. This is the stateless analogue of surprise-based
  misconception interventions [32].

**Statefulness verdict: stateless — in.** Prediction is covert; the reveal
supplies the outcome. No state required. Note the distinction from quizzing:
these are not assessments of the learner (no score, no memory of the answer) —
they are comprehension devices whose feedback is the content itself.

### 2.8 Concrete examples and analogy

**Mechanism.** Abstract principles are learned and transferred more reliably
when grounded in concrete instances. Gick & Holyoak's classic studies showed
that people can transfer a solution from a concrete analog (a military story)
to a structurally identical problem (the radiation problem) — but spontaneous
transfer is fragile and usually requires the analogy to be noticed or
prompted [34]. Comparing **two or more** analogs induces an abstract schema
that transfers better than any single example (schema induction; Gick &
Holyoak 1983 [35]). Gentner's structure-mapping theory explains why: analogy
works by mapping relational structure, not surface features, so the mapping
must be made explicit to be used [36].

**Evidence level: mixed.** The underlying effects are real but the headline
promise — "give learners a concrete example and they'll transfer" — fails
without support: single-example spontaneous transfer is unreliable [34], and
analogies can actively mislead when surface features dominate or the mapping
is wrong. What is well supported is the *scaffolded* version: multiple
compared examples plus explicit correspondence mapping [35][36]. (Note:
Dunlosky et al. 2013 did not rate concrete examples among their ten
techniques, so there is no utility rating to cite [1].)

**Candidate artifact types.**

- **Analogy-mapping artifact** — source and target presented side by side with
  explicit correspondence links (e.g., "query vector ↔ the question you ask a
  librarian; key vectors ↔ the index cards; attention weights ↔ how much each
  card matters"), each link clickable to expand. Explicit mapping is the
  evidence-backed part [36].
- **Multi-example schema-induction carousel** — two or three concrete analogs
  or worked instances of the same principle, followed by a card that strips
  away the surface features and states the induced schema [35].

**Statefulness verdict: stateless — in.** All mapping and comparison content
is payload-internal.

### 2.9 Concept and knowledge maps

**Mechanism.** Node-link diagrams externalize the relational structure of a
domain, letting learners see how concepts connect rather than processing
relations implicitly from prose. Constructing or studying such maps is a
generative activity in Mayer's sense (generating relations among ideas)
[17][37].

**Evidence level: moderate.** Nesbit & Adesope's meta-analysis (55 studies,
5,818 participants, 67 effect sizes) found that learning with concept/knowledge
maps was associated with increased knowledge retention across grade levels and
domains, but effect sizes varied from small to large depending on how maps
were used (constructed vs. viewed) and on the comparison treatment, with
significant heterogeneity [37]. Honest summary: maps help, mostly, but are not
a guaranteed win and their benefit depends on design and use.

**Candidate artifact types.**

- **Interactive concept map** — the concepts of the captured passage as nodes,
  relations as labeled edges, click-to-expand each node into its definition or
  its role in the passage. For a dense paper section, this is the artifact
  that answers "how do these pieces fit together?" — a question no
  single-paragraph Depth Dive text answers well.

**Statefulness verdict: stateless — in.** A node-link graph is a compact JSON
payload. (The stronger "learner constructs the map" variant [37] would need
drag-and-drop with evaluation — out under the stateless lock; the viewing +
expanding variant remains.)

### 2.10 Desirable difficulties

**Mechanism.** Bjork's desirable-difficulties framework holds that conditions
which make learning feel slower and harder — spaced practice, interleaved
practice, varied conditions, testing, generation — can produce *better*
long-term retention and transfer, because they induce deeper encoding and
discrimination. The term originates with Bjork (1994) [38], building on the
Bjork & Bjork disuse theory of memory [39]. Soderstrom & Bjork (2015) sharpen
the key distinction: immediate *performance* can worsen under a desirable
difficulty while long-term *learning* improves — the two must not be confused
[40].

**Evidence level: well-replicated for the specific instantiations** — spacing
[46][1], interleaving [45][1], testing [41][1], and generation [28] each
have strong literatures. The umbrella framing is sound but has boundary
conditions: a difficulty is only desirable if the learner can meet it;
otherwise it is merely undesirable [38][40].

**Candidate artifact types: none standalone.** Desirable difficulties do not
imply an artifact type of their own; they are a *lens*. Their within-session
instantiations — generation and prediction (§2.6, §2.7) and variation of
examples (§2.8) — are exactly the interaction moments already identified as
embeddable in stateless artifacts.

**Statefulness verdict: split.** The canonical desirable difficulties —
spacing, interleaving, repeated testing — are defined over *time and repeated
episodes* and fundamentally require scheduling and per-learner state →
**out** for Depth Dive (§2.11, §2.12). The within-session instantiations are
stateless-compatible and survive as components of other artifact types.

### 2.11 Retrieval practice (testing effect) — surveyed, then judged against the lock

**Mechanism.** Actively retrieving information from memory strengthens the
memory trace and its cues far more than re-exposure does; testing is a
learning event, not just an assessment [41][42][43].

**Evidence level: well-replicated — this is one of the strongest effects in
the learning sciences.** Roediger & Karpicke (2006): with educational prose,
repeated studying beat repeated testing on a 5-minute test, but on delayed
tests (2 days, 1 week) prior testing produced substantially greater retention
— while repeated studying produced *higher confidence*, an illusion of
learning [41]. Karpicke & Roediger (2008) showed retrieval practice is what
produces long-term retention, not repeated re-exposure [42]. Dunlosky et al.
2013 gave practice testing a **high utility** assessment — one of only two
techniques so rated [1].

**Statefulness verdict: OUT — and this is the survey's most significant
rejection.** Two-part reasoning:

1. A *single* self-contained recall prompt with reveal ("Without looking back,
   reconstruct the argument of this passage → reveal") is technically
   stateless. But the charting session locked **no quizzing** for Depth Dive
   outputs, and a recall prompt is a quiz by any honest description. So the
   stateless version is out by product decision.
2. The *full-strength* version — the one that earns the high-utility rating —
   is repeated, spaced retrieval across sessions [41][46][1], which
   fundamentally requires per-learner state (a learner model, a schedule).
   `CONTEXT.md` already records Retrieval Practice / Spaced Repetition as
   post-MVP for exactly this reason.

The downstream taxonomy ticket should carry this trade explicitly: the single
most robust lever in the surveyed evidence is deliberately left on the table
by the statelessness/no-quizzing lock. The stateless substitutes that retain
*some* of the same cognitive machinery are prediction-reveal (§2.7) and
generation-with-reveal (§2.6) — retrieval-adjacent, but not assessment-shaped
and not scheduled.

### 2.12 Interleaving and distributed practice — surveyed, then judged against the lock

**Interleaving.** Mixing different problem types or topics within a practice
session improves discrimination and long-term retention relative to blocked
practice, robustly in mathematics (Rohrer, Dedrick & Stershic 2015 [45]) and
categorization; Dunlosky et al. 2013 rated it
**moderate utility**, noting the benefits had only begun to be systematically
explored at the time [1]. **Verdict: OUT — fundamentally stateful.**
Interleaving is defined over the *sequencing of multiple practice episodes*;
there is nothing to interleave inside a single stateless payload. (Its weak
within-payload cousin — varying examples to induce a schema — survives under
§2.8.)

**Distributed practice (spacing).** Spacing study episodes over time is one of
the two Dunlosky **high-utility** techniques, supported by over a century of
research and a large quantitative meta-analysis (Cepeda et al. 2006 [46];
Dunlosky et al. 2013 [1]). Rohrer & Taylor (2006) confirmed it for
mathematics retention, with overlearning decaying rapidly [44]. **Verdict: OUT
— fundamentally stateful.** Spacing requires scheduling resurfacing of a
captured passage across days/weeks — durable per-concept state, exactly what
the stateless lock excludes and what `CONTEXT.md` defers to post-MVP.

## 3. Synthesis

### 3.1 Shortlist of evidence-backed candidate artifact types

For the "Output-type taxonomy & routing" ticket, alongside the accepted
flagship **interactive (parameter-manipulation) animation**:

| Candidate artifact type | Justifying method(s) | Evidence strength |
|---|---|---|
| **Dual-coding explanation** (text + passage-specific diagram; already the MVP format per `CONTEXT.md`) | Dual coding [10][12]; Mayer multimedia [13] | Well-replicated |
| **Step-through worked-example player** (hidden next steps, optional model self-explanation per step, user-controlled detail level) | Worked-example effect [18][19][20]; self-explanation [23][24]; cognitive load / segmenting [13][15] | Well-replicated (novices); expertise reversal handled via user control [22] |
| **Prediction-reveal layer** (predict-the-next-step / predict-the-output pauses with reveal; misconception → prediction → resolution shape) — composable onto animation, derivations, code | Generation effect [28]; pretesting [30][31]; prediction effect [29][32][33] | Well-replicated in memory/prediction paradigms; comprehension transfer partly extrapolated |
| **Segmented carousel with concept primer** (one idea per user-paced card; key-term primer first) | Mayer segmenting + pre-training principles [13]; cognitive load [15] | Well-replicated |
| **Analogy-mapping artifact** (explicit source↔target correspondences; multi-example schema induction) | Analogical transfer [34]; schema induction [35]; structure mapping [36] | Mixed — only the scaffolded/explicit-mapping variant is well supported |
| **Interactive concept map** (node-link, click-to-expand) | Concept-map meta-analysis [37]; dual coding [10] | Moderate, heterogeneous |
| **Elaborative / self-explanation prompt cards with reveal** | Elaborative interrogation [25][26][1]; self-explanation [23][24][1] | Moderate (Dunlosky moderate utility) |

Payload feasibility: every item above is emittable as structured JSON + text +
diagram/graph specs + code in one harness response, rendered client-side
without further server calls. The only Mayer principle pointing beyond current
payload capabilities is **modality** (narration), which would require TTS
audio — flagged as post-MVP for the flagship animation, not a shortlist
blocker.

### 3.2 Notable rejections (verdicts that are themselves findings)

| Rejected | Why |
|---|---|
| **Quizzing / practice-testing artifacts** | No-quizzing lock. The stateless one-shot variant is technically feasible but is a quiz by any honest description; the high-utility variant (spaced repeated retrieval [41][46][1]) fundamentally requires scheduling state. The strongest evidence-backed lever in the entire survey is deliberately unavailable — the taxonomy ticket should record this as a known cost of the stateless decision, not an oversight. |
| **Interleaving scheduler** | Fundamentally stateful: interleaving is defined over sequencing across practice episodes/sessions [45]. |
| **Spaced resurfacing of captured passages** | Fundamentally stateful: requires durable per-concept state and a schedule [46][1]; already deferred post-MVP in `CONTEXT.md`. |
| **Learner-generated imagery prompts** ("picture this in your head") | Dunlosky low utility [1]; not an artifact type. Present visuals instead. |
| **Feedback-on-learner-output loops** (grade the learner's self-explanation, correct their generated answer) | Requires capturing and evaluating learner responses across turns — per-learner state. Only the model-answer-reveal half survives. |

### 3.3 What the taxonomy ticket should know (surprising, contested, unresolved)

1. **The two Dunlosky high-utility techniques (practice testing, distributed
   practice) are both stateful** [1]. Everything compatible with the
   stateless lock comes from the moderate-utility tier or from multimedia/
   cognitive-load research. Expect honest, moderate effect sizes from the
   shortlist — not the blockbuster numbers of the testing literature.
2. **Expertise reversal is real and unresolvable without a learner model** [22].
   Worked-example scaffolding that helps a novice actively hurts an expert.
   With no learner model allowed, the only mitigation is user-controlled
   detail levels (show/hide steps, skip primer). Routing decisions should
   treat "reader expertise" as a *request-time* signal (user says so) rather
   than an inferred state.
3. **The generation effect's evidence base is memory-shaped.** Canonical
   paradigms are word lists and paired associates [28]; transfer to
   comprehension of dense technical prose is extrapolation. Prediction/
   pretesting evidence is closer to educational material [29][30][31] but
   newer. If Depth Dive ever gets an evaluation harness, generation/prediction
   moments are the highest-uncertainty component and the most worth measuring.
4. **Seductive details cut both ways for interactive products** [16]. The
   coherence principle says interesting-but-irrelevant material harms learning.
   For an interactive artifact, the risk is *decorative interactivity* —
   animations and controls that are fun but off-topic. The taxonomy's routing
   rules should require every interactive element to be load-bearing for the
   captured passage.
5. **Dual coding is confirmed but narrower than its reputation.** Presented
   visuals: strong [12][13]. Instructed imagery: low utility [1]. Depth Dive's
   MVP dual-coding format is on the right side of that line; the taxonomy
   should keep it there (render, don't prescribe).
6. **Analogy is a power tool with a safety catch.** Single analogs fail to
   transfer spontaneously [34]; analogies can mislead via surface features.
   Only explicit mapping + multiple compared analogs is well supported
   [35][36]. An analogy artifact that ships without explicit correspondence
   links is shipping the unsupported variant.
7. **Prediction vs. quizzing is a framing distinction, not a mechanical one.**
   Both are "prompt → reveal." The survey's position: prediction-reveal is in
   because its function is comprehension (the reveal *is* the content, nothing
   is recorded or scored); quizzing is out because its function is assessment
   and its strongest form needs scheduling [29][30][41]. The taxonomy ticket
   should encode this distinction in the artifact definitions, or the
   no-quizzing lock will be quietly violated by another name.

## 4. Limitations

- Most primary articles are paywalled; citations rest on verified abstracts
  and landing pages (PubMed, Crossref, publisher pages) plus the monograph
  abstracts retrieved for this survey. Effect-size figures were deliberately
  not quoted except where an abstract supplied them.
- The surveyed literatures were mostly built on classroom and laboratory
  materials (prose passages, algebra, science texts), not on LLM-rendered
  interactive payloads for self-directed adults reading AI/ML papers. The
  artifact mapping is therefore an evidence-grounded extrapolation, and the
  shortlist is ranked by evidence strength in §3.1 accordingly.
- Dunlosky et al.'s utility ratings date to 2013 [1]; post-2013 work (e.g.,
  the prediction literature [29][33]) was surveyed separately and labeled as
  newer.

## 5. References

Utility ratings and monograph anchor:

1. Dunlosky, J., Rawson, K. A., Marsh, E. J., Nathan, M. J., & Willingham,
   D. T. (2013). Improving Students' Learning With Effective Learning
   Techniques: Promising Directions From Cognitive and Educational Psychology.
   *Psychological Science in the Public Interest, 14*(1), 4–58.
   https://doi.org/10.1177/1529100612453266 (abstract verified via PubMed
   PMID 26173288; full text paywalled)

Dual coding and multimedia learning:

10. Paivio, A. (1986). *Mental Representations: A Dual Coding Approach.*
    Cambridge University Press. (Book; no DOI.)
11. Paivio, A. (1971). *Imagery and Verbal Processes.* Holt, Rinehart &
    Winston. (Book; origin of dual coding theory.)
12. Paivio, A., & Csapo, K. (1969). Concrete image and verbal memory codes.
    *Journal of Experimental Psychology, 80*(2, Pt. 1), 279–285.
    https://doi.org/10.1037/h0027273
13. Mayer, R. E. (2009). *Multimedia Learning* (2nd ed.). Cambridge
    University Press. https://doi.org/10.1017/CBO9780511811678 (abstract
    verified via Crossref: 12 principles organized as reducing extraneous,
    managing essential, and fostering generative processing, with boundary
    conditions)
14. Mayer, R. E. (2014). The Cognitive Theory of Multimedia Learning. In R. E.
    Mayer (Ed.), *The Cambridge Handbook of Multimedia Learning* (2nd ed.).
    Cambridge University Press. https://doi.org/10.1017/CBO9781139547369.005
    (3rd-ed. 2021 version: https://doi.org/10.1017/9781108894333.008)
15. Mayer, R. E., & Moreno, R. (2003). Nine Ways to Reduce Cognitive Load in
    Multimedia Learning. *Educational Psychologist, 38*(1), 43–52.
    https://doi.org/10.1207/s15326985ep3801_6
16. Harp, S. F., & Mayer, R. E. (1998). How seductive details do their
    damage: A theory of cognitive interest in science learning. *Journal of
    Educational Psychology, 90*(3), 414–434.
    https://doi.org/10.1037/0022-0663.90.3.414
17. Fiorella, L., & Mayer, R. E. (2021). The Generative Activity Principle in
    Multimedia Learning. In R. E. Mayer (Ed.), *The Cambridge Handbook of
    Multimedia Learning* (3rd ed.). Cambridge University Press.
    https://doi.org/10.1017/9781108894333.036

Worked examples, cognitive load, expertise reversal:

18. Sweller, J. (1988). Cognitive Load During Problem Solving: Effects on
    Learning. *Cognitive Science, 12*(2), 257–285.
    https://doi.org/10.1207/s15516709cog1202_4 (1981 origin: Sweller, J.
    (1981). Some cognitive processes and their consequences for the
    organisation and presentation of information. *Australian Journal of
    Psychology, 33*(1), 1–8 — no resolvable DOI found)
19. Sweller, J., & Cooper, G. A. (1985). The Use of Worked Examples as a
    Substitute for Problem Solving in Learning Algebra. *Cognition and
    Instruction, 2*(1), 59–89. https://doi.org/10.1207/s1532690xci0201_3
20. Renkl, A. (2014). Toward an Instructionally Oriented Theory of
    Example-Based Learning. *Cognitive Science, 38*(1), 1–37.
    https://doi.org/10.1111/cogs.12086
21. Chi, M. T. H. (2000). Self-explaining expository texts: The dual
    processes of generating inferences and repairing mental models. In M. L.
    Kamil, P. D. Pearson, E. B. Barr, & P. P. Afflerbach (Eds.), *Handbook of
    Reading Research, Vol. III.* Erlbaum. (Book chapter; no DOI.)
22. Kalyuga, S., Ayres, P., Chandler, P., & Sweller, E. (2003). The Expertise
    Reversal Effect. *Educational Psychologist, 38*(1), 23–31.
    https://doi.org/10.1207/s15326985ep3801_4

Self-explanation and elaborative interrogation:

23. Chi, M. T. H., Bassok, M., Lewis, M. W., Reimann, P., & Glaser, R.
    (1989). Self-explanations: How students study and use examples in learning
    to solve problems. *Cognitive Science, 13*(2), 145–182.
    https://doi.org/10.1207/s15516709cog1302_1
24. Chi (2000), full citation at reference 21 — cited here for the
    dual-process account of self-explanation (generating inferences +
    repairing mental models).
25. McDaniel, M. A., & Donnelly, C. M. (1996). Learning with analogy and
    elaborative interrogation. *Journal of Educational Psychology, 88*(3),
    508–519. https://doi.org/10.1037/0022-0663.88.3.508
26. Woloshyn, V. E., Wood, E., & Willoughby, T. (1994). Considering prior
    knowledge when using elaborative interrogation. *Applied Cognitive
    Psychology, 8*(1). https://doi.org/10.1002/acp.2350080104
27. Dunlosky et al. (2013) utility ratings for elaborative interrogation and
    self-explanation (both moderate utility) — full citation at reference 1.

Generation effect, prediction, pretesting:

28. Slamecka, N. J., & Graf, P. (1978). The generation effect: Delineation of
    a phenomenon. *Journal of Experimental Psychology: Human Learning and
    Memory, 4*(6), 592–604. https://doi.org/10.1037/0278-7393.4.6.592
    (citation verified via DOI resolution; full text paywalled)
29. Brod, G., Greve, A., Jolles, D., Theobald, M., & Galeano-Keiner, E. M.
    (2022). Explicitly predicting outcomes enhances learning of
    expectancy-violating information. *Psychonomic Bulletin & Review, 29*(6),
    2192–2201. https://doi.org/10.3758/s13423-022-02124-x (open access,
    PMC9722848)
30. Richland, L. E., Kornell, N., & Kao, L. S. (2009). The pretesting effect:
    Do unsuccessful retrieval attempts enhance learning? *Journal of
    Experimental Psychology: Applied, 15*(3), 243–257.
    https://doi.org/10.1037/a0016496
31. Kornell, N., Hays, M. J., & Bjork, R. A. (2009). Unsuccessful retrieval
    attempts enhance subsequent learning. *Journal of Experimental
    Psychology: Learning, Memory, and Cognition, 35*(4), 989–998.
    https://doi.org/10.1037/a0015729
32. Theobald, M., & Brod, G. (2021). Tackling Scientific Misconceptions: The
    Element of Surprise. *Child Development, 92*(5), 2128–2141.
    https://doi.org/10.1111/cdev.13582
33. Shing, Y. L., Brod, G., & Greve, A. (2023). Prediction error and memory
    across the lifespan. *Neuroscience & Biobehavioral Reviews, 155*, 105462.
    https://doi.org/10.1016/j.neubiorev.2023.105462

Analogy and concrete examples:

34. Gick, M. L., & Holyoak, K. J. (1980). Analogical problem solving.
    *Cognitive Psychology, 12*(3), 306–355.
    https://doi.org/10.1016/0010-0285(80)90013-4
35. Gick, M. L., & Holyoak, K. J. (1983). Schema induction and analogical
    transfer. *Cognitive Psychology, 15*(1), 1–38.
    https://doi.org/10.1016/0010-0285(83)90002-6
36. Gentner, D. (1983). Structure-mapping: A theoretical framework for
    analogy. *Cognitive Science, 7*(2), 155–170.
    https://doi.org/10.1016/s0364-0213(83)80009-3

Concept maps:

37. Nesbit, J. C., & Adesope, O. O. (2006). Learning With Concept and
    Knowledge Maps: A Meta-Analysis. *Review of Educational Research, 76*(3),
    413–448. https://doi.org/10.3102/00346543076003413 (abstract verified
    via Crossref: 55 studies, 5,818 participants, 67 effect sizes; effects
    small to large depending on use and comparison treatment)

Desirable difficulties, spacing, interleaving, retrieval practice:

38. Bjork, R. A. (1994). Memory and metamemory considerations in the training
    of human beings. In J. Metcalfe & A. Shimamura (Eds.), *Metacognition:
    Knowing About Knowing* (pp. 181–205). MIT Press. (Book chapter; origin of
    the term "desirable difficulties"; no DOI.)
39. Bjork, R. A., & Bjork, E. L. (1992). A new theory of disuse and an old
    theory of stimulus fluctuation. In A. Pick, L. Krasnor, & I. E. Sigel
    (Eds.), *Developmental Psychology: An Evolving Synthesis* (pp. 89–121).
    Erlbaum. (Book chapter; no DOI.)
40. Soderstrom, N. C., & Bjork, R. A. (2015). Learning Versus Performance: An
    Integrative Review. *Perspectives on Psychological Science, 10*(2),
    176–199. https://doi.org/10.1177/1745691615569000
41. Roediger, H. L., & Karpicke, J. D. (2006). Test-Enhanced Learning: Taking
    Memory Tests Improves Long-Term Retention. *Psychological Science, 17*(3),
    249–255. https://doi.org/10.1111/j.1467-9280.2006.01693.x (abstract
    verified via PubMed PMID 16507066)
42. Karpicke, J. D., & Roediger, H. L. (2008). The Critical Importance of
    Retrieval for Learning. *Science, 319*(5865), 966–968.
    https://doi.org/10.1126/science.1152408
43. Karpicke, J. D. (2012). Retrieval-Based Learning: Active Retrieval
    Promotes Meaningful Learning. *Current Directions in Psychological
    Science, 21*(3), 157–163. https://doi.org/10.1177/0963721412443552
    (see also Karpicke & Grimaldi 2012,
    https://doi.org/10.1007/s10648-012-9202-2)
44. Rohrer, D., & Taylor, K. (2006). The effects of overlearning and
    distributed practise on the retention of mathematics knowledge. *Applied
    Cognitive Psychology, 20*(9), 1209–1224.
    https://doi.org/10.1002/acp.1266
45. Rohrer, D., Dedrick, R. F., & Stershic, S. (2015). Interleaved practice
    improves mathematics learning. *Journal of Educational Psychology,
    107*(3), 900–908. https://doi.org/10.1037/edu0000001
46. Cepeda, N. J., Pashler, H., Vul, E., Wixted, J. T., & Rohrer, D. (2006).
    Distributed practice in verbal recall tasks: A review and quantitative
    synthesis. *Psychological Bulletin, 132*(3), 354–380.
    https://doi.org/10.1037/0033-2909.132.3.354
47. Bjork, R. A., Dunlosky, J., & Kornell, N. (2013). Self-Regulated
    Learning: Beliefs, Techniques, and Illusions. *Annual Review of
    Psychology, 64*, 417–444.
    https://doi.org/10.1146/annurev-psych-113011-143823 (abstract verified
    via publisher; full text paywalled)
