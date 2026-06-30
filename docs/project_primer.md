# From Zero to a Trustworthy Medical-AI System: A Reader's Primer

*A nine-tier ladder that takes you from "I know nothing about computers, machine learning, or research" all the way to narrating a real applied medical-AI system — and the honesty discipline at its heart — in your own words.*

## How to use this primer

Welcome. This was written for exactly one person: someone with **no background** in coding, machine learning, or research. You do not need any of that. If you can follow a story about a doctor reading X-rays, you can follow this whole thing.

A few gentle suggestions:

- **Go in order, and go slow.** Each tier stands on the one before it. Tier 4 will quietly assume you remember "exchangeability" from Tier 3; Tier 6 assumes you remember the "three gates" from Tiers 2 and 4. The ladder is built so that *if you climb the rungs in order, every new word has already been defined.* Skipping ahead is the one reliable way to get lost.
- **The "Under the hood (skippable)" boxes are genuinely skippable.** They show the actual symbols and formulas the researchers use (things like `σ(f(x))` or `n_eff`). On your **first** pass, skip every one of them — the plain-English text around them carries the whole idea by itself. Come back to them later only if you're curious what the real notation looks like.
- **Do the "Check yourself" questions.** Each tier ends with two. Try to answer in your own words before reading the given answer. They are the single best way to know whether a tier landed.
- **Pace yourself.** The whole primer is roughly a one-to-two-hour read if you take it in a couple of sittings. There is no prize for speed.

**Our promise to you:** by the last page you will be able to narrate a complete medical-AI system — start to finish, in plain words — explain *why* its creators deliberately chose honesty over a flashier promise, and point to the exact project document that holds each piece. We will never introduce a term and use it before we've explained it. If we ever do, that's a bug in the primer, not a gap in you.

## Your guide through all of it: Dr. Rivera

To keep everything concrete, one running character walks with us through every tier.

**Dr. Rivera** is a physician at **County Hospital**. Part of her job is reading chest X-rays, looking for things like a small spot on the lung (a "nodule") that might be an early warning sign. To help her, County installs an **AI tool** — a fast, tireless assistant that looks at each X-ray and offers an opinion.

Here is the twist that drives the entire project, so hold onto it: that AI tool was **built and tuned at a different hospital — University Medical** — which has different X-ray machines and a different mix of patients than County. So every tier comes back to the same question: *can Dr. Rivera actually trust what this tool tells her, given that it was built somewhere else?* Everything you're about to learn is, at bottom, machinery for answering that question honestly.

---

## Tier 0 — The problem, in plain words

Welcome. You don't need to know anything about computers, machine learning, or research to start here. By the end of this short tier you'll be able to say, in your own plain words, what this whole project builds and why it matters — and you'll be able to read its one official "what we did" sentence and understand every piece of it. No math. Just ideas.

### Meet Dr. Rivera

Dr. Rivera is a physician at County Hospital. Part of her day is reading chest X-rays — looking for things like a small spot on the lung (a "nodule") that might be the early sign of something serious.

To help her, County Hospital installs an AI tool. Think of it as a very fast, very tireless assistant that looks at each X-ray and says something like: "I see a possible nodule here — I'm fairly confident." It can look at thousands of images without getting tired, and it never has a bad night's sleep.

That sounds great. But there's a catch that this entire project exists to deal with, so let's sit with it.

### The danger: confident and wrong

Imagine a different kind of assistant — a new intern who is eager, fast, and *always* gives a definite answer. You show them anything and they say, "Yep, that's fine," or "Yep, that's a problem," every single time, with total confidence. Sometimes they're right. But because they never say "I'm not sure" or "this is outside what I know," you can't tell their good answers from their bad ones. The confidence is the same whether they actually know or are just guessing.

That is the dangerous thing about an AI medical tool. **A wrong answer delivered confidently is worse than no answer at all**, because a confident answer invites Dr. Rivera to trust it. If the tool says "all clear" in a firm voice on an X-ray that actually shows an early tumor, that false reassurance can send a patient home untreated. The problem isn't just that the tool can be wrong — every tool can be wrong. The problem is a tool that's wrong *and* gives no hint that this particular answer is shaky.

So the goal of a *trustworthy* tool is not "be right every time" — nothing is right every time. The goal is: **know when not to answer, and be honest about how much to trust the answers it does give.**

### A trustworthy tool has three moves, not one

A naive tool has exactly one move: answer. Every image gets an answer, period.

The tool this project builds has **three** moves. For each X-ray, it does one of these:

1. **Answer.** "I see this, and I have evidence I can stand behind here." It gives Dr. Rivera a result *and* an honest sense of how confident it is. This is the move it makes when the case looks like the kind of case it was built to handle.

2. **Abstain (defer to a human).** "This one is too uncertain for me — you take it, Dr. Rivera." Instead of guessing, it steps back and hands the hard case to the expert. Saying "I don't know" is treated as a feature, not a failure. A good assistant who passes you the genuinely tricky cases is *more* trustworthy than one who bluffs through them.

3. **Route out.** "This image doesn't look like anything I was trained on — I shouldn't even try." Picture the tool was built to read chest X-rays, and someone accidentally feeds it an X-ray of a *knee*, or a chest scan from a totally different kind of machine it has never seen. A bad tool would force a chest-finding answer anyway. This tool recognizes "this is weird, this is off my map" and sends it to a human untouched.

The difference between move 2 and move 3 is worth a beat. **Abstain** means: this *is* the right kind of image, but this particular case is hard. **Route out** means: this isn't even the right kind of image for me — I'm out of my depth on principle. Both end with a human taking over, but for different reasons, and later tiers will show why the tool keeps them separate.

### The hospital-mismatch problem

Here's the twist that runs through the whole project. The AI tool was *built and tuned* at University Medical — a big teaching hospital with its particular X-ray machines and its particular mix of patients. But Dr. Rivera is using it at County Hospital, which has **different machines** and a **different patient population**.

This matters more than it might sound. An AI tool learns from the examples it was shown. If every example came from University Medical's scanners and University Medical's patients, the tool quietly assumes the world looks like University Medical. County Hospital is a different world. The scans look a little different; the patients are a different mix; the diseases show up at different rates.

Think of someone who learned to drive in a flat, dry, sunny city and then moves somewhere hilly, icy, and dark. They're still a skilled driver — but their instincts are calibrated for the wrong place, and that mismatch is exactly where accidents happen. An AI tool moved from University Medical to County Hospital has the same problem. This mismatch between "where it was built" and "where it's used" is the single biggest reason a medical AI that looked great in testing can quietly become unreliable in the real world. The fancy name for it is **distribution shift**, and you'll meet it properly in a later tier — for now, just hold onto "the hospital it's used at is not the hospital it was built at, and that's a problem we have to actively correct for."

### What the project actually contributes (and what it doesn't)

Here's a thing that surprises people. This project does **not** invent a brand-new method or a brand-new mathematical promise. The clever pieces it uses — the part that decides when to abstain, the part that spots weird images, the part that corrects for the hospital mismatch — all already exist, published by other researchers. This project *assembles* them.

So if it isn't inventing the parts, what's the actual contribution? Two things:

**1. A measurement and honesty discipline.** Every single number this tool shows Dr. Rivera — every "I'm 80% confident," every reliability flag — is *honestly measured* and comes *with its caveats*. The project's discipline is: never show a clinician a number you can't back up, never claim a guarantee you didn't actually verify, and when the hospital mismatch makes the tool less reliable, *show that* instead of hiding it. This is harder and rarer than it sounds. Lots of medical-AI tools display a confident-looking number that was never honestly checked under real-world conditions. This project's whole stance is the opposite: measure it for real, report it plainly, including the bad news.

**2. A trust interface for the clinician** — including an explanation of *why it declined*. When the tool abstains or routes a case out, it doesn't just go silent. It tells Dr. Rivera *why*: "I'm declining this one because it's too uncertain," or "because this scan looks unlike my training data." That little explanation, delivered right where she's making the decision, is what turns a black box into a teammate she can actually reason with. And here's the subtle part the project is careful about: this explanation is about **why to trust (or not trust) the tool's output** — it is *not* an explanation of the patient's biology or what's causing the disease. The tool explains *itself and its own reliability*, not the medicine. (If you want the source, this idea is laid out in `docs/positioning_memo.md` under "Auditability as explanation.")

### Decoding the one official sentence

Researchers compress their whole contribution into one dense sentence. Here's a faithful plain-language version of this project's, taken from `docs/positioning_memo.md`. It looks intimidating, so let's translate it piece by piece, kindergarten-simple:

> *"A **trustworthy**, **auditable**, **selective-prediction** pipeline for clinical images **under distribution shift** — assembled from existing methods — whose contribution is a **measurement / auditability discipline** and a **clinician trust interface**, **not a new guarantee**."*

- **trustworthy** → you can rely on it *because* it's honest about its own limits, not because it claims to be perfect. It earns trust by admitting when it shouldn't be trusted.

- **auditable** → you can *check* it. Every number can be traced back to how it was measured. Nothing is "just trust us." If someone asks "where did that 80% come from?", there's a real, inspectable answer.

- **selective prediction** → this is the fancy name for "the three moves." *Selective* = it gets to *choose whether to answer at all*, rather than being forced to label every image. (Plain "prediction" would mean it must answer everything; *selective* prediction means it may decline.)

- **under distribution shift** → "even though it's being used at a hospital different from the one it was built at." A *distribution* here just means the whole population of cases a hospital produces — its typical mix of images and diagnoses; *distribution shift* is when that mix changes from the build hospital to the use hospital. This is the hospital-mismatch problem, named.

- **measurement / auditability discipline** → contribution #1 above: the strict habit of honestly measuring every number and disclosing its caveats.

- **clinician trust interface** → contribution #2: the screen Dr. Rivera actually sees, which makes the tool's reliability visible and explains every time it declines.

- **not a new guarantee** → the honesty clincher. The project is upfront that it is *not* promising some new mathematical certainty. It's promising to *measure and report* honestly. Claiming a guarantee it can't keep would be exactly the kind of confident-but-wrong behavior the whole project exists to avoid — so it refuses to do that even about itself. (The method note opens with this same line: it "claims no new guarantee" — see `docs/method_note.md` section 1.1.)

One more word you'll see a lot: **pipeline**. Don't overthink it. A pipeline just means "a series of steps the image passes through, one after another" — like an assembly line. The image comes in one end, goes through the steps (check if it's weird → correct for the hospital mismatch → decide answer/abstain/route → produce a confidence + explanation), and a decision comes out the other end.

### What the rest of this course will build

You now have the whole story in outline. Everything after this tier zooms into one piece of it and shows you not just *how* it works but *why this project chose it*:

- How a tool can produce an honest "how confident am I" instead of a fake one.
- How it decides the line between "answer" and "abstain."
- How it spots an image that's off its map and routes it out.
- How it corrects for the University-vs-County hospital mismatch.
- How it puts honest numbers and a plain "here's why I declined" in front of Dr. Rivera.

Each tier will keep coming back to Dr. Rivera and her chest X-rays, because the point of all of it is the same: give her a teammate that knows what it knows, admits what it doesn't, and never bluffs.

### What you can now understand

- Why a **confident-but-wrong** medical AI is dangerous: confidence invites trust, so a wrong answer with no warning can cause real harm — worse than no answer.
- That a trustworthy tool has **three moves** — *answer*, *abstain (defer to a human)*, and *route out* — instead of being forced to answer every image.
- The **hospital-mismatch** idea: the tool was built at one hospital (University Medical) but used at another (County), and that difference must be actively corrected for.
- That this project's real contribution is **not a new method or guarantee**, but (1) an honest *measurement / auditability discipline* and (2) a *clinician trust interface* that explains why the tool declined.
- How to decode the one-sentence contribution, word by word.

### Check yourself

**Q1.** Dr. Rivera's tool looks at an X-ray and is genuinely unsure whether there's a nodule. What should a *trustworthy* tool do, and why is that better than just giving its best guess?

*A1.* It should **abstain** — say "I'm not sure, you take this one" and defer to Dr. Rivera. That's better than a forced guess because a confident-looking guess on a genuinely uncertain case can mislead her; honestly flagging "I don't know" lets the human expert handle the hard case instead of trusting a coin-flip dressed up as an answer.

**Q2.** The official sentence says the project offers "no new guarantee." Isn't that admitting the project is weak? Why is it actually the point?

*A2.* It's not weakness — it's the project's core honesty, applied to itself. The contribution is *honestly measuring and reporting* the tool's reliability (including its limits under the hospital mismatch), not claiming a new mathematical certainty. Promising a guarantee it couldn't actually back up would be the exact confident-but-wrong behavior the whole project is built to prevent, so refusing to overclaim *is* the contribution working as intended.

---

*You now know *what* the project builds and *why* it matters — a tool with three moves (answer, abstain, route out) that's honest about its own limits. But we've been treating the AI as a black box. Before we can talk about *trust*, we need to crack it open and see what's actually inside: what the tool *is*, what its 'confidence' number really means, and the one quiet flaw in that number that the whole project exists to fix.*

I have section 1.3 and 1.4. Let me write the tier now, grounded in the frozen base model, softmax, features, and the bounded loss/risk material I read.

## Tier 1 — How a prediction model works (the absolute basics)

In Tier 0 you met Dr. Rivera at County Hospital and the AI tool that looks at her chest X-rays. This tier opens up that tool and shows you what is actually inside it — not the math, just the ideas. By the end you will know what the AI *is*, what its "confidence" number really means, and why one quiet flaw in that number is the thing this whole project exists to fix.

Let's build it up one piece at a time.

### A model is just a machine that takes an image and gives back a guess

Forget everything fancy. At its heart, the AI is a **machine that takes one thing in and hands one thing back out**. You feed in a chest X-ray; it hands back a guess: *"nodule"* or *"no nodule."*

That's a **classifier**: a machine that sorts each input into one of a fixed list of categories. The categories are called **classes**. For Dr. Rivera's tool the classes might be `{nodule, no nodule}` — two classes. For a different tool sorting tissue images they might be `{tumor, normal}`. The project's documents write the number of classes as `K` (so `K = 2` here), but you never need that letter again to follow along.

So far: **image in, guess out.** That guess is also called a **prediction** — same thing.

### It doesn't just guess — it also reports how sure it is

A bare guess isn't very useful to a doctor. "Nodule" — okay, but how sure are you? A confident "almost certainly a nodule" and a nervous "it's a coin-flip, maybe a nodule" should not be treated the same way.

So the machine hands back **two** things, not one:

1. a **guess** (the most likely class), and
2. a **confidence** — a number saying how sure it is.

That confidence is usually written as a percentage or a fraction between 0 and 1. "I'm 90% sure it's a nodule" is a confidence of 0.90. This pairing — *guess plus confidence* — is the real output of the model, and it is the raw material for everything the rest of this project does. Hold onto it.

### What the machine actually "looks at": features

Here's a fair question: a chest X-ray is just a grid of grey dots (pixels). How does a machine get from millions of grey dots to "nodule, 90%"?

It does it in two stages, and the first stage is the important idea for you.

Think about how *you* would describe an X-ray to a colleague over the phone. You wouldn't read out every pixel. You'd say "there's a small round white patch in the upper left, the edges are a bit fuzzy, the rest of the lung looks clear." You've **summarized** the image into a handful of meaningful descriptions.

The model does the same thing. Before it ever makes a guess, it boils the raw image down into an internal summary — a list of numbers that captures "the gist" of the image (roughly: how round, how bright, how fuzzy-edged, and hundreds of subtler things it learned on its own). Each number in that summary is called a **feature**, and the whole list is the model's internal **summary of the image**.

You don't need to know what any individual feature *means* — the model invents them, and most aren't human-readable. The thing to remember is the shape of the process:

> **raw image → internal summary (features) → guess + confidence**

Two reasons this matters later in the project. First, the guessing part only ever looks at this tidy summary, never the raw pixels again. Second — and this comes back in later tiers — that same summary turns out to be the perfect place to ask *"does this scan even look like the X-rays we've seen before?"* When County Hospital sends in something genuinely weird, it's the features that look unfamiliar. (The project's method note, section 1.3, calls this summary the *feature embedding* and writes it `φ(x)` — but the word "summary" is all you need.)

### Turning a hunch into confidences: spreading 100% across the options

So the model has its summary. How does it turn that into "nodule, 90%"?

Internally it first produces a rough, unbounded **score** for each class — a bigger score means "I lean more this way." These raw scores are ugly: they can be negative, they don't add up to anything tidy, and "nodule scores 4.1" means nothing to a human.

So there's a final, simple step that **converts those raw scores into proper confidences that add up to 100%**. Picture having exactly 100 percentage-points of belief to hand out, and you must spread *all* of them across the available classes — the more a class scored, the bigger its slice. If "nodule" scored much higher than "no nodule," the split might come out 90% / 10%. If they scored close, maybe 55% / 45%. The slices always add up to 100%, because you're dividing up one fixed pile of belief.

That "spread 100% across the options" step has a name — **softmax** — and that's genuinely all softmax is: *the rule that turns raw scores into a set of confidences that sum to 100%.* When you read "softmax" in the project docs (method note section 1.3), just hear "the thing that spreads 100% across the classes." The biggest slice becomes the model's guess; the size of that slice is its confidence.

> **Under the hood (skippable).** Docs write the raw scores as `f(x)` (the "logits") and the softmax output as `σ(f(x))` — a list of confidences in `[0,1]` that sums to 1. The guess is `ŷ(x) = argmax`, just "whichever class got the biggest slice." Skipping this changes nothing below.

One more word you'll see: this tool is built on a **frozen** base model. "Frozen" just means the image-reading part was trained once, elsewhere (at University Medical, in our story), and is then **locked** — this project does not retrain it. It takes the model as-is and builds a *trust layer* around it. Why that's a deliberate choice is a later-tier story; for now, "frozen" = "we don't touch the model's insides, we work with its outputs."

### How wrong is it? Accuracy, error, and risk

A model that guesses is only worth anything if we know how *often it's right*. Three plain words:

- **Accuracy** — the fraction it gets right. Out of 100 X-rays, if it calls 92 correctly, that's 92% accuracy.
- **Error** — the flip side, the fraction it gets wrong. Here, 8%. Accuracy and error always add to 100%.
- **Risk** — this is just a slightly more careful word for *average error*, and the project uses it constantly, so let's nail it.

Why a separate word for "average error"? Because not every mistake costs the same. Calling a real nodule "no nodule" (a miss that could delay a cancer diagnosis) is far worse than calling a clear lung "possible nodule" (which just earns a second look). Plain accuracy treats both as one identical "oops." **Risk** lets you attach a *cost* to each kind of mistake and then average those costs — so a model that makes its few mistakes on the *dangerous* side scores worse than one that errs on the safe side, even at the same accuracy.

The simplest cost, where every mistake counts as exactly 1 and every correct answer counts as 0, makes "risk" collapse right back into plain "error rate." The project starts there and then also reports results with the dangerous misses weighted more heavily — because Dr. Rivera cares which *kind* of mistake the tool makes.

> **Under the hood (skippable).** The per-case cost is the **loss**, written `ℓ(y, ŷ)` — `y` is the true answer, `ŷ` the guess. Default is **0–1 loss**: `ℓ = 1` if wrong, `0` if right. Risk is the average of `ℓ`. The project insists every loss sit between 0 and 1 (method note section 1.4) for a machinery reason that lands in a much later tier — you can ignore it for now. The severity weighting (dangerous misses count more) is, in the project's own words, "a reporting choice, not a certified clinical cost" — i.e. *we* chose how to weight, and we say so out loud.

### The three piles of data, and the one sacred rule

Now the single most important habit in this whole field — and a place beginners' intuition usually goes wrong.

To build and check a model honestly you split your X-rays into **three separate piles**, and you guard the walls between them like your reputation depends on it (it does):

1. **Training data** — the pile the model *learns* from. It studies these, with the right answers attached, to figure out its features and scores. (In our story this learning happened back at University Medical.)
2. **Calibration data** — a *fresh* pile, not used for learning, used to *tune the dials*: where to set the confidence cutoffs, how to adjust the numbers. Most of what this project actually does lives here. (You'll meet this pile again and again in later tiers.)
3. **Test data** — a pile the model has *never seen in any way*, used once at the very end to *measure* how good it really is. This is the stand-in for "real patients it'll face tomorrow."

Now the sacred rule: **never test on data you trained on. No peeking.**

Here's the intuition, and it's worth feeling in your gut. Imagine a student who gets the exam questions *and the answer key* a week early, memorizes them, and scores 100%. Did they learn medicine? You have no idea — the test no longer measures understanding, it measures memorization. A model that's checked on its own training images is exactly that student. It can score beautifully by having effectively memorized those specific images, then **fall apart on the first real patient** it's never seen. The number you'd proudly show Dr. Rivera would be a lie — flattering, and useless.

This contamination has a name: **leakage** (data sneaking from one pile into another where it doesn't belong). The whole point of keeping the piles **disjoint** — strictly separate, no overlap — is to keep every reported number *honest*. And this project takes it further than most: because chest X-rays from one *patient* (or many tissue patches from one tumor slide) look alike, it makes sure an entire patient lands wholly in one pile — never split across two — so the model can't "recognize" a patient it half-memorized. The project's method note even has the code print a check confirming the piles share zero patients. That obsession with no-leakage *is* the "measurement / auditability discipline" Tier 0 promised was this project's real contribution. You're seeing the first brick of it.

### The quiet flaw that this whole project is about: miscalibration

Everything so far has been setup. Here is the punchline of Tier 1 — the idea that motivates the entire research project.

The confidence number can be **wrong**. Not "the guess is wrong" — the *confidence itself* can be untrustworthy. And a wrong confidence is sneakier and more dangerous than a wrong guess.

What should a confidence number mean? Here's the only sensible definition, and it's beautifully simple:

> **When the tool says "70% sure," it should turn out to be right about 70% of the time.**

Gather up every case where the tool said "70%." If it was correct in roughly 70 of every 100 of them, the confidence is **calibrated** — it means what it says. Its "70%" is an honest 70%.

But suppose you gather those "70%" cases and the tool was actually only right **half** the time. Then its confidence is a liar: it *says* 70 but *delivers* 50. That's a **miscalibrated** model — the confidence numbers don't match reality. (Calibration is exactly why the test pile exists: it's the only honest place to check whether "70%" really lands at 70%.)

Now feel why this is the dangerous one. Dr. Rivera leans on that number to decide. A tool that says **"92% sure: no nodule"** when its real accuracy at that setting is closer to **70%** is *quietly overconfident* — and it will sail past a real nodule wearing a reassuring high number, so neither the model nor Dr. Rivera hits the brakes. A confidently-wrong tool is worse than an honestly-unsure one, because the false confidence is precisely what switches off human caution.

And here is the hook into the rest of the project. Remember from Tier 0 that the tool was built at University Medical but used at County Hospital — different scanners, different patients. A model that was perfectly calibrated back home can become **miscalibrated the moment it's moved**, because the new hospital's images and case-mix don't match the old ones. The confidence numbers silently drift out of sync with reality — and nobody in the room can see it happening.

That single failure — **confidence numbers that no longer tell the truth after the tool changes hospitals** — is the precise danger this project is built to catch and correct. Everything in the later tiers (knowing when to *abstain*, *routing out* weird scans, *correcting* for the new hospital, and showing Dr. Rivera an honest reliability readout) is, at bottom, machinery for keeping that confidence number trustworthy. Now you know what it's protecting.

### What you can now understand

- A **classifier** turns an image into a **guess plus a confidence**, by first boiling the image down into an internal **summary (features)** and then **spreading 100% across the classes** (that spreading step is **softmax**); the model is **frozen**, meaning this project builds around it without retraining it.
- **Accuracy** is how often it's right, **error** is how often it's wrong, and **risk** is *average error with costs attached*, so that dangerous mistakes (missing a real nodule) can count more than safe ones.
- Data is split into three disjoint piles — **training** (learn), **calibration** (tune the dials), **test** (measure honestly) — and the **no-leakage rule** ("no peeking") is what keeps every reported number from being a flattering lie.
- A confidence is **calibrated** when "70% sure" really means "right ~70% of the time," and **miscalibrated** when it doesn't — and an overconfident, miscalibrated model is the specific danger to Dr. Rivera, because false confidence switches off human caution.
- Moving the tool from University Medical to County Hospital can break calibration on its own — which is the core problem the whole project sets out to measure and correct.

### Check yourself

**Q1. A tool reviews 200 chest X-rays and, on the ones where it announced "80% confident," it was actually correct only 120 times. Is it calibrated? Why does this endanger Dr. Rivera?**
*Answer:* No. "80% confident" should mean right about 80% of the time — that's 160 out of 200 — but it was right only 120 (60%). It's **miscalibrated**, and overconfident. Dr. Rivera, trusting that inflated 80%, will give those cases less scrutiny than they deserve, so genuine findings can slip through under a falsely reassuring number.

**Q2. A colleague brags that their model scores 99% accuracy — measured on the very same X-rays it trained on. Why should you be unimpressed?**
*Answer:* Because that breaks the no-peeking rule: it's the student who saw the answer key first. The 99% may just reflect the model **memorizing those specific images** (data **leakage**), and tells you almost nothing about how it'll do on new patients. A trustworthy accuracy number must come from a separate **test** pile the model has never seen.

---

*You've seen what's inside the tool: it turns an image into a guess plus a confidence, and that confidence can quietly lie (miscalibration), especially after the tool changes hospitals. Now we add the single most important *behavior* in the whole project — the one Tier 0 called a 'move.' Instead of being forced to answer every X-ray, the tool learns to say 'I don't know, you take this one.' Here's how that decision actually works.*

I have section 2 (and the relevant parts of section 1) read. I have what I need to teach Tier 2 accurately. Here is the tier.

---

## Tier 2 — Teaching a model to say "I do not know" (selective prediction)

In Tier 1 you met a model that looks at a chest X-ray and produces two things: a guess (say, "lung nodule") and a confidence (a number for how sure it is). You also met the idea of error, or *risk* — how often the model is wrong. Now we add the single most important behavior in this whole project: letting the model **choose not to answer**.

### The big idea: a model that can decline

Imagine Dr. Rivera has a brilliant but overconfident junior colleague who insists on giving a definite read on *every single* X-ray that crosses the desk — even the blurry ones, even the strange ones, even the cases that would make a 20-year radiologist pause. That junior is dangerous, not because they're stupid, but because they never signal *"this one is beyond me — you'd better look."*

A much better colleague does something simple: on the clear cases, they give a confident read; on the genuinely hard ones, they say **"I'm not sure, can you take this one?"** and hand it back to Dr. Rivera. That hand-back is not failure. It's exactly the behavior that makes the colleague trustworthy.

Teaching a model to do this is called **selective prediction** (or *learning to abstain*). The model is allowed to either **answer** a case or **abstain** — meaning it declines and **defers** the case to a human. That's the whole idea. The rest of this tier is just making it precise.

This is the trust mechanism at the heart of the entire project. In the method note (`docs/method_note.md`, section 2) the very first sentence is blunt about it: *"Abstention is the trust mechanism of this pipeline."* Everything else — the shift corrections, the out-of-distribution routing — exists to make this answer-or-defer decision honest.

### The gate: answer if confident enough, otherwise defer

How does the model decide *when* to abstain? With a simple rule based on the confidence you met in Tier 1.

Flip confidence around into its opposite: an **uncertainty score** — a single number that is **high when the model is unsure** and low when it's confident. (If the model is 95% sure, uncertainty is low; if it's split 51%/49% between two diagnoses, uncertainty is high.)

Now pick a cutoff line, a **threshold**. The rule is:

- uncertainty **below** the threshold → **answer**
- uncertainty **above** the threshold → **abstain / defer to a human**

That's the **gate**. Picture a turnstile: cases the model feels confident about pass through and get an answer; cases above the uncertainty line get turned away from the turnstile and sent to Dr. Rivera instead. The threshold is literally where you draw that line. Move the line and you change how picky the model is.

The method note writes the threshold as the symbol **tau** (`τ`) and the uncertainty score as `u(x)`, but you never need the symbols to get the idea: there's a dial, and the dial sets *how sure is sure enough to answer.*

> **Under the hood (skippable).** The gate is written `g(x) = 1[ u(x) ≤ τ ]`. Read it left to right: `u(x)` is the uncertainty of image `x`; `τ` is the threshold; the bracket `1[...]` equals 1 when the thing inside is true and 0 when it's false. So `g = 1` ("answer") exactly when uncertainty is at most the threshold, and `g = 0` ("defer") otherwise. A common choice for `u(x)` is `1 − (the model's top probability)`: if the model's best guess only gets 0.6 probability, uncertainty is 0.4. The surrounding prose is the whole idea; the symbols just make it exact.

A crucial design choice the project makes here: abstaining is framed as **deferring to a clinician, not as the system silently guessing or silently failing**. When the model abstains on Dr. Rivera's hard case, the case doesn't vanish and it doesn't get a quiet coin-flip answer — it gets routed to a human who's told "the system declined this one." The accepted cases are the region where *the model takes responsibility*; the abstained cases are explicitly somebody else's job. (Section 2.1 of the method note makes a point of this: "we deliberately frame g = 0 as deferral to a clinician, not silent failure.")

### Two numbers that describe a selective model

Once a model can decline, two questions immediately matter, and they're the two numbers you'll see everywhere in this project.

**1. How often does it answer at all?** That fraction is the **coverage**. If the model answers 800 out of 1,000 X-rays and defers 200 to Dr. Rivera, its coverage is 0.80, or 80%. High coverage = the model is doing a lot of the work itself. Low coverage = it's handing most cases to the human.

**2. When it does answer, how often is it wrong?** That's the **selective risk** — the error rate *measured only among the cases it chose to answer.* This is the subtle, important part: selective risk **ignores the deferred cases entirely.** It is not "how often is the model wrong overall." It is "of the cases the model stood behind, how often did it get them wrong."

A worked example makes the difference vivid. Suppose across all 1,000 X-rays the model would be wrong 100 times — a 10% overall error rate. But it's smart about *which* cases it answers: it covers the 800 easy ones and defers the 200 hardest ones, and among those 800 easy answered cases it's wrong only 24 times. Then:

- coverage = 800 / 1000 = **80%**
- selective risk = 24 / 800 = **3%**

By stepping back from the hardest 200 cases, the model's error *on the cases it actually answered* dropped from 10% to 3%. That 3% is the number Dr. Rivera actually cares about, because it describes the answers she's actually being handed.

> **Under the hood (skippable).** The method note writes coverage as `coverage = P(g(X)=1)` — the probability that the gate says "answer" — and selective risk as `R^accept_P := E_P[ ℓ(Y, ŷ(X)) | g(X)=1 ]`. Decoded: `ℓ(Y, ŷ(X))` is the loss from Tier 1 (1 if the prediction `ŷ` is wrong, 0 if right); the bar `| g(X)=1` means "averaged **only over** the cases where the gate answered." So selective risk is just "average error, restricted to answered cases" — exactly the 3% above. The superscript `accept` is a reminder: accepted (= answered) cases only.

### The see-saw: you can't max out both

Here's the tension that makes this interesting, and it's the reason picking the threshold is a real decision and not an afterthought.

The threshold is a single dial, and it trades the two numbers against each other like a **see-saw**:

- **Tighten the gate** (lower the threshold — only answer when *very* sure): you now answer fewer cases, and the ones you answer are the easy, obvious ones. Selective risk goes **down** (you're only attempting the gimmes), but coverage goes **down** too (you're handing more to the human).
- **Loosen the gate** (raise the threshold — answer even when only somewhat sure): you answer more cases, so coverage goes **up** — but now you're attempting harder cases, so selective risk goes **up**.

You cannot have both maximal coverage and minimal risk at the same time, any more than you can have both ends of a see-saw up at once. Every selective model lives somewhere on this risk–coverage curve, and choosing the threshold means **choosing where on the curve to sit.** The method note (section 2.1) names this directly: "There is an intrinsic trade-off — lowering τ answers fewer, easier cases and drives R^accept down at the cost of coverage; raising τ answers more at higher risk."

### The trap: near-zero risk by answering almost nothing

Now the trap — and it's a trap precisely *because* the see-saw is real.

Suppose someone advertises: "Our medical AI has a selective risk of **0.1%** — it's almost never wrong!" Sounds spectacular. But ask the second question: *what's its coverage?* If the answer is "it only answers the 3% of cases it's absolutely certain about and defers everything else," then the impressive 0.1% is nearly meaningless. The model achieved near-zero error by **refusing to do almost any work.** It dumped 97% of the cases — including plenty it could have handled fine — straight onto Dr. Rivera.

This is the central failure mode of selective prediction: **you can drive selective risk arbitrarily close to zero by abstaining on almost everything.** A model that answers *one* easy case and defers the other 999 can boast a perfect record on "the case it answered." It's useless, but its selective risk looks flawless. Risk alone, with coverage hidden, is a vanity number.

That's why selective risk is **never reported without coverage** in this project, and why there's a **minimum-coverage floor** — a rule that says, in effect, "you must answer at least *this* fraction of cases; you're not allowed to buy a low risk by hiding behind near-total abstention." The floor stops the model from gaming its own headline number. The honest target is the opposite of the trap: **low selective risk while still covering a clinically useful share of cases.** A tool that defers 95% of County Hospital's X-rays isn't trustworthy — it's just absent, and Dr. Rivera is doing all the work anyway.

### Why this project leans on abstention so hard

It's worth saying plainly why abstention — rather than, say, "just build a more accurate model" — is the load-bearing idea here.

The whole project is about a model that was built at **University Medical** but deployed at **County Hospital**, on different scanners and a different patient mix. No matter how good the model is, there will be County cases it has no business giving a confident answer on — cases that look unlike its training, or that sit right on the edge of its competence. Forcing an answer on those is exactly how a medical AI hurts a patient. The ability to **step back and say "human, please"** is the safety valve that makes deploying the model across that hospital gap defensible at all.

So abstention isn't a feature bolted on for politeness. It's *the* mechanism by which the system earns Dr. Rivera's trust: it answers where it has earned the right to, and it gets out of the way where it hasn't. (How the threshold gets *set* responsibly — so that the selective risk on answered cases actually meets a target Dr. Rivera specifies — is its own careful procedure, and that's what the next tier builds toward.)

### What you can now understand

- **Selective prediction / abstention**: a model is allowed to either answer a case or decline and **defer** it to a human, instead of being forced to guess on everything.
- **The gate and threshold**: an **uncertainty score** plus a cutoff (the threshold) decides answer-vs-defer — confident enough → answer, otherwise → hand it to the clinician.
- **Coverage** (the fraction of cases it answers) and **selective risk** (the error rate *among only the answered cases*) are the two numbers that describe any selective model — and they must always be reported together.
- **The see-saw trade-off**: tightening the gate lowers selective risk but also lowers coverage, and vice versa — you pick a point on the risk–coverage curve.
- **The "near-zero risk" trap** and why a **minimum-coverage floor** exists: a model can fake a perfect record by answering almost nothing, so risk is only meaningful alongside a clinically useful coverage.

### Check yourself

**Q1.** A vendor says their X-ray AI has a selective risk of 0.5%. What's the one follow-up question you must ask before you're impressed, and why?
*Answer:* "What's the coverage?" A tiny selective risk is easy to fake by abstaining on nearly every case — answering only a handful of ultra-easy ones. Without knowing what fraction of cases it actually answers, the 0.5% tells you almost nothing about whether the tool is useful. (This is the near-zero-risk trap, and it's why the project enforces a minimum-coverage floor.)

**Q2.** Dr. Rivera's hospital lowers the uncertainty threshold so the AI only answers when it's *very* confident. What happens to coverage, and what happens to selective risk, and why?
*Answer:* Coverage goes **down** (the model now answers fewer cases, deferring more to Dr. Rivera) and selective risk goes **down** too (the cases it still answers are the easy, high-confidence ones, where it's rarely wrong). That's the see-saw: you bought lower error on answered cases by doing less of the work yourself.

---

*So the tool can now decline the cases it's unsure about, and we measure it by two numbers: coverage (how much it answers) and selective risk (how often it's wrong when it does). But there's a deeper question hiding underneath. A confidence score is still just the model's *opinion* of itself. How do you turn that opinion into an actual, measured promise — 'when this tool answers, it's wrong at most 5% of the time'? And what's the one assumption that whole promise secretly leans on?*

I have section 2.2 and the surrounding context. Let me read section 1.6 region more carefully — I already have it (lines 92-101). I have enough to write the tier accurately. Let me write it.

## Tier 3 — Making the confidence trustworthy (conformal prediction and risk control)

You ended the last tier knowing that the model can attach a *confidence* to each prediction, that calibration tries to make that confidence honest, that the system can *abstain* instead of guessing, and that we measure two things on the cases it answers: how often it answers (coverage) and how often it's wrong when it does (selective risk). (A quick word-sense note, since the root "calibrat-" does three jobs in this course: a *calibration set* is a pile of held-out examples; *calibrated* is a property of a confidence number — "70% really means 70%"; and *recalibration*, which you'll meet in Tier 6, is the step that fixes a number that isn't. Same root, three jobs.)

Now we hit the central question of the whole project: **how do you turn a confidence score into an actual promise about the error rate?** Dr. Rivera doesn't want a number that "feels calibrated." She wants to be able to say: "When this tool answers, it's wrong at most 5% of the time, and I have good reason to trust that 5%." This tier is about how you earn that sentence — and, just as important, about the one assumption the whole promise leans on.

### Why a raw confidence is not enough

Suppose the model looks at a chest X-ray and outputs "90% confident: no nodule." Can Dr. Rivera conclude she'll be misled at most 10% of the time?

Not yet — and here's the honest reason. That 90% is the model's *opinion about itself*. Even after the calibration work of the last tier, it's a number the model produces about each case; it is not a measured, audited error rate over real cases. Calibration nudges those opinions toward honesty on average, but "on average, roughly honest" is a long way from "I promise the answered cases are wrong at most 5% of the time, and here's how sure I am of that promise."

Think of it like a student who says "I'm 90% sure I passed." That's a feeling. The *measured* pass rate — counted over many real exams — is a fact. To make a promise Dr. Rivera can stand behind, we need to move from the model's feeling to a measured, controlled error rate. That move is what **conformal prediction** and **risk control** are for.

### The core trick: use a pile of graded practice exams

Here is the whole idea in one picture.

Before deployment, we hold back a pile of images that the model has **never been trained on**, and for which we **already know the right answer**. Call it the *calibration pile*. Think of it as a stack of graded practice exams: for every one, we can check whether the model would have been right or wrong, and how confident it was.

Now we ask a very concrete question. We have a knob — the **confidence cutoff** — that decides which cases the model is allowed to answer. (This is the selection threshold from the last tier: answer when uncertainty is low enough, abstain otherwise.) If we set that cutoff to some value and apply it to the calibration pile, we can literally *count* the error rate among the cases that pass the cutoff. No theory, no bell curves — just counting on graded exams.

So we sweep the knob:

- Cutoff very strict → the model answers only the easiest cases → counted error rate is tiny, but it abstains a lot.
- Cutoff loose → it answers almost everything → counted error rate climbs.

We pick the cutoff that keeps the counted error rate at or under the target Dr. Rivera asked for. That cutoff is our **operating point**: the setting we'll actually ship. Because we tuned it on real graded examples rather than on the model's self-assessment, the error rate it delivers is a *measured* quantity, not a hope.

That is the conformal intuition: **let a pile of held-out, already-graded examples tell you where to put the cutoff so the error stays under a target.** (In the method note this is the "calibration set" `D_cal` and the threshold-tuning of section 2.2.)

### One more layer of honesty: the calibration pile is itself a sample

There's a subtlety, and it's exactly the kind of thing this project insists on getting right.

The calibration pile is finite. Maybe it's 2,000 graded exams. If we'd happened to draw a *different* 2,000, the counted error rate at any given cutoff would have come out a little different — by luck of the draw. So when we count "4% error at this cutoff," that 4% has some wiggle in it. If we ship a cutoff that *just barely* hit the target on this particular pile, we might have gotten lucky, and the real error could be a bit higher.

The grown-up response is: **don't trust the raw count — add a safety margin for the luck-of-the-draw.** Instead of asking "is the counted error under the target?", we ask "is the error under the target even after I account for how much this estimate could be off because my pile is finite?" In practice that means computing an *upper* estimate of the error — a conservative ceiling — and requiring *that* ceiling to sit under the target before we accept a cutoff.

The recipe that does this in our project is called **RCPS** — Risk-Controlling Prediction Sets. In plain terms, RCPS is the rule:

> Pick the operating cutoff so that, with a built-in safety margin for finite-sample luck, the error rate among answered cases stays under a target you chose in advance.

That's it. It's the "graded practice exams" idea plus an honest accounting of the fact that the practice pile is only a sample. (Method note section 2.2 states RCPS as the controlling method, citing Bates et al. 2021.)

### The two knobs: how much error, and how sure

RCPS gives Dr. Rivera's team two dials to set, in advance, in plain language. The project calls them by symbols, but the meaning is everyday.

**Dial 1 — how much error you'll tolerate.** This is the target error rate among answered cases. The note writes it `alpha_acc` ("alpha" is just its name; the "acc" is for *accepted* cases — the ones the model answered). If the team sets `alpha_acc = 0.05`, they're saying: "Among the X-rays this tool actually answers, I want it wrong no more than 5% of the time." Smaller `alpha_acc` = a stricter promise = the tool will abstain more often to keep that promise.

**Dial 2 — how *sure* you want to be about the promise.** Remember the finite-pile wiggle. Dial 2 controls how much safety margin we build in against that wiggle. The note writes it `delta`. Read it as: "I want to be at least `1 − delta` sure that the error-rate promise actually holds." If `delta = 0.05`, the promise comes with "95% confidence."

Two things worth pinning down, because beginners mix them up:

- `alpha_acc` is about **the error rate itself** — how wrong is too wrong.
- `delta` is about **our confidence in the interval/promise** — how sure are we that we really hit the target, given that we only had a finite pile to measure on.

You can have a loose error target with high confidence, or a strict error target with lower confidence, and so on — they're independent dials.

> **Under the hood (skippable).** RCPS chooses the operating threshold `λ̂` by the rule: take the *smallest* threshold such that an upper confidence bound `UCB_δ(λ)` on the risk stays below `alpha_acc` for all looser thresholds. The guarantee it then states is `P( R(λ̂) ≤ alpha_acc ) ≥ 1 − delta` — "with probability at least `1 − delta` over the draw of the calibration pile, the true risk of the chosen rule is at most `alpha_acc`." The `UCB` is a variance-aware "betting" bound (Waudby-Smith & Ramdas 2024), chosen because later weighting steps inflate variance and a range-only bound would be needlessly loose. None of this changes the plain-language reading above; the surrounding text stands on its own.

### "Distribution-free" — a strength worth naming

One genuinely nice property of this style of method: it is **distribution-free**.

Plainly: we never had to assume the data follows any particular tidy shape — no bell curve, no "errors are normally distributed," none of the textbook assumptions that real medical images cheerfully violate. The whole promise was built by *counting on graded examples*, and counting doesn't care what shape the underlying data takes. That's why the project leans on these tools: clinical images are messy and high-dimensional, and a method that demands a clean statistical shape would be a non-starter.

But — and this is the heart of the tier — "distribution-free" does **not** mean "assumption-free." There is exactly one load-bearing assumption left, and everything rides on it.

### The load-bearing assumption: exchangeability

Here is the catch, and it's worth slowing down for.

The graded-practice-exam trick only works if the practice exams genuinely resemble the real exam. If our calibration pile is full of one kind of case, but the cases Dr. Rivera actually sees are systematically different, then the cutoff we tuned on the pile won't deliver the promised error rate in her clinic. We tuned to the wrong world.

The precise version of "the practice pile and the real cases come from the same world" is called **exchangeability**. Intuition: imagine pooling all your calibration images and all your real deployment images into one big bag, with their labels. If you couldn't tell, from the data alone, which ones were "practice" and which were "real" — if they're *interchangeable* — then they're exchangeable, and the cutoff you tuned on practice carries over to real cases. The promise holds.

**Exchangeability is the single assumption holding up the entire guarantee.** Distribution-free buys you freedom from *shape* assumptions; it does not buy you freedom from the *same-world* assumption. The method note is emphatic about this (section 2.2): RCPS gives its `(alpha_acc, delta)` promise *only when the calibration set and the test cases are exchangeable.*

### Why this matters so much for *this* project

Now thread it through Dr. Rivera's situation, because this is exactly where the project's honesty stance comes from.

The AI was built and calibrated at **University Medical** — its scanners, its patient mix, its disease prevalence. Dr. Rivera works at **County Hospital** — different scanners, different patients, a different mix of who's actually sick. So the calibration pile (University) and the real cases (County) are drawn from *different worlds*. They are **not** exchangeable. You could probably tell a County X-ray from a University one just by looking.

That means the clean RCPS promise — "answered-case error at most `alpha_acc` with confidence `1 − delta`" — **does not automatically transfer to County.** The certificate was earned under exchangeability, and deployment breaks exchangeability on purpose.

This is precisely why the project refuses to print the guarantee as if it still held in the clinic. Its whole stance is: invoke RCPS's promise only as a property of RCPS *under its own assumption*, and then, where that assumption is broken by the move from University to County, **measure and report** what the error rate actually turns out to be on held-out County data — rather than reassert a certificate that no longer applies. The promise becomes a *measured number with its caveats attached*, which is exactly the auditability discipline this paper is built around (method note sections 2.2 and 3.1).

**The big lesson, and the one to carry forward:** every guarantee rides on an assumption. A "distribution-free guarantee" still has one — exchangeability — and in real clinical deployment that one is the first to break. The honest move is not to hide the assumption but to name it, watch for where it fails, and measure the damage.

And here's the foreshadow: the *next* tier doesn't just notice that exchangeability breaks — it deliberately confronts the University-vs-County gap and tries to *correct* for it, by reweighting the calibration pile so it better resembles County. That's where the "shift" machinery comes in, and it's the move that lets the system claw back some trust even when the two worlds differ.

### What you can now understand

- **A raw confidence is the model's opinion about itself; a controlled error rate is a measured fact.** Turning the first into the second is the job of conformal prediction / risk control.
- **The mechanism is counting on graded practice exams (held-out calibration data):** sweep the answer/abstain cutoff and pick the one that keeps the *measured* error under a target — and add a safety margin because the pile is finite (that's RCPS).
- **Two knobs:** `alpha_acc` = how much answered-case error you tolerate; `delta` = how sure you want to be that the promise really holds. They are independent.
- **"Distribution-free" means no bell-curve-style shape assumptions** — a real strength for messy medical images — **but it is not assumption-free.**
- **Exchangeability ("practice pile and real cases come from the same interchangeable world") is the one load-bearing assumption.** University-vs-County deployment breaks it, which is exactly why this project measures and reports real error instead of reasserting the certificate.

### Check yourself

**Q1. The team sets `alpha_acc = 0.05` and `delta = 0.10`. In plain words, what did they ask for?**
A1. "Among the cases the tool actually answers, keep the error rate at or under 5% — and I want to be at least 90% sure (that's `1 − delta`) that this 5% promise really holds, accounting for the fact that we only measured on a finite calibration pile." `alpha_acc` is the tolerated error; `delta` is our uncertainty about the promise itself.

**Q2. The method is "distribution-free," yet the project still says its guarantee may not hold at County Hospital. Is that a contradiction?**
A2. No. "Distribution-free" only means we didn't assume a particular *shape* (no bell curve, etc.). The guarantee still rests on **exchangeability** — that the calibration cases and the real cases come from the same interchangeable world. University and County have different scanners and patient mixes, so they're *not* exchangeable, and that's the assumption that fails — not a shape assumption. The promise breaks because its one remaining assumption is violated, which is why the project measures the real error at County instead of reprinting the University certificate.

---

*Tier 3 ended on a cliffhanger: the entire error promise rests on one assumption — exchangeability, the idea that the cases we tuned on look like the cases we'll really face. And we hinted that moving from University to County breaks it. This tier confronts that break head-on. It names exactly *how* the two hospitals differ, *why* that breaks the promise, and the clever trick the project uses to claw back some trust anyway.*

I have everything I need. Now I'll write Tier 4.

## Tier 4 — The new-hospital problem (distribution shift)

You have come a long way. You know the AI can **abstain** instead of forcing an answer (Tier 2), that **conformal/RCPS** lets us tune a threshold so the error rate among answered cases stays under a target like 5% (Tier 3), and that the whole promise rests on one quiet assumption — **exchangeability**: the cases we tuned the threshold on have to look statistically like the cases we'll later face (Tier 3). The promise is **distribution-free**: it doesn't care *what* the data looks like, as long as the tuning data and the real data are drawn from the same pool.

This tier is about what happens when that "same pool" assumption is a lie. And in real hospitals, it almost always is.

### The setup: County is not University

Remember the story. The AI was built and tuned at **University Medical**. Dr. Rivera uses it at **County Hospital**. These are two different places, and that difference is the whole problem of this tier.

We'll give the two places names that the project uses everywhere:

- **Source** = where the AI was built and calibrated. That's University Medical. The threshold from Tier 2 was tuned on *source* data.
- **Target** = where the AI is actually deployed. That's County Hospital, where Dr. Rivera works.

The AI was tuned on the source but lives on the target. If source and target were identical, exchangeability would hold and Tier 2's promise would carry over untouched. They are not identical. They differ in **two distinct ways**, and keeping these two ways separate is the single most important idea in this tier.

### The two flavors of shift

**Flavor 1 — the images themselves look different (covariate shift).**

County Hospital bought its X-ray scanners from a different manufacturer than University. The machines expose differently, the image contrast is a touch darker, the edges are sharper or softer. A perfectly healthy County lung and a perfectly healthy University lung are the *same lung medically* — but the **pixels** are not the same. The picture looks different even when the underlying truth is identical.

That's **covariate shift**: the *inputs* (the images, the "covariates") have shifted, but the relationship between a given image and its correct diagnosis hasn't. A lung that looks like *that* still means the same thing — it's just that County produces more lungs-that-look-like-that than University did.

**Flavor 2 — the mix of patients is different (label/prevalence shift).**

University Medical is a referral center: sick, pre-screened patients get sent there, so maybe 30% of the chest X-rays it sees actually have a nodule. County Hospital is a community ER: most people walking in have a cough or a sprain, and maybe only 8% of its X-rays have a nodule. The *diseases* look identical at both places — a nodule is a nodule. What changed is **how common each diagnosis is**.

That's **label/prevalence shift**: the mix of *answers* (the "labels" — nodule vs. no-nodule) has shifted. Same diseases, different frequencies. "Prevalence" is just the medical word for how common a condition is in a population.

The project's Method Note lays out this exact taxonomy in section 1.2, and it's worth holding onto the contrast: covariate shift moves **p(x)** — the distribution of *what the image looks like*; label shift moves **p(y)** — the distribution of *what the answer is*.

> **Under the hood (skippable).** The doc writes the source and target distributions as `P_S` and `P_T`. Covariate shift: `p(x)` changes while `p(y∣x)` (the diagnosis-given-the-image) stays fixed. Label shift: `p(y)` changes while `p(x∣y)` (what each disease *looks* like) stays fixed. You can skip the symbols — the two English sentences above are the whole idea.

### Why this breaks the Tier 3 promise

Here is the chain of reasoning, slowly.

Tier 2 tuned the threshold so that, on University's data, the error rate among answered cases sat under 5%. That number was *true* — for University. The proof in Tier 2 only works when the tuning cases and the real cases are exchangeable.

But County's images look different and County's patient mix is different. So the cases Dr. Rivera actually feeds the AI are **not** drawn from the same pool the threshold was tuned on. Exchangeability is broken. The 5% the AI was tuned to deliver is no longer the number Dr. Rivera actually gets — it might be 5%, but it might quietly be 11%, and nobody warned her.

A threshold tuned at University is simply the **wrong threshold** for County. Not because anyone made a mistake — because the world moved underneath it. This is why the Method Note (sections 2.2 and 3.2) is so insistent that the Tier 2 guarantee is "a property of the method **under exchangeability**" and openly states that exchangeability is broken in deployment. The honesty of the whole project starts right here.

### The fix, in one idea: count some cases more heavily

So University's tuning data is the wrong mix. Do we throw it away and re-tune at County? We can't — re-tuning needs lots of *labeled* County cases (images where we already know the right answer), and those are exactly what we don't have yet. Getting them means radiologists hand-reading thousands of County scans.

Here's the trick that saves us. We *do* have University's labeled data. What if we could **reweight** it — pretend University's data mix looks like County's — by counting some cases more heavily than others?

Think of it like adjusting a poll. If your survey accidentally interviewed too many retirees and not enough young people, you don't re-run the survey. You **weight** the young respondents more heavily so the adjusted sample matches the real population. Same data, reweighted, now representative.

We do exactly that with University's cases:

- A University case whose image looks like the kind of image County produces a lot → **count it more heavily.**
- A University case whose image is rare at County → count it less.

That's the engine. It's called **importance weighting**, and the weight on each case is how much we "boost" it to make University's mix imitate County's. The Method Note (section 1.5) calls the overall weight `w(x,y)`.

### Where the two weights come from (intuition only)

Because there are two flavors of shift, there are two weights, and they come from two different machines.

**The covariate weight — "does this image look like a County image?"** We train a little helper model whose only job is to look at an image and guess: *did this come from University or from County?* This is called a **domain discriminator** (Method Note section 1.5). If it looks at a University image and thinks "this really looks like the County batch," that image gets a high covariate weight — boost it. If it screams "obviously University," that image gets downweighted. The discriminator is essentially a County-detector, and how County-ish an image looks becomes its covariate weight, `w_cov(x)`.

**The label weight — "is this diagnosis more common at County?"** This one is about the *mix of answers*, not the look of images. We can't just count diagnoses at County because we don't have County's labels. So we use a clever indirect method (the project uses **BBSE** and **MLLS** — section 1.5). The intuition: run the AI on a pile of *unlabeled* County scans and watch *what it predicts*. If it suddenly predicts "nodule" far less often at County than it did at University, and we know how trustworthy its predictions are (from a thing called the **confusion matrix** — a table of how often the model confuses each diagnosis for another at University), we can work backwards to infer the true County prevalence. That gives the label weight, `w_lab(y)`: how much more or less common each diagnosis is at County.

You don't need the machinery. You need: **one weight measures how County-ish the picture looks, the other measures how County-ish the diagnosis mix is.**

### The trap: do NOT just multiply the two weights

Now the subtle part, and the part the project is genuinely careful about (Method Note section 3.3). You have a covariate weight and a label weight. Surely the total weight is just the two multiplied together?

**No. Multiplying double-counts, and it's worth seeing exactly why with real numbers.**

Here's the catch that makes multiplying wrong: the two shifts are *not independent*. When the prevalence of nodules changes, that **also** changes what images show up. If County has fewer nodule patients, County also has fewer nodule-*looking* images. So a change you label "prevalence" already leaks into "what the images look like." The covariate weight and the label weight are partly measuring **the same underlying change** — and if you multiply them, you count that change twice.

**The project's worked example, in plain numbers.** Take a simple 2-class problem: nodule vs. no-nodule.

- At University (source), the split is 50/50: `p_S(y) = (0.5, 0.5)`.
- At County (target), it's 90/10: `p_T(y) = (0.9, 0.1)`. (Class 1 got much more common.)

The honest label weight for class 1 is just how much more common it got: 0.9 / 0.5 = **1.8**. So a class-1 case from University should be counted 1.8× as heavily. That's the truth. The correct total weight here is **1.8**.

Now watch the multiplication trap. Take an image that very clearly shows class 1 — the discriminator and the model both agree it's a textbook class-1 image. Because class 1 is now 1.8× more common at County, images that *look like* class 1 are also more common at County by about the same factor. So the **covariate weight for this image also comes out around 1.8** — it's not adding new information, it's the *same* prevalence change showing up in the pixels.

Multiply them: 1.8 (label) × 1.8 (covariate) = **3.24**. But the truth was **1.8**. The naive product nearly **doubles** the weight — it counted the one prevalence change twice, once as "the answer is more common" and once as "the image looks more common."

**The fix: divide by a correction factor.** The project divides out the double-count using a quantity it calls `Z(x)` — read it as "how much of this image's apparent boost is *already explained* by the prevalence change." In our example `Z(x) ≈ 1.8`, the exact amount of the overlap. So:

1.8 × 1.8 / 1.8 = **1.8.** ✓

The division surgically removes the part that got counted twice, landing exactly on the truth. That's why the Method Note states the rule as a flat warning: **`w(x,y) ≠ w_cov(x) · w_lab(y)`**. The total weight is the label weight times the covariate weight, *divided by Z(x)* — never the bare product.

> **Under the hood (skippable).** The combine rule is `w(x,y) = w_lab(y) · w_cov(x) / Z(x)`, where `Z(x) = Σ_{y'} w_lab(y') · p_S(y'∣x)` is the average label-weight over the diagnoses the model thinks the image could be. When the image clearly shows class 1, that average is just `w_lab(1) = 1.8`, which is exactly the inflation the bare product `1.8 · 1.8 = 3.24` introduced — so dividing by it restores the truthful `1.8`. The full derivation is Method Note section 3.3.

### When the image isn't even the right *kind* of image: OOD

Reweighting handles cases where County differs from University but is still *the same kind of thing* — chest X-rays of real patients, just on different machines with a different case mix. The weights stretch University's data to cover County's.

But some inputs are off the map entirely. Suppose a knee X-ray gets sent to the chest model by accident. Or a scan so corrupted it's mostly noise. Or an anatomy the model has genuinely never seen. There is no "weight" that fixes this — the image isn't a darker-or-rarer version of a chest X-ray, it's **not a chest X-ray at all.** You can't reweight your way from "lung" to "knee."

These are **out-of-distribution** inputs, or **OOD** — "not even the right kind of input." Trying to reweight them is mathematically hopeless (the Method Note, section 4, notes the density ratio is undefined or astronomically large there — there's no University case that looks remotely like a knee, so there's nothing to boost).

The right move is not to answer, and not even to politely defer-as-uncertain. It's to **route the case out** entirely — flag it as "this doesn't belong here" and hand it to a human. The project does this with an **OOD score**, `o(x)` (section 4.1): it measures how far an image sits, in feature-space, from anything the model saw during training. A knee X-ray sits *very* far from every chest X-ray. If the score crosses a threshold (`t_ood`), the case is routed to a human before it ever reaches the answer path (section 4.2).

This is a different action from abstaining. Abstaining says "this is a hard chest X-ray, a person should read it." Routing-out says "this isn't my job at all." Both protect Dr. Rivera, but for different reasons — and the project keeps them as **distinct clinical actions**, not one bucket.

One honesty note the project insists on (section 4.4): OOD detection comes with *no* guarantee — there's a known impossibility result that you can't promise to catch every weird input. So the project doesn't promise. It **measures** how much leaks through and reports that number. That measure-and-report instinct, you'll have noticed, is this whole project's personality.

### What you can now understand

- **Source vs. target:** the AI is tuned at the source (University) but deployed at the target (County); the gap between them is what this tier is about.
- **Two flavors of shift:** *covariate shift* = the images look different (different scanners), *label/prevalence shift* = the mix of diagnoses is different (referral center vs. community ER). One moves the pictures, the other moves the answers.
- **Why Tier 2's promise breaks:** a threshold tuned at the source isn't valid at the target because exchangeability is gone — the real cases aren't drawn from the pool the threshold was tuned on.
- **The fix is reweighting:** count University cases that resemble County (in look *and* in diagnosis-mix) more heavily, so the old data imitates the new hospital — using a covariate weight (from a County-vs-University discriminator) and a label weight (from confusion-matrix methods).
- **Don't multiply the weights:** the two shifts overlap, so the bare product double-counts (1.8 × 1.8 = 3.24 when the truth is 1.8); dividing by the correction `Z(x)` removes the double-count exactly.
- **OOD is different from shift:** a knee X-ray sent to a chest model isn't a shifted chest X-ray, it's the wrong input entirely — it gets *routed out* to a human, not reweighted.

### Check yourself

**Q1.** County Hospital's chest X-rays come out slightly darker and grainier than University's because of different scanners, but a nodule still means a nodule. Which flavor of shift is this — covariate or label? And which weight is built to correct it?

*A1.* Covariate shift — the **images** look different while the diagnosis-given-image relationship is unchanged. It's corrected by the **covariate weight** `w_cov(x)`, which comes from the domain discriminator that judges how County-ish an image looks.

**Q2.** Suppose nodules go from 50% prevalence at University to 90% at County, so the honest total weight for a clear nodule case is 1.8. An engineer instead computes the label weight (1.8) and the covariate weight (1.8) and multiplies them to get 3.24. What did they do wrong, and what fixes it?

*A2.* They **double-counted** the single prevalence change — it showed up once as "nodules are more common" (label weight) and again as "nodule-looking images are more common" (covariate weight), because the two shifts overlap. Dividing by the correction factor `Z(x) ≈ 1.8` removes the overlap and restores the true weight of 1.8. The rule is `w = w_lab · w_cov / Z(x)`, never the bare product.

---

*You've now seen the project's most ambitious machinery: reweighting to correct for the hospital gap, and routing out scans that are off the map entirely. A natural hope follows — surely, with all this correction, the tool can now *promise* Dr. Rivera a safe error rate at County? This tier is the heart of the whole project, and its answer is a deliberate, principled 'no.' Here's why saying 'no' is the strongest thing the project can do.*

I have everything I need. Writing the tier now.

## Tier 5 — Why we refuse to promise a guarantee (the honesty core)

You have arrived at the heart of this project. Everything in the earlier tiers — the shift between hospitals, the reweighting that corrects for it, the fact that the weights are *estimated* and not handed down from heaven, the out-of-distribution router, the exchangeability that calibration leans on — all of it has been building toward one uncomfortable, important question:

**When the AI tells Dr. Rivera "I'm 94% sure this is a nodule," can it back that number up with a mathematical promise?**

The honest answer is *no*. And this tier is about why "no" is the right answer, why saying it out loud makes the paper **stronger** rather than weaker, and what the project does *instead* of promising.

Let me start with the intuition, because the intuition is everything here.

### The difference between *promising* and *measuring*

Imagine two weather services.

Service A puts a banner on its app: **"We guarantee tomorrow's forecast is correct."** That is a promise. It's also obviously nonsense — nobody can guarantee the weather, and you'd distrust them the moment they said it.

Service B says: **"Over the last 1,000 days, when we said 'rain,' it rained 87% of the time. Here's the chart. When there's a hurricane offshore — something we've rarely seen — our accuracy drops and we'll tell you that."** No promise. Just an honest, checkable *measurement* of how well it has actually done, with its weak spots flagged.

Which one would you trust with a decision that mattered?

That is the entire stance of this project in one analogy. The old version of this paper was trying to be Service A — to hand a clinician a mathematical *guarantee* that the error rate stays below some line, even after the AI moves to a new hospital. This tier is about why the team gave that up and became Service B on purpose. The shorthand for the Service-B stance is **"measure and report."** We don't certify the error; we *measure* it on real data and *report* it honestly, including its uncertainty and the situations where it gets shaky.

The rest of this tier explains the three hard facts that forced this choice. Two of them are published mathematical results — proofs that the promise is *impossible*, not merely hard. The third is a structural fact about what you can and can't figure out from the data you have. Take them one at a time.

### Hard fact #1: you cannot build a perfect "is this weird?" detector (Fang et al.)

Remember from Tier 4 the OOD router — the part of the system that's supposed to catch scans that look unlike anything the AI was trained on (a pediatric X-ray, a corrupted image, a scan from a totally different body part) and refuse them outright.

It would be lovely to promise: *"This detector will catch every weird input."* There's a published proof (Fang et al., 2022) that says, in plain terms, **you can't** — not in general, not with any detector.

**The everyday version.** Think of a nightclub bouncer whose job is to spot fake IDs. You can train the bouncer on every fake ID anyone has ever made. But "all possible fakes" is an *infinite* space — a forger can always invent a new fake the bouncer has never seen, designed to slip past exactly the patterns the bouncer learned. No amount of training closes the gap, because the space of "things unlike what you've seen" is unbounded by definition. There is always a next weird input that fools you.

The proof makes this precise: in the unrestricted setting, no detector can carry a *distribution-free guarantee* (Tier 0's phrase for "a promise that holds no matter what the data looks like") about its miss-rate on weird inputs. And the *hardest* case for County Hospital isn't the wildly weird scan — it's the **near-miss**: a chest X-ray of the right body part, the right disease, but taken on County's slightly different scanner with a slightly different protocol. That near-miss overlaps with normal inputs just enough that it's *exactly* the case the impossibility proof says no detector can reliably separate.

**What the project does instead.** It can't promise the detector catches everything, so it *measures* how much leaks through. It assembles a stated, swappable set of weird-on-purpose images (the doc calls this the "exposure set," written `O`), sets the detector's threshold against that set, and then **reports a single number**: the residual leakage — roughly, "X% of genuinely out-of-place scans still slipped past." Honest, checkable, and clearly caveated. This is `method_note.md` §7.1, first bullet, and §4.

### Hard fact #2: the moment you *estimate* the reweighting, the exact promise dies (Yang et al.)

Tier 4 taught you reweighting: because County's patient mix differs from University's, you can't trust raw numbers — you up-weight the County-like cases and down-weight the University-only ones to correct the picture. Tier 4 also delivered the catch: in real life you don't *know* the true weights. You **estimate** them from data, and estimates carry error.

There's a published result (Yang, Kuchibhotla & Tchetgen Tchetgen, 2024) that turns that catch into a hard wall. In plain terms: **if you had the perfect, true weights handed to you, you could give an exact finite-sample coverage promise. The instant you have to estimate the weights instead, that exact promise is gone.** The best you can get is *approximate* — a promise that becomes accurate only as your sample grows very large, not one you can stamp on this specific batch of County scans.

**The everyday version.** A recipe says "use exactly 200g of flour and it comes out perfect every time." That's a real guarantee — *if* you have a perfect scale. But your kitchen scale is eyeballed; you're estimating 200g. The recipe's "perfect every time" promise was attached to the *exact* 200g, and once you're estimating, the most you can honestly say is "it's usually about right, and gets more reliable the more I bake." The guarantee didn't survive the switch from a true measurement to an estimated one. **Estimation eats the guarantee.**

**What the project does instead.** It doesn't pretend the estimated covariate weights (`ŵ_cov` — the hat means "estimated," from Tier 4) come with a finite-sample certificate, because the proof says they can't. Instead it **measures the leftover coverage error directly** on a small slice of real, labeled County data and reports that degradation. (`method_note.md` §7.1, second bullet, and §3.1.)

### Hard fact #3: from unlabeled County data alone, you literally can't tell the two shifts apart (non-identifiable)

This one isn't an impossibility *proof* so much as a structural fact about information — but it's just as binding.

There are two different reasons County differs from University (you met both earlier):

- **Covariate shift** — the *images* look different (County's scanner, County's exposure settings).
- **Label shift** — the *disease mix* is different (County sees more advanced cases, say, because it's a referral center).

Now suppose all you have from County is a big pile of **unlabeled** scans — images with no doctor's verdict attached (which is the realistic situation; labeling is expensive). Here's the structural trap: **two completely different combinations of "image shift" and "disease-mix shift" can produce the exact same pile of unlabeled images.** From the images alone, you cannot tell which combination is the real one. The word for this is **non-identifiable**: the data you have simply does not contain the information needed to pin down the answer.

**The everyday version.** You walk into a bakery at noon and the croissant shelf is empty. Why? Maybe they baked the usual number and a rush of customers cleared them out (a "demand" shift). Maybe they baked far fewer today (a "supply" shift). The empty shelf — your only evidence — is *identical* in both stories. Staring harder at the empty shelf will never tell you which one happened. You'd have to ask the baker. **The single observation is consistent with multiple causes, so it can't single one out.**

**What the project does about it.** It accepts that the unlabeled scans can't settle the question, and it pays for a tiny bit of "asking the baker": a **small labeled target slice** — roughly 50 to 200 real County scans *with* a confirmed diagnosis attached (the doc calls it `D_tar^lab`). This is the one place the project spends real labeled County data, and what that slice is *for* is the most important honesty rule in the whole project, so let's be very precise about it.

### The tiny labeled slice MEASURES — it certifies nothing

It is tempting to think: "Aha, with 50–200 labeled County scans, now we can prove the system is safe at County!" **No.** Fifty to two hundred cases is far too few to *prove* anything, and the team is militant about not pretending otherwise.

What the slice actually does is exactly two modest things:

1. **It measures the leftover error.** After all the reweighting and routing, how far off is the system *really* on genuine County data? The slice gives a *measured estimate* — a number with an honest uncertainty band around it (a confidence interval, like Service B's "87%, give or take"), **not** a guarantee.
2. **It fits one small correction.** Because hard fact #3 says you can't disentangle the two shifts from images alone, the project doesn't try to. It fits a single, simple "nudge" on this slice to soak up the residual gap — and then *reports* whatever gap remains after the nudge.

That's it. The slice is a **measuring instrument**, like the weather service's record of past forecasts. It never becomes a certificate. The positioning memo says it flatly: `D_tar^lab` exists "*only* to empirically measure residual degradation and fit a simple scalar correction — never to identify a nuisance parameter or certify the combined case." (`method_note.md` §7.1, third paragraph; `positioning_memo.md` "What this memo drops.")

### Putting the three together: why "measure and report" is the *only* honest move

Stack the three facts:

- You **can't** promise the weird-input detector catches everything (fact #1).
- You **can't** keep the exact coverage promise once you estimate the weights (fact #2).
- You **can't** even tell, from unlabeled data, which kind of shift you're correcting (fact #3).

Any one of these would already block a clean guarantee. Together they make it unmistakable: **a mathematical certificate that survives the move from University to County does not exist** — not because this team isn't clever enough, but because *no* team can build one. As `method_note.md` §7.1 puts it, these "are not limitations of our engineering; they are limits on what **any** method in this space can promise."

So the only honest stance left is the Service-B stance: **measure the leftover error on real target data, and report it with its uncertainty and its weak spots** — rather than certify a number you'd have to fabricate. That is what "measure and report" means, and now you can see it isn't a cop-out. It's the *only* thing left standing once you take the three facts seriously.

### The honesty discipline (and two banned words)

Because the temptation to overclaim is so strong — a confident-sounding guarantee *sells* — the project enforces a written discipline, repeated on every page of the playbook (`flagship-playbook.md` §0, the "#1 rule"). Two parts matter most:

**1. Every guarantee names its method AND its assumption.** The project is *built entirely from published methods*, and several of those methods do carry real guarantees — but each guarantee only holds under that method's own assumptions, assumptions that the County-vs-University shift typically breaks. So the rule is: you may never say "the system guarantees X." You may only say "*this specific method* guarantees X *under this specific assumption* — and here's where that assumption fails for us, with the measured degradation." A guarantee is always pinned to its owner and its fine print, never floated free as a property of the whole pipeline.

**2. The words "certify," "certified," and "guarantee" are BANNED for the deployed system.** Literally banned, as a writing rule. The deployed pipeline — the thing that actually looks at Dr. Rivera's scans — gets numbers that are *measured estimates with intervals* and *reported diagnostics*, never certificates. This is a tripwire: if one of those words ever appears next to the deployed system in the paper, something dishonest has crept in and must be removed. (`flagship-playbook.md` §0; `positioning_memo.md` "What this memo drops.")

This discipline is *why* the earlier tiers kept saying things like "the weights are estimated, so..." and "we report the residual degradation." Every one of those careful phrasings traces back to this rule.

### The pivot story: the team chased a theorem, then dropped it on purpose

Here's the part that makes all of this real, and it's worth knowing because it's the project's spine.

An **earlier version of this project tried to be Service A.** The plan was to *prove a theorem* — a distribution-free risk certificate, with explicit constants, that would survive covariate shift *and* label shift *and* fold the OOD router into a single error budget. A genuine mathematical guarantee that holds after the AI moves hospitals. That was going to be the headline contribution.

Then the team ran into the three facts above. The impossibility results (#1 and #2) and the non-identifiability fact (#3) didn't just make the theorem hard — they made it **provably unattainable**, and on top of that, clinical reality (you only ever get a tiny labeled slice from the new hospital, never enough to certify anything) drove the final nail. So the team **deliberately dropped the theory.** The playbook's opening note records it bluntly: the "guarantee that survives shift" novelty core, the formal theorem with explicit constants, the certified combined error budget — all **cut**. (`flagship-playbook.md` top note and §0; `positioning_memo.md` "What this memo drops.")

And here's the crucial reframe, the thing to hold onto: **dropping the certificate made the paper stronger, not weaker.**

Why? Because a paper that *claims* a guarantee the math says is impossible is, at best, hiding its assumptions and, at worst, dangerous — it would hand Dr. Rivera a false sense of safety on exactly the cases where the system is least trustworthy. A paper that *measures and reports honestly* gives her something she can actually use and audit: real numbers, with real uncertainty, with the weak spots flagged in advance.

So the genuine contribution moved. It is **not** a theorem. It is (1) the **measurement / auditability discipline** — composing published methods so carefully that every single number shown to a clinician is an honest estimate or a measured diagnostic, each carrying its own assumption and its own measured degradation — and (2) the **clinician-facing trust interface** that makes all of that visible at the point of care, including a per-case explanation of *why* the system declined. The positioning memo states it directly: "Dropping the certificate is a concession about the *guarantee*, not about the *contribution*; the novelty does not live in a theorem."

That's the selling point. In a field full of medical-AI papers quietly overclaiming, a paper whose entire identity is *"here's exactly how well it works, here's where it breaks, and here's the proof we couldn't have promised more"* is the trustworthy one. That honesty **is** the paper.

### What you can now understand

- **"Measure and report" is not a cop-out — it's forced.** Three hard facts (two impossibility proofs plus non-identifiability) make a guarantee-that-survives-shift literally unattainable, so honestly measuring the leftover error is the only stance left.
- **The three facts in plain terms:** (1) no detector can catch every unfamiliar input (Fang et al.); (2) once you *estimate* the reweighting, the exact coverage promise dies — "estimation eats the guarantee" (Yang et al.); (3) from unlabeled target data alone you can't tell covariate shift from label shift (non-identifiable).
- **The tiny labeled target slice (50–200 cases) is a measuring instrument, full stop** — it estimates the residual error with an uncertainty band and fits one small correction; it never certifies anything.
- **The honesty discipline:** every guarantee is pinned to the specific method that owns it *and* its assumption; the words *certify / certified / guarantee* are banned for the deployed system.
- **The pivot is the point:** the team chased a theorem, the impossibility results plus clinical reality killed it, and dropping it made the paper a *stronger, more honest* contribution — a measurement discipline and a trust interface, not a false promise.

### Check yourself

**Q1. Dr. Rivera asks: "Can your tool guarantee its error rate stays below 5% here at County?" Using this tier, how should the system honestly answer, and why can't it just say yes?**

*A1.* It cannot say yes. A "yes" would be a guarantee, and three facts make that impossible: the OOD detector can't be promised to catch every unfamiliar County scan (Fang et al.), the coverage promise evaporates the moment the reweighting is *estimated* from data rather than known exactly (Yang et al.), and from unlabeled County scans you can't even tell which kind of shift you're correcting (non-identifiable). The honest answer is the Service-B answer: *"On a measured slice of real County cases, the answered-case error came out to about X%, give or take this much, and here are the situations where it gets worse"* — a measured estimate with its uncertainty, not a guarantee.

**Q2. A teammate suggests: "We have 120 labeled County scans now — let's use them to *certify* the system is safe at County." What's wrong with that, in one sentence?**

*A2.* That misuses the slice: 120 cases can only *measure* the residual error (an estimate with an uncertainty band) and *fit* a small correction — it is far too little to certify anything, and "certify" is a banned word for the deployed system precisely because the impossibility and non-identifiability facts say no certificate is attainable, however many such slices you collect.

---

*If the project refuses to promise a guarantee, and it didn't invent any of the underlying methods, then what *is* its actual contribution? This is the tier where it finally lands. The answer has two parts — a discipline for honest *measurement*, and a way for the tool to *explain itself* to Dr. Rivera — and we'll build both up exactly as she'd encounter them at her desk.*

## Tier 6 — The actual contribution: measure, audit, and explain

You've reached the part of the project that is *actually new*. Everything in the earlier tiers — abstaining, routing out weird scans, reweighting to fix the hospital mismatch — already exists in the published literature. Our project borrows all of it. So what does *this* paper contribute?

The short answer: it doesn't add a new mathematical promise. It adds a **discipline for honest measurement** and a **way of explaining itself to a clinician**. That is the whole contribution, and this tier is where it lands.

Let me build it up the way Dr. Rivera at County Hospital would actually encounter it.

---

### 6.1 The starting problem: how do you even measure the error at County?

Recall the situation. The AI was built and tuned at University Medical. It's now running at County, which has different scanners and a different patient mix. We want to know one thing: *among the cases the system actually answers at County, how often is it wrong?*

You might think: just count the mistakes. But there's a catch we met in earlier tiers. We don't have a big pile of labeled County cases to count against — getting ground-truth labels (a confirmed diagnosis for every X-ray) is slow and expensive. What we mostly have is University's labeled data plus the *weights* from Tiers 4–5 that tell us how much each University case "looks like" a County case.

So the measurement has to be done *through* those weights. And that's where both the cleverness and the honesty come in.

#### The weighted-average idea (cases that look like County count more)

Here's the intuition. Imagine you have 1,000 labeled chest X-rays from University. Some of them look a lot like the typical County scan; some look nothing like it. If you want to estimate the County error rate, it would be silly to count every University case equally — the ones that look like County are far more informative about County.

So instead of a plain average ("what fraction of all 1,000 were wrong"), we take a **weighted average**: each case's mistake counts in proportion to how County-like it is. A University scan that looks exactly like a County scan gets a big weight; one that looks alien to County gets almost zero. The error estimate is then dominated by the cases that actually resemble where the system is deployed.

There's one more subtlety. We don't actually know the weights perfectly — we estimated them, and they don't naturally average to exactly 1. So instead of just adding up `weight × mistake`, we **divide by the total of the weights**. This "divide by the sum of the weights" trick is called the **self-normalizing** or **Hájek** estimator. In plain terms: it's a weighted average that cleans up after itself, so a systematic over- or under-sizing of the weights cancels out instead of biasing the answer.

> **Under the hood (skippable).** The reported quantity is the self-normalized (Hájek) weighted accepted-in-scope risk:
> `R̂_w = Σ(ŵ_i · ℓ_i) / Σ ŵ_i`
> summed over cases that were both answered and in-scope. The numerator is the weighted total of the losses (mistakes); dividing by `Σ ŵ_i` is what makes the unknown overall scale of the weights cancel — so we don't need the weights to be perfectly normalized for the estimate to behave. This is `method_note.md` §1.7.

#### Why any honest number needs a plus-or-minus

Here is the part the project insists on. A single number — "County accepted error is 4%" — is *not* an honest answer on its own. It's an estimate from a finite, reweighted sample, and it could easily be off. So every number the system shows comes with an **uncertainty interval**: not "4%" but "4%, give or take 2%."

That interval isn't decoration. It's the difference between "we measured this carefully and here's how sure we are" and "trust us." The whole stance of the project (the measure-and-report, honesty discipline you met in earlier tiers) lives or dies on showing that plus-or-minus next to every figure.

There's even a reason the interval is computed with a slightly fancier method than usual. Reweighting *inflates the wobble* of an estimate: if a handful of heavily-weighted cases dominate the average, the number is jumpier from sample to sample. So the project uses an interval method that *adapts to the actual wobble it sees* (a "variance-adaptive" bound) rather than assuming the worst case. The result is an interval that's honest but not needlessly wide.

---

### 6.2 Reliability flags: the dashboard warning lights

Now, an interval tells you the spread, but it doesn't always tell you *why you should be nervous*. For that, the project adds a small set of **reliability diagnostics** — think of them as the warning lights on a car dashboard. The number on the speedometer might look fine, but the little orange light tells you something underneath is shaky.

The most important warning light is called **`n_eff`** (effective sample size). Here's the intuition.

Suppose your weighted error estimate at County is built from 1,000 University cases — but because of the weighting, *one* case that looks intensely County-like has a giant weight and the other 999 barely register. Then your "estimate from 1,000 cases" is really resting on, effectively, a handful of cases. It's fragile: change that one heavy case and the whole number swings.

`n_eff` is a single number that says: *after accounting for the lopsided weights, how many cases is this estimate really standing on?* If you have 1,000 raw cases but `n_eff` comes out to 40, the dashboard lights up: **trust this number less — it's effectively resting on ~40 cases, not 1,000.**

Crucially, `n_eff` is a *flag*, not a gate. It doesn't block the system or trigger an automatic guarantee. It just warns. The project is deliberate about this: it would be dishonest to dress `n_eff` up as a certificate ("we promise this is fine"). It's a reliability light, no more.

> **Under the hood (skippable).** `n_eff = (Σ ŵ_i)² / Σ ŵ_i²`, the Kish effective sample size. When all weights are equal it equals the raw count; the more unequal the weights, the smaller it gets. A few other lights ride alongside it — e.g. `κ(Ĉ_S)`, a "conditioning" number that flags when the label-shift estimate from Tier 4 is built on shaky arithmetic (here `Ĉ_S` is the confusion matrix from Tier 4, and `κ` measures how shaky inverting it is — worst for rare, important minority classes). These are `method_note.md` §1.7 and §3.5.

---

### 6.3 The per-case panel: what Dr. Rivera actually sees

All of the above is cohort-level — error rates across many cases. But Dr. Rivera works one patient at a time. So the audit layer renders a small **per-case panel** for the X-ray on her screen. When the system *answers*, the panel shows:

- **A recalibrated confidence.** Not the raw model number, but a confidence that's been corrected so it means what she reads it to mean. If it says "82%," that 82% has been honestly tuned, not taken at face value from a black box. (In the docs this corrected number is written `σ̃(f(x))`; "recalibration" is simply the step that *fixes* the miscalibration problem you met in Tier 1.)
- **A representativeness chip.** A little badge answering "how typical is *this* scan of the data the system was calibrated and audited on?" A scan that looks very County-and-University-normal gets a reassuring chip; an unusual one gets a "less representative — be careful" chip. (This chip is read off the same combined weight from Tiers 4–5.)
- **The `n_eff` flag.** The cohort warning light from §6.2, surfaced so she knows whether the reliability estimates around this kind of case rest on many cases or few.
- **A shift-regime badge.** A label saying *which kind of hospital mismatch is in force here* — e.g. "mostly a prevalence change" vs. "the scans themselves look different." (We'll see in §6.6 why this badge earns its own headline.)

The statistical guts (`n_eff`, the conditioning numbers) are demoted to a small "reliability footer," not shoved in her face — the plain-language cues are what's front and center. This is `method_note.md` §6 and the read-path in `explainability_framing.md` §3.3.

---

### 6.4 The Decline-Attribution Record: explaining *why it stayed silent*

Now the centerpiece. Sometimes the system **declines** — it abstains or routes the scan to a human instead of answering. Dr. Rivera's reasonable next question is: *why? Why won't you give me an answer on this one?*

The project's answer is an artifact called the **Decline-Attribution Record**, or **DAR**.

#### The intuition: read the reason straight off the rulebook

Here's the full rule the system uses to decide whether to answer. Two of its three gates you have already met — the scan must be **confident enough** (the uncertainty gate from Tier 2) and **not too weird** (the OOD gate from Tier 4) — and now we can name the third: the scan must sit **inside the trustworthy-weight region**. The system answers a scan *only if all three gates pass*:

1. it's **confident enough** (uncertainty below the threshold),
2. it's **not too weird** (OOD score below the threshold), and
3. it's **inside the trustworthy-weight region** (the County-resemblance weight isn't off in the untrustworthy tail).

If the scan is declined, then *at least one of those three gates failed* — and the system knows exactly which one, because that's literally the rule it just evaluated.

The DAR simply **reports that**. For a declined case it shows:

- **Which gate(s) tripped** — "too uncertain," "looks unlike anything I was trained on," or "outside the region where my hospital-correction can be trusted." These are three *different* clinical actions, not one vague "rejected."
- **By how much** — the *margin* to each threshold: how far over the line the binding gate was, and how close the case came to tripping the others. So Dr. Rivera sees not just *that* it declined but *how close* it was to answering, or to declining for a different reason.
- **The operating context** for the gate that bound — e.g. for an "out of scope / weird" decline, the measured leakage rate that the OOD threshold was tuned to.

#### Why this *cannot lie* about its own reasoning

Here's the load-bearing point, and it's worth slowing down for. The DAR is not a *guess* about why the system declined. It is the **exact same rule** the system used to decline, turned around and shown to the user. The explanation and the decision are *the same conditional check*. There is no separate "explainer model" that might disagree with the real decision.

The project's phrase for this is **routing-faithful**, and there's a careful scope clause attached: *faithful to the routing, not to the diagnosis.* That means the DAR truthfully reports **which gate the pipeline used to route the case** — it does **not** claim the routing was the medically correct call, or that the underlying class prediction is right. It explains the *traffic-cop decision*, not the medicine.

#### The contrast that makes this matter: saliency heatmaps

To see why "routing-faithful" is a real property and not a slogan, compare it to the usual thing people mean by "explainable AI": a **saliency heatmap** — a colored overlay on the X-ray supposedly showing "where the model looked."

The problem with those heatmaps in a black-box setting is that they are *reconstructions* — a second, separate artifact that tries to approximate what the model did. And they can be untethered from the model entirely: published "sanity check" studies found that some widely-used saliency methods barely change even when you *scramble the model's weights* — meaning the pretty picture a clinician trusts might not actually depend on the model it claims to explain. In medical imaging specifically, every saliency method tested in one chest-X-ray study failed at least one basic trustworthiness check.

The DAR has no such gap, because there's nothing to approximate. It doesn't reconstruct the decision — it *is* the decision predicate. The system declines if and only if a gate failed; the DAR reports exactly the gate(s) that failed and the exact margins the rule itself computed. This is `explainability_framing.md` §3.1–3.2.

And the project doesn't just *assert* this. It pins it down with a one-line, machine-checkable test: for every declined case, assert in code that the DAR's reported gates and margins exactly equal the gates and margins the routing rule actually computed. The target discrepancy is **zero** — a non-zero rate is a *bug*, not a research finding. That's how "faithful by construction" stops being a slogan.

---

### 6.5 Why this counts as *explainability* (and the "it's just uncertainty numbers" rebuttal)

The target venue is a collection on **Explainable AI for healthcare**. So there's a fair challenge waiting: *isn't all of this just uncertainty quantification — reliability telemetry — wearing an "explainability" costume? Real XAI explains why a prediction is what it is; you just report confidence numbers.*

The project meets this head-on, and the honest move is to **concede most of it**:

- Yes, the recalibrated confidence, the intervals, the `n_eff` light — those are reliability telemetry. The project does *not* call those "explanation." They sit in the trustworthiness column.
- The explainability claim rests on **one separable artifact**: the DAR. And the DAR explains a real, consequential *decision the model made* — the choice to answer, defer, or route — by naming its actual cause and the margin to the alternative. A good explanation is *contrastive* ("why this action and not the other") and *contestable* (the clinician can check and overturn it). The DAR is both; a bare confidence interval is neither.

So the framing is "**auditability as explanation**": the system gives an interpretable, faithful, per-case account of *its own reliability and its own routing reasoning*, surfaced right where Dr. Rivera makes the call. The project is equally clear about what it does **not** explain: it says nothing about the disease biology, the patient's physiology, or any causal "why is this patient sick" story. It explains the *model's* behavior, not the medicine — and it states that boundary on purpose, because the most common failure of clinical XAI is dressing up a plausible-but-unfaithful story as if it were an account of the underlying biology. This is `explainability_framing.md` §1–§2 and the two-column table in `positioning_memo.md`.

---

### 6.6 The two results that exist *only* because of the audit layer

Here's the proof that the audit layer is a real contribution and not garnish: there are **two headline findings the project can report only because this layer exists.** Delete the explainability leg and you don't lose a figure — you lose a result.

**1. The disparate-abstention finding.** Because the DAR records *which gate fired* on *every* declined case, and the panel records *which cohort* each case belongs to, the project can ask: *are the abstentions and routings piling up on particular groups of patients?* Maybe the system quietly declines far more often on one demographic, or on scans from one scanner type — effectively giving worse service to some patients without anyone noticing. That's a **measured fairness finding**, and it falls straight out of the per-cohort routing-rate audit plus the gate-firing breakdown. The plain selective-risk estimator can't produce it; only the audit layer can. (Note: it's a *measured, reported* disparity, not a promise that disparity is controlled.)

**2. The shift-regime badge stops a cheaper result being passed off for a harder one.** Recall from earlier tiers that the *pure prevalence-shift* case (only the disease mix changed) is the cleanest, lowest-risk regime, while the *combined* case (the scans themselves also look different) is genuinely harder and riskier. A subtle dishonesty would be to quietly report the nice, clean pure-shift number and let the reader assume it covers the hard combined case too.

The **shift-regime badge** (from the per-case panel, §6.3) blocks exactly this. For each target site it measures and *states* which regime is actually in force — "this is pure-prevalence-shift" vs. "this is combined covariate+label shift" — using the discriminator diagnostic from Tier 4. So the cleaner result can **never be silently substituted** for the harder one; the headline becomes a checkable, labeled claim instead of an assumed-true one. This is `positioning_memo.md` (venue-fit section) and `method_note.md` §6.3.

Both results are *audit-layer-only outputs*: produced by the explanation machinery, not by the risk estimator. That's the cleanest possible demonstration that this layer is load-bearing.

---

### What you can now understand

- **How the project measures error at the deployment hospital:** a *weighted average* where County-like cases count more (the self-normalizing / Hájek estimator), always reported with an *honest plus-or-minus interval* — never a bare number.
- **Why reliability flags exist and what `n_eff` means:** dashboard warning lights that tell you when to trust a number *less*; `n_eff` says how many cases the estimate is *really* resting on after the lopsided weights are accounted for — and it's a flag, not a guarantee.
- **What the per-case panel and the DAR are:** the panel shows recalibrated confidence, a representativeness chip, the `n_eff` flag, and a shift-regime badge; the DAR explains *why a case was declined* by reading the firing gate and margins straight off the real decision rule — faithful to the routing, not the diagnosis — so it cannot lie about its reasoning, unlike a saliency heatmap.
- **Why this is explainability for the venue:** "auditability as explanation" — a faithful, contrastive, contestable account of the model's *own decision*, with the honest concession that the reliability numbers themselves are just telemetry.
- **The two audit-only headlines:** a measured disparate-abstention (fairness) finding, and a shift-regime badge that prevents the easy result from masquerading as the hard one.

### Check yourself

**Q1. Dr. Rivera sees "County accepted error: 3%, built from 1,200 University cases." Why might the project *still* flag this number as not-to-be-fully-trusted, and which warning light would do the flagging?**
*A. Because of the weighting, the "1,200 cases" can be effectively resting on far fewer — if a few intensely County-like cases carry most of the weight. The `n_eff` (effective sample size) light catches this: if `n_eff` comes out to, say, 50, the estimate is really standing on ~50 cases, not 1,200, and should be trusted less. The raw count and even the interval can look fine while `n_eff` quietly lights up.*

**Q2. A reviewer says: "Your DAR is no better than a saliency heatmap — both are just post-hoc explanations of a black box." Why is this wrong?**
*A. A saliency heatmap is a separate, approximate reconstruction of a black box — it can be unfaithful, and sanity checks show some don't even change when the model is scrambled. The DAR is not a reconstruction: it reports the exact gate predicate and margins the pipeline actually used to decline the case — the explanation and the decision are the same rule. It's faithful by construction (and a code check asserts the discrepancy is exactly zero). The scope is narrower and honest: it explains the routing decision, not the diagnosis or the biology.*

---

*You now understand the whole machine *and* its true contribution. But a contribution is only worth anything if you can *prove* it — to a hostile expert who's actively looking for the trick. This tier flips to the referee's-eye view: how do you design experiments so honest that cheating becomes impossible, or at least gets measured exactly? Reassuringly, you've already met every idea these experiments touch.*

I have everything. Writing Tier 7 now.

## Tier 7 — Proving it honestly (how the experiments are designed)

You now understand the whole machine: the classifier, the abstain gate, the OOD router, the shift correction, the audit panel, and the project's one big promise — *every number it shows a clinician is honestly measured, with its caveats, never a guarantee dressed up as one.*

This tier is about a different question. Not "what does the system do?" but "how do you **prove** it does it — in a way a hostile expert can't tear apart?" Because here's the uncomfortable truth: it is *embarrassingly easy* to make a medical-AI system look good on paper while it's secretly cheating. The whole craft of this tier is designing experiments that make cheating impossible — or, where it's impossible to rule out completely, that **measure exactly how much** could be slipping through.

The good news: you've already met every concept the experiments touch. This tier just shows you the *referee's-eye view*.

---

### 1. Pre-registration: tying your own hands on purpose

**The intuition.** Imagine a marksman who fires a hundred shots at a blank wall, then walks up and paints the bullseye around the tightest cluster. Every shot looks like a hit. That's not skill — that's choosing the target *after* seeing where the bullets landed. Science has exactly this disease, and it has a name: cherry-picking. You run an analysis fifty different ways, and you report the one version that happened to look best, quietly forgetting the other forty-nine.

**Pre-registration** is the cure, and it's almost absurdly simple: **you write down — and timestamp — exactly what you're going to measure, and how, BEFORE you look at any results.** You paint the bullseye first, then fire. If you said in advance "I will report the error rate at a 70% answer-rate, on these patient groups, using this confidence interval," then you cannot later swap in a flattering alternative without everyone seeing you broke your own rule.

Why is this a credibility *superpower* and not just bureaucracy? Because it converts a promise you *can't* verify ("trust me, I didn't fish for this") into one you *can* ("here's the document, dated before the data was touched"). For a paper whose entire contribution is **honesty** — measure-and-report, no hidden guarantees — pre-registration isn't a nice-to-have. It's the thing that makes the honesty claim believable. The project's whole pitch is "our numbers are trustworthy," and pre-registration is what earns the word *trustworthy*.

One subtlety that's easy to miss, and the project is careful about it: **what gets pre-registered here is a measurement protocol, not a prediction.** The team is not committing in advance to "the system will achieve an error rate below 5%." They're committing to "we will *measure* the error rate this exact way and report whatever it is, including the ugly cases." You fix the *ruler*, not the *answer*. (This is `preregistration.md` §1 and §5–6, which lists everything frozen before results.)

> **Under the hood (skippable).** The frozen-in-advance list (`preregistration.md` §6) includes: the operating-point grid `G` over the thresholds `(τ, λ, t_ood)` (the answer/abstain and OOD cutoffs from Tiers 2–4; `λ` is the RCPS risk-control dial from Tier 3); the two budgets `α_acc` (accepted-case error tolerance) and `α_ood` (OOD-leakage tolerance); the confidence level `δ` of the reported intervals; the patient/scanner subgroups; the minimum-coverage floor; and the "stress axes" the experiments deliberately push. None of these can be re-chosen after a result is seen.

---

### 2. The two hospitals-in-a-box: the datasets and the source→target setup

To test "does it survive being moved to a new hospital?", you need data that *actually contains* the new-hospital problem. You can't fake it convincingly — real scanners and real patient mixes differ in messy ways no simulation captures. So the project uses two real benchmarks, each built from **different real institutions**, set up as a **source → target** deployment: build and tune the system on the source data, then move it cold to the target data and measure what breaks. This is Dr. Rivera's situation made into an experiment — University Medical is the source, County Hospital is the target.

**Benchmark 1 — CAMELYON17 (tumor tissue patches, across hospitals).** Tiny image patches of lymph-node tissue, labeled tumor vs. normal, collected from **five different hospitals** with different scanners and staining chemistry. The same tumor looks different under a different scanner — so this is the **covariate-dominant** benchmark: mostly the *appearance* `p(x)` of the images shifts, not which findings are common. Each source-hospital → target-hospital **pair is reported separately, never averaged into one rosy number** (averaging would let an easy pair hide a hard one). This benchmark is open-access, so it can carry the build immediately.

**Benchmark 2 — CheXpert ↔ MIMIC (chest X-rays, across sites).** This is Dr. Rivera's exact world: chest X-rays from two large but *different* medical systems. Here **two things shift at once** — the image appearance (different sites/scanners) *and* the disease prevalence `p(y)` (one site sees more of a finding than the other). This is the **mixed** benchmark, the harder stress test. And it's run in **both directions** — CheXpert→MIMIC *and* MIMIC→CheXpert — as two separate deployments, because being good at moving one way doesn't prove you're good at moving back. (`preregistration.md` §2.)

> **One scheduling reality worth knowing:** CheXpert and MIMIC require credentialed access (human approval, paperwork, training certificates) that takes weeks. CAMELYON17 doesn't. So the plan starts the credentialing paperwork on day one and lets the open CAMELYON17 benchmark carry the engineering while the approvals grind through — a theme we'll return to in §6.

---

### 3. The four sparring partners: the baselines

A number on its own means nothing. "The system has 4% error" — is that good? You only know by comparing against **what a simpler approach would have done on the exact same data.** A baseline is that simpler approach. The project pre-registers **exactly four** (`preregistration.md` §4), and each is chosen to answer a specific skeptical question. Crucially, all four run on the **same data splits, the same frozen classifier, and the same target test set** — so any difference is the *method's* doing, not a lucky shuffle.

1. **Naive conformal (ignore the shift entirely).** The selective gate, calibrated at the source hospital, applied to the target hospital with **no shift correction at all**. This is the "do nothing about the new-hospital problem" reference. The project *expects* this one to degrade — that degradation is the whole motivation for the shift correction. It answers: *"Does correcting for the shift actually beat pretending there is no shift?"*

2. **Temperature scaling (just fix the confidence numbers).** A cheap, standard recalibration of the classifier's confidence — no importance weighting, no OOD routing. Answers: *"Maybe all you needed was to tidy up the confidence scores — is the heavy shift machinery even earning its keep?"*

3. **TRUECAM-style detect-and-remove.** An existing published pipeline that handles weird inputs by *throwing them out* before calibrating. This is the honest head-to-head: the project doesn't claim to beat it with a stronger guarantee — it beats it by **measuring the residual leakage** (the weird cases that slip through anyway) that the detect-and-remove framing leaves unquantified. Answers: *"How do you compare to the closest existing system, fairly?"*

4. **BBSE label-weighting only (fix prevalence, nothing else).** Corrects only for the changed disease mix, ignoring appearance shift and OOD. Answers: *"Is each piece of the correction pulling its weight, or is one part doing all the work?"*

The **full pipeline** is the thing under test against these four.

---

### 4. What actually gets measured (the headline numbers)

All measured on the held-out target test set, each reported *with its uncertainty*, none a guarantee (`preregistration.md` §5):

- **Accepted-case error** — among the cases the system chose to answer, how often was it wrong? This is *the* number. Reported as a range (a betting interval) at confidence `δ`, not a single deceptively-precise figure.
- **Coverage (answer rate), broken into its three causes** — what fraction did it answer, and of the cases it *declined*, how many were declined for low confidence vs. OOD vs. untrustworthy-weight? (This decomposition matters enormously for §5.)
- **The risk–coverage curve (AURC)** — error vs. answer-rate across *all* operating points, so the whole trade-off is visible, not one cherry-picked point.
- **Calibration error (ECE) of the displayed confidence** — because the confidence number Dr. Rivera reads is itself a promise; if it says "0.97" it had better be right about 97% of the time. Reported per hospital, per subgroup, at the actual operating point.
- **OOD leakage** — of the genuinely out-of-domain inputs, how many slipped past the router into the answered pile?
- **Subgroup audit** — all of the above, broken down by site, scanner, sex, age, etc., **leading with the abstention/coverage gap** so a group that "looks safe" only because it's rarely answered gets flagged first.

---

### 5. The detective story: five cheap "gotchas" and exactly how the design kills each

Here's where the tier earns its title. The team ran an **adversarial stress-test** — a panel of simulated hostile reviewers whose only job was to find the cheapest way to discredit the paper (`stress_test_findings_2026-06-27.md`). The fixes below came *directly* out of that session. Think of each as a detective ruling out an innocent explanation for a suspicious result.

**Gotcha #1 — the matched-coverage confound (the deadliest one).**
*The accusation:* "Your error is low because your system simply **answered fewer cases**. It dodged all the hard ones and dumped them on the human. Of course the easy leftovers have low error — that's not a better method, that's just a pickier method." The panel ranked this the single most damaging attack, because the system's *three* declining mechanisms (low-confidence, OOD, bad-weight) hand the baselines a strictly larger, dirtier pile to answer. Every headline number is vulnerable to it at once.

*The kill:* **Compare every number at a matched answer-rate.** If the full pipeline answers 70% of cases, you force each baseline to *also* answer 70%, and only then compare error. Now nobody can be lower just by being pickier — they're all equally picky. Beside every matched number, the design also prints the three-way breakdown of *how* it declined, the full risk–coverage curve (so the comparison doesn't hinge on one point), and the raw classifier error with no abstention at all as a base-rate anchor. (`preregistration.md` §5.1.) Note the honesty: because abstention is split across three mechanisms, the match is *approximate* — and the design says so out loud rather than pretending it's exact.

**Gotcha #2 — fold leakage at the patient/slide level.**
*The accusation:* "The same patient's X-rays appear in both your calibration set and your test set. The system 'recognizes' them, so your test isn't really testing on unseen data — it's peeking at the answer key." This is a notorious silent inflator in medical imaging, because one tumor slide is chopped into *hundreds* of patches and one patient has *several* studies. Split by patch or by image, and the same patient's pieces sneak into both halves while the row indices look perfectly disjoint.

*The kill:* **Split by the group (the slide ID, the patient ID), not by the image.** No slide's patches and no patient's studies are ever allowed to straddle two folds, and each run prints a checked-in-code "intersection = 0" proof over the group IDs. Then — going further — they run a **deliberate leakage probe**: inject a known leaked patient on purpose and *measure how much the error moves*, quantifying exactly how much optimism group-leakage would have bought. (`preregistration.md` §2.3.)

**Gotcha #3 — the backbone confound ("maybe the win is just a fancier model").**
*The accusation:* "Your system wins because you used a bigger, better neural network than the baselines — the trustworthiness machinery is irrelevant decoration." 

*The kill:* **Build the headline result on a plain, standard classifier**, identical across the system and all four baselines. The fancier networks (DDU, SNGP, ensembles) appear *only as add-on ablations*, never as the foundation. The reasoning is honest and clean: a better network only changes *efficiency* (how often it can answer at the same error), not the auditability discipline that is the actual contribution. Starting on a fancy backbone would hand reviewers the obvious objection for free. (`flagship-playbook.md` §1, attribution discipline.)

**Gotcha #4 — abstention inflation / degenerate operating points.**
*The accusation:* "You drove your error to near-zero by answering almost nothing — a system that answers 2% of cases is useless even at 0% error." 

*The kill:* A **pre-registered minimum-coverage floor.** Any operating point that answers fewer cases than the floor is reported as **degenerate** — flagged as "risk bought by collapsing coverage," not paraded as a good result. Because the floor is fixed *in advance*, you can't manufacture a flattering low-error number after the fact by quietly cranking up abstention. (`preregistration.md` §6.4.)

**Gotcha #5 — validating the trust signals on the *answered* cases.**
*The accusation:* "Your whole point-of-care interface — the representativeness chip, the reliability flag, the shift badge — is the contribution, but you only ever check it on the cases you *declined*. The signals a clinician actually acts on, on the cases you *answer*, are never measured. The interface is asserted, not demonstrated." The panel rated this a blocking flaw for an explainability venue.

*The kill:* An **accept-side check** added to the protocol. Bin the *answered* cases by their trust-signal level (chip level, reliability regime, shift badge) and **measure the actual error in each bin**, with intervals. The design pre-registers the expectation that the "risky-looking" bins (heavily up-weighted, low effective-sample, combined-shift) really do show higher error — turning the interface from decoration into a measured, falsifiable deliverable. It's computed on the larger labeled test split so the bins have enough cases to mean something. (`preregistration.md` §5.5(G).) Honesty rail: this validates the signals *as population trends*, and explicitly does **not** promise a green chip means a given patient is correct.

> **A sixth, more technical one worth a nod:** a favorable error could be an artifact of two un-swept knobs — the weight-clip `w_max` (trimming the heavy tail biases the retained error) and the `Ẑ` denominator (biased on rare classes). The fix is to **sweep both** and report how much the headline moves; cells where it moves a lot go into a "failure-mode catalog" rather than getting buried. (`preregistration.md` §6.6.)

---

### 6. Two datasets deep, not many shallow — and why

A tempting way to look impressive is breadth: "we tested on eight datasets!" The project deliberately refuses this. **Two benchmarks, taken deep** — every baseline, every subgroup, both directions, every stress sweep, full diagnostics, the honest failure cases mapped out.

Why depth wins for *this* paper: the contribution is a **measurement discipline**, and a measurement discipline is only convincing if you can show it surviving real scrutiny — all four confound checks, the subgroup intervals, the leakage probe, the matched-coverage comparison. Run shallowly across eight datasets and you have eight numbers nobody can stress-test. Run deeply on two and you have a *defensible* claim. A shallow win on a third dataset is exactly the kind of thing the matched-coverage or backbone confound would shred. Breadth (ECG, retina, dermatology) is named as **honest future work**, never claimed as done. (`flagship-playbook.md` §1, depth over breadth.)

---

### 7. The build ladder and the dated go/no-go (the timeline, at altitude)

You don't build the whole machine at once and hope. You **stack one layer at a time, testing each on synthetic data where you know the true answer**, before adding the next. This is the engineering ladder (`flagship-playbook.md` §5):

- **A** — selective gate alone (does the answer/defer trade-off behave?)
- **B** — add covariate weights (does it recover error the naive path loses under known appearance shift?)
- **C** — add label weights (does correcting prevalence beat covariate-only?)
- **D** — add the OOD screen (does it cut leakage?)
- **R** — the real-data headline on both benchmarks
- **S** — the subgroup audit

Each rung is a **debugging gate, not a proof**: if layer B can't beat the naive path on synthetic data where you *control* the shift, the real-data result can't be trusted either — so you stop and fix B before stacking C. Synthetic data is the laboratory where you control the truth; the benchmarks are the real exam.

**The one dated decision that protects the whole timeline.** The deadline is **2026-10-05**, about 3.5 months out. The CheXpert/MIMIC credentialing is a weeks-long human-approval bottleneck entirely outside the team's control — the classic way a two-dataset plan quietly drifts into a half-finished one. So the plan replaces a vague "if it slips…" with a **dated, artifact-keyed go/no-go**: *by 2026-07-31, GO two-benchmark **only if** both the PhysioNet approval and the Stanford AIMI access are physically in hand; otherwise COMMIT to CAMELYON17 as the primary benchmark, with the X-ray data dropping in if and when it arrives* (`flagship-playbook.md` §8). The trigger is the **artifact** (the approval emails), not the calendar mood. And because the entire CORE deliverable — full pipeline, four baselines, OOD screen, subgroup audit, the runnable trust panel — is built credentialing-independent on CAMELYON17 by roughly week five, the second benchmark becomes a **re-run on new data, not a fresh build.** The binding risk, as the stress-test concluded, is schedule discipline — not the science.

---

### What you can now understand

- **Pre-registration** = fixing your measurement ruler (datasets, baselines, metrics, operating points) *before* seeing results — and why that single discipline is what makes the project's honesty claim believable rather than just asserted.
- The **two real benchmarks** (CAMELYON17 tumor patches, covariate-dominant; CheXpert↔MIMIC X-rays, mixed shift, both directions) and the **source→target** setup that turns "moved to a new hospital" into a measurable experiment.
- The **four baselines** and the specific skeptical question each one answers — including why "ignore the shift" is *expected* to degrade.
- The **five reviewer gotchas** and the exact design move that kills each — above all the **matched-coverage confound** (compare at equal answer-rate) and group-level fold disjointness — and that these checks came straight out of an adversarial stress-test.
- Why **two datasets deep beats many shallow** for a measurement-discipline paper, and the high-level **build ladder + dated go/no-go** that keeps the whole thing shippable by the deadline.

### Check yourself

**Q1.** Dr. Rivera's AI reports a 3% error rate; a "do-nothing" baseline reports 6%. A reviewer sneers, "Your 3% is fake — you just answered fewer cases." What single, pre-registered design choice already defeats this, and how?
**A1.** The **matched-coverage comparison.** Both systems are forced to answer the *same* fraction of cases before their error is compared, so a lower error can't come from simply being pickier — the "answered less" explanation is mechanically ruled out. (Bonus: the risk–coverage curve and the three-way decline breakdown are printed alongside, so the comparison doesn't even rest on one point.)

**Q2.** Why is splitting the data **by patient ID** rather than **by image** treated as sacred, and why does the team *deliberately inject a leaked patient* on top of that?
**A2.** One patient contributes many images/patches; if pieces of the same patient land in both the calibration and test folds, the system effectively peeks at the answer key and the error looks falsely low — even though the image-level rows look disjoint. Splitting by patient ID prevents this. The deliberate **leakage probe** goes further: by injecting a known leak on purpose and measuring how much the error shifts, the team *quantifies* how much optimism leakage would have introduced — turning an invisible risk into a reported, bounded number, in keeping with the measure-and-report stance.

---

*This is the summit. No new machinery — just two final jobs. First, we thread every single piece you've learned into one continuous story: one doctor, one X-ray, start to finish, so you can narrate the whole system yourself. Then we pull back the camera to see the *strategic* choices behind the project and what actually happened in the work session that produced it.*

I now have everything I need. Writing the tier.

## Tier 8 — The whole thing in one breath, and what we did together (and why)

You've climbed seven tiers. You've met confidence scores, the abstain gate, shift reweighting, the out-of-distribution (OOD) screen, measured risk with its honest interval, and the audit panel a clinician actually reads. This last tier does two jobs. First, it threads **all of it** into a single story — one X-ray, one doctor, start to finish — so you can narrate the whole system yourself. Second, it pulls back the camera and shows you the *strategic* choices behind the project, and tells you what actually happened during **this work session**, so you understand not just the machine but the people building it and the decisions they made.

No new machinery here. Just assembly, rationale, and a tour.

---

### One breath: Dr. Rivera and one chest X-ray

Dr. Rivera is a physician at County Hospital. A new chest X-ray lands in her queue. The AI tool — built and tuned at University Medical, a *different* hospital with different scanners and a different patient mix — takes a look. Here is everything that happens, in order, with each stage named.

**1. The model forms a raw opinion (confidence).** The neural network looks at the pixels and produces a raw score — "I think this is a lung nodule, and I'm 0.88 sure." That 0.88 is a *confidence*. On its own it is not yet trustworthy: a raw network confidence is often over-stated, and it was learned at University Medical, not County.

**2. Recalibration fixes what the number means.** Before anyone reads that 0.88, the system passes it through a recalibration step so the displayed confidence — written in the docs as `σ̃(f(x))` (say "sigma-tilde of f of x") — actually means what a clinician reads it to mean. A displayed "0.90" should be right about 90% of the time, not 70%.

**3. Shift reweighting corrects for "County is not University."** County's patients and County's scanners differ from the data the model was built on. The system reweights — it asks "how typical is *this* case of the data the model was calibrated on, and how has the disease mix shifted between the two hospitals?" That correction combines a covariate part (the image looks different) and a label part (the disease prevalence is different). This is the stage that tries to undo the home-field disadvantage of being built somewhere else.

**4. The OOD screen asks "is this even the kind of thing I was built for?"** Before answering, the system runs a screen for *out-of-distribution* inputs — a scan that looks unlike anything in its training world. If County accidentally sends an abdominal scan, a wrong-modality image, or a scan from a scanner so unusual the model has never seen its like, the screen catches it and **routes** it to a human untouched. The system does not guess on aliens.

**5. The gate decides: answer, abstain, or route.** Now three reliability gates fire or don't:
- Is the case too *uncertain* (uncertainty `u(x)` above its threshold `τ̂`, "tau-hat")? → abstain, defer to Dr. Rivera.
- Is it *out-of-distribution* (the OOD score `o(x)` above its threshold)? → route out.
- Is its *covariate weight* untrustworthy — so weird that the shift correction can't be relied on (`ŵ_cov(x)` above a ceiling `w_max`)? → decline.

If none fires, the system **answers**: "Possible nodule, displayed confidence 0.90." If one fires, it stays silent and hands the case to the human, on purpose.

**6. The number Dr. Rivera trusts is a measured risk with an interval.** Crucially, the system never *promises* it is correct on her case. What it offers instead is an honestly **measured** track record: on held-out County-like cases, among the ones it chose to answer, its error rate was (say) 4%, give or take a stated margin — a *confidence interval*, not a guarantee. That measured answered-case risk is the load-bearing deliverable. It comes with caveats attached, by design.

**7. The audit panel and the Decline-Attribution Record (DAR) make it all visible.** At the point of care, Dr. Rivera sees a small trust panel: the recalibrated confidence, a *representativeness chip* (how typical her case is of what the model was audited on), a cohort-reliability flag (`n_eff`, the effective sample size behind this kind of case), and a shift-regime badge (which correction regime is actually in force at County). And if the system **declined**, she gets a **Decline-Attribution Record** — a plain statement of *which gate fired and by how much*: "Declined: out-of-distribution score exceeded threshold by a wide margin." That explanation is **routing-faithful** — it reports the exact predicate the pipeline evaluated to decline, not a plausible-sounding story bolted on afterward.

That's the whole thing. Confidence → recalibrate → reweight for shift → OOD screen → answer-or-route → measured risk with interval → audit panel + DAR. If you can retell those seven beats in your own words, you understand the system.

---

### A one-line map of every project document

The project keeps its thinking in separate documents, each with one job. When you want the detail behind any stage above, here's where to look:

- **`docs/method_note.md`** — *the method itself, plus all the notation.* The single source of truth for every symbol, weight, threshold, and budget (`σ̃`, `ŵ`, `τ̂`, `α_acc`, `n_eff`, and the rest). Section 6 holds the audit layer; section 7 holds the honest limitations.
- **`docs/positioning_memo.md`** — *the pitch and the venue fit.* What the paper claims in one sentence, what it deliberately does **not** claim, and why this is the right journal for it.
- **`docs/preregistration.md`** — *the measurement plan, written before any results.* Which datasets, which baselines, which metrics, what gets locked down in advance so nobody can cherry-pick after seeing the numbers.
- **`docs/explainability_framing.md`** — *the explainable-AI argument.* Why "explaining the model's own reliability and why it declined" counts as real clinical explanation.
- **`flagship-playbook.md`** — *the build plan.* The staged engineering ladder, the schedule to the deadline, what's built first.
- **`analysis/competitor_matrix.csv`** — *the related-work landscape.* One row per prior system, what ingredient each does or doesn't supply (this is where the honest TRUECAM contrast comes from).
- **`docs/stress_test_prompt.md`** and **`docs/stress_test_findings_2026-06-27.md`** — *the self-review.* The reusable adversarial-reviewer prompt, and the findings from running it (more on that below).

---

### The big strategic choices — and *why* each one

A primer that only taught the machine would miss the point. The interesting decisions are the ones about *what kind of paper to write at all*. Four of them matter most.

**Why applied measure-and-report, not a new theorem.** An earlier version of this project tried to prove a new mathematical *guarantee* — a certificate that the system's risk stays bounded even under hospital-to-hospital shift. That ambition was **dropped in full**. Why? Because the honest math says it can't be done in this setting. Three published "impossibility" results stand in the way: OOD detection is provably not learnable in general; finite-sample distribution-free coverage under covariate shift isn't attainable once you have to *estimate* the reweighting; and combined covariate-and-label shift isn't even identifiable from unlabeled target data alone. So promising a guarantee would mean overclaiming. The project's response is the only honest one: **measure** the degradation, report it with intervals and caveats, and make it visible — rather than assert it away. The contribution is a *discipline*, not a *bound*. (This is the heart of `positioning_memo.md`.)

**Why these specific off-the-shelf methods.** Every piece is a published, citable method (RCPS / weighted conformal for the risk control, BBSE and MLLS+BCTS for the label-shift correction, a standard Mahalanobis detector for OOD). Nothing is invented. The novelty is in the *composition and the measurement discipline* — wiring these together under strict data-hygiene rules and reporting every number honestly. Building on a fancy custom backbone would invite the obvious reviewer question, "is the win just your fancy net?" Using standard parts keeps the result *attributable* to the discipline, which is what the paper is actually about.

**Why an Explainable-AI healthcare venue.** The target is a Springer *Discover Computing* collection on machine learning and explainable AI for healthcare (deadline 2026-10-05). The fit is not cosmetic: the project's whole reason for existing is to make a model's reliability *legible at the point of care* — the recalibrated confidence, the representativeness chip, and above all the routing-faithful explanation of *why the system declined*. That's a form of explainability — explaining the model's own trustworthiness — and it's exactly what this collection asks for.

**Why two datasets deep, not ten shallow.** The plan commits to two imaging benchmarks taken thoroughly — CAMELYON17-WILDS (cross-hospital tumor patches) and CheXpert ↔ MIMIC-CXR (cross-site chest X-rays, both directions) — rather than a shallow sweep across many. These two were chosen because they exhibit *exactly* the mixed covariate-and-label shift that breaks the component methods' assumptions — which is precisely where a measure-and-report discipline earns its keep. Going deep lets the project actually demonstrate the measurement story instead of gesturing at it. Breadth (ECG, retina, dermatology) is named as honest future work, not a v1 obligation.

---

### What we did together this session — and why each step

This primer was written alongside a real working session on the project. Here is what happened, in plain terms.

**We ran a 6-lens adversarial reviewer panel against the docs.** Before submitting any paper, you want to find its weak spots *before* a hostile reviewer does. So the project has a reusable "stress-test" prompt that role-plays six different skeptical reviewers at once — one obsessed with clinical realism, one with reproducibility and hidden confounds, one with venue/explainability fit, one with honesty and novelty, one with scope and feasibility, one with measurement soundness. Each attacks the design in parallel; an independent skeptic verifies every complaint is real; a chair de-duplicates and ranks them by how likely they are to block acceptance. The run is recorded in `docs/stress_test_findings_2026-06-27.md`.

**The verdict: REVISE — but fundamentally sound.** The panel found *no fatal flaw and no smuggled-in guarantee*. The honesty discipline held up under attack. What it surfaced was 37 valid findings: two genuine "confound" gaps plus a pile of execution and disclosure hardening — all fixable before the deadline.

**The single thing a hostile reviewer hits first: the matched-coverage confound.** Here's the cleverest attack, in Dr. Rivera's terms. The system *abstains* on hard cases. The baselines it's compared against don't — they're forced to answer everything. So of course the system looks more accurate on the cases it chose to answer: it's grading itself only on the easy ones it kept. A sharp reviewer reads the whole headline result as "bought by abstaining more, not by actually correcting for shift." The defusing fix is a single pre-registered commitment: compare the system and the baselines at the **same answer rate**, and print the breakdown beside every number — turning the most damaging attack into a demonstrated control.

**We applied all of the findings to the docs.** That confound fix plus the other ~23 findings (validate the accept-side trust signals, not just the decline ones; test the OOD screen in the *hard* near-OOD regime where it actually matters, not just where it's trivially easy; pin the fold-disjointness to patient/slide IDs so the same patient can't leak across splits; check the displayed confidence's calibration per-subgroup; name a single labelled "what is new here" paragraph; and more) were written into the documents — as **pre-registered commitments** where they were promises about how to measure, as **clearer prose** where they were positioning, and as **planning** where they were schedule discipline (most importantly: set a *dated* credentialing go/no-go so the long-pole dataset approval can't silently slip the deadline).

**Then a polish pass, and we pushed it all to the repo.** Loose ends were tightened — consistent self-limiting terminology like "routing-faithful" instead of bare "faithful," removing a convenience sum that could be misread as a guarantee — and the whole updated package was committed to the project's git repository. Each step had a reason: stress-test to find the weak spots, apply fixes to close them as *advance commitments* (so they can't be quietly dropped later), polish so nothing reads as an overclaim, push so the work is saved and shared.

---

### A closing word, and how to keep learning

You started this primer with no coding, no machine learning, and no research background. You can now narrate a real medical-AI system end to end, name the document that holds each piece, explain *why* the project chose honesty-and-measurement over a flashier guarantee, and describe how a team pressure-tests its own work before the world does. That is genuinely a lot.

If you want to go deeper, the natural next moves are: open `docs/method_note.md` and find a symbol you met here (try `n_eff` or `σ̃`) and read its precise definition; skim `docs/preregistration.md` to see what "committing to a measurement in advance" actually looks like on the page; and read one row of `competitor_matrix.csv` to see how the project positions itself against a real prior system. Each will now make sense, because you have the scaffolding. The hard part — knowing what all of it is *for* — is behind you.

---

### What you can now understand

- **The full pipeline as one story:** confidence → recalibrate → reweight for shift → OOD screen → answer-or-route → measured risk with interval → audit panel + Decline-Attribution Record, all through Dr. Rivera and one chest X-ray.
- **Where every detail lives:** which of the seven project documents holds the method, the pitch, the measurement plan, the explainability argument, the build plan, the related work, and the self-review.
- **The four strategic choices and their rationale:** applied measure-and-report (because guarantees are provably impossible here), off-the-shelf methods (to keep the win attributable), an explainable-AI healthcare venue (because explaining the model's *own* reliability is the contribution), and two datasets deep (because that's where the discipline earns its keep).
- **What this session did and why:** a 6-lens adversarial review returned "revise but sound," exposed the matched-coverage confound as the first-strike attack plus ~23 other findings, all of which were applied to the docs as pre-registered commitments / clearer prose / planning, then polished and pushed.

### Check yourself

**Q1. Dr. Rivera sees the AI display "92% confident — possible nodule." A colleague says, "So it's guaranteeing it's right 92% of the time on this scan." What's wrong with that reading?**
*A. Two things. The 92% is a recalibrated confidence meant to be honest about what the number means, but it is still an estimate, not a promise about her one specific case. And the system's real trustworthiness claim isn't per-case at all — it's a* measured *answered-case error rate on held-out County-like data, reported with a confidence interval and its caveats. The whole project deliberately refuses to "certify" or "guarantee" any single prediction; it measures and reports.*

**Q2. The stress-test panel said the "matched-coverage confound" is the first thing a hostile reviewer attacks. In one sentence, what is the confound and what's the fix?**
*A. Because the system abstains on hard cases while the baselines must answer everything, its accuracy on the cases it kept could just be "bought by abstaining more" rather than by correcting for shift — and the fix is to pre-register a comparison at the* same answer rate *for both, printing the routing breakdown beside every headline number.*

---

## Glossary

A friendly one-liner for every key term and symbol in the primer. If a word ever stumps you mid-tier, this is your safety net.

- **Abstain (defer):** The tool's move of declining a case and handing it to a human because it's too *uncertain* — "this is a hard one, you take it, Dr. Rivera." Treated as a feature, not a failure.
- **Accuracy:** The fraction of cases the tool gets right (e.g. 92 out of 100 = 92%). The flip side of *error*.
- **`alpha_acc` (α_acc):** A dial set in advance saying *how much error you'll tolerate* among the cases the tool answers — e.g. 0.05 means "wrong at most 5% of the time on answered cases."
- **AURC (risk–coverage curve):** A chart showing error versus answer-rate across *all* threshold settings at once, so the whole trade-off is visible instead of one cherry-picked point.
- **Auditability / auditable:** The property that every number the tool shows can be *checked* and traced back to how it was measured. Nothing is "just trust us."
- **Baseline:** A simpler approach run on the *exact same data*, so you can tell whether the full system's numbers are actually good. The project pre-registers four.
- **BBSE / MLLS:** Two published methods for estimating *label shift* — figuring out how the disease mix changed at County without needing County's labels. (You only need: "they estimate the new prevalence indirectly.")
- **Calibration data (calibration pile):** A fresh pile of graded examples, never used for learning, used to *tune the dials* (like where to set thresholds). Most of what this project does happens here.
- **Calibrated:** A confidence number means what it says — "70% sure" really is right about 70% of the time. The opposite is *miscalibrated*.
- **CAMELYON17:** A real benchmark of tumor-tissue image patches from five different hospitals; the project's *covariate-dominant* (appearance-shift) test set. Open-access.
- **CheXpert ↔ MIMIC:** Two real chest-X-ray datasets from different medical systems; the project's *mixed-shift* (appearance + prevalence) test set, run in both directions.
- **Class:** One of the fixed categories the tool sorts images into (e.g. `nodule` / `no nodule`). The count of classes is written `K`.
- **Classifier:** The core machine that sorts each input image into one of the classes — "image in, guess out."
- **Confidence:** A number (0 to 1, or a percentage) for how sure the tool is about its guess. The raw material the whole project works on.
- **Confusion matrix:** A table of how often the model mistakes each diagnosis for another (measured at University); used to back out County's true prevalence.
- **Conformal prediction:** The family of methods that turns a confidence score into a *measured* error promise by counting on held-out graded examples.
- **Coverage:** The fraction of cases the tool actually answers (rather than deferring). High coverage = it's doing a lot of the work itself.
- **Covariate shift:** When the *images* look different between hospitals (different scanners) but a given image still means the same diagnosis. Moves "what the picture looks like."
- **DAR (Decline-Attribution Record):** A plain statement, for a declined case, of *which gate tripped and by how much* — read straight off the real decision rule, so it can't misrepresent why the tool declined.
- **`delta` (δ):** A dial for *how sure you want to be* that the error promise actually holds, given that the calibration pile is finite. `delta = 0.05` means "95% confidence in the promise."
- **Disjoint:** Strictly separate, no overlap — the property the three data piles must have (and that patients must not straddle).
- **Distribution:** The whole population of cases a hospital produces — its typical mix of images and diagnoses. (Not the everyday "handing things out.")
- **Distribution-free:** A guarantee that needs no assumption about the *shape* of the data (no bell curve). A real strength — but it still needs *exchangeability*.
- **Distribution shift:** The umbrella term for "the hospital it's used at is not the hospital it was built at." Splits into *covariate shift* and *label shift*.
- **Domain discriminator:** A small helper model that guesses whether an image came from University or County; how "County-ish" it judges an image becomes that image's *covariate weight*.
- **ECE (calibration error):** A measured number for how far the displayed confidence drifts from the truth (does "0.97" really mean right 97% of the time?).
- **Error:** The fraction of cases the tool gets wrong. Accuracy + error = 100%.
- **Exchangeability:** The one load-bearing assumption behind the error promise — the calibration cases and the real cases come from the same "interchangeable" world. Moving hospitals breaks it.
- **Features (summary / embedding):** The internal list of numbers the model boils an image down to before guessing — "the gist" of the image. Written `φ(x)`.
- **Frozen:** The image-reading model was trained once (at University) and then *locked*; this project builds a trust layer around it without retraining it.
- **Gate:** The rule that decides answer-vs-decline by comparing a score to a *threshold*. The system has three gates: uncertainty, OOD, and untrustworthy-weight.
- **Guess (prediction):** The most likely class the tool picks for an image. Same thing as "prediction."
- **Hájek / self-normalizing estimator:** A weighted average that "divides by the sum of the weights" so the unknown overall scale of the weights cancels out — keeping the error estimate honest.
- **Importance weighting (reweighting):** Counting some University cases more heavily than others so University's data mix imitates County's — like reweighting a poll. The weight is `w(x,y)`.
- **Label shift (prevalence shift):** When the *mix of diagnoses* differs between hospitals (same diseases, different frequencies). Moves "what the answer is." "Prevalence" = how common a condition is.
- **Leakage:** Data sneaking from one pile into another where it doesn't belong (e.g. the same patient in both calibration and test), which inflates results dishonestly.
- **Loss:** The cost of a single case — `1` if the guess is wrong, `0` if right (the default "0–1 loss"). Averaging losses gives *risk*. Written `ℓ`.
- **Mahalanobis detector:** The specific published method the project uses to compute the OOD score (how far an image sits from the training data in feature-space).
- **`n_eff` (effective sample size):** A warning light saying how many cases an estimate is *really* resting on after lopsided weights are accounted for — small `n_eff` means trust the number less.
- **Miscalibrated:** A confidence number that doesn't match reality ("says 70, delivers 50"). The specific danger to Dr. Rivera, because false confidence switches off human caution.
- **Minimum-coverage floor:** A pre-set rule that the tool must answer at least a useful fraction of cases, so it can't fake a perfect record by abstaining on almost everything.
- **OOD (out-of-distribution):** An input that's *not even the right kind of thing* (a knee X-ray sent to a chest model). Can't be reweighted — it gets *routed out* to a human.
- **OOD score (`o(x)`) / threshold (`t_ood`):** A number for how far an image sits from the training world, and the cutoff above which the case is routed out.
- **Operating point:** The actual threshold setting the system ships — the chosen spot on the risk–coverage trade-off.
- **Pipeline:** The series of steps an image passes through, one after another, like an assembly line — image in one end, decision out the other.
- **Pre-registration:** Writing down and timestamping exactly what you'll measure, and how, *before* looking at any results — so you can't cherry-pick afterward.
- **Prevalence:** The medical word for how common a condition is in a population.
- **RCPS (Risk-Controlling Prediction Sets):** The published recipe that picks the threshold so the answered-case error stays under your target — *with a safety margin* for the fact that the calibration pile is only a finite sample.
- **Recalibration:** The step that corrects the raw confidence so the displayed number means what a clinician reads it to mean. Written `σ̃(f(x))` ("sigma-tilde"). (Same root as *calibrated*, but it's the fixing *step*, not the pile or the property.)
- **Representativeness chip:** A point-of-care badge saying how *typical* this particular scan is of the data the tool was calibrated and audited on.
- **Risk:** Average error, with *costs* attached so that dangerous mistakes (missing a real nodule) can count more than safe ones. With the default 0–1 loss it's just the plain error rate.
- **Risk control:** Tuning a threshold so a measured error rate stays under a chosen target — the job conformal/RCPS does.
- **Route out:** The tool's move of refusing a case because it's *the wrong kind of input* (OOD), distinct from abstaining (a hard but valid case).
- **Routing-faithful:** The DAR truthfully reports *which gate the pipeline used to route the case* — faithful to the routing decision, **not** a claim that the diagnosis or biology is correct.
- **Saliency heatmap:** The usual "explainable AI" picture — a colored overlay claiming to show where a model looked. The project contrasts it with the DAR because heatmaps can be *unfaithful* (some don't even change when the model is scrambled).
- **Selective prediction:** The fancy name for letting the tool *choose whether to answer at all* — answer, or abstain. The trust mechanism at the project's core.
- **Selective risk:** The error rate measured *only among the cases the tool chose to answer* — ignoring deferred cases. Must always be reported with *coverage*.
- **Shift-regime badge:** A point-of-care label stating which kind of shift is actually in force ("pure prevalence" vs. "combined"), so the easy case can't be passed off as the hard one.
- **Softmax:** The final step that spreads 100% of belief across the classes, turning raw scores into confidences that sum to 100%. Written `σ`.
- **Source / target:** *Source* = where the AI was built and calibrated (University Medical). *Target* = where it's deployed (County Hospital). Written `P_S` / `P_T`.
- **Test data (test pile):** A pile the model has *never seen in any way*, used once at the end to measure how good it really is. The stand-in for "real patients tomorrow."
- **Threshold (`τ`, "tau"):** The cutoff line on the uncertainty score that sets how sure is sure enough to answer. The dial for the answer/abstain gate. (RCPS's chosen answer/abstain cutoff is also written `λ`, "lambda.")
- **Training data:** The pile the model *learns* from (with answers attached). In the story, this learning happened at University.
- **TRUECAM:** An existing published pipeline that handles weird inputs by throwing them out; one of the four baselines and the closest prior system.
- **Trustworthy:** You can rely on the tool *because* it's honest about its own limits, not because it claims to be perfect.
- **Uncertainty score (`u(x)`):** The flip side of confidence — high when the model is unsure, low when confident. What the answer/abstain gate compares to the threshold.
- **`w_cov` / `w_lab`:** The *covariate weight* (how County-ish an image looks) and the *label weight* (how County-ish a diagnosis is). Combined into the total weight — but *not* by simply multiplying them.
- **`Z(x)` (correction factor):** The quantity you divide by when combining the two weights, to remove the *double-counting* that simple multiplication would introduce.

---

## The whole pipeline in one breath

You can now read this sentence and feel every word click into place:

A new chest X-ray lands in Dr. Rivera's queue at County. The frozen model (built at University) **forms a raw confidence**; a **recalibration** step makes that confidence mean what she'll read it to mean; **shift reweighting** corrects for County not being University — counting University cases more or less heavily by how County-ish they look (covariate weight) and how County-ish the diagnosis mix is (label weight), combined *without* double-counting by dividing out the correction factor; an **OOD screen** catches anything that isn't even a chest X-ray and routes it out; then the **three gates** fire or stay quiet — too uncertain → abstain, off-the-map → route out, untrustworthy weight → decline — and if none fires, the tool **answers**. The number Dr. Rivera trusts isn't a promise but a **measured answered-case error rate with an honest interval**, and the **audit panel + Decline-Attribution Record** show her the recalibrated confidence, how representative her case is, how many cases the estimate really rests on, which shift regime is in force, and — if it declined — exactly which gate tripped and by how much. No certificate. No guarantee. Just honest measurement, made visible.

If you just understood that whole paragraph, you understand the system. That's the entire ladder, from Tier 0's three moves to Tier 8's one breath.

## The one-line doc map

When you want the detail behind any piece, here is where each one lives:

- **`docs/method_note.md`** — the method itself and every symbol (the single source of truth).
- **`docs/positioning_memo.md`** — the one-sentence pitch, what it deliberately doesn't claim, and the venue fit.
- **`docs/preregistration.md`** — the measurement plan, frozen before any results (datasets, baselines, metrics).
- **`docs/explainability_framing.md`** — why "explaining the model's own reliability and why it declined" counts as real clinical explanation.
- **`flagship-playbook.md`** — the staged build plan and the schedule to the deadline.
- **`analysis/competitor_matrix.csv`** — one row per prior system, including the honest TRUECAM contrast.
- **`docs/stress_test_prompt.md`** + **`docs/stress_test_findings_2026-06-27.md`** — the reusable adversarial-reviewer prompt and the findings from running it.

## Where to go next

You started with no coding, no machine learning, and no research background, and you can now narrate a real medical-AI system end to end and explain *why* its honesty stance is a strength, not a weakness. That is genuinely a lot.

Three gentle next steps, each of which will now make sense because you have the scaffolding:

1. Open `docs/method_note.md` and find a symbol you met here — try `n_eff` or `σ̃` — and read its precise definition.
2. Skim `docs/preregistration.md` to see what "committing to a measurement in advance" actually looks like on the page.
3. Read one row of `analysis/competitor_matrix.csv` to see how the project positions itself against a real prior system.

And if you'd like a patient tutor to quiz you, deepen any tier, or meet you wherever your level is — the reusable prompt below is built for exactly that. The hard part, knowing what all of this is *for*, is already behind you.

---

## Tutor mode (reusable prompt)

Copy everything in the box below and paste it to any capable AI assistant. It turns that assistant into a patient, personal tutor for *this specific project*, able to meet you at any level — total beginner to confident.

```
You are my patient, encouraging personal tutor for one specific project: an applied
medical-AI system described in a nine-tier learning primer (Tiers 0–8). I may be a
complete beginner with no background in coding, machine learning, or research — adapt
to whatever level I show.

THE PROJECT (your subject matter — never contradict this):
An applied medical-AI pipeline for clinical images (chest X-rays, tumor-tissue patches)
that does THREE things instead of always answering: it ANSWERS when confident, ABSTAINS
(defers a hard case to a human), or ROUTES OUT inputs that aren't even the right kind of
image (out-of-distribution). It was built/tuned at one hospital ("source" = University
Medical) but deployed at another ("target" = County Hospital) with different scanners and
a different patient mix — this gap is "distribution shift" (covariate shift = images look
different; label/prevalence shift = the disease mix differs). The pipeline reweights the
source data to imitate the target (importance weighting; combine the covariate and label
weights by DIVIDING OUT a correction factor Z(x), never by plain multiplication), screens
for out-of-distribution inputs, and chooses answer/abstain/route via three gates
(uncertainty, OOD, and untrustworthy-covariate-weight).

Crucially: the project invents NO new method and makes NO new guarantee. Every component
is published (RCPS / weighted conformal for risk control, BBSE and MLLS for label shift,
a Mahalanobis detector for OOD). Its real contribution is TWO things: (1) an honest
MEASUREMENT / AUDITABILITY discipline — every number shown to a clinician is an honestly
measured estimate WITH its uncertainty interval and caveats, never a certificate; the
words "certify/guarantee" are banned for the deployed system — and (2) a CLINICIAN TRUST
INTERFACE, including a "Decline-Attribution Record" (DAR) that explains WHY the tool
declined by reading the exact gate that tripped straight off the real decision rule
("routing-faithful" — faithful to the routing decision, not a claim the diagnosis is
correct). It refuses a guarantee because three published facts make one impossible here:
OOD detection isn't reliably learnable; estimating the reweighting kills the exact coverage
promise; and covariate-vs-label shift isn't identifiable from unlabeled target data alone.

THE RUNNING EXAMPLE: Always teach through Dr. Rivera, a physician at County Hospital
reading chest X-rays for lung nodules, using the AI built at University Medical. Keep her
consistent.

THE TIER LADDER (so you know where each idea lives):
0 = the problem in plain words (three moves, hospital mismatch, "no new guarantee")
1 = how a prediction model works (classifier, features, softmax, calibration, the data piles)
2 = selective prediction (abstain; coverage vs. selective risk; the see-saw; the near-zero trap)
3 = making confidence trustworthy (conformal/RCPS; alpha_acc & delta; distribution-free; exchangeability)
4 = distribution shift (source/target; covariate vs. label shift; reweighting; don't multiply the weights; OOD)
5 = why we refuse to guarantee (the three impossibility/non-identifiability facts; measure-and-report)
6 = the actual contribution (weighted error estimate; n_eff flag; the per-case panel; the DAR; "auditability as explanation")
7 = proving it honestly (pre-registration; two benchmarks; four baselines; the five reviewer "gotchas")
8 = the whole thing in one breath + the strategic choices + what the work session did

HOW TO TUTOR ME:
1. Start by asking what I already feel comfortable with and what I want to focus on. If I
   don't know, propose starting at Tier 0 and climbing.
2. NEVER use a term before you've defined it in plain words. If you must use one, define it
   in the same breath with a tiny everyday analogy (the primer leans hard on analogies:
   the overconfident intern, the see-saw, the graded practice exams, reweighting a poll,
   the empty croissant shelf, the nightclub bouncer, Service A vs. Service B weather apps).
3. Teach intuition FIRST, symbols only if I ask. Keep math optional and clearly labeled.
4. After each idea, ask me a short "check yourself" question and wait for my answer before
   moving on. Gently correct misconceptions; tell me when I'm right and why.
5. If I get something wrong, don't just give the answer — find the smaller missing piece,
   re-explain THAT, and re-ask.
6. Let me drive depth: I can say "deeper," "simpler," "skip the math," "quiz me," "give me
   an analogy," or "where does this live in the docs?" and you adapt instantly.
7. Keep the honesty spirit of the project itself: if something genuinely can't be promised,
   say so and explain why that's a strength.
8. Be warm, plain, and encouraging. I started from zero; treat that as completely normal.

Begin now by greeting me as Dr. Rivera's tutor and asking where I'd like to start.
```
