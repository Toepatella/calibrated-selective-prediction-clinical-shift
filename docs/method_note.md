# Method Note — Calibrated Selective Prediction Under Clinical Distribution Shift

*Realizes [flagship-playbook.md](../flagship-playbook.md) §3 (Method formalization), the **full superset** — covariate **and** label shift, an integrated OOD error budget, weighted RCPS, formal theory. Internal sections `3.1`–`3.7` map one-to-one to the playbook's subsections and are complete. The theory in §3.2–§3.6 was independently derived three times and adversarially red-teamed from four lenses (conformal-theorist, shift-identifiability, OOD-budget, Reviewer-2); §3.7 records the surviving weakest claims and the **honesty fixes folded back into §3.2–§3.6**. Every guarantee sentence here carries its assumption — the playbook's honesty constraint.*

**Conventions.** `monospace` = symbol/variable, identical to the names in `conformal/` and `ood/`. A hat (`x̂`) **always** denotes a quantity *estimated from finite data*; the un-hatted symbol is the population truth, and the gap between them is the approximation-error budget characterized in §3.6. `1[·]` = 1 if the bracket is true else 0 · `⌈·⌉` = round up · `≤/≥` = at most/least · `∝` = proportional to · `Σ` = sum · `∘` = elementwise product. Every weight, threshold, and budget defined in §3.1 is reused verbatim by §3.2–§3.7 and by the code; this section is the single source of truth for symbols.

---

## 3.1 Notation

### Spaces, distributions, and the shift

- `𝒳` — input space (an image). `𝒴 = {1,…,K}` — label space, `K` classes. `Δ^{K−1}` — the probability simplex over `𝒴`.
- `P_S`, `P_T` — the **s**ource and **t**arget joint distributions over `𝒳 × 𝒴`. The entire project is a promise *calibrated on `P_S`* that must still hold *on `P_T`*.
- `p_S(x), p_T(x)` — input (covariate) marginals · `p_S(y), p_T(y)` — label (prevalence) marginals · `p_S(y∣x), p_T(y∣x)` — posteriors · `p_S(x∣y), p_T(x∣y)` — class-conditionals.
- **Shift taxonomy** (which factor moves decides which weight is correct — see §3.3):
  - *Covariate shift* — `p(x)` changes, `p(y∣x)` invariant ⇒ the correct weight depends on `x` only.
  - *Label / prevalence shift* — `p(y)` changes, `p(x∣y)` invariant ⇒ the correct weight depends on `y` only.
  - *Both at once* — neither single weight suffices, and the combined weight is **not** their product (§3.3, the double-counting trap).

### Frozen base model and its outputs

- `f` — the **frozen** base model, `f : 𝒳 → ℝ^K` (logits). Trained on the source *train* split; parameters fixed before any calibration. Validity comes from the conformal layer and is base-model-agnostic (playbook §1, attribution discipline) — `f` only buys efficiency.
- `σ(f(x)) ∈ Δ^{K−1}` — softmax probabilities · `ŷ(x) = argmax_k f(x)_k` — the predicted label.
- `φ(x) ∈ ℝ^p` — penultimate **feature embedding** (`p` = feature dim), optionally spectral-normalized. The OOD score `o(·)` and the domain discriminator `d(·)` both act on `φ(x)`, never on raw pixels.
- `u(x) ∈ ℝ` — scalar **uncertainty** (high = less certain), e.g. softmax-response `u(x) = 1 − max_k σ(f(x))_k` or energy `u(x) = −log Σ_k exp f(x)_k`.

### Loss, selection, and selective risk

- `ℓ(y, ŷ) ∈ [0,1]` — bounded **loss**; default 0–1 loss `ℓ = 1[y ≠ ŷ]`. Boundedness in `[0,1]` is exactly what the RCPS / WSR concentration bound requires.
- `τ` — **selection threshold** on `u`. `g(x) = 1[u(x) ≤ τ]` — the **selection gate**: `g=1` ⇒ answer, `g=0` ⇒ abstain (defer). Condition (1).
- `coverage = P(g(X)=1)` — fraction answered. **Selective (accepted) risk** on a distribution `P`:
  ```
  R^accept_P  :=  E_P[ ℓ(Y, ŷ(X)) | g(X)=1 ]
  ```
  Selection makes the answered subset non-exchangeable with full calibration ⇒ the controlling threshold must be calibrated *on the accepted region* (§3.3).

### Distribution-shift weights (the core)

- **Master importance-weighting identity** — what every weighted method rests on. The Radon–Nikodym derivative
  ```
  w(x,y)  :=  dP_T/dP_S (x,y)  =  p_T(x,y) / p_S(x,y)
  ```
  transports any expectation from source to target:  `E_{P_T}[ h(X,Y) ] = E_{P_S}[ w(X,Y) · h(X,Y) ]`.  `w(x,y)` is the **combined weight**.
- `w_cov(x) := p_T(x) / p_S(x)` — **covariate density ratio**. Equals the full `w(x,y)` *iff* the shift is pure covariate. Estimated from a domain discriminator `d(x) = P(domain = T ∣ x)`:
  ```
  ŵ_cov(x) = clip( (d(x) / (1 − d(x))) · c , 0 , w_max ),     c = π_S / π_T   (domain base-rate correction)
  ```
  where `π_S, π_T` are the source/target proportions in the discriminator's training pool (`c ≈ n_S/n_T`). Unreliable in the far tail ⇒ clip at `w_max` and route the overflow to abstain (§3.3). *(Tibshirani et al. 2019, weighted conformal.)*
- `w_lab(y) := p_T(y) / p_S(y)` — **label ratio**. Equals the full `w(x,y)` *iff* the shift is pure label. Estimated by **BBSE** (Lipton et al. 2018) from the source confusion matrix and target predictions:
  ```
  ŵ_lab = Ĉ_S^{−1} q̂_T,      then   p̂_T(y) = ŵ_lab(y) · p_S(y)
  ```
  - `C_S` — **source confusion matrix**, `C_S[i,k] = P_S(ŷ = i, y = k)` (joint form). Invertibility of `Ĉ_S` is the BBSE **identifiability** condition.
  - `q̂_T` — distribution of `f`'s **predicted** labels on the unlabeled target sample, `q̂_T[i] = P_T(ŷ = i)`.
  - *(This is the precise weight form of the playbook Appendix-F relation `Ĉ_S · p̂_T = q̂_T`: under label shift `q_T = C_S · w_lab`, so `w_lab = Ĉ_S^{−1} q̂_T` and `p̂_T = w_lab ∘ p_S`.)*
- `w(x,y)` — **combined weight** under the *stated structural shift model*. ⚠ `w(x,y) ≠ w_cov(x) · w_lab(y)` in general — the naïve product double-counts when both shifts coexist; the valid form is derived in §3.3 / §3.6, not multiplied.

### OOD score, routing, and clipping

- `o(x) ∈ ℝ` — **OOD score**, a feature-space distance on `φ(x)` (Mahalanobis++ primary, kNN ablation); high = more out-of-distribution.
- `t_ood` — **OOD threshold**: `o(x) > t_ood` ⇒ route to abstain (far-OOD). Calibrated to spend the `α_ood` budget.
- `w_max` — clip cap / **routing cap** on the covariate weight: where `ŵ_cov(x) > w_max` the density ratio is untrustworthy ⇒ route to abstain. This removed tail is exactly the mass that would blow up the weighted bound.
- **In-scope / answered event** (recurs throughout; the conditioning event of the §3.2 guarantee):
  ```
  A(x)  :=  { g(x) = 1 }  ∩  { o(x) ≤ t_ood }  ∩  { ŵ_cov(x) ≤ w_max }
  ```

### Risk-control thresholds and budgets

- `λ` — the **risk-control threshold** tuned on calibration (the RCPS / CRC `λ̂`); indexes a nested family of decision rules whose risk `R(λ)` is monotone in `λ`.
- `R(λ)` — risk as a function of `λ`. `UCB_δ(λ)` — the `(1−δ)` **upper confidence bound** on `R(λ)`, supplied by the **WSR / betting (hedged-capital)** bound — tighter than Hoeffding–Bentkus and variance-adaptive, which matters because importance weights inflate the risk-estimate variance *(Waudby-Smith & Ramdas 2024)*.
  ```
  λ̂ = inf{ λ : UCB_δ(λ′) < α_acc  for all λ′ ≥ λ }      ⇒      P( R(λ̂) ≤ α_acc ) ≥ 1 − δ     (RCPS / PAC)
  ```
  *(Bates et al. 2021.)*
- `α` — total **error budget**, split into the integrated form (§3.2):
  ```
  α = α_acc + α_ood
  ```
  `α_acc` = budget for accepted in-distribution error (controlled by the weighted RCPS above) · `α_ood` = budget for residual far-OOD the detector missed.
- `δ` — RCPS **confidence**: the PAC promise holds with probability `≥ 1 − δ` over the calibration draw.
- Jointly-tuned knobs: `(τ, λ, t_ood)`. Tuning all three on one calibration set is multiple looks ⇒ wrap the search in **LTT** (multiple-testing) to keep the bound valid (playbook §3.5).

### Calibration data, sample sizes, estimates

- `D_cal = {(X_i, Y_i)}_{i=1}^n`  iid `∼ P_S` — **labeled source calibration set**; `n = |D_cal|`.
- `D_tar = {X̃_j}_{j=1}^m`  iid `∼ p_T(·)` — **unlabeled target sample** (covariates only) for estimating `ŵ_cov`, running BBSE, and setting `t_ood`; `m` = target sample size. Small `m` is a primary stress axis (playbook §6.2).
- `D_tar^lab = {(X̃_k,Ỹ_k)}_{k=1}^{m_lab}`  iid `∼ P_T` — **small labeled target slice** (`m_lab ≈ 50–200`), disjoint from `D_tar`. Its sole role is to **identify the otherwise-unidentifiable FJS nuisance parameters** (§3.3): it pins the scale constant in the FJS identity and supplies an honest, *measured* `ε_M` instead of an assumed one. Optional but strongly recommended; without it, only the label-shift corollary (§3.6) is certifiable.
- **Disjointness (leakage discipline).** The subsets used to fit the discriminator `d`, estimate the **source confusion `Ĉ_S`** and the **target predicted-label histogram `q̂_T`** for BBSE, fit the OOD detector `o`, calibrate `(τ, λ, t_ood)`, the auxiliary **OOD-exposure set `O`** (§3.2), the **labeled target slice `D_tar^lab`**, and the final test set are mutually **disjoint** — asserted in code (playbook risk register: "data leakage"). In particular `Ĉ_S` is fit on a *source* fold disjoint from `D_cal`, so every weight is a deterministic function of folds independent of the calibration losses — the precondition for the §3.6 martingale.
- **Hat = estimate.** `ŵ_cov, ŵ_lab, p̂_T, q̂_T, Ĉ_S, λ̂, τ̂, t̂_ood` are finite-sample estimates of their un-hatted population counterparts; their errors compose into the approximation bound of §3.6 (weight error + BBSE error + detector error → `α_ood`).

### Quick-reference table

| Symbol | Type | Meaning |
|---|---|---|
| `P_S, P_T` | distribution | source / target joint over `𝒳×𝒴` |
| `f, σ(f(x)), ŷ(x)` | model | frozen logits · softmax · predicted label |
| `φ(x)` | vector `∈ ℝ^p` | penultimate features (input to `o`, `d`) |
| `u(x)` | scalar | uncertainty (high = unsure) |
| `ℓ(y,ŷ) ∈ [0,1]` | scalar | bounded loss (default `1[y≠ŷ]`) |
| `g(x)=1[u(x)≤τ]`, `τ` | gate / threshold | selection: 1 = answer, 0 = abstain |
| `R^accept_P`, `coverage` | scalar | selective risk `E_P[ℓ∣g=1]` · fraction answered |
| `w_cov(x)=p_T/p_S(x)` | weight | covariate density ratio (discriminator) |
| `w_lab(y)=p_T/p_S(y)` | weight | label ratio (BBSE) |
| `w(x,y)=dP_T/dP_S` | weight | combined weight (**≠** product, §3.3) |
| `d(x), c` | prob. / const | domain-discriminator posterior · base-rate const `π_S/π_T` |
| `C_S, q̂_T` | matrix / vector | source confusion · target predicted-label dist. (BBSE) |
| `o(x)` | scalar | OOD score (feature distance) |
| `t_ood, w_max` | thresholds | OOD cutoff · weight-clip / routing cap |
| `λ, R(λ), UCB_δ` | thresh. / fn | risk-control threshold · risk · WSR `(1−δ)` upper bound |
| `α = α_acc + α_ood` | budgets | total = accepted-error + OOD-leakage |
| `δ` | scalar | PAC confidence (promise holds w.p. `≥ 1−δ`) |
| `n, m` | counts | source-cal size · unlabeled target-sample size |
| `D_cal, D_tar` | datasets | labeled source cal · unlabeled target sample |
| `A(x)` | event | answered & in-scope: `g=1 ∧ o≤t_ood ∧ ŵ_cov≤w_max` |

---

## 3.2 Target guarantee (integrated)

**The aspirational target.** Over the observable in-scope/answered gate `A(x) = {g=1} ∩ {o≤t_ood} ∩ {ŵ_cov≤w_max}` (§3.1), we want the answered-case target risk controlled:
```
R_T^accept := E_{P_T}[ ℓ(Y,ŷ(X)) | A(X) ]  ≤  α = α_acc + α_ood.
```

**What is actually certifiable (the honest box).** The finite-sample certificate controls the **plug-in** self-normalized weighted accepted-in-scope risk `R̂_w` (built from the *estimated* weight `ŵ`, §3.3) and the **audited** far-OOD leakage `L̂_O` (measured on an exposure set `O`, below). The gap from these to the true `R_T^accept` is a set of **deterministic biases the confidence level δ does not cover** — so the bare `≤ α` box overclaims and is replaced by:
```
CERTIFIED  (finite-sample, marginal over the disjoint calibration draws):
    P_{D_cal, O}(  R̂_w(τ̂,λ̂) ≤ α_acc   AND   L̂_O(t̂_ood) ≤ α_ood  )  ≥  1 − δ,     δ = δ_acc + δ_ood.

DEPLOYMENT  (the TRUE answered-case risk), as a deterministic inequality under (A3, A3′):
    R_T^accept  ≤   α_acc + α_ood   +   ε_w + ε_BBSE + ε_M
                    └── certified ──┘     └── named, non-δ-covered approximation biases ──┘
    with ε_det (detector proxy-vs-deployment error) folded INTO α_ood under (A3, A3′).
```
In words: among target cases answered and judged in-scope, the *plug-in* expected loss is controlled at `α_acc` (conf. `1−δ_acc`) and the *audited* far-OOD leakage at `α_ood` (conf. `1−δ_ood`); the **true** risk equals that budget **plus** three named deterministic biases — covariate-weight `ε_w`, BBSE `ε_BBSE`, shift-model misspecification `ε_M` — which sit *outside* the δ-event and are either reported or absorbed by calibrating against the deflated target `α_acc − ε_w − ε_BBSE − ε_M`. §3.6 characterizes each ε; §3.7 records why this honest box, not `≤ α`, survives red-team.

**The TRUECAM resolution, stated honestly.** TRUECAM removes OOD heuristically and concedes the result *"cannot be guaranteed in deployment."* We do **not** assume detector error is zero — we **budget and audit** it: the far-OOD mass the detector misses is bounded by a certified miss-rate `α_ood` against a *stated, swappable* exposure model `O`, leaving a single **named, auditable residual** `ε_det` under (A3) rather than an unbounded gap. The contribution is therefore not *"guaranteed under any deployment"* but *"the previously-unbounded OOD gap reduced to one budget line plus one named residual under a stated exposure model"* — still strictly beyond every audited competitor.

**Standing assumptions (load-bearing; carried by every guarantee sentence).**
- **(A1) Bounded loss** `ℓ ∈ [0,1]` — turns missed-mass *probability* into a *risk* bound.
- **(A2) Bounded post-routing weight** `w(x,y) ≤ W` on the retained region — the precondition of the WSR bound; secured *only* by routing the high-ratio/far tail (§3.3).
- **(A3) Detector-transfer / proxy conservativeness** — `O` over-approximates the true far-OOD set on the answered region and stochastically dominates deployment far-OOD in *detectability*. Untestable without labeled target OOD; **stated, not assumed away**.
- **(A3′) OOD prevalence** — `P_T(far-OOD)/P_T(A) ≤ 1`, so the OOD-*conditional* miss-rate `α_ood` upper-bounds the answered-population far-OOD *fraction*; otherwise the residual prevalence factor is named in `ε_det`.
- **(A4) Identifiable shift model** — the combined weight is identifiable under **Factorizable Joint Shift (FJS, Tasche 2022/2026)**: the covariate-shift mechanism and the label-shift mechanism are independent (no joint shift in the *interaction* of the two), **anchored by the small labeled target slice `D_tar^lab`** which resolves FJS's scale ambiguity and its non-identifiability from unlabeled target covariates alone (Tasche's "Problem B"). Without `D_tar^lab`, FJS is *not* identifiable and only the pure-label-shift corollary (§3.6) is certifiable.
- **(A5) Disjoint folds + union budget** — all estimators are fit on mutually disjoint folds; `α=α_acc+α_ood`, `δ=δ_acc+δ_ood` compose by a union bound over the two independent calibration events, with a **single coherent `t_ood`** feeding both (§3.4).

---

## 3.3 The four exchangeability-breakers + neutralizers

Each mechanism below breaks the exchangeability split-conformal/RCPS needs between calibration and a fresh test point; each is neutralized, leaving an explicit **residual assumption**.

| Breaker | Why it breaks exchangeability | Neutralizer | Residual assumption |
|---|---|---|---|
| **Selection** `g=1[u≤τ]` answers only low-uncertainty (easier) cases | A bound on the *marginal* risk does **not** bound the *conditional* (accepted) risk `P(·∣g=1)` | Calibrate **on the accepted region**: compute `R̂_w` and the WSR UCB using only cal points with `g=1` (and in-scope `A=1`); selection becomes part of the risk functional; `(τ,λ)` co-selected via LTT | Cal and test accepted+in-scope points share `P_T(·∣A)`; `u,g,A` fixed on disjoint folds; **coverage `P(g=1)` is not itself guaranteed** — only risk given answering; need enough `n_eff` |
| **Covariate shift** `p(x)` moves, `p(y∣x)` fixed | Cal `∼P_S`, test `∼P_T`: unweighted risk is biased; exchangeability → *weighted* exchangeability (Tibshirani 2019) | Reweight every cal contribution by clipped `ŵ_cov`; clip+route the far tail | Overlap `supp(p_T)⊆supp(p_S)` on the in-scope region; `d` calibrated, bounded off `{0,1}`; `c≈n_S/n_T`. Leaves **`ε_w`** |
| **Label / prevalence shift** `p(y)` moves, `p(x∣y)` fixed | Covariate-only reweighting cannot capture a *label-marginal* change; naïvely multiplying `w_cov·w_lab` **double-counts** | BBSE `ŵ_lab` (`p̂_T=Ĉ_S⁻¹q̂_T`), **combined** as `w=w_lab·w_cov/Z` — not the product (below) | `Ĉ_S` invertible & well-conditioned; `p(x∣y)` invariant (anti-causal); **class-homogeneity** (untestable). Leaves **`ε_BBSE`** and **`ε_M`** |
| **Far-OOD tail + imperfect detector** `p_S≈0`, or huge density ratio, or a leaky detector | On target-only support `dP_T/dP_S` is undefined/∞; one huge weight drives `n_eff→1` and the UCB explodes (TRUECAM's gap); a leaky detector passes far-OOD into the answered set | **Route** to abstain on `UCB[ŵ_cov]>w_max` OR `o>t_ood` (restores `w≤W`), **and budget** the residual: calibrate `t_ood` on disjoint `O` so the audited far-OOD leakage `≤ α_ood` | (A3)+(A3′). **OOD protection rests on `o(x)`** — the discriminator can be confidently wrong (`d≈0`) on novel inputs, so `w_max` alone does *not* catch far-OOD. Honest residual **`ε_det`** |

Two routing mechanisms, **decoupled**: `w_max` is a *variance/boundedness* control on the in-scope weighted estimator (its routed mass costs only coverage — correctly excluded from `A`, no α budget needed); `t_ood` is the *sole far-OOD guard* whose miss-rate is charged to `α_ood`. Far-OOD is defined **operationally** (the observable gate), not by the support boundary `{p_S=0}`: routing on `UCB[ŵ_cov]>w_max` puts the dangerous near-OOD (finite-but-enormous ratio) *inside* the α_ood-charged routed mass.

**Deriving the combined weight (do not multiply).** Under a *stated* structural model the combined weight is the label ratio times a **corrected** covariate factor.
- **Why FJS, not the class-homogeneous model M, and not Sparse Joint Shift (SJS).** Three identifying assumptions were weighed (playbook honesty constraint — name the choice and why):
  - *Class-homogeneous model M* (`p_T(x∣y)/p_S(x∣y)` the same for every `y`) is **untestable** from unlabeled target data and is in direct tension with BBSE's own `C_T=C_S` assumption (§3.7 #4) — rejected as the primary identifying claim.
  - *SJS* (Chen, Zaharia & Zou 2022; Tasche 2023) buys label-free identifiability by assuming only a **sparse subset of features** shifts. That premise fits **tabular** clinical features (a few named labs/vitals) but not **raw-image covariate shift** (scanner/stain/protocol changes act on the *entire* pixel/embedding space at once — dense, not sparse). Adopting SJS here would mean assuming away the exact shift this project targets — rejected for the imaging tracks.
  - **FJS** (Tasche 2022/2026): the covariate-shift mechanism (`r`, below) and the label-shift mechanism (`w_lab`) are statistically **independent processes**, with no claim about *how many* dimensions move or *how* they move within a class. This is the weakest of the three premises for imaging data — a scanner swap and a local prevalence change have no obvious causal coupling — so FJS is the adopted identifying assumption. **Caveat carried forward:** FJS is still not identifiable from unlabeled target covariates alone (Tasche's "Problem B": the recovered factors are determined only up to an unknown positive scale constant `c`) — this is resolved operationally by the small labeled target slice `D_tar^lab` (§3.1), not by FJS independence alone.
  - Model M is retained only as the special case `r(x)` constant-within-class — i.e., **FJS restricted to a degenerate, class-blind covariate factor** — making M a strict (and strictly less defensible) sub-case of FJS, not a separate competing model.
- **FJS factorization.** Joint factors as `p(x,y)=p(y)p(x∣y)`, with (i) free prevalence change `p_T(y)≠p_S(y)` and (ii) a covariate distortion factor `r(x) := p_T(x∣y)/p_S(x∣y)` whose *mechanism* is independent of the label-shift mechanism — `r` need not be class-homogeneous, only generated by a process independent of what moves `p(y)`. Pure label shift (`r≡1`) and pure covariate shift (`p_T(y)=p_S(y)`) are the two boundary cases.
- **The valid weight.** By the master identity `w(x,y)=w_lab(y)·r(x)`. Writing the observable marginal ratio `w_cov(x)=r(x)·Z(x)` with the **per-`x` normalizer**
  ```
  Z(x) := Σ_{y'} w_lab(y') · p_S(y'∣x) = E_{p_S(·∣x)}[ w_lab ]      (estimated  Ẑ(x)=Σ_{y'} ŵ_lab(y') σ̃(f(x))_{y'})
  ```
  gives `r(x)=w_cov(x)/Z(x)` and the **FJS-identified combined weight**
  ```
  w(x,y) = w_lab(y) · w_cov(x) / Z(x)   ·   ĉ_scale     (ĉ_scale fit from D_tar^lab, §3.4 step 2.5)
  ```
  `Z(x)` is exactly the double-count corrector the naïve product omits; `ĉ_scale` is the FJS scale constant unidentifiable from `D_tar` alone. Transport is then exact, `E_{P_S}[w·h]=E_{P_T}[h]`, **given FJS + the labeled-slice scale anchor**. Collapses to `w_lab(y)` (pure label, `w_cov=Z`, `ĉ_scale=1`) and to `w_cov(x)` (pure covariate, `w_lab≡1,Z≡1`).
- **Why the product is wrong (worked).** Even under *pure* label shift (truth `w=w_lab(y)`, `x`-independent), prevalence change moves `p(x)`, so `w_cov(x)=Z(x)` is non-constant; the naïve product inflates by exactly `Z(x)`. E.g. `p_S(y)=(.5,.5)→p_T(y)=(.9,.1)`, invariant class-conditionals: at an `x` where class 1 dominates, `w̃(x,1)≈1.8·1.8=3.24` vs truth `1.8`. Dividing by `Z(x)` removes it.
- **The identifiability caveat (the weakest link, §3.7).** FJS independence is itself **untestable** without labeled target data — it relocates, rather than removes, the identifiability problem (Tasche 2022/2026 §3.5: "FJS does not uniquely determine the shift" from target covariates alone). The labeled slice `D_tar^lab` is the load-bearing fix: it (i) pins `ĉ_scale`, and (ii) gives a **measured** `ε_M` — the gap between the FJS plug-in estimate and ground truth on the slice — instead of an assumed-zero or unbounded one. With `m_lab` small, `ε_M`'s own estimation variance is reported, not silently absorbed. **Adopted consequence:** the **pure-label-shift corollary** (`ε_w=ε_M=0`, fully identifiable, no labeled slice needed) remains the *primary certified* claim; the FJS combined covariate+label case, anchored by `D_tar^lab`, is the *secondary certified* claim (certified conditional on the slice, not assumption-free); the old class-homogeneous-model-M combined case without any labeled slice is no longer claimed at all.

---

## 3.4 Calibration + inference algorithm

```
CALIBRATION   (mutually DISJOINT folds: D_disc · D_bbse^src (labeled source) · D_tar (unlabeled target)
               · D_tar^lab (small labeled target slice) · O (OOD-exposure) · D_cal (labeled source, accepted-region)
               · held-out test)

  1. DISCRIMINATOR.   Fit d(x) on D_disc (source-vs-target).
                      ŵ_cov(x) = clip( (d/(1−d))·ĉ , 0 , w_max ),  ĉ = n_S/n_T.
                      Route on an UPPER confidence bound UCB[ŵ_cov(x)] > w_max  (so retained pts have true w ≤ w_max w.h.p.).

  2. BBSE.            Ĉ_S on D_bbse^src (labeled source, disjoint from D_cal);  q̂_T on D_tar.
                      ŵ_lab = Ĉ_S⁻¹ q̂_T;  PROJECT p̂_T onto the simplex and FLOOR ŵ_lab ∈ [w_lab,min , w_lab,max], w_lab,min>0
                      (else ill-conditioned Ĉ_S makes Z, W explode or go negative).
                      DIAGNOSTIC: compare q̂_T vs Ĉ_S p̂_T to flag gross anti-causal / C_T≠C_S violation.

  3. COMBINE (not multiply).  Ẑ(x) = Σ_{y'} ŵ_lab(y')·σ̃(f(x))_{y'} with σ̃ a TEMPERATURE/Platt-recalibrated softmax
                      fit on a held-out source fold (raw softmax is a non-vanishing denominator bias).
                      ŵ_0(x,y) = ŵ_lab(y)·ŵ_cov(x)/Ẑ(x).   Sharp cap  W = w_max·(max_y ŵ_lab / min_y ŵ_lab).

  3.5 FJS SCALE ANCHOR.  On D_tar^lab (small labeled target slice, disjoint from all above), fit the FJS scale
                      constant ĉ_scale = argmin_c Σ_{(x,y)∈D_tar^lab} ( c·ŵ_0(x,y)·ℓ̃(x,y) − 1 )²  (or the EM-style
                      Saerens/Tasche-2026 refinement seeded by D_tar^lab and run to convergence over D_tar).
                      REPORT measured ε_M = | plug-in risk using ĉ_scale·ŵ_0  −  empirical risk on D_tar^lab |.
                      ŵ(x,y) = ĉ_scale · ŵ_0(x,y).   If D_tar^lab unavailable: set ĉ_scale=1, FLAG combined-shift
                      certificate as "label-shift-corollary only" (§3.6) — do not certify the combined case.

  4. OOD.            Fit o(x) on spectral-norm features.  On O, set t_ood = smallest cutoff whose (1−δ_ood) WSR UCB
                      on the leakage rate ≤ α_ood  (monotone/fixed-sequence scan — anytime-valid, no extra multiplicity).

  5. RISK GRID (single LTT family).  On accepted+in-scope cal pts {g=1, o≤t_ood, ŵ_cov≤w_max}:
                      R̂_w(τ,λ) = Σ ŵ·ℓ(λ) / Σ ŵ   (self-normalized / Hájek).
                      GATE on n_eff (Kish) ≥ n_eff,min  — else DECLARE "no certificate" (do NOT emit a vacuous bound).
                      Search (τ,λ,t_ood) over a FIXED grid G under ONE LTT/FWER family at δ_acc, each config supplying a
                      betting e-value for the risk null AND the SAME t_ood feeding the α_ood leakage cert;
                      λ̂ = inf{ λ : UCB_{δ_acc}(λ′) < α_acc  ∀ λ′ ≥ λ }.

  6. RECORD  (τ̂, λ̂, t̂_ood, w_max, ĉ, [w_lab,min,w_lab,max], α_acc, α_ood, δ_acc, δ_ood, realized n_eff).

INFERENCE on target x:
  if UCB[ŵ_cov(x)] > w_max  or  o(x) > t̂_ood :  abstain (far-OOD / unbounded weight)     # (4)
  elif u(x) > τ̂ :                               abstain (defer)                            # (1)
  else :                                         answer ŷ(x);  risk ≤ α + ε  per §3.2 box  # (2)+(3a)+(3b)
```

---

## 3.5 Backbone decision

- **Weighted RCPS** for the deployed `(α,δ)` PAC claim, with the **WSR / hedged-capital UCB** (`gostevehoward/confseq`, `src/confseq/betting.py`) instead of Hoeffding–Bentkus — tighter and variance-adaptive, which matters *more* here because importance weights inflate the risk-estimate variance and deflate `n_eff` (HB's range-only width `W·√(log(1/δ)/2n)` is far too loose). WSR validity holds on the **rescaled bounded** variable `Z_i = ŵ_iℓ_i/W ∈ [0,1]` **conditional on the weight-fold** — the one piece that survived red-team untouched: given weights fit on disjoint folds, the `Z_i` are i.i.d.-bounded and the hedged-capital process is a nonnegative martingale (Ville). **Boundedness is secured only by routing** — retain `w>W` and no betting/Hoeffding bound holds.
- **Self-normalized (Hájek) risk** so the unknown `E[w]=1` cancels; certify by a **direct betting bound on the ratio** (preferred) or paired numerator-UCB / denominator-LCB with a `δ_acc/2` split — valid **only when the `(1−δ_acc/2)` LCB on the accepted weight-mass is `> 0`**; below that, declare no certificate.
- **LTT** wraps the joint `(τ,λ,t_ood)` search: a single RCPS bound is invalid under multiple looks. Each grid config supplies a valid betting e-value; FWER control over the **fixed** grid gives simultaneous validity (cost `+log|G|` in the constants). The **same** `t_ood` feeds both the risk and the leakage certificate (so the two budgets union cleanly).
- **CRC** kept only as the less-conservative comparison row (controls the *expectation*, not the PAC tail). Refs: RCPS `aangelopoulos/rcps` (`core/` bounds + `λ̂`), WSR `gostevehoward/confseq`, LTT `aangelopoulos/ltt`.

---

## 3.6 Theorem + formal statement

**Theorem (selective + weighted RCPS under bounded post-routing weight, integrated OOD budget).** Fix `α=α_acc+α_ood`, `δ=δ_acc+δ_ood`; assume (A1)–(A5). Let `A` be the observable in-scope gate and `I={p_S>0}`. On `A∩I` the estimated, simplex-floored, clipped combined weight `ŵ(x,y)=ŵ_lab(y)ŵ_cov(x)/Ẑ(x)` is bounded `0 ≤ ŵ ≤ W` with the **sharp cap**
```
W = w_max · ( max_y ŵ_lab(y) / min_y ŵ_lab(y) ),
```
tighter than the naïve `w_max·max_y ŵ_lab` because `Ẑ(x)=E_{p_S(·∣x)}[ŵ_lab] ≥ min_y ŵ_lab(y)`. With `D_cal` (`n` labeled, disjoint from the BBSE source fold) and a fixed grid `G` of `(τ,λ,t_ood)`, choose `(τ̂,λ̂,t̂_ood)` by the single-family LTT/RCPS rule. Then **marginally over the disjoint calibration draws** (weights treated as fixed functions of their own folds),
```
P( R̂_w^{in-scope}(τ̂,λ̂) ≤ α_acc ) ≥ 1 − δ_acc,
```
and composed with the OOD-leakage certificate `P(L̂_O(t̂_ood) ≤ α_ood) ≥ 1−δ_ood` via union bound,
```
P( R̂_w^{in-scope} + L̂_O ≤ α ) ≥ 1 − δ.
```
The gap to the **true** `R_T^accept` is the deterministic `ε_w + ε_BBSE + ε_M + ε_det` of the §3.2 box.

**Explicit constants.**
- **(C1) Sharp weight cap** `W` as above; `ŵℓ ∈ [0,W]`; rescale `Z_i=ŵ_iℓ_i/W ∈ [0,1]`.
- **(C2) WSR / hedged-capital UCB** (no closed form): capital `K_t(m)=Π_{i≤t}(1+λ_i^{bet}(h_i−m))` with a predictable bet (truncated GROW / aGRAPA, `λ_i^{bet}=clip((μ̂_{i−1}−m)/(σ̂²_{i−1}+(m−μ̂_{i−1})²), 0, ½/m)`); `U=inf{m: max_t K_t(m)≥1/δ_acc}` by Ville. Variance-adaptive width `≈ √(2σ̂_w² log(1/δ_acc)/n_eff) + (W/3)·log(1/δ_acc)/n_eff` — scales with the realized weighted-loss **variance** `σ̂_w²`, not HB's range.
- **(C3) Effective sample size (Kish)** `n_eff = (Σ_{A∩I} ŵ_i)² / Σ_{A∩I} ŵ_i² ≤ |A∩I|`; worst case `≈|A∩I|/W`. **Certification precondition:** require `n_eff ≥ n_eff,min(W,α_acc,δ_acc)`; below it, abstain from certifying (report `n_eff`).
- **(C4) Self-normalization** ratio bias `O_p(1/n_eff)`; rigorous via a direct betting bound on the Hájek ratio (or `δ_acc/2` numerator-UCB / denominator-LCB split, requiring `LCB_den > 0`).
- **(C5) LTT union factor** `log(1/δ_acc) → log(|G|/δ_acc)` (Bonferroni `δ_acc/|G|`, or sharper graphical / fixed-sequence) — an added `(W/n_eff)·log|G|`.
- **(C6) OOD-leakage UCB** WSR on the Bernoulli leakage indicator over `m_ood` disjoint exposure points, range `[0,1]`, `UCB_{δ_ood} ≤ α_ood`.
- **(C7) Budget** `δ=δ_acc+δ_ood`, `α=α_acc+α_ood` by union bound.

**Proof sketch.** (i) *Transport on `A∩I`:* by the master identity restricted to `A∩I` (where `p_S>0`, `w≤W`), `R_T^{in-scope}(λ)=E_{P_S}[wℓ(λ)1_{A∩I}]/E_{P_S}[w1_{A∩I}]` — a ratio of bounded source means (the unknown `E[w]=1` cancels in the Hájek form). (ii) *WSR validity:* `Z_i=ŵ_iℓ_i/W∈[0,1]` i.i.d. given the fixed weight folds; the hedged-capital process is a nonnegative martingale under `H0: mean=m`, so Ville gives an anytime-valid `(1−δ_acc)` UCB on `E[Z]`; `×W`. Same for the denominator → an LCB. (iii) *RCPS selection:* the nested family makes `λ↦R(λ)` monotone; `λ̂=inf{λ:U(λ′)<α_acc ∀λ′≥λ}` and `{R(λ̂)>α_acc}⊆{U fails at the truth}`, prob `≤δ_acc`. (iv) *Transport back:* `R(λ̂)=R_T^{in-scope}(λ̂)` by (i). (v) *LTT:* each config's e-value `e_c=1/K_t`; FWER over `G` ⇒ every selected config meets its bound simultaneously at `1−δ_acc`. (vi) *Integrate:* union with the OOD-leakage certificate (`+δ_ood`, `+α_ood`). The only inexactness is `ŵ≠w`, carried as the ε's.

**Marginal over what.** PAC over the random draw of the disjoint calibration splits (`D_cal`, the BBSE source/target folds, `D_tar` for `ŵ_cov`/`t_ood`, and `O`); **conditional on the frozen `f`** (validity is base-model-agnostic — `f` moves efficiency, not validity) and on the realized estimators as deterministic functions of their own folds. **Not** per-test-point / conditional coverage, **not** an average over retraining `f`.

**Finite-sample vs asymptotic.** *Finite-sample exact* (given true bounded weights): the WSR UCB + Ville, the RCPS `λ̂`, the LTT FWER, and the OOD-leakage UCB — all non-asymptotic with constants (C1)–(C7). *Approximate*: the weights are **estimated**, so the certificate is finite-sample-exact for the **plug-in** weighted risk; the gap to the true target risk is `ε_w+ε_BBSE+ε_M`, consistency-based (`→0` as `m,n→∞` with `κ(C_S)` bounded) **except** the fixed softmax-calibration part of `Ẑ`, which does not vanish. Honest one-liner: *finite-sample exact for the bounded-weighted problem it solves; approximately valid for the true target risk up to a characterized weight + BBSE + misspecification + detector bias.*

**Approximation error (coupled, not merely additive).** Writing `ŵ=w+Δw` on `A∩I`, the cleanest bound is the single `L1` weight-perturbation
```
|R_T^accept − R̂_target|  ≤  E_{P_S}[ |ŵ − w| · 1_{A∩I} ]  +  ε_det,
```
which to first order separates as `ε_w + ε_BBSE` but **couples through `Z`** (shared denominator): the cross term carries a `1/min_x Ẑ(x)` amplification, largest exactly in the rare-class / heavy-up-weight regime — so the additive split is a *small-ε* approximation, not generically tight. Components:
- **`ε_w`** (covariate) — discriminator miscalibration of `d`, base-rate `ĉ`, clipping bias at `w_max` (benign: routed, charged to `α_ood`), and the `Ẑ` softmax plug-in (a **fixed, non-`√m`-vanishing** denominator bias, amplified on high-weight classes). `ε_w ≤ W·E_{P_S}[|Δw_cov|/w_cov + |ΔẐ|/Z]`; with discriminator excess risk `ε_d`, `‖ŵ_cov−w_cov‖_{L2}=O(√ε_d)`.
- **`ε_BBSE`** (label) — `‖p̂_T−p_T‖ ≲ (κ(C_S)/σ_min(C_S))·O_p(√(K/min(n,m)))·(1/min_y p_S(y))`; blows up as `C_S→singular`; `→0` with `κ(C_S)` bounded.
- **`ε_M`** (shift-model misspecification) — under FJS (§3.3), `ε_M` is the **measured** residual between the FJS plug-in risk and ground truth on `D_tar^lab`, reported with its own slice-size variance (not assumed zero, not unbounded) — a strict improvement over the rejected class-homogeneous model M, where `ε_M` was unmeasurable in principle. Still **not** certifiable to arbitrary precision without a *larger* labeled target sample; treat as a reported point estimate with CI, not a PAC bound. **Zero under pure label shift** (no `D_tar^lab` needed there).
- **`ε_det`** (detector) — the **only** estimation error *budgeted* (`=α_ood`) rather than added; residual proxy/transfer mismatch under (A3, A3′).

**Label-shift corollary (the primary certified claim).** Set `w_cov≡1` (no covariate shift) under the anti-causal assumption (`p(x∣y)` invariant). Then `Z(x)=1`, the combined weight collapses **exactly** to `w(x,y)=w_lab(y)=p̂_T(y)/p_S(y)`, the covariate cap is inert (`W=max_y ŵ_lab/min_y ŵ_lab`, no `w_max` factor), and **`ε_w=ε_M=0`**:
```
P( R_T^accept(λ̂) ≤ α_acc + α_ood ) ≥ 1 − δ,      deployment gap only ε_BBSE  (+ ε_det inside α_ood).
```
The cleanest, fully-identifiable instance — `w_lab` identifiable whenever `Ĉ_S` is invertible, **one** untestable assumption (anti-causal) not two — and the **recommended primary validation/deployment regime**. *Honest caveat:* anti-causal `p(x∣y)`-invariance still fails under genuine cross-scanner appearance change; report the BBSE consistency diagnostic (step 2). The combined covariate+label case is the stress extension whose extra slack is `ε_w` and extra assumption/residual is class-homogeneity / `ε_M`.

---

## 3.7 Red-team (Appendix D-3 results)

The D-3 method red-team was run as four independent adversarial lenses — conformal-theorist (RCPS/LTT/WSR validity), shift-identifiability, OOD-budget leakage, skeptical Reviewer-2 — against the synthesized §3.2–§3.6. Consensus verdict: **the machinery is valid; the *bare* `≤ α` presentation overclaimed.** The two weakest claims are exactly the playbook's predictions. Findings, with the fix **already folded into §3.2–§3.6**:

| # | Weakest claim (attacked first) | Verdict | Fix adopted |
|---|---|---|---|
| 1 | Boxed `R_T^accept ≤ α` for the **true** target risk | leaks (fatal) | Box restated for the **plug-in** risk; biases `ε_w+ε_BBSE+ε_M` named **outside** the δ-envelope (§3.2). |
| 2 | `α_ood` "bounds residual far-OOD risk" in deployment | holds-w/-assumption (major) | `α_ood` bounds the **audited** miss-rate on `O`; deployment transfer via **named (A3)**; residual `ε_det`. |
| 3 | far-OOD `:= {p_S=0}` (support-theoretic) | leaks (fatal) | Redefined **operationally**: route on `{UCB[ŵ_cov]>w_max}∪{o>t_ood}`; near-OOD blow-up now inside the α_ood-charged routed mass (§3.3). |
| 4 | Combined weight identifiable under class-homogeneity | leaks (fatal) | Class-homogeneous model M **replaced by FJS** (independence of the two shift mechanisms, Tasche 2022/2026) plus a **small labeled target slice `D_tar^lab`** anchoring FJS's scale constant; SJS considered and rejected (its sparsity premise doesn't fit dense pixel-space covariate shift). Pure-label corollary remains primary-certified; FJS-combined is secondary-certified (conditional on the slice) with **measured** `ε_w`, `ε_M` (§3.1, §3.3, §3.6). |
| 5 | OOD audit denominator | leaks (major) | Audit gives an OOD-*conditional* miss-rate; answered-population fraction needs **(A3′)** on prevalence; residual named (§3.2). |
| 6 | Two budgets union-bounded with a **shared** `t_ood` | invalid (major) | **Single LTT family** over `(τ,λ,t_ood)`, the *same* `t_ood` feeding both certificates (§3.4, §3.5). |
| 7 | "Sharp" cap `W` with `min_y ŵ_lab` in the denominator | leaks (major) | **Simplex-project / floor** `ŵ_lab ≥ w_lab,min>0` before forming `Z`,`W` (§3.4); else ill-conditioned `Ĉ_S` breaks the `Z_i∈[0,1]` rescaling. |
| 8 | Softmax normalizer `Ẑ` "a small bias" | leaks (major) | `Ẑ` is a **non-vanishing denominator** bias amplified on high-weight classes; **recalibrate `f`** (temperature/Platt) on a held-out source fold; reported inside `ε_w` (§3.4, §3.6). |
| 9 | WSR validity on weighted loss | **holds** (couldn't break) | Prose tightened to "conditional on the weight-fold"; **BBSE source-confusion fold added to disjointness** (§3.1). |
| 10 | `n_eff` collapse → vacuous-but-valid bound | holds (minor) | **Report `n_eff`**; `n_eff ≥ n_eff,min` promoted to a theorem precondition (§3.6 C3). |

**Net.** The integrated guarantee is defensible as *"approximately valid given stated assumptions,"* and the honest headline — *"survives covariate+label shift up to a characterized weight+BBSE+misspecification bias, with OOD error reduced to one budget line plus one named residual under a stated exposure model"* — is still strictly beyond every audited competitor. Two items remain genuine **limitations** (for §7.6 "what remains open," not bugs): `ε_M` is **uncertifiable** without labeled target data (hence the label-shift corollary is the primary certified claim), and `ε_det` rests on the **untestable (A3)** (hence `O` must be swappable and its hardness reported). These are the open edges to name honestly in the discussion.
