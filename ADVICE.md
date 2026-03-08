# Advice On Using "Crazy" Target-Xent Curves In Search

## Short answer

Yes: the right object is **not** the average target cross-entropy, and not even "is it high right now?".
The right object is a **pathwise rare-event statistic**: repeated unusually high target-xent events, plus the size of the upward moves that create them.

If you want to improve on pure projlen while still using the model, the best practical direction is:

1. Keep a strong pure-projlen lane so you do not lose the algebraic signal.
2. Add a second lane that explicitly rewards **rare spike evidence**.
3. Rank prefixes by a **posterior / evidence score built from spike-process features**, not by avg5 alone.

## What the saved data says

I compared the 6 saved kernel-hit curves against the 5 saved `random_*` curves used for the overlay.

In this tiny sample, the following statistics perfectly separated kernels from randoms:

- curve mean
- curve max
- top-3 average
- 90th percentile
- total positive jump mass
- largest upward jump
- number of prefixes above fixed high thresholds

For these, the separation was:

- AUC = `1.0`
- exact one-sided permutation `p = 0.0022`

What **did not** separate them:

- `max(curve) - median(curve)` within each curve

That had:

- AUC = `0.5`
- no meaningful significance

### Interpretation

This matters.

The useful signal is **not** "one isolated spike relative to that curve's own background."
The useful signal is closer to:

- the curve spends more time in the extreme regime
- the curve has more upward shock mass
- the upper tail is consistently heavier

So if you optimize only `avg5`, you are still washing out part of the signal.
But if you optimize pure `max`, you will overreact to one noisy excursion.

The correct middle ground is an **extreme-value / exceedance score over the path so far**.

## The core statistical point

You cannot select for a **future** spike directly.
What you can do is maintain, for each prefix, the evidence:

`history up to time t -> how likely is this prefix to eventually lie on a kernel trajectory?`

That is a sequential inference problem, not a plain averaging problem.

So the search objective should be something like:

`good prefix = low projlen + unusually spike-like target-xent history`

not:

`good prefix = low projlen + high running average`

## Best practical approach

## 1. Use level-normalized surprise, not raw xent

At level `t`, raw target-xent has a background distribution across all children.
Normalize each child's current xent against that level:

- z-score
- robust z-score using median/MAD
- or empirical quantile / percentile

Call this normalized surprise `z_t`.

This is better than raw xent because it measures:

`how rare is this child at this level among its peers?`

That is the quantity the reservoir should care about.

## 2. Track spike-process state for each prefix

For each prefix, maintain online state such as:

- `M_t`: running max of `z_t`
- `E_t`: cumulative exceedance mass `sum max(0, z_t - tau)`
- `C_t`: count of exceedances `sum 1[z_t > tau]`
- `J_t`: cumulative positive jump mass `sum max(0, z_t - z_{t-1})`
- `Jmax_t`: largest upward jump so far
- `R_t`: recency-weighted exceedance mass

This is exactly the type of information your current `avg5` state is discarding.

## 3. Score by spike evidence, not by avg5

The best immediate score is something like:

`spike_score_t = w1 * M_t + w2 * E_t + w3 * C_t + w4 * J_t + w5 * z_t`

with `tau` chosen as a high level-relative threshold, for example:

- top 5% at that level
- or median + `2 * MAD`

Then use:

`total_score = projlen_penalty - lambda * spike_score_t`

with the search engine's lower-is-better convention.

### Why this is better

- `M_t` captures the presence of extreme events
- `E_t` and `C_t` capture repeated extreme behavior
- `J_t` captures the "crazy curve" upward bursts you care about
- `z_t` keeps some responsiveness to the current step

This targets exactly the phenomenon you described.

## 4. Do not replace pure projlen entirely

Pure projlen is still the strongest structural prior.
The safest design is a **two-lane reservoir**:

- lane A: pure projlen
- lane B: spike-aware score

Then merge survivors from both lanes each level.

Recommended starting split:

- 70% to 85% of budget for pure projlen
- 15% to 30% for spike-aware selection

This avoids the failure mode where a noisy xent score kills the genuinely promising algebraic prefixes too early.

## 5. Promote rare events immediately

Add an event-trigger rule:

- if `z_t > tau_hi`, or
- if `J_t` jumps by more than a large threshold,

then force that prefix into an elite queue for the next round, regardless of its ordinary bucket competition.

This is the search analogue of anomaly detection.
It is exactly what you want if kernel trajectories are characterized by rare but highly informative events.

## Best algorithmic options

## Option A: Softmax / top-k history objective

This is the smallest code change.

Replace `avg5` with something like:

- mean of top 2 values in the last 8 steps
- or `logsumexp(beta * history) / beta`

This interpolates between average and max.

Pros:

- very easy to implement
- much closer to "spikes matter"

Cons:

- still does not explicitly reward repeated exceedances
- still not level-normalized

This is a good quick baseline, but not my top recommendation.

## Option B: Exceedance-process objective

Use the spike-state summary above (`M_t`, `E_t`, `C_t`, `J_t`, `R_t`) and optimize it directly.

Pros:

- matches the observed phenomenon best
- still simple enough to implement in the current search code
- should dominate avg5 if the real signal is "rare but structured bursts"

Cons:

- requires new frontier state beyond just `xent_history` / `xent_max`

This is my recommended first serious change.

## Option C: Learn `P(kernel eventually | history so far)`

This is the most principled medium-term approach.

From archived searches, create training examples:

- input: prefix history up to level `t`
- label: whether this prefix eventually leads to a saved kernel hit

Features can include:

- projlen
- current xent percentile
- running max
- exceedance count
- jump mass
- factor identity / last factor
- recent trajectory summary

Then fit:

- logistic regression
- small MLP
- isotonic / calibrated model

and use the predicted probability as the spike-aware lane score.

Pros:

- directly answers the real search question
- uses only information available up to time `t`
- naturally combines projlen and spike history

Cons:

- needs archived search traces with labels
- more engineering

This is the best long-term formulation.

## Option D: Change-point / CUSUM detector

Treat high xent as a sequential anomaly process and maintain a CUSUM-like statistic:

`S_t = max(0, S_{t-1} + z_t - k)`

or a jump-sensitive variant.

Pros:

- elegant sequential detection framework
- very compact state

Cons:

- less interpretable than exceedance summaries
- needs tuning

Worth trying as a compact baseline.

## What I would do next

## Immediate next experiment

Implement a new score type, conceptually:

- `score_type = target_xent_spike_maximize`

State per prefix:

- current normalized xent
- running max
- exceedance mass
- exceedance count
- positive jump mass

Selection:

- after bootstrap, allocate most of the budget to pure projlen
- allocate the rest to spike-aware ranking
- add automatic promotion for very rare events

## Minimal viable formula

At level `t`, with level-normalized surprise `z_t`:

`S_t = 0.35 * M_t + 0.30 * E_t + 0.20 * C_t + 0.10 * J_t + 0.05 * z_t`

Then search with:

`priority_t = projlen_t - lambda * S_t`

with `lambda` tuned so spike evidence influences ranking but does not swamp projlen.

If you want an even safer version:

- use pure projlen until level 20 or 25
- then turn on the spike-aware lane

## What not to do

- Do not use raw `max xent` alone. It will be too noise-sensitive.
- Do not use only `avg5`. It smooths away the event structure you care about.
- Do not collapse to model score only. You will likely lose the algebraic prior that pure projlen is exploiting.
- Do not use a fixed absolute threshold without level normalization. The background scale can drift with depth.

## Concrete recommendation

If the goal is to beat pure projlen by exploiting the model signal while respecting what the data actually shows, I would rank the approaches like this:

1. **Two-lane search: projlen + exceedance-process spike score**
2. **Learn a prefix-level posterior of eventual kernel hit**
3. **Softmax / top-k history objective as a quick baseline**
4. **Pure max-based objective**

The best immediate bet is clearly **not** "raise the average."
It is:

**preserve low projlen, but explicitly keep prefixes whose target-xent history shows repeated rare excursions and large upward bursts.**

That is the search rule that matches the observed kernel curves.
