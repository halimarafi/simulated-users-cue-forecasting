---
title: "Analysis specification"
---

*Draft 4. Companion to the README. It says exactly what will be run, in what order, and how each result will be judged. Items marked **[confirm]** need a detail from the Brands et al. paper.*

## 1. Overview

There are two analyses.

- **Analysis 1, fidelity.** Four kinds of simulated user learn from basic trials and forecast cued trials. Each is scored on three checks against real behaviour and against a human ceiling.
- **Analysis 2, who responds.** Each person's traits are placed on a map. We test whether the map predicts how strongly that person responds to the cue.

Rule that holds throughout: **a simulated user never learns from the cued trials of the person it copies.**

## Order of work, and where the attention model comes in

1. **Prepare the data** (section 2).
2. **Fit the four models to basic trials** (section 4). The hidden Markov model is the last part of rung 3, so it is fitted here, alongside the others.
3. **Check each model can be trusted** (section 5).
4. **Build copies and make the forecasts** (sections 3 and 6).
5. **Analysis 1** (section 7). The attention model is tested in the "whole session" check: does adding it make simulated sessions unfold more like real ones?
6. **Analysis 2** (section 9). The attention model contributes one extra trait to the map: how much of the session each person spends engaged. If it failed its checks in step 3, the map is drawn without that trait.

## 2. Data preparation

| Step | What is done |
| --- | --- |
| Load | `dd_rawdata.txt`: 42 people, 2 sessions, 16,128 trials. |
| Trial order | Rows are treated as the order the trials were shown, within each person and session. |
| Fast responses | Drop trials faster than 0.25 s (9 trials, from 3 people). Nobody can read an offer and choose that quickly, so these are not real decisions. |
| Slow responses | Drop trials slower than 10 s (124 trials, 0.8%). Most responses take 1 to 3 s, so a response over 10 s reflects mind-wandering or an interruption rather than the decision process being modelled. Lapses shorter than 10 s are kept, and rung 3 models them. |
| One unusual person | Participant 25 is much slower than everyone else (typical response 4 s against 1.5 s) and loses 47 trials to the 10 s rule. They stay in. Every main result is also reported without them. |
| Conditions | N = basic, F = future cue, P = past cue. F and P are pooled as "cued" in the main analysis and split in a secondary one. |
| Sessions | Each session is modelled as its own sequence. A person's traits are estimated per session, tied together by a person-level average. |

Exclusions are fixed before fitting and applied the same way to real and simulated data.

## 3. Models and simulated agents

The four rungs are four **models**. Each model is fitted to every real person's basic trials, which gives each person a set of traits under that model. Traits are never mixed between models.

**Simulated agents** are built from those models downstream. There are two kinds:

| Kind | Where its traits come from | Used for |
| --- | --- | --- |
| **Copies** | One per real person, per model: that person's estimated traits. Each copy does exactly the trials its real person saw. | Check 2, the forecasts, and Analysis 2 |
| **Invented agents** | Traits we choose ourselves | Check 3 (parameter recovery) only |

Population-level comparisons are made by pooling the 42 copies, so a separate simulated crowd is not needed. (Draft 2 of this plan also had a crowd of 1,000 new agents. It was dropped to keep the project small.)

## 4. The four simulated users

All four are fitted to basic trials only, with people nested in a group (hierarchical Bayesian fitting in Stan).

**Rung 0: simple statistics.**
Chance of choosing "later" = logistic function of (later amount, delay), with person-level intercepts and slopes. It has no model of time taken, so its response times are drawn at random from that person's basic-trial times.

**Rung 1: patience model.**
Value of the later reward = amount / (1 + k × delay). `k` is the person's discount rate, estimated as log(k). Choices follow a softmax rule with a consistency parameter. Response times are handled as in rung 0.

**Rung 2: patience plus decision time.**
The drift-diffusion model published by Brands et al. Evidence builds towards "later" or "now" at a speed that depends on the value difference between the options, through an S-shaped function. Per-person traits:

- log(k): patience
- boundary: how much evidence is needed before deciding (caution)
- drift scaling: how strongly value differences drive the decision (two parameters)
- non-decision time: time for seeing and pressing
- starting bias: a lean towards "later" or "now" before evidence arrives

We adopt this model on the paper's authority and do not re-run their model comparison. **[confirm exact drift formula and priors]**

**Rung 3: patience plus decision time plus attention.** Built in two steps so each part can be judged separately.

- **3a, practice.** The boundary and non-decision time can decline smoothly across a session. This is needed because people speed up a lot: median response time falls from about 1.9 s to 1.3 s. Without it, the next step would mistake practice for attention.
- **3b, attention states (hidden Markov model).** On each trial the person is in one of two unseen states.
    - *Engaged:* behaves as rung 3a.
    - *Lapsing:* the choice does not depend on the offer (drift fixed at zero). How long a lapsed choice takes is left for the data to decide. It could be fast (guessing) or slow (mind elsewhere).
    - Per-person traits: the chance of staying engaged, and the chance of staying lapsed, from one trial to the next.
    - Cued trials sit between basic trials in the sequence. During fitting they are treated as unobserved: the state still moves through them, but they add no information.
    - The state restarts at the beginning of each session.

## 5. Checks before trusting any model

A model moves on only if it passes all three.

**Check 1: fitting worked.** Standard convergence diagnostics. Pass mark: R-hat below 1.01 and no divergent transitions.

**Check 2: parameter recovery.** This asks: when the model tells us someone is patient, can we believe it? It is like testing a kitchen scale with a weight you already know.

1. We create 42 invented agents and choose their traits ourselves. For example, invented agent 1 gets patience = 0.2 and caution = 1.5. These are the "known traits": known because we made them up.
2. Each invented agent does the task inside the computer, using the real trial lists. This gives simulated data where we know the right answer.
3. We hand the simulated data to the model without telling it the traits, and ask it to estimate them.
4. We compare the model's estimates with the numbers we chose.

Pass mark: across the invented agents, the estimated and true values should line up, with a correlation above 0.8 (1.0 would be perfect). This is checked for each trait separately. A trait that fails is not used in the trait map in Analysis 2, because we could not trust it.

**Check 3: basic-trial match.** The copies redo the basic trials they learned from. Pass mark: real choices and response times fall inside the range the model produces 95% of the time.

For rung 3b there is one more check: on simulated data with known lapses, the model must recover which trials were lapses better than chance and must not report lapses when none were simulated. **If 3b fails, it is reported as not supported by these data and 3a stands as rung 3.**

## 6. The three forecasts for cued trials

A simulated user has never seen a cue. To make it respond to one, we have to give it an instruction of the form: "when you see a cue, become this much more patient." The three forecasts differ only in **where "this much" comes from**.

- **A. No effect.** "This much" is zero. This is the comparison point: any useful forecast has to beat it.
- **B. Earlier studies.** We look the number up. The same research group ran six older experiments with different participants, and in each one the cue made people somewhat more patient. We take the typical size of that change and give it to every simulated user. Nothing from the cued trials in our dataset is used. This is like predicting how a new feature will do from how similar features did in the past.
- **C. Learned from other people.** We measure the number in part of our own dataset. Take 35 of the 42 people, see how much the cue changed them, and give that amount to the copies of the other 7. Then compare those 7 copies with the 7 real people, whose cued trials we never touched. Repeat with a different 7 each time until everyone has had a turn. This is like trying a feature on a small pilot group and using that to predict everyone else.

B is the harder and more honest test. C will usually do better, and the gap between them shows how much a pilot is worth.

| Forecast | What the simulated user is told | Applies to |
| --- | --- | --- |
| **A. No effect** | Treat cued trials like basic trials. | Rungs 0 to 3 |
| **B. Earlier studies** | The cue lowers log(k) by a fixed amount, the same for everyone. Nothing else changes. Main value −0.175, with a low case of −0.07 and a high case of −0.28. A re-analysis of six earlier datasets (Peters, 2020) reports average shifts in log(k) from −0.07 to −0.28; the main value is the midpoint of that range, because the per-dataset table could not be accessed. | Rungs 1 to 3 |
| **C. Learned from other people** | Participants are split into 6 groups of 7. For each group in turn, the cue's average effect is measured in the other 35 people and applied to the 7 left out. Rung 0: the cue shifts the intercept. Rung 1: it shifts log(k). Rungs 2 and 3: it shifts log(k) and the boundary. | Rungs 0 to 3 |

Forecast B is the main test, because it uses no cued data from this study at all. Forecast C shows how much a small pilot adds.

**What Forecast B tests.** The cue enters as a shift in one named parameter, log(k), and nothing else. The test is whether the full behavioural signature follows from that single shift: choice curves at every reward size and delay, response-time distributions, and the spread of cue effects across people. A correct shift in log(k) does not guarantee the rest. Brands et al. report that cues also raise the boundary, so Forecast B is expected to miss the slowing on cued trials. Forecast C allows both shifts, so the gap between B and C on timing measures this directly.

**Why 6 groups of 7 and not two halves of 21.** About five people respond strongly to the cue. A single 21/21 split could put most of them in one half by chance, which would make the effect look as if it does not carry over. Six rotating groups use 35 people for each estimate and predict every person once.

For Forecast C, the average cue shift in the 35 training people is estimated with each person's basic-trial traits held fixed at their estimates. Only the cued trials of those 35 people are used. This keeps it fast and keeps the 7 people being predicted completely untouched.

Future and past cues are pooled as "cued" in the forecasts.

## 7. Analysis 1: fidelity

Every measure is computed on the real cued trials and on the simulated ones, for each rung and each forecast.

**Check 1: each decision**

| Measure | In plain words |
| --- | --- |
| Choice curve error | No median split. The task already has built-in levels: 16 sizes of later reward and 6 delays per person (shortest to longest). For each size, and for each delay, compare the share choosing "later" in simulated and real data. Average the gaps. |
| Choice agreement | Share of trials where the copy and the real person made the same choice. |
| Choice prediction score | How much probability the model put on what the person actually chose (log score). |
| Timing error | Compare response-time distributions (10th, 30th, 50th, 70th, 90th percentiles), separately for "later" and "now" choices and for easy and hard offers. |
| Cue effect error | Predicted minus real change, from basic to cued trials, in the share choosing "later" and in median response time. |

**Check 2: the whole session**

| Measure | In plain words |
| --- | --- |
| Time course | Split each session into 8 blocks. Compare the simulated and real curves for response time and for share choosing "later". |
| Carry-over | Do slow trials follow slow trials? Correlation between each trial's response time and the next few, after removing the overall speed-up. |
| Streaks | Lengths of runs of very fast responses, and of choices that go against the offer's value. |

**Check 3: run to run**

Each simulation is repeated 200 times with different random seeds. For every measure, report the average and the range covering 95% of runs.

**Comparing rungs.** Differences between rungs are tested by resampling participants (bootstrap, 5,000 draws). An added layer "helps" only if the 95% interval of the improvement excludes zero.

## 8. Human reference lines

| Reference | How it is computed | Meaning |
| --- | --- | --- |
| Same-day ceiling | Split a person's basic trials into two halves within a session and compare the halves. | Best possible agreement when nothing about the person has changed. |
| Across-day ceiling | Compare a person's basic trials on day 1 with day 2. The offers differ between days, so this uses rates and fitted traits, not item-by-item matches. | How stable a person is over time. |
| Population floor | Split the 42 people into two random halves many times and compare the halves. | How different two real groups look by chance. |

Scores are reported raw and as a percentage of the relevant ceiling.

## 9. Analysis 2: who responds

**Step 1: traits.** For each person, from basic trials only: patience, caution, drift scaling, non-decision time and, if rung 3b passed, share of time engaged.

**Step 2: are they traits?** Correlate each trait between day 1 and day 2. Only traits with a correlation above 0.5 are kept as axes.

**Step 3: the map.** Plot people on the two most stable axes, with a third shown by colour. People are not forced into groups. As a check, we test whether the data support 1, 2, 3 or 4 clusters, and report the answer even if it is "one".

**Step 4: the outcome.** Each person's response to the cue, measured two ways: change in share choosing "later", and shift in log(k). The second avoids the problem that very patient people have no room to increase.

**Step 5: prediction.** Predict each person's response from their map position using a simple linear model with at most three predictors, tested on people left out of the fit (the same 6 groups as Forecast C).

**Step 6: judging it.** Report the correlation between predicted and real response in left-out people. The ceiling is how well a person's response to the future cue matches their response to the past cue, measured on different days. The comparison point is "everyone responds by the average amount".

## 10. Figures

| Figure | Shows |
| --- | --- |
| 1 | The task and the study design. |
| 2 | Headline: fidelity of each rung on each check, as a share of the human ceiling. |
| 3 | Headline figure. One point per person: predicted vs real cue effect, for choices and for timing, under forecasts A, B and C. The human ceiling is drawn as a band around the diagonal. |
| 4 | Session time course: real people vs each rung. |
| 5 | One example participant: the hidden Markov model's trial-by-trial probability of being engaged. |
| 6 | The trait map. |
| 7 | Predicted vs real response to the cue, per person. |

## 11. Code layout

Python 3.11 in Jupyter notebooks. Models in Stan through `cmdstanpy`. All random seeds fixed.

| Notebook | Content |
| --- | --- |
| `01_data.ipynb` | Loading, exclusions, descriptive plots. |
| `02_models_fit.ipynb` | Fit rungs 0 to 3 to basic trials. |
| `03_model_checks.ipynb` | Convergence, parameter recovery, basic-trial match. |
| `04_forecasts.ipynb` | Forecasts A, B and C: copies predict the cued trials. |
| `05_fidelity.ipynb` | Analysis 1 and human reference lines. |
| `06_who_responds.ipynb` | Analysis 2. |

## 12. Results of the checks so far, and known limitations

- All three fitted models (rungs 0 to 2) pass check 1 and check 3.
- Check 2 (basic-trial match) fails the strict "inside the 95% band" pass mark for all three models. The bands are very narrow (each is built from about 8,000 trials), so most misses are 2 to 4 percentage points. The verdict is kept as fixed, and the size of each gap is reported next to it.
- **Known limitation, not fixed:** for the smallest later rewards (only a little above €20), real people choose "later" about 15% of the time, while every model predicts about 30–35%. The models treat each euro as equally valuable and have no extra pull towards "now". This is reported as a limitation rather than corrected, to keep the project small.

### Changes made when building notebook 05

These simplify the plan above. They were decided before notebook 05 was run on the full data.

- **Human references.** Section 8 planned a same-day ceiling, an across-day ceiling and a population floor. Notebook 05 instead uses two references that fit the forecasts directly. (1) For each person's cue effect: its measurement error, estimated by resampling their trials. From this come the best possible person-by-person correlation (the square root of the reliability) and the noise band in the headline figure. (2) For the group's choice curves: how far two equally valid samples of 42 people would differ, estimated by resampling people. The day 1 vs day 2 consistency is used in Analysis 2 instead.
- **Response times** are compared for "later" and "now" choices, without a further split into easy and hard offers.
- **Whole-session check:** the time course and carry-over are included. Streaks of fast responses are left for rung 3.
- **Comparing models:** the bootstrap over people is applied to the prediction score and to each person's cue-effect error.

### Changes made when building notebook 06

Decided before notebook 06 was run on the full data.

- **Cue response is measured two ways:** the change in "later" choices (simple, but limited by a ceiling for very patient people), and each person-day's shift in log(k) with their other basic-trial traits held fixed (rung 1's patience model, one shift per person-day, tied together by a group average and spread).
- **Stability** is the share of each trait's variation that is between people rather than between days, taken from notebook 02's model (sigma_p and sigma_s). A plain day 1 vs day 2 correlation of the estimates overstates stability, because the model ties a person's two days together.
- **Predictors are fixed in advance:** patience, caution and non-decision time. These recovered best in check 3, and each has a reason to matter for this cue. A predictor is dropped if its stability is below 0.5.
- **Strong responders:** a cue effect above the notebook 05 noise band (1.96 times the typical measurement error of one person's cue effect).
- **Added check:** whether the basic-trial share of "later" choices alone predicts the cue response, to detect a ceiling effect.
- **Clusters:** Gaussian mixture models with 1 to 4 groups, compared by BIC.

## 13. Open items

1. Brands et al.'s exact drift formula and priors. The model used here follows the S-shaped drift of Peters & D'Esposito (2020).
2. Forecast B's main value is the midpoint of the reported range (−0.175). If Table 3 of Peters (2020) becomes available, replace it with the reported average before notebook 04 is run for real.
3. **Decision after notebook 06: rung 3 (practice and attention states) is not built in this version.** Notebooks 04 to 06 used rungs 0 to 2. The notebook 05 results (the session speed-up, carry-over between trials and the slow tail of response times) set out what rung 3 would need to fix. Its design is described in the README under "What I would do next".
