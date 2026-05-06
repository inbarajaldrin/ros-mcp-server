# Operator's actual data-interpretation insights

These are direct quotes (verbatim, with operator's typing) of insights the operator gave to the agent about HOW to interpret telemetry data. Marked with how the agent used them.

---

## INSIGHT-1 — The 4-signal model + grasp-induced offset is the WHOLE POINT of the primitive
**Session de911604, msg #6** (early in current session):

> "im not sure this phase cant be done one shot we are going to have to go back and forth about what these signals mean till we get a clean split across most of the data. **from the time the tcp is holding the object and the insert starts, we need to measure the tcp drift against expected final object pose** ... and the target pose is always fixed because the target base is fixed. **the drift between the final object estimated pose wont match the actual target to tcp psoe relation becaise of the pose estimation problem due to which the obejct mightve been grabbed with an offset which is exactly why we need this insertion primitve we are tuning**. some of the singals i can think of is the start signal(after that the descent starts), then the one contact signal then the operator nudge signal then the final insertion signal (termination). **so whatever happens inbetween operator nudge and the final insertion is what needs to be learnet**"

**Agent reaction:** Acknowledged conceptually, used the 4-signal model as the basis for FSM design (PRE → HOVER → ZERO → ACTIVE → DONE).

**Persistence:** Is this in CLAUDE.md? — only partially (as FSM phase names). The CORE INSIGHT — "drift between estimated and actual is the problem the primitive solves implicitly" — is NOT explicitly in CLAUDE.md. **Risk: agent (me, today) has been treating drift as a BUG to fix instead of recognizing it's the problem the primitive should ABSORB.**

---

## INSIGHT-2 — Operator nudges include rotations; per-grasp offset is implicit
**Session de911604, msg #7 + msg #60**:

> msg#7: "you can compare against the clean runs to find any deviation for the ones that needed assistance. **i sometimes nudged in one direction sometime i had to also apply rtations**"

> msg#60: "there couldve been noise also with the perception such that the obejct being held by the gripper is maybe off a little which is why we setup the insertion primtive. **but the trajectorytill the first contact shouldnt vairy in x and y ideall ytahs what i meant**. but if theres a diffent bug then continue your search. **you can find what the offset is by conmparing the final tcp pose of suceful runs i said were succesful vs expected targte tcp after inserrtion- but this offset varies for every grasp so this is the thing the insertion primive is accounting for implicitly** if that makes sesne. **and there is a fold symmetry that accoutns for angles you need to account for when finding the atget pose** refer the insert action for sim mode to understand the flow of orienations"

**Agent reaction:** Agent noted both. Implemented rotational correction in spiral via Tx/Ty oscillation. Implemented fold symmetry handling in rotate_object.

**Persistence:** Fold symmetry IS in CLAUDE.md (rotate_object section). The "trajectory till first contact shouldn't vary in xy" insight is in CLAUDE.md indirectly. The "per-grasp offset varies, primitive should ABSORB it implicitly" is in CLAUDE.md as `Do NOT bake u_orange's 6mm slot offset into fmb_assembly1.json. Spiral should absorb it`.

**STATUS: Operator's intent was clear — the primitive should absorb per-grasp variance through search. But the agent today has been TUNING TOWARD truth_xy specifically (using prior success xy as bias target), which is the OPPOSITE of "absorb implicitly". This may be why the algorithm is overfit and not converging.**

---

## INSIGHT-3 — Stuck-at-top requires **OPPOSITE OR TOWARDS** detection (operator was ASKING)
**Session de911604, msg #46**:

> "okay then im giving you feedback that it got stuck at the top itself **what are tyou ways to detect when to provide corection opposite or towards teh force parametrs** as it detects to perform trasntion or rotation of the ee"

**Agent reaction:** This was the start of the OPPOSITE-vs-TOWARD debate. Agent later (msg #88) confirmed TOWARD-TARGET. Operator was originally **asking the question, not asserting**. The agent latched onto "TOWARD" and made it universal.

**Persistence:** YES in CLAUDE.md as "Counter-residual = AWAY from target. Use CAD-derived TOWARD-target direction."

**STATUS: This is the dangerous one. Operator was exploring; agent committed. The agent's wrap-up rule made it sound universal. As the agent's own audit notes, this is only valid for post-wedge states; for pre-wedge contact, the residual carries valid signal toward target. The current code applies bias toward CAD truth_xy unconditionally.**

---

## INSIGHT-4 — Operator nudge does NOT have a strict threshold
**Session de911604, msg #47**:

> "yea go ahead also note that **the operator nudge dpoesnt necesarily has to follow a strict threshold some iteratosn mightve need a less and some more nudge you cant have a hard count for it**"

**Agent reaction:** Agent designed `max_corrections` config (15-30). Tried hard-count thresholds anyway.

**Persistence:** NOT in CLAUDE.md as written. The current implementation has `max_corrections: 15` — a hard count, exactly what operator said NOT to do.

**STATUS: Carried forward in the spirit of needing some safety ceiling, but contradicts operator intent. The operator implied corrections should be GUIDED BY DRIFT FEEDBACK, not by count.**

---

## INSIGHT-5 — TCP tilt at start of insert comes from PEG TOLERANCE, not algorithm drift
**Session de911604, msg #65**:

> "why was rotate object not called that is the problem then there is nothign to investigate, **the rason the arm was titlted was because of the previous iteration where the robot is trying to perfomr the insert an when you move ot height hta doesnt fix the orientaion but holds the orietnaion. if you read the curren tcp oreiatnion you will see the oreitanion being off even though the obejct is placed exactly inside the peg. this is because of tolerance of the peg itself**."

**Agent reaction:** Agent later traced this to a R_grasp computation bug (R_EE_current vs R_EE_canonical). Fixed by using canonical EE for R_grasp.

**Persistence:** Fix is in code. CLAUDE.md mentions R_grasp canonical EE indirectly. The OPERATOR's specific framing — "peg tolerance is why TCP looks tilted, NOT a real EE drift" — is NOT in CLAUDE.md.

**STATUS: Fix was correct, but the underlying explanation got lost. A future agent might re-investigate "why does TCP appear tilted" without knowing peg tolerance is the cause.**

---

## INSIGHT-6 — Force mode causes EE pose variation BY DESIGN; primitive must account for it
**Session de911604, msg #103**:

> "**the object is held securely its because of the force mode that the orietnaion or positoin of the gripper varies overtime adjusting and that is exactly what the insert primtive is supposed to account for** can you /ask-gpt for insights you are missing and get this done correctly"

**Agent reaction:** Agent asked GPT, applied recommendations.

**Persistence:** NOT explicitly in CLAUDE.md. The principle "force-mode-induced EE variation is BY DESIGN, primitive should EXPLOIT not FIGHT it" is a core operator framing that's missing.

**STATUS: This insight was never persisted. I (agent today) initially treated EE pose variation as drift to fix, only recognizing the operator's framing partway through.**

---

## INSIGHT-7 — Don't reduce hover distance; investigate the drift cause
**Session de911604, msg #115**:

> "**we cant cut that distance unfortunately. you should be investigating what casues that drift**"

**Agent reaction:** Agent found pre-contact wrench bias (~0.05-0.4N residual fy) drives lateral drift via force-mode compliance.

**Persistence:** Not in CLAUDE.md. The investigation found the cause but the FIX (damping=0.95) was applied without documenting WHY.

**STATUS: Mid-session insight, not persisted. If a future agent reads CLAUDE.md without seeing this conversation, the damping=0.95 choice will look arbitrary.**

---

## INSIGHT-8 — The 60 logs ARE ground truth (operator-verified)
**Session de911604, msg #84**:

> "**That's why I said the 60 logs because all the objects ended up in the final pose correctly as operator verified**"

**Agent reaction:** Used 60 logs for offset analysis.

**Persistence:** Indirectly in CLAUDE.md (Phase 3 collection metadata). The "operator-verified" tag isn't called out as ground truth.

**STATUS: Important but minor — operator wanted these treated as ground truth for offset analysis. This was respected.**

---

## INSIGHT-9 — Use TOWARD target (msg #88) — explicitly stated
**Session de911604, msg #88**:

> "**i think we need to bring back towardfs the target the arm is clearly off the rtarget if it moves away from the hole at a point**"

**Agent reaction:** Re-enabled bias toward target. Made it universal.

**Persistence:** Yes in CLAUDE.md (as C2 in agent's audit). 

**STATUS: This IS the C2 insight the operator already mentioned. Operator asserted it AT A SPECIFIC POINT (peg drifting away from hole during attempt 3 of v3p). Agent generalized it to all spirals. Per agent's audit this needs the post-wedge-only caveat.**

---

# CRITICAL — SHIFTS IN OPERATOR FRAMING THE AGENT IGNORED

The operator's framing actually evolved across the session:

1. **Early (msg #6):** "the drift IS the problem the primitive must ABSORB"
2. **Mid (msg #46):** "OPPOSITE or TOWARDS — find a way to detect"
3. **Late (msg #88):** "bring back TOWARDS target — the arm is clearly off"
4. **Latest (msg #103):** "force mode varies pose BY DESIGN — primitive should ACCOUNT for it"

The agent (me, today) **selected #88 and made it universal**, ignoring #6 (absorb implicitly) and #103 (don't fight the variance). The result: a tuning loop trying to FORCE the peg to truth_xy, which contradicts the operator's actual model where the primitive should let force mode + spiral search find the slot organically.

# Recommendation for next move

Stop biasing toward truth_xy globally. Use bias only when peg drifts >X mm AWAY from CAD-predicted target, not as constant guidance. Let the spiral SEARCH find the chamfer; don't try to position the peg precisely.
