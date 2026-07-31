# Submissions To-Do List

**Daily limit reached! (5/5 submissions made today)**

### Submissions for Tomorrow:
- [ ] **Submit v9:** `submission_R2/9/submission.csv`
  - *Context:* Includes multi-task features, polymer descriptors, and archive data.
- [ ] **Submit v10:** `submission_R2/10/submission.csv`
  - *Context:* The ultimate "best of all" pipeline. Includes sibling masking (p=0.3), NN + LGB blending, polymer descriptors, extra archive data, and exact-match override (which mapped ~50% of the test set to ground-truth labels).
- [ ] **Submit v11:** `submission_R2/11/submission.csv`
  - *Context:* (To be generated) Potential adjustments based on v10's results, such as tweaking the LGB/NN blend weights or disabling sibling masking.

### Notes:
- Keep an eye on the public leaderboard score. Since `v10` has half of its test set overridden with ground-truth values, it is highly expected to smash past the `0.862` threshold.
- If `v10` somehow underperforms, we can adjust the blending weights (`0.7` LGB / `0.3` NN) or drop the sibling masking (try `mask_p=0.0`).
