# icl-w2s

### Overcomplete list of things to try

- [x] take subset of dataset, evaluate all models. narrow down to cleanest pair of models.
  - Depends on how long it takes to do a full test set eval
- [x] strong model + zero-shot baseline? This may invalidate the setup.
- [x] measure PGR on cleanest model pair
- [ ] vary few-shot prompt
- [ ] repeat exps for specific MATH splits, levels etc
- [ ] "self-distillation" - how good is the communication channel of few-shot examples?
  - If this works, do it for multiple rounds, see how far the improvement goes.
    - If it goes far, see sample complexity of effective oversight
- [ ] easy-to-hard? strong model + easy questions -> strong model + hard questions?
- [ ] zero-shot + few-shot, try both, only use disagreements
- [ ] majority voting - this would touch temperature which introduces a new hparam; not sure if I'll have time
- [ ] vs uniform label noise?