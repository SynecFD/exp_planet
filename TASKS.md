# Tasks

- [x] NIKLAS: Planner bauen
- [x] NIKLAS: RSSM anpassen, damit Multi-GPU Packing möglich ist und korrekt unpacked wird.
- [x] RSSM + MaskedLoss Function schreiben, sodass RSSM Padding/Padding nicht für backward-Path berücksichtig wird.
- [x] Population based Optimizer schreiben (selber wie David Ha?)
- [ ] Previous Action im Planner auf echte previous action der observation setzen, falls Observation nicht t=0 ist?
- [ ] LightningModule und DataModule schreiben
- [x] Args/Config-DataClass oder entsprechendes Click/Argparse dict einbauen
- [ ] Korrekte Prior/Posterior Belief Terminologie
- [ ] one-off errors mit den Time-Step inputs für die Nets korrigieren
- [ ] belief / recurrent_hidden zwischen batches resetten? Kai-Issue in Github
- [ ] testing implementieren und als reward loggen
- [ ] readme schreiben, wie man startet, envs auswählt und Tensorboard startet
- [ ] env.close für renders in den reset-calls für gym-classic envs implementieren (evtl. gar nicht render callen, weil
  das RGB array immer dasselbe ist oder sonst neuen Wrapper für classic-gyms schreiben, wo env.close nach env.render("
  rgb_array) gerufen wird)

#### Optional:

- [ ] Sämtliche Anregungen aus den 3 Artikeln von Reddit zu Pytorch Improvements und dem Reinforcement Example Code von
  Lightning
- [ ] 
  Use [torchvision for data extraction](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html#input-extraction)
  ?
- [ ] Das Auslagern in Massenspeicher implementieren, falls der RAM voll ist mit Daten, so wie David Ha es in Word
  Models machen wollte und dann einen PR dazu gab. (Issue #19)
- [ ] Experimental PyTorch Feature: [Named Tensors](https://pytorch.org/docs/stable/named_tensor.html) für explizite
  Dimensions-Axen: (B)atch (oder N), (S)equence, (C)olor, (H)eight, (W)idth
  


