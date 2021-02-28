# Tasks

- [x] NIKLAS: Planner bauen
- [x] NIKLAS: RSSM anpassen, damit Multi-GPU Packing möglich ist und korrekt unpacked wird.
- [ ] RSSM + MaskedLoss Function schreiben, sodass RSSM Padding/Padding nicht für backward-Path berücksichtig wird.
- [ ] Population based Optimizer schreiben (selber wie David Ha?)  
- [ ] Previous Action im Planner auf echte previous action der observation setzen, falls Observation nicht t=0 ist?
- [ ] LightningModule und DataModule schreiben
- [ ] Args/Config-DataClass oder entsprechendes Click/Argparse dict einbauen
- [ ] Korrekte Prior/Posterior Belief Terminologie

#### Optional:

- [ ] Sämtliche Anregungen aus den 3 Artikeln von Reddit zu Pytorch Improvements und dem Reinforcement Example Code von
  Lightning
- [ ] Use [torchvision for data
  extraction](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html#input-extraction)?
- [ ] Das Auslagern in Massenspeicher implementieren, falls der RAM voll ist mit Daten, so wie David Ha es in Word
  Models machen wollte und dann einen PR dazu gab. (Issue #19)
- [ ] Experimental PyTorch Feature: [Named Tensors](https://pytorch.org/docs/stable/named_tensor.html) für explizite Dimensions-Axen: (B)atch (oder N), (S)equence, (C)olor, (H)eight, (W)idth
  


