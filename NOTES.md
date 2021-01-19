# Notes

## Theory:

PlaNet learns:
* a transition model $`p\left( s_t \mid s_{t-1}, a_{t-1} \right)`$
* an observation model $`p\left( o_t \mid s_t \right)`$
* a reward model $`p\left( r_t \mid s_t \right)`$ from previously experienced episodes
* an encoder $`q\left( s_t \mid o_{\leq t}, a_{\lt t} \right)`$ to infer an approximate belief over the current hidden state from the history using filtering 

We use model-predictive control (MPC) to allow the agent to adapt its plan based on new observations, meaning we replan at each step.
## Implementation:

`config` is an `AttrDict`: A Dictionary, which allows access to its keys as attributes (i.e. `dict.key` instead of `dict['key']`). 
`AttrDict` is a complex class hand-written by the authors. It's might be a better idea to use [`dataclasses`](https://docs.python.org/3.7/library/dataclasses.html) from Python 3.7+ (cf. [StackOverflow](https://stackoverflow.com/a/14620633)).

### Reimplementation:

* Kai Arulkumaran (_PyTorch_): 
[[Reddit/r/ML](https://www.reddit.com/r/MachineLearning/comments/bgoyym/p_pytorch_implementation_of_planet_a_deep/)], 
[[Code](https://github.com/Kaixhin/PlaNet)]
* Kaito Suzuki (_PyTorch_): 
[[Twitter](https://twitter.com/63556poiuytrewq/status/1264231488305819650)], 
[[Code](https://github.com/cross32768/PlaNet_PyTorch)]
