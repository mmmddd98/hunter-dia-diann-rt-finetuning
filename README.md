# HUNTER DIA: DIA-NN RT Fine-Tuning

External fine-tuning and evaluation of DIA-NN 2.5.0 retention-time (RT) models
for the [HUNTER DIA model-adaptation thesis](https://github.com/mmmddd98/hunter-dia-model-adaptation).
The trainer consumes prepared Carafe2-format RT observations and exports a
DIA-NN TorchScript model with its matching token dictionary, including
N-terminal and lysine heavy dimethylation (UniMod:510).

## Install

Use Python 3.11 and a PyTorch build suited to your CPU/CUDA system.
See the [setup guide](docs/usage.md#install) for the tested environment.

```bash
git clone https://github.com/mmmddd98/hunter-dia-diann-rt-finetuning.git
cd hunter-dia-diann-rt-finetuning
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e '.[test,evaluation]'
```

Obtain `rt.d0.pt` and `dict.txt` from DIA-NN 2.5.0 and place them in `models/`,
or pass your own model directory. Pretrained weights are not bundled.

## Train with prepared partitions

```bash
python train_carafe_rt.py \
  --train-split inputs/train.tsv \
  --validation-split inputs/validation.tsv \
  --model-dir models --output-dir outputs/rt_run \
  --epochs 40 --warmup-epochs 10 --batch-size 512 \
  --learning-rate 1e-4 --loss mae --seed 2024 --device cpu
```

Use `--device cuda` for GPU training. Training and validation must be
peptidoform-disjoint and use the same RT target coordinate. Alignment is fitted
on training observations during [data preparation](https://github.com/mmmddd98/hunter-dia-workflows/tree/main/data_collect),
not by this trainer. Keep final-test data separate from model selection.

Deploy `rt.d0.finetuned.pt` together with the generated `dict.txt`.
The original dictionary does not encode the added heavy-dimethyl tokens.

## Guides

- [Input format](docs/data-format.md) and [outputs](docs/usage.md#model-and-output-contract)
- [Training recipes: layers, losses, weighting and sampling](docs/training-recipes.md)
- [Evaluation and attribution](captum_attribution/README.md)
- [Development and synthetic tests](docs/development.md)

The separate [Carafe2 fork](https://github.com/mmmddd98/hunter-dia-carafe2) trains
Carafe2/AlphaPeptDeep models. Those checkpoints are not interchangeable with
DIA-NN TorchScript models. W&B tracking is optional.
