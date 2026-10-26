import torch

from diann_finetune.training import (
    build_loss_function,
    configure_optimizer,
    distributed_batch_shard,
    heavy_checkpoint_selection_scores,
    round_robin_organism_batches,
    sample_organism_indices,
    save_portable_torchscript,
    training_sample_weights,
)


class TinyRTModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(256, 4)
        self.backbone = torch.nn.Linear(4, 4)
        self.cnn = torch.nn.Linear(4, 4)
        self.cnn2 = torch.nn.Linear(4, 4)
        self.rnn = torch.nn.Linear(4, 4)
        self.prepare = torch.nn.Linear(4, 4)
        self.out = torch.nn.Linear(4, 1)


def test_portable_checkpoint_has_cpu_tensor_storage(tmp_path):
    class ScriptableDeviceModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([2.0]))
            self.device = "cpu"

        def forward(self, values):
            return values * self.weight

    model = torch.jit.script(ScriptableDeviceModel())
    path = tmp_path / "portable.pt"
    save_portable_torchscript(model, path)

    # No map_location: serialized storages themselves must already be CPU.
    restored = torch.jit.load(path)
    assert next(restored.parameters()).device.type == "cpu"
    assert restored.device == "cpu"


def test_head_strategy_updates_only_custom_embedding_rows_and_output():
    model = TinyRTModel()
    optimizer, parameters, base_lrs = configure_optimizer(
        model, "head", {40: 38, 41: 39}, 1e-4, None, 0.0
    )
    assert [group["name"] for group in optimizer.param_groups] == ["custom_embeddings", "output"]
    assert base_lrs == [1e-4, 1e-4]
    loss = model.embedding(torch.tensor([5, 40, 41])).sum() + model.out(torch.ones(1, 4)).sum()
    loss.backward()
    assert torch.count_nonzero(model.embedding.weight.grad[5]) == 0
    assert torch.count_nonzero(model.embedding.weight.grad[40]) > 0
    assert torch.count_nonzero(model.embedding.weight.grad[41]) > 0
    assert model.backbone.weight.requires_grad is False
    parameter_ids = {id(parameter) for parameter in parameters}
    assert all(id(parameter) in parameter_ids for parameter in model.out.parameters())


def test_discriminative_strategy_has_separate_learning_rates():
    model = TinyRTModel()
    optimizer, _, base_lrs = configure_optimizer(
        model, "discriminative", {40: 38, 41: 39}, 1e-4, 1e-5, 0.0
    )
    assert [group["name"] for group in optimizer.param_groups] == [
        "backbone",
        "head",
        "custom_embeddings",
    ]
    assert base_lrs == [1e-5, 1e-4, 1e-4]
    assert all(parameter.requires_grad for parameter in model.parameters())


def test_partial_layer_strategies_select_expected_blocks():
    cases = {
        "embeddings": {"embedding"},
        "rt_head": {"embedding", "prepare", "out"},
        "cnn_head": {"embedding", "cnn", "cnn2", "prepare", "out"},
        "gru_head": {"embedding", "rnn", "prepare", "out"},
    }
    for strategy, expected_modules in cases.items():
        model = TinyRTModel()
        optimizer, _, _ = configure_optimizer(
            model, strategy, {40: 38, 41: 39}, 1e-4, 1e-5, 0.0, 2e-4
        )
        trainable_modules = {
            name.split(".")[0]
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        assert trainable_modules == expected_modules
        embedding_group = next(
            group for group in optimizer.param_groups if group["name"] == "custom_embeddings"
        )
        assert embedding_group["lr"] == 2e-4


def test_selectable_losses_are_unreduced():
    predictions = torch.tensor([0.0, 0.2])
    targets = torch.tensor([0.1, 0.0])
    for name in ("mae", "huber", "mse"):
        values = build_loss_function(name, 0.03)(predictions, targets)
        assert values.shape == predictions.shape
        assert torch.all(values > 0)


def test_heavy_dimethyl_category_weights():
    import pandas as pd

    frame = pd.DataFrame(
        {
            "mods": [
                "",
                "Dimethyl:2H(4)13C(2)@N-term",
                "Dimethyl:2H(4)13C(2)@K",
                "Dimethyl:2H(4)13C(2)@N-term;Dimethyl:2H(4)13C(2)@K",
            ]
        }
    )
    weights, counts = training_sample_weights(
        frame,
        heavy_nterm_weight=2.0,
        heavy_lysine_weight=1.25,
        heavy_both_weight=3.0,
    )
    assert weights.tolist() == [1.0, 2.0, 1.25, 3.0]
    assert counts == {
        "unweighted": 1,
        "heavy_nterm_only": 1,
        "heavy_lysine_only": 1,
        "heavy_nterm_and_lysine": 1,
    }


def test_normal_weight_can_be_reduced_for_target_focused_training():
    import pandas as pd

    frame = pd.DataFrame({
        "mods": ["", "Dimethyl:2H(4)13C(2)@N-term"],
    })
    weights, _ = training_sample_weights(
        frame,
        normal_weight=0.05,
        heavy_nterm_weight=3.0,
        heavy_lysine_weight=2.0,
        heavy_both_weight=4.0,
    )
    assert torch.allclose(weights, torch.tensor([0.05, 3.0]))


def test_heavy_checkpoint_selection_grid_spans_overall_to_heavy_priority():
    metrics = {
        "mae": 0.10,
        "heavy_nterm_mae": 0.20,
        "heavy_lysine_mae": 0.30,
        "heavy_both_mae": 0.40,
    }
    scores = heavy_checkpoint_selection_scores(metrics)
    # Heavy-Macro = (0.20 + 0.30 + 0.40) / 3 = 0.30.
    expected = {"100_0": 0.10, "75_25": 0.15, "50_50": 0.20, "25_75": 0.25, "0_100": 0.30}
    assert scores.keys() == expected.keys()
    assert all(abs(scores[label] - value) < 1e-12 for label, value in expected.items())


def test_distributed_batch_shard_covers_rows_once_and_masks_padding():
    indices = [10, 11, 12, 13, 14]
    shards = [distributed_batch_shard(indices, rank, 3) for rank in range(3)]
    assert [len(shard) for shard, _ in shards] == [2, 2, 2]
    valid_indices = sorted(
        index
        for (shard, valid) in shards
        for index, keep in zip(shard, valid)
        if keep
    )
    assert valid_indices == indices
    assert sum(int((~valid).sum()) for _, valid in shards) == 1


def test_distributed_batch_shard_is_identity_for_one_process():
    indices, valid = distributed_batch_shard([3, 5, 7], rank=0, world_size=1)
    assert indices == [3, 5, 7]
    assert valid.tolist() == [True, True, True]


def test_fixed_organism_ratios_have_exact_epoch_quotas():
    organisms = ["celegans"] * 12 + ["human"] * 3 + ["triplemix"] * 6
    for strategy, expected in {
        "ratio_1_1_1": {"celegans": 12, "human": 12, "triplemix": 12},
        "ratio_2_1_1": {"celegans": 18, "human": 9, "triplemix": 9},
        "ratio_4_1_1": {"celegans": 24, "human": 6, "triplemix": 6},
        "ratio_8_1_1": {"celegans": 32, "human": 4, "triplemix": 4},
        "ratio_1_2_1": {"celegans": 10, "human": 20, "triplemix": 10},
        "ratio_1_1_2": {"celegans": 10, "human": 10, "triplemix": 20},
    }.items():
        sample_rows = sum(expected.values())
        indices = sample_organism_indices(organisms, strategy, sample_rows, seed=7)
        observed = {name: sum(organisms[index] == name for index in indices) for name in expected}
        assert observed == expected


def test_inverse_frequency_sampling_is_reproducible_and_balanced_in_expectation():
    organisms = ["celegans"] * 100 + ["human"] * 10 + ["triplemix"] * 25
    first = sample_organism_indices(organisms, "inverse_frequency", 9000, seed=11)
    second = sample_organism_indices(organisms, "inverse_frequency", 9000, seed=11)
    assert first == second
    counts = {name: sum(organisms[index] == name for index in first) for name in set(organisms)}
    assert all(abs(count - 3000) < 200 for count in counts.values())


def test_temperature_sampling_interpolates_uniform_and_natural_source_mix():
    organisms = ["celegans"] * 100 + ["human"] * 10 + ["triplemix"] * 25
    low = sample_organism_indices(organisms, "temperature_0_25", 30000, seed=13)
    high = sample_organism_indices(organisms, "temperature_0_75", 30000, seed=13)
    low_ce = sum(organisms[index] == "celegans" for index in low)
    high_ce = sum(organisms[index] == "celegans" for index in high)
    assert 10000 < low_ce < high_ce < 30000 * 100 / 135


def test_curriculum_reaches_requested_endpoints():
    organisms = ["celegans"] * 100 + ["human"] * 10 + ["triplemix"] * 25
    start = sample_organism_indices(
        organisms, "curriculum_ce4_to_uniform", 600, seed=17, progress=0.0
    )
    end = sample_organism_indices(
        organisms, "curriculum_ce4_to_uniform", 600, seed=17, progress=1.0
    )
    start_counts = [sum(organisms[index] == name for index in start) for name in ("celegans", "human", "triplemix")]
    end_counts = [sum(organisms[index] == name for index in end) for name in ("celegans", "human", "triplemix")]
    assert start_counts == [400, 100, 100]
    assert end_counts == [200, 200, 200]


def test_round_robin_batches_are_source_homogeneous_and_fixed_length():
    encoded = [torch.zeros(5, dtype=torch.long) for _ in range(18)]
    organisms = ["celegans"] * 6 + ["human"] * 6 + ["triplemix"] * 6
    batches = list(
        round_robin_organism_batches(encoded, organisms, 2, steps=9, seed=3)
    )
    assert len(batches) == 9
    assert [organisms[batch[0]] for batch in batches] == [
        "celegans", "human", "triplemix",
        "celegans", "human", "triplemix",
        "celegans", "human", "triplemix",
    ]
    assert all(len({organisms[index] for index in batch}) == 1 for batch in batches)
