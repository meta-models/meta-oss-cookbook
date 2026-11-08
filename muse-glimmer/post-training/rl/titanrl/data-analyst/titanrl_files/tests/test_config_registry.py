# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Construct real recipe configs without building models or GPU actors."""

from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple

import pytest
from renderers import CustomRendererConfig
from torchtitan.components.loss import ChunkedLossWrapper, Loss
from torchtitan.components.optim import (
    AdamW,
    LRSchedulersContainer,
    Optim,
    OptimizersContainer,
)
from torchtitan.components.renderer import RenderersConfigAdapter
from torchtitan.config import ConfigLoader
from torchtitan.distributed.activation_remat import FullActivationRematConfig
from torchtitan.distributed.local_compile import LocalCompileConfig
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.models.common.decoder_sharding import (
    pre_lm_head_norm_sharding_config,
    token_embedding_sharding_config,
)
from torchtitan.models.muse_glimmer import MuseGlimmerModel
from torchtitan.models.muse_glimmer.model import SoftCappedLinear
from torchtitan.rl.controller.controller import Controller
from torchtitan.rl.examples.glimmer_data_analyst import config_registry
from torchtitan.rl.examples.glimmer_data_analyst.data import GlimmerDataAnalystDataset
from torchtitan.rl.examples.glimmer_data_analyst.env import GlimmerDataAnalystEnv
from torchtitan.rl.examples.glimmer_data_analyst.rubric import RewardCorrectArtifact
from torchtitan.rl.generator.generator import Generator
from torchtitan.rl.generator.vllm.engine import VLLMEngine
from torchtitan.rl.rollout.advantage import AdvantageEstimator
from torchtitan.rl.rollout.data import IterableRLDataLoader
from torchtitan.rl.rollout.recorder import RolloutSampleRecorder
from torchtitan.rl.rollout.types import Rollout, RolloutGroup, RolloutStatus
from torchtitan.rl.trainer.losses import DAPOLoss
from torchtitan.rl.trainer.trainer import Trainer

CONFIG_MODULE = "torchtitan.rl.examples.glimmer_data_analyst.config_registry"


class RecipeCase(NamedTuple):
    factory_name: str
    steps: int
    prompts: int
    samples: int
    validation_steps: int
    checkpoint_interval: int


@pytest.fixture(
    params=[
        pytest.param(
            RecipeCase("rl_grpo_muse_glimmer_30b_data_analyst", 15, 8, 8, 64, 50),
            id="full",
        ),
        pytest.param(
            RecipeCase(
                "rl_grpo_muse_glimmer_30b_data_analyst_smoke", 10, 4, 4, 16, 1000
            ),
            id="smoke",
        ),
    ]
)
def recipe_case(request: pytest.FixtureRequest) -> RecipeCase:
    return request.param


@pytest.fixture
def recipe(recipe_case: RecipeCase) -> Controller.Config:
    return getattr(config_registry, recipe_case.factory_name)()


def test_factory_constructs_modern_training_config(
    recipe: Controller.Config, recipe_case: RecipeCase
) -> None:
    assert isinstance(recipe, Controller.Config)
    assert isinstance(recipe.trainer, Trainer.Config)
    assert isinstance(recipe.generator, Generator.Config)
    assert recipe.async_loop.num_training_steps == recipe_case.steps
    assert recipe.async_loop.num_prompts_per_train_step == recipe_case.prompts
    assert recipe.async_loop.num_samples_per_prompt == recipe_case.samples
    assert recipe.async_loop.validation.steps == recipe_case.validation_steps
    assert recipe.async_loop.target_offpolicy_steps == 1
    # Keep real GRPO filtering: the bounded demo must not farm zero-advantage
    # updates merely to make a long requested budget terminate successfully.
    assert (
        recipe.async_loop.batcher.training_sample_builder.drop_zero_std_reward_groups
        is True
    )

    trainer = recipe.trainer
    assert trainer.parallelism.data_parallel_shard_degree == 3
    assert trainer.parallelism.tensor_parallel_degree == 2
    assert trainer.training.disable_cuda_graphs is True
    assert trainer.training.max_context_length == 4096
    assert trainer.training.num_tokens_per_microbatch_per_dp_rank == 4096
    assert isinstance(trainer.activation_remat, FullActivationRematConfig)
    assert trainer.activation_remat.save_regions == []

    assert isinstance(trainer.optim, Optim.Config)
    assert isinstance(trainer.optim.optimizer, OptimizersContainer.Config)
    assert len(trainer.optim.optimizer.optimizers) == 1
    optimizer = trainer.optim.optimizer.optimizers[0]
    assert isinstance(optimizer, AdamW.Config)
    assert optimizer.pattern == r".*"
    assert optimizer.lr == pytest.approx(1e-6)
    assert isinstance(trainer.optim.lr_scheduler, LRSchedulersContainer.Config)
    assert trainer.optim.lr_scheduler.warmup_steps == 2
    assert trainer.optim.lr_scheduler.decay_type == "linear"
    assert trainer.optim.lr_scheduler.min_lr_factor == 1.0

    assert trainer.checkpointer is not None
    assert trainer.checkpointer.initial_load_in_hf is True
    assert trainer.checkpointer.interval == recipe_case.checkpoint_interval
    assert trainer.checkpointer.last_save_model_only is False
    assert trainer.checkpointer.keep_latest_k == 3

    assert isinstance(trainer.loss, Loss.Config)
    assert isinstance(trainer.loss, ChunkedLossWrapper.Config)
    assert trainer.loss.num_chunks == 8
    assert isinstance(trainer.loss.loss_fn, DAPOLoss.Config)
    assert trainer.loss.loss_fn.ratio_clip_low == pytest.approx(0.2)
    assert trainer.loss.loss_fn.ratio_clip_high == pytest.approx(0.28)
    assert trainer.loss.loss_fn.global_vocab_size == decoder_vocab_size(trainer.model)

    generator = recipe.generator
    assert generator.parallelism.data_parallel_degree == 1
    assert generator.parallelism.tensor_parallel_degree == 2
    assert generator.model_dtype == "bfloat16"
    assert generator.checkpointer is None
    assert isinstance(generator.engine, VLLMEngine.Config)
    assert generator.engine.cuda_graph.mode == "NONE"
    assert generator.sampling.temperature == 1.0
    assert generator.sampling.top_p == 1.0
    assert generator.sampling.max_tokens == 384
    assert recipe.metrics.enable_wandb is True


def test_role_models_are_independent_and_sharded_for_their_roles(
    recipe: Controller.Config,
) -> None:
    trainer_model = recipe.trainer.model
    generator_model = recipe.generator.model
    assert isinstance(trainer_model, MuseGlimmerModel.Config)
    assert isinstance(generator_model, MuseGlimmerModel.Config)
    assert trainer_model is not generator_model
    assert trainer_model.layers is not generator_model.layers
    assert trainer_model.layers[0] is not generator_model.layers[0]
    assert trainer_model.lm_head is not generator_model.lm_head
    assert trainer_model.max_context_length == generator_model.max_context_length

    for model, parallelism in (
        (trainer_model, recipe.trainer.parallelism),
        (generator_model, recipe.generator.parallelism.to_training()),
    ):
        assert model.max_context_length >= recipe.trainer.training.max_context_length
        assert model.vocab_size == 202048
        assert len(model.layers) == 52
        assert isinstance(model.lm_head, SoftCappedLinear.Config)
        assert isinstance(model.local_compile, LocalCompileConfig)
        assert model.local_compile.regions == []
        assert model.norm.sharding_config == pre_lm_head_norm_sharding_config(
            enable_sp=parallelism.enable_sequence_parallel
        )
        assert (
            model.tok_embeddings.embedding.sharding_config
            == token_embedding_sharding_config(
                enable_sp=parallelism.enable_sequence_parallel
            )
        )

    assert trainer_model.local_compile is not generator_model.local_compile
    assert (
        trainer_model.local_compile.regions is not generator_model.local_compile.regions
    )
    trainer_model.local_compile.regions.append("loss")
    assert generator_model.local_compile.regions == []


def test_repeated_factory_calls_do_not_share_mutable_configs(
    recipe_case: RecipeCase,
) -> None:
    factory = getattr(config_registry, recipe_case.factory_name)
    first = factory()
    second = factory()
    assert first is not second
    assert first.trainer.model is not second.trainer.model
    assert first.generator.model is not second.generator.model
    assert first.async_loop is not second.async_loop
    assert first.trainer.checkpointer is not second.trainer.checkpointer
    assert first.rollouter.worker.token_env is not second.rollouter.worker.token_env

    first.async_loop.num_training_steps = 1
    first.rollouter.worker.token_env.max_num_turns = 1
    assert second.async_loop.num_training_steps == recipe_case.steps
    assert second.rollouter.worker.token_env.max_num_turns == 6


def test_rollout_wiring_preserves_grpo_and_reserves_completion_headroom(
    recipe: Controller.Config,
) -> None:
    rollouter = recipe.rollouter
    assert isinstance(rollouter.training_dataloader, IterableRLDataLoader.Config)
    training_dataset = rollouter.training_dataloader.dataset
    assert isinstance(training_dataset, GlimmerDataAnalystDataset.Config)
    assert training_dataset.split == "train"
    assert training_dataset.seed == 42
    assert isinstance(rollouter.validation_dataset, GlimmerDataAnalystDataset.Config)
    assert rollouter.validation_dataset.split == "validation"
    assert rollouter.validation_dataset.seed == 42
    assert training_dataset is not rollouter.validation_dataset

    worker = rollouter.worker
    assert isinstance(worker.message_env, GlimmerDataAnalystEnv.Config)
    assert isinstance(worker.advantage, AdvantageEstimator.Config)
    assert worker.advantage.should_std_normalize is True
    assert worker.rubric.truncation_reward is None
    assert len(worker.rubric.reward_fns) == 1
    reward = worker.rubric.reward_fns[0]
    assert isinstance(reward, RewardCorrectArtifact.Config)
    assert reward.weight == 1.0
    assert worker.token_env.max_num_turns == 6
    assert worker.token_env.step_timeout_s == 120.0
    assert worker.token_env.max_rollout_tokens == 3712
    assert (
        worker.token_env.max_rollout_tokens + recipe.generator.sampling.max_tokens
        <= recipe.trainer.training.max_context_length
    )

    assert isinstance(recipe.renderer, RenderersConfigAdapter)
    renderer_config = recipe.renderer.get_renderers_config()
    assert isinstance(renderer_config, CustomRendererConfig)
    assert (
        renderer_config.import_path
        == "torchtitan.models.muse_glimmer.renderer.MuseGlimmerRenderer"
    )


def test_recorder_filter_keeps_all_scored_siblings_and_unscored_errors(
    recipe: Controller.Config, recipe_case: RecipeCase
) -> None:
    assert isinstance(recipe.rollout_recorder, RolloutSampleRecorder.Config)
    groups = [
        RolloutGroup(
            group_id=group_id,
            rollouts=[
                Rollout(
                    group_id=group_id,
                    rollout_id=index,
                    status=(
                        RolloutStatus.ERROR
                        if group_id == 1 and index == recipe_case.samples - 1
                        else RolloutStatus.COMPLETED
                    ),
                    reward=(
                        None
                        if group_id == 1 and index == recipe_case.samples - 1
                        else index / recipe_case.samples
                    ),
                )
                for index in range(recipe_case.samples)
            ],
        )
        for group_id in range(2)
    ]
    # Building a CPU filter does not build the recorder or any training actors.
    selected = recipe.rollout_recorder.filter.build()(groups)
    assert len(selected) == 2 * recipe_case.samples
    assert {(rollout.group_id, rollout.rollout_id) for rollout in selected} == {
        (group.group_id, rollout.rollout_id)
        for group in groups
        for rollout in group.rollouts
    }


def test_loader_loads_exact_factory_module_and_output_directory(
    recipe_case: RecipeCase, tmp_path: Path
) -> None:
    output_dir = tmp_path / "training-output"
    config = ConfigLoader().load(
        [
            "--module",
            CONFIG_MODULE,
            "--config",
            recipe_case.factory_name,
            "--output-dir",
            str(output_dir),
        ]
    )
    assert isinstance(config, Controller.Config)
    assert config.async_loop.num_training_steps == recipe_case.steps
    assert config.async_loop.validation.steps == recipe_case.validation_steps
    assert config.trainer.checkpointer.interval == recipe_case.checkpoint_interval
    assert config.dump_folder == str(output_dir)
    # The loader constructs config objects, not the controller or its recorder.
    assert not output_dir.exists()


def test_loader_config_snapshot_serializes_before_building_runtime(
    recipe_case: RecipeCase, tmp_path: Path
) -> None:
    output_dir = tmp_path / "snapshot-only-output"
    config = ConfigLoader().load(
        [
            "--module",
            CONFIG_MODULE,
            "--config",
            recipe_case.factory_name,
            "--output-dir",
            str(output_dir),
        ]
    )
    # Exercise the serialization used by runtime logging. CPython pprint on
    # this PyTorch wheel's SDPBackend enums is an upstream issue, independent
    # of recipe loading; do not monkeypatch it to manufacture a CLI pass.
    snapshot = json.loads(json.dumps(config.to_dict()))
    assert snapshot["async_loop"]["num_training_steps"] == recipe_case.steps
    assert snapshot["trainer"]["model"]["local_compile"]["regions"] == []
    assert snapshot["generator"]["model"]["local_compile"]["regions"] == []
    assert snapshot["trainer"]["loss"]["num_chunks"] == 8
    assert snapshot["dump_folder"] == str(output_dir)
    assert not output_dir.exists()


@pytest.mark.parametrize("flag", ["--dump-folder", "--training.steps"])
def test_loader_rejects_obsolete_or_generic_field_flags(
    flag: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as error:
        ConfigLoader().load(
            [
                "--module",
                CONFIG_MODULE,
                "--config",
                "rl_grpo_muse_glimmer_30b_data_analyst_smoke",
                flag,
                "10",
            ]
        )
    assert error.value.code == 2
    assert "unrecognized arguments" in capsys.readouterr().err
