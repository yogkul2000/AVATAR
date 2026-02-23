# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

import json
import random
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, List, Optional, Tuple

import torch
from accelerate.utils import gather, gather_object
from trl.trainer.grpo_trainer import nanmax, nanmin, nanstd

from swift.template import TemplateInputs
from swift.utils import get_logger, remove_response, to_device
from .grpo_trainer import GRPOTrainer
from .rollout_mixin import DataType
from .utils import compute_chord_loss, profiling_context

logger = get_logger()


@dataclass
class TASReplayEntry:
    batch: dict
    score: float
    reward_proxy: float
    kl_mean: float
    mean_rbar: float
    tier: str


class StratifiedReplayBuffer:
    def __init__(self, max_size: int, easy_ratio: float, medium_ratio: float, hard_ratio: float,
                 strategy: str, temperature: float, rng: random.Random):
        self._rng = rng
        self._strategy = strategy
        self._temperature = max(1e-6, temperature)

        easy_size = int(round(max_size * easy_ratio))
        medium_size = int(round(max_size * medium_ratio))
        hard_size = max_size - easy_size - medium_size

        self._tiers = {
            'easy': deque(maxlen=max(1, easy_size)),
            'medium': deque(maxlen=max(1, medium_size)),
            'hard': deque(maxlen=max(1, hard_size)),
        }
        self._tier_ratios = {'easy': easy_ratio, 'medium': medium_ratio, 'hard': hard_ratio}

    def __len__(self) -> int:
        return sum(len(tier) for tier in self._tiers.values())

    def add(self, entry: TASReplayEntry) -> None:
        self._tiers[entry.tier].append(entry)

    def sample(self, n: int) -> List[TASReplayEntry]:
        if n <= 0:
            return []
        total_available = len(self)
        if total_available == 0:
            return []
        n = min(n, total_available)

        targets = self._target_counts(n)
        result: List[TASReplayEntry] = []
        remaining = n
        for tier, count in targets.items():
            if count <= 0:
                continue
            sampled = self._sample_from_tier(tier, count)
            result.extend(sampled)
            remaining -= len(sampled)

        if remaining > 0:
            all_entries = []
            for tier_entries in self._tiers.values():
                all_entries.extend(list(tier_entries))
            if all_entries:
                result.extend(self._rng.sample(all_entries, k=min(remaining, len(all_entries))))
        return result

    def _target_counts(self, n: int) -> Dict[str, int]:
        counts = {tier: int(round(n * ratio)) for tier, ratio in self._tier_ratios.items()}
        total = sum(counts.values())
        while total > n:
            tier = max(counts, key=counts.get)
            if counts[tier] > 0:
                counts[tier] -= 1
                total -= 1
            else:
                break
        while total < n:
            tier = max(self._tier_ratios, key=self._tier_ratios.get)
            counts[tier] += 1
            total += 1
        return counts

    def _sample_from_tier(self, tier: str, n: int) -> List[TASReplayEntry]:
        entries = list(self._tiers[tier])
        if not entries:
            return []
        n = min(n, len(entries))
        if self._strategy == 'fifo':
            return entries[:n]
        if self._strategy != 'priority':
            return self._rng.sample(entries, k=n)

        scores = [e.score for e in entries]
        min_score = min(scores)
        shifted = [s - min_score for s in scores]
        weights = [pow(2.71828, s / self._temperature) for s in shifted]
        return self._rng.choices(entries, weights=weights, k=n)


class GRPOTASTrainer(GRPOTrainer):
    """GRPO trainer implementing AVATAR off-policy replay + TAS."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tas_rng = random.Random(self.args.seed + 17)
        self._tas_buffer = StratifiedReplayBuffer(
            max_size=self.args.tas_replay_buffer_size,
            easy_ratio=self.args.tas_replay_easy_ratio,
            medium_ratio=self.args.tas_replay_medium_ratio,
            hard_ratio=self.args.tas_replay_hard_ratio,
            strategy=self.args.tas_replay_strategy,
            temperature=self.args.tas_priority_temperature,
            rng=self._tas_rng,
        )
        self._vcrs: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=self.args.tas_vcrs_window))
        self._prompt_stats: Dict[str, Dict[str, float]] = {}
        self._zero_reward_streak: Dict[str, int] = defaultdict(int)
        self._tas_last_log = {}

    def _preprocess_inputs(self, inputs: DataType) -> DataType:
        if not self.model.training or not self.args.tas_hint_key:
            return super()._preprocess_inputs(inputs)

        for input_item in inputs:
            if input_item.get('_tas_hint_applied'):
                continue
            prompt_msgs = input_item['messages']
            if prompt_msgs and prompt_msgs[-1].get('role') == 'assistant':
                prompt_msgs = prompt_msgs[:-1]
            prompt_key = self._get_prompt_key(prompt_msgs)
            if not self._should_apply_hint(prompt_key):
                continue
            hint = input_item.get(self.args.tas_hint_key)
            if not hint:
                continue
            if isinstance(hint, list):
                hint = '\n'.join(str(h) for h in hint)
            hint_text = f"{self.args.tas_hint_prefix}{hint}"
            hint_msg = {'role': self.args.tas_hint_role, 'content': hint_text}
            input_item['messages'] = [hint_msg] + input_item['messages']
            input_item['_tas_hint_applied'] = True
        return super()._preprocess_inputs(inputs)

    def _prepare_inputs(self, generation_batch):
        if not self.args.tas_enable or not self.model.training:
            return super()._prepare_inputs(generation_batch)

        num_rollout_samples = self.args.steps_per_generation * self.template.sequence_parallel_size
        generate_every = num_rollout_samples * self.num_iterations
        if self._step % generate_every == 0 or self._buffered_inputs is None:
            on_policy_batches = self._generate_and_score_completions(generation_batch)
            mixed_batches = self._mix_with_replay(on_policy_batches)
            self._buffered_inputs = mixed_batches
            self._add_to_replay_buffer(on_policy_batches)
            self._log_avatar_status()
        inputs = self._buffered_inputs[self._step % num_rollout_samples]
        self._step += 1
        return inputs

    def _generate_and_score_completions(self, inputs: DataType) -> DataType:
        if self.template.truncation_strategy == 'raise':
            inputs = self.resample_encode_failed_inputs(inputs)

        inputs = self._generate_completions(inputs)
        total_rewards_per_func = self._score_completions(inputs)
        mode = 'train' if self.model.training else 'eval'

        if self.dynamic_sample and mode == 'train':
            inputs, total_rewards_per_func = self._dynamic_sampling(inputs, total_rewards_per_func)  # noqa

        batch_encoded_inputs = self._prepare_batch_inputs(inputs)
        total_advantages = self._compute_advantages(inputs, total_rewards_per_func, batch_encoded_inputs)

        local_advantages = self._get_even_process_advantages(inputs, total_advantages)
        for i, advantage in enumerate(local_advantages):
            inputs[i]['advantages'] = advantage
        self._logs['advantages'].extend(total_advantages.tolist())

        gas_chunks = self.split_by_mini_batches(inputs)
        assert len(gas_chunks) == len(batch_encoded_inputs), \
            f'Mismatch: {len(gas_chunks)} chunks vs {len(batch_encoded_inputs)} batches'

        rewards = (total_rewards_per_func * self.reward_weights.unsqueeze(0)).nansum(dim=1)
        offset = 0
        for batch, batch_encoded in zip(gas_chunks, batch_encoded_inputs):
            all_advantages = torch.stack([data['advantages'] for data in batch])
            batch_encoded['advantages'] = all_advantages

            prompt_keys = []
            for data in batch:
                prompt_msgs = data['messages']
                if prompt_msgs and prompt_msgs[-1].get('role') == 'assistant':
                    prompt_msgs = prompt_msgs[:-1]
                prompt_keys.append(self._get_prompt_key(prompt_msgs))
            batch_encoded['prompt_keys'] = prompt_keys
            slice_rewards = rewards[offset:offset + len(batch)]
            offset += len(batch)
            batch_encoded['rewards'] = slice_rewards.to(batch_encoded['advantages'].device)

        with profiling_context(self, 'log_metrics'):
            self._log_rollout_metrics(inputs)

        return batch_encoded_inputs

    def _get_even_process_advantages(self, inputs: DataType, total_advantages: torch.Tensor):
        from .utils import get_even_process_data
        local_advantages = get_even_process_data(self, total_advantages)
        assert len(local_advantages) == len(inputs)
        return local_advantages

    def _log_rollout_metrics(self, inputs: DataType) -> None:
        from copy import deepcopy

        messages = [inp['messages'][:-1] for inp in inputs]
        completions = [deepcopy(inp['messages'][-1]['content']) for inp in inputs]
        for i, completion in enumerate(completions):
            if isinstance(completion, str):
                continue
            if isinstance(completion, list):
                token_ids = completion
            elif isinstance(completion, dict):
                token_ids = completion['token_ids']
            completions[i] = self.processing_class.decode(token_ids)
        valid_messages = self._gather_and_flatten(messages, flatten_level=0)
        valid_completions = self._gather_and_flatten(completions, flatten_level=0)

        prompts_text = []
        for msgs in valid_messages:
            remove_response(msgs)
            template_inputs = TemplateInputs.from_dict({'messages': msgs})
            res = self.template.encode(template_inputs)
            prompts_text.append(self.template.safe_decode(res['input_ids']))
        self._logs['prompt'].extend(prompts_text)
        self._logs['completion'].extend(valid_completions)

        metrics_for_logs_to_gather = {}
        if all('solution' in inp for inp in inputs):
            metrics_for_logs_to_gather['solution'] = [inp['solution'] for inp in inputs]
        if all('rollout_infos' in inp and 'num_turns' in inp['rollout_infos'] for inp in inputs):
            metrics_for_logs_to_gather['num_turns'] = [inp['rollout_infos']['num_turns'] for inp in inputs]

        if metrics_for_logs_to_gather:
            for key, value in metrics_for_logs_to_gather.items():
                if key not in self._logs:
                    self._logs[key] = deque(maxlen=self.args.generation_batch_size)
                self._logs[key].extend(self._gather_and_flatten(value, flatten_level=0))

    def _mix_with_replay(self, on_policy_batches: List[dict]) -> List[dict]:
        if not self._replay_ready():
            self._tas_last_log = {
                'replay_used': 0,
                'on_policy_used': len(on_policy_batches),
                'off_policy_used': 0,
                'buffer_size': len(self._tas_buffer),
            }
            return on_policy_batches

        total_batches = len(on_policy_batches)
        off_policy_batches = min(self.args.tas_off_policy_batches, total_batches)
        if self.args.tas_on_policy_batches is not None:
            on_policy_batches_target = min(self.args.tas_on_policy_batches, total_batches)
            off_policy_batches = min(off_policy_batches, total_batches - on_policy_batches_target)

        if off_policy_batches <= 0:
            self._tas_last_log = {
                'replay_used': 0,
                'on_policy_used': len(on_policy_batches),
                'off_policy_used': 0,
                'buffer_size': len(self._tas_buffer),
            }
            return on_policy_batches

        replay_entries = self._tas_buffer.sample(off_policy_batches)
        if not replay_entries:
            self._tas_last_log = {
                'replay_used': 0,
                'on_policy_used': len(on_policy_batches),
                'off_policy_used': 0,
                'buffer_size': len(self._tas_buffer),
            }
            return on_policy_batches
        replace_indices = self._tas_rng.sample(range(total_batches), k=len(replay_entries))
        replay_batches = [self._to_device_batch(entry.batch) for entry in replay_entries]
        for replay_batch in replay_batches:
            self._apply_vcrs_to_replay_batch(replay_batch)

        mixed = list(on_policy_batches)
        for idx, replay_batch in zip(replace_indices, replay_batches):
            mixed[idx] = replay_batch
        self._tas_last_log = {
            'replay_used': 1,
            'on_policy_used': len(on_policy_batches) - len(replay_batches),
            'off_policy_used': len(replay_batches),
            'buffer_size': len(self._tas_buffer),
        }
        return mixed

    def _add_to_replay_buffer(self, on_policy_batches: List[dict]) -> None:
        for batch in on_policy_batches:
            prompt_keys = batch.get('prompt_keys', [])
            rewards = batch.get('rewards')
            if rewards is not None and prompt_keys:
                for key, reward in zip(prompt_keys, rewards.tolist()):
                    self._vcrs[key].append(float(reward))
                    if reward == 0.0:
                        self._zero_reward_streak[key] += 1
                    else:
                        self._zero_reward_streak[key] = 0

            mean_rbar = self._mean_rbar(prompt_keys)
            tier = self._assign_tier(mean_rbar)

            reward_proxy, kl_mean, score = self._score_replay_batch(batch)
            if self.args.tas_store_score_threshold is not None and score < self.args.tas_store_score_threshold:
                continue
            cpu_batch = self._detach_to_cpu(batch)
            self._tas_buffer.add(
                TASReplayEntry(
                    batch=cpu_batch,
                    score=score,
                    reward_proxy=reward_proxy,
                    kl_mean=kl_mean,
                    mean_rbar=mean_rbar,
                    tier=tier,
                ))
            for key in prompt_keys:
                self._prompt_stats[key] = {'rbar': mean_rbar, 'kl_mean': kl_mean}

    def _replay_ready(self) -> bool:
        if self.args.tas_off_policy_batches <= 0:
            return False
        return len(self._tas_buffer) >= self.args.tas_replay_min_size

    def _apply_vcrs_to_replay_batch(self, batch: dict) -> None:
        prompt_keys = batch.get('prompt_keys')
        rewards = batch.get('rewards')
        if rewards is None or not prompt_keys:
            return
        device = rewards.device
        rbar = torch.tensor([
            self._get_rbar(key) for key in prompt_keys
        ], dtype=torch.float32, device=device)
        std = rewards.std() if rewards.numel() > 1 else torch.tensor(1.0, device=device)
        advantages = (rewards - rbar) / (std + 1e-4)
        advantages = advantages * self.args.tas_off_policy_alpha
        batch['advantages'] = advantages

    def _mean_rbar(self, prompt_keys: Iterable[str]) -> float:
        values = [self._get_rbar(key) for key in prompt_keys]
        return float(sum(values) / max(1, len(values)))

    def _get_rbar(self, prompt_key: str) -> float:
        values = self._vcrs.get(prompt_key)
        if not values:
            return 0.0
        return float(sum(values) / len(values))

    def _assign_tier(self, rbar: float) -> str:
        all_rbars = sorted([self._get_rbar(k) for k in self._vcrs.keys()])
        if not all_rbars:
            return 'hard'
        hard_q = self._quantile(all_rbars, 0.40)
        medium_q = self._quantile(all_rbars, 0.75)
        if rbar <= hard_q:
            return 'hard'
        if rbar <= medium_q:
            return 'medium'
        return 'easy'

    @staticmethod
    def _quantile(values: List[float], q: float) -> float:
        if not values:
            return 0.0
        idx = int(round(q * (len(values) - 1)))
        return values[max(0, min(len(values) - 1, idx))]

    def _score_replay_batch(self, batch: dict) -> Tuple[float, float, float]:
        advantages = batch.get('advantages')
        reward_proxy = float(advantages.mean().item()) if advantages is not None else 0.0

        kl_mean = 0.0
        ref_per_token_logps = batch.get('ref_per_token_logps')
        if ref_per_token_logps is not None:
            old_per_token_logps = batch.get('old_per_token_logps')
            completion_mask = batch.get('completion_mask')
            if old_per_token_logps is not None and completion_mask is not None:
                per_token_kl = old_per_token_logps - ref_per_token_logps
                kl_per_seq = (per_token_kl * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)
                kl_mean = float(kl_per_seq.mean().item())

        score = (
            self.args.tas_priority_reward_weight * reward_proxy
            - self.args.tas_priority_kl_weight * kl_mean
        )
        return reward_proxy, kl_mean, score

    def _should_apply_hint(self, prompt_key: str) -> bool:
        if self.args.tas_hint_always:
            return True
        zero_patience = self.args.tas_hint_zero_patience
        if zero_patience is not None and self._zero_reward_streak.get(prompt_key, 0) >= zero_patience:
            return True
        if self.args.tas_hint_reward_threshold is None and self.args.tas_hint_kl_threshold is None:
            return False
        rbar = self._get_rbar(prompt_key)
        if self.args.tas_hint_reward_threshold is not None and rbar > self.args.tas_hint_reward_threshold:
            return False
        if self.args.tas_hint_kl_threshold is not None:
            kl = self._prompt_stats.get(prompt_key, {}).get('kl_mean', 0.0)
            if kl > self.args.tas_hint_kl_threshold:
                return False
        return True

    def _get_prompt_key(self, messages: List[dict]) -> str:
        return json.dumps(messages, sort_keys=True)

    def _detach_to_cpu(self, batch: dict) -> dict:
        cpu_batch = {}
        for key, value in batch.items():
            if torch.is_tensor(value):
                cpu_batch[key] = value.detach().cpu()
            else:
                cpu_batch[key] = value
        return cpu_batch

    def _to_device_batch(self, batch: dict) -> dict:
        return to_device(batch, self.accelerator.device)

    def _compute_tas_weights(self, completion_mask: torch.Tensor) -> torch.Tensor:
        if not self.args.tas_enable or self.args.tas_lambda <= 0:
            return torch.ones_like(completion_mask, dtype=torch.float32)
        bsz, seq_len = completion_mask.shape
        positions = torch.arange(seq_len, device=completion_mask.device).unsqueeze(0).expand(bsz, -1)
        lengths = completion_mask.sum(-1).clamp(min=1)
        denom = (lengths - 1).clamp(min=1).unsqueeze(1)
        t_norm = positions.float() / denom.float()
        weights = 1.0 + self.args.tas_lambda * (2.0 * t_norm - 1.0) ** 2
        return weights

    def _log_avatar_status(self) -> None:
        if not self.args.tas_enable or not self.accelerator.is_main_process:
            return
        stats = self._tas_last_log or {}
        replay_used = stats.get('replay_used', 0)
        on_policy = stats.get('on_policy_used', 0)
        off_policy = stats.get('off_policy_used', 0)
        buffer_size = stats.get('buffer_size', len(self._tas_buffer))
        vcrs_prompts = len(self._vcrs)
        color = '\x1b[92m' if replay_used else '\x1b[93m'
        reset = '\x1b[0m'
        logger.info(
            f"{color}[AVATAR] replay={replay_used} on={on_policy} off={off_policy} "
            f"buffer={buffer_size} vcrs_prompts={vcrs_prompts}{reset}")

    def _compute_loss_and_metrics(self, model, inputs):
        mode = 'train' if self.model.training else 'eval'
        completion_mask = inputs['completion_mask']
        truncated_mask = inputs['truncated_mask']
        per_token_logps, entropies = self._get_per_token_logps_and_entropies(
            model, inputs, compute_entropy=self.compute_entropy)

        entropy_mask = None
        entropy_metrics = {}

        if self.compute_entropy:
            entropies = entropies.masked_fill(completion_mask == 0, float('nan'))
            if self.args.log_entropy:
                per_completion_entropies_mean = torch.nanmean(entropies, dim=1)
                global_per_completion_entropies_mean = gather(per_completion_entropies_mean)
                entropy_metrics = {
                    'entropy_logs': global_per_completion_entropies_mean.tolist(),
                    'entropy_mean': global_per_completion_entropies_mean.nanmean().item(),
                    'entropy_max': nanmax(global_per_completion_entropies_mean).item(),
                    'entropy_min': nanmin(global_per_completion_entropies_mean).item()
                }

            if self.args.top_entropy_quantile < 1.0:
                entropy_threshold = torch.nanquantile(entropies.flatten().float(), 1 - self.top_entropy_quantile)
                entropy_metrics['entropy_threshold'] = entropy_threshold.item()
                entropy_mask = entropies >= entropy_threshold

        if self.overlong_filter and any(truncated_mask):
            if all(truncated_mask):
                logger.info('All completions are overlong and truncated, resulting in NaN some values for some metrics.')
            truncated_mask = truncated_mask.unsqueeze(-1).expand_as(completion_mask)
            completion_mask = completion_mask & (~truncated_mask)

        if self.beta != 0.0 and not self.kl_in_reward:
            ref_per_token_logps = inputs['ref_per_token_logps']
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1)
        else:
            per_token_kl = None

        advantages = inputs['advantages']
        old_per_token_logps = (
            per_token_logps.detach() if inputs['old_per_token_logps'] is None else inputs['old_per_token_logps'])

        rollout_correction_metrics = {}
        should_compute_rollout_metrics = (
            self.rollout_importance_sampling_mode is not None or self.log_rollout_offpolicy_metrics)

        local_has_rollout_per_token_logps = inputs.get('rollout_per_token_logps') is not None
        all_has_rollout_per_token_logps = gather_object([local_has_rollout_per_token_logps])

        should_compute_rollout_metrics = should_compute_rollout_metrics and all(all_has_rollout_per_token_logps)
        if (not self.disable_rollout_importance_sampling and should_compute_rollout_metrics):
            rollout_per_token_logps = inputs['rollout_per_token_logps']
            rollout_correction_metrics = self._compute_rollout_offpolicy_metrics(old_per_token_logps,
                                                                                 rollout_per_token_logps,
                                                                                 completion_mask)
            if self.rollout_importance_sampling_mode is not None:
                rollout_log_ratio = old_per_token_logps - rollout_per_token_logps
                rollout_is_weights = self._apply_rollout_importance_sampling(rollout_log_ratio, completion_mask)
                is_metrics = self._compute_is_correction_metrics(rollout_log_ratio, rollout_is_weights, completion_mask)
                rollout_correction_metrics.update(is_metrics)
                inputs['rollout_is_weights'] = rollout_is_weights
            else:
                inputs['rollout_is_weights'] = None
        else:
            inputs['rollout_is_weights'] = None

        log_ratio = per_token_logps - old_per_token_logps
        if self.importance_sampling_level == 'token':
            log_importance_weights = log_ratio
        elif self.importance_sampling_level in ['sequence', 'sequence_token']:
            seq_level_log_weights = ((log_ratio * completion_mask).sum(-1)
                                     / completion_mask.sum(-1).clamp(min=1.0)).unsqueeze(-1)
            if self.importance_sampling_level == 'sequence':
                log_importance_weights = seq_level_log_weights
            else:
                seq_level_log_weight = seq_level_log_weights.detach()
                log_importance_weights = per_token_logps - per_token_logps.detach() + seq_level_log_weight
        else:
            raise ValueError(
                f"Unknown importance sampling level: {self.importance_sampling_level}. Possible values are 'token' "
                "and 'sequence'.")

        coef_1 = torch.exp(log_importance_weights)

        if self.loss_type == 'cispo':
            clamped_ratios = torch.clamp(coef_1, max=self.epsilon_high).detach()
            per_token_loss = -clamped_ratios * advantages.unsqueeze(1) * per_token_logps
        elif self.loss_type == 'sapo':
            advantages_expanded = advantages.unsqueeze(1)
            gate_pos = torch.sigmoid(self.tau_pos * (coef_1 - 1)) * (4.0 / self.tau_pos)
            gate_neg = torch.sigmoid(self.tau_neg * (coef_1 - 1)) * (4.0 / self.tau_neg)
            is_positive = advantages_expanded > 0
            soft_gate = torch.where(is_positive, gate_pos, gate_neg)

            per_token_loss = -soft_gate * advantages_expanded
        elif self.loss_type in ['grpo', 'bnpo', 'dr_grpo', 'dapo']:
            coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
            if self.args.delta is not None:
                coef_1 = torch.clamp(coef_1, max=self.args.delta)

            per_token_loss1 = coef_1 * advantages.unsqueeze(1)
            per_token_loss2 = coef_2 * advantages.unsqueeze(1)
            per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        if entropy_mask is not None:
            per_token_loss = per_token_loss * entropy_mask

        tas_weights = self._compute_tas_weights(completion_mask)
        per_token_loss = per_token_loss * tas_weights

        if per_token_kl is not None:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        if inputs.get('rollout_is_weights') is not None and self.rollout_importance_sampling_mode is not None:
            rollout_is_weights = inputs['rollout_is_weights']
            per_token_loss = per_token_loss * rollout_is_weights

        if self.off_policy_sequence_mask_delta is not None:
            rollout_per_token_logps = inputs.get('rollout_per_token_logps')
            old_policy_per_token_logps = rollout_per_token_logps if rollout_per_token_logps is not None \
                else old_per_token_logps
            off_policy_seq_mask = self._compute_off_policy_sequence_mask(per_token_logps, old_policy_per_token_logps,
                                                                         completion_mask, advantages)
            off_policy_seq_mask_expanded = off_policy_seq_mask.unsqueeze(-1).expand_as(completion_mask)
            completion_mask = completion_mask & off_policy_seq_mask_expanded

        if self.loss_type in ['grpo', 'sapo']:
            loss = ((per_token_loss * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)).mean()
        elif self.loss_type == 'bnpo':
            loss = (per_token_loss * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
        elif self.loss_type == 'dr_grpo':
            batch_size = completion_mask.shape[0]
            loss = (per_token_loss * completion_mask).sum() / (batch_size * self.max_completion_length)
        elif self.loss_type in ['cispo', 'dapo']:
            normalizer = inputs['num_items_in_batch'] / self.accelerator.num_processes
            loss = (per_token_loss * completion_mask).sum() / normalizer
        else:
            raise ValueError(f'Unknown loss type: {self.loss_type}')

        completion_token_count = completion_mask.sum().clamp(min=1.0)

        def masked_batch_mean(x):
            if x.shape[1] == 1:
                return x.mean()
            else:
                return (x * completion_mask).sum() / completion_token_count

        metrics_data = {
            'mode': mode,
            'entropy': entropy_metrics,
            'completion_mask': completion_mask,
            'completion_token_count': completion_token_count,
        }

        if per_token_kl is not None:
            mean_kl = masked_batch_mean(per_token_kl)
            metrics_data['kl'] = self.accelerator.gather_for_metrics(mean_kl).nanmean().item()

        if rollout_correction_metrics:
            metrics_data['rollout_correction'] = rollout_correction_metrics

        if self.loss_type == 'cispo':
            is_cispo_clipped = (coef_1 > self.epsilon_high) & (advantages.unsqueeze(1) > 0)
            cispo_clip_ratio = masked_batch_mean(is_cispo_clipped.float())
            gathered_cispo_clip_ratio = self.accelerator.gather_for_metrics(cispo_clip_ratio)
            metrics_data['clipping'] = {'cispo_clip_ratio': gathered_cispo_clip_ratio.nanmean().item()}
        elif self.loss_type == 'sapo':
            pass
        else:
            is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (advantages.unsqueeze(1) < 0)
            is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (advantages.unsqueeze(1) > 0)
            is_region_clipped = is_low_clipped | is_high_clipped

            low_clip = masked_batch_mean(is_low_clipped.float())
            high_clip = masked_batch_mean(is_high_clipped.float())
            clip_ratio = masked_batch_mean(is_region_clipped.float())

            gathered_low_clip = self.accelerator.gather_for_metrics(low_clip)
            gathered_high_clip = self.accelerator.gather_for_metrics(high_clip)
            gathered_clip_ratio = self.accelerator.gather_for_metrics(clip_ratio)

            metrics_data['clipping'] = {
                'low_clip_mean': gathered_low_clip.nanmean().item(),
                'low_clip_min': nanmin(gathered_low_clip).item(),
                'high_clip_mean': gathered_high_clip.nanmean().item(),
                'high_clip_max': nanmax(gathered_high_clip).item(),
                'region_clip_mean': gathered_clip_ratio.nanmean().item()
            }
        if mode == 'train' and self.chord_sft_iterator is not None:
            loss = compute_chord_loss(self, grpo_loss=loss)

        return loss, metrics_data
