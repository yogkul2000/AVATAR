import json
import math
import os
import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional

import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoConfig, AutoModel, AutoTokenizer

from swift.rewards import ORM, orms
from swift.utils import get_logger

logger = get_logger()


class CustomAccuracyReward(ORM):
    def __call__(self, completions, solution, **kwargs) -> List[float]:
        problem_types = kwargs.get('problem_type', [])
        rewards = []

        for idx, (completion, sol) in enumerate(zip(completions, solution)):
            completion_answer = self._extract_answer(completion)
            solution_answer = self._extract_answer(sol)

            problem_type = problem_types[idx] if idx < len(problem_types) else "default"

            if problem_type == "multiple_choice":
                reward = 1.0 if completion_answer.strip().upper() == solution_answer.strip().upper() else 0.0
            elif problem_type == "numerical":
                reward = self._evaluate_numerical(completion_answer, solution_answer)
            elif problem_type in {"ocr", "text_recognition"}:
                reward = self._evaluate_ocr(completion_answer, solution_answer)
            elif problem_type in {"free_form", "descriptive"}:
                reward = self._evaluate_free_form(completion_answer, solution_answer)
            elif problem_type == "code":
                reward = self._evaluate_code(completion_answer, solution_answer)
            else:
                reward = 1.0 if completion_answer.strip() == solution_answer.strip() else 0.0

            rewards.append(reward)

        return rewards

    def _extract_answer(self, text: str) -> str:
        pattern = r'<answer>\s*(.*?)\s*</answer>'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return text.strip()

    def _normalize_number(self, num_str: str) -> Optional[float]:
        try:
            clean_str = re.sub(r'[^\d.+-]', '', num_str)
            return float(clean_str)
        except (ValueError, TypeError):
            return None

    def _evaluate_numerical(self, completion: str, solution: str) -> float:
        comp_num = self._normalize_number(completion)
        sol_num = self._normalize_number(solution)

        if comp_num is None or sol_num is None:
            return 0.0

        if abs(sol_num) < 1e-10:
            absolute_error = abs(comp_num - sol_num)
            return 1.0 if absolute_error < 1e-5 else 0.0
        else:
            relative_error = abs(comp_num - sol_num) / abs(sol_num)
            return 1.0 if relative_error < 1e-3 else 0.0

    def _evaluate_ocr(self, completion: str, solution: str) -> float:
        def word_error_rate(reference, hypothesis):
            ref_words = reference.lower().split()
            hyp_words = hypothesis.lower().split()

            m, n = len(ref_words), len(hyp_words)
            dp = [[0] * (n + 1) for _ in range(m + 1)]

            for i in range(m + 1):
                dp[i][0] = i
            for j in range(n + 1):
                dp[0][j] = j

            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if ref_words[i - 1] == hyp_words[j - 1]:
                        dp[i][j] = dp[i - 1][j - 1]
                    else:
                        dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

            error_rate = dp[m][n] / max(1, m)
            return 1.0 - min(1.0, error_rate)

        return word_error_rate(solution, completion)

    def _evaluate_free_form(self, completion: str, solution: str) -> float:
        try:
            from rouge_score import rouge_scorer
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            scores = scorer.score(solution, completion)
            avg_score = (scores['rouge1'].fmeasure + scores['rouge2'].fmeasure + scores['rougeL'].fmeasure) / 3
            return float(avg_score)
        except ImportError:
            return self._simple_text_similarity(completion, solution)

    def _simple_text_similarity(self, text1: str, text2: str) -> float:
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        overlap = len(words1.intersection(words2))
        union = len(words1.union(words2))

        return overlap / union if union > 0 else 0.0

    def _evaluate_code(self, completion: str, solution: str) -> float:
        def normalize_code(code: str) -> str:
            code = re.sub(r'#.*$', '', code, flags=re.MULTILINE)
            code = re.sub(r'\s+', ' ', code).strip()
            return code

        norm_completion = normalize_code(completion)
        norm_solution = normalize_code(solution)
        return 1.0 if norm_completion == norm_solution else 0.0


class FormatRewardORM(ORM):
    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        def _clean(text: str) -> str:
            text = re.sub(r"<\|im_start\|>|<\|im_end\|>", "", text)
            return text.strip()

        pattern = r"^\s*<think>[\s\S]*?</think>\s*<answer>[\s\S]*?</answer>\s*$"

        out = []
        for content in completions:
            clean = _clean(content)
            out.append(1.0 if re.fullmatch(pattern, clean, re.DOTALL) else 0.0)
        return out


# -----------------------------
# Stage 2 reward: Rself
# -----------------------------


class SelfRewardORM(ORM):
    def __call__(self, completions, prompt_id, **kwargs) -> List[float]:
        answers = [self._extract_answer(c) for c in completions]
        groups: Dict[str, List[int]] = defaultdict(list)
        for i, pid in enumerate(prompt_id):
            groups[pid].append(i)

        rewards = [0.0] * len(completions)
        for pid, idxs in groups.items():
            votes = Counter([answers[i] for i in idxs])
            majority = votes.most_common(1)[0][0]
            for i in idxs:
                rewards[i] = 1.0 if answers[i] == majority else 0.0
        return rewards

    @staticmethod
    def _extract_answer(text: str) -> str:
        match = re.search(r'<answer>\s*(.*?)\s*</answer>', text, re.DOTALL)
        return match.group(1).strip() if match else text.strip()



IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))

    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=8):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    if bound:
        start, end = bound
    else:
        start, end = -100000, 100000

    start_idx = max(0, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([
        int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
        for idx in range(num_segments)
    ])

    pixel_values_list = []
    num_patches_list = []
    transform = build_transform(input_size=input_size)

    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
        img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)

    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list


def split_model(model_path):
    device_map = {}
    world_size = torch.cuda.device_count()
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    num_layers = config.llm_config.num_hidden_layers
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for _ in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.model.rotary_emb'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

    return device_map


class InternVL3Judge:
    def __init__(self):
        model_path = os.getenv('INTERNVL3_PATH', 'OpenGVLab/InternVL3-2B')
        device_map = split_model(model_path)
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            load_in_8bit=False,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
            device_map=device_map).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
        self.generation_config = dict(max_new_tokens=512, do_sample=False)

    def score(self, video_path: str, prompt: str) -> Dict[str, float]:
        pixel_values, num_patches_list = load_video(video_path, num_segments=8, max_num=1)
        pixel_values = pixel_values.to(torch.bfloat16).cuda()
        video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
        question = video_prefix + prompt
        response = self.model.chat(
            self.tokenizer,
            pixel_values,
            question,
            self.generation_config,
            num_patches_list=num_patches_list,
            history=None,
            return_history=False,
        )
        try:
            return json.loads(response)
        except Exception:
            logger.warning(f'InternVL3 judge output parse failed: {response[:200]}')
            return {}


_JUDGE = None


def _get_judge():
    global _JUDGE
    if _JUDGE is None:
        _JUDGE = InternVL3Judge()
    return _JUDGE


JUDGE_PROMPT = (
    "You are a meticulous and precise Audio-Visual Grounding Evaluator. Your task is to provide a granular, "
    "stepwise evaluation of a model's attempt to localize an object based on audio and visual cues.\n\n"
    "Given the following inputs, you will score the model's reasoning and final answer on several criteria.\n\n"
    "[User Query]: {query}\n"
    "[Audio Caption]: {audio}\n"
    "[Ground Truth Answer]: {gt}\n"
    "[Model's Reasoning]: {reason}\n"
    "[Model's Final Answer]: {answer}\n\n"
    "Your Task:\n"
    "Based on your analysis of all the provided information, you must evaluate the model's performance on the "
    "following four criteria. Provide your judgment as a single JSON object with a score from 0.0 (complete failure) "
    "to 1.0 (perfect) for each criterion.\n\n"
    "1. Audio Cue Grounding (audio_grounding_score): Did the model's reasoning correctly identify key descriptive "
    "words in the Audio Caption that point to the target?\n"
    "2. Visual Object Identification (visual_id_score): Based on the audio cue, did the model's reasoning correctly "
    "identify the corresponding visual object in the scene?\n"
    "3. Location Accuracy (location_acc_score): How accurate is the final spatial location in the Model's Final "
    "Answer compared to the Ground Truth Answer?\n"
    "4. Caption Correctness (caption_corr_score): How well does the model's final textual description match the "
    "Ground Truth Answer?\n\n"
    "Output Format:\n"
    "Return ONLY a single JSON object with your scores.\n"
    "Example Output:\n"
    "{"
    "\"audio_grounding_score\": 0.9, "
    "\"visual_id_score\": 1.0, "
    "\"location_acc_score\": 1.0, "
    "\"caption_corr_score\": 0.8"
    "}"
)


class InternVL3JudgeReward(ORM):
    def __call__(self, completions, solution, question, audio_cues, videos, **kwargs) -> List[float]:
        rewards = []
        judge = _get_judge()
        for comp, gt, q, audio, vid_list in zip(completions, solution, question, audio_cues, videos):
            think = self._extract_tag(comp, 'think')
            answer = self._extract_tag(comp, 'answer')
            prompt = JUDGE_PROMPT.format(
                query=q,
                audio=audio if audio is not None else '',
                gt=gt,
                reason=think,
                answer=answer,
            )
            video_path = vid_list[0] if isinstance(vid_list, list) and vid_list else vid_list
            scores = judge.score(video_path, prompt)
            score = self._aggregate(scores)
            rewards.append(score)
        return rewards

    @staticmethod
    def _extract_tag(text: str, tag: str) -> str:
        match = re.search(rf'<{tag}>\s*(.*?)\s*</{tag}>', text, re.DOTALL)
        return match.group(1).strip() if match else ''

    @staticmethod
    def _aggregate(scores: Dict[str, float]) -> float:
        keys = ['audio_grounding_score', 'visual_id_score', 'location_acc_score', 'caption_corr_score']
        vals = [scores.get(k) for k in keys if k in scores]
        if not vals:
            return 0.0
        return float(sum(vals) / len(vals))


# Register reward functions
orms['custom_accuracy_reward'] = CustomAccuracyReward
orms['custom_format_reward'] = FormatRewardORM
orms['self_reward'] = SelfRewardORM
orms['judge_reward'] = InternVL3JudgeReward
