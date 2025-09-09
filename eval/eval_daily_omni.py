import torch
from transformers import (
    Qwen2_5OmniForConditionalGeneration,
    Qwen2_5OmniProcessor,
    Qwen2_5OmniThinkerForConditionalGeneration,
)

try:
    from qwen_omni_utils import process_mm_info
except ImportError:
    print("Warning: qwen_omni_utils not found. Multimedia processing might fail.")

    def process_mm_info(*args, **kwargs):
        print("Error: process_mm_info is not available.")
        return None, None, None


from typing import List, Dict, Any
import sys
import argparse
import json
from tqdm import tqdm
import os
import re
import av


def load_json_data(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Error: File not found at '{file_path}'")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in '{file_path}'")
        return None


def get_video_path(video_id, base_path):
    if not base_path:
        raise ValueError("Video base path cannot be empty.")
    return os.path.join(base_path, video_id, f"{video_id}_video.mp4")


def evaluate_answer(model_answer, correct_answer):
    if not model_answer:
        return False
    return model_answer.strip().upper() == correct_answer.strip().upper()


def _check_if_video_has_audio(video_path):
    try:
        with av.open(video_path) as container:
            return len(container.streams.audio) > 0
    except Exception:
        return False


def extract_final_answer(text):
    if "assistant\n" in text:
        assistant_response = text.split("assistant\n", 1)[-1]
    else:
        assistant_response = text

    pattern = r"<answer>\s*(.*?)\s*</answer>"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()

    letter_matches = re.findall(r"\b([A-Z])\b", assistant_response)
    if letter_matches:
        return letter_matches[-1]

    any_letter_match = re.search(r"([A-Z])", assistant_response)
    if any_letter_match:
        return any_letter_match.group(1)

    return ""


def test_all_questions(model, processor, args):
    qa_type_count = {}
    qa_type_correct = {}
    video_cat_count = {}
    video_cat_correct = {}

    data = load_json_data(args.json_file_path)
    if not data:
        print(f"Failed to load data from {args.json_file_path}. Exiting.")
        return

    total_questions = len(data)
    correct_answers = 0
    failed = 0
    VIDEO_CAT = []
    QA_TYPE = []

    for item in data:
        video_category = item.get("video_category")
        qa_type = item.get("Type")
        if video_category and video_category not in VIDEO_CAT:
            VIDEO_CAT.append(video_category)
        if qa_type and qa_type not in QA_TYPE:
            QA_TYPE.append(qa_type)

    VIDEO_CAT.sort()
    QA_TYPE.sort()

    for qa_type in QA_TYPE:
        qa_type_count[qa_type] = 0
        qa_type_correct[qa_type] = 0
    for video_category in VIDEO_CAT:
        video_cat_count[video_category] = 0
        video_cat_correct[video_category] = 0

    total_questions = len(data)
    correct_answers = 0
    failed = 0
    qa_duration_count = {"30s": 0, "60s": 0}
    qa_duration_correct = {"30s": 0, "60s": 0}

    print(f"Starting evaluation on {args.json_file_path}...")
    print(f"Using video base directory: {args.video_base_dir}")
    print(f"Use audio input: {args.use_audio_in_video}")

    for item in tqdm(data, desc="Evaluating Questions"):
        question = item.get("Question")
        choices = item.get("Choice")
        correct_answer = item.get("Answer")
        video_id = item.get("video_id")
        qa_type = item.get("Type")
        video_category = item.get("video_category")
        video_duration = item.get("video_duration")

        if not all(
            [
                question,
                choices,
                correct_answer,
                video_id,
                qa_type,
                video_category,
                video_duration,
            ]
        ):
            print(
                f"\nWarning: Skipping item due to missing fields. Item Index: {data.index(item)}, Video ID: {video_id or 'Unknown'}"
            )
            failed += 1
            continue

        try:
            video_path = get_video_path(video_id, args.video_base_dir)
            if not os.path.exists(video_path):
                print(
                    f"\nWarning: Video file not found for ID {video_id} at path {video_path}. Skipping."
                )
                failed += 1
                continue
        except ValueError as e:
            print(
                f"\nError constructing video path: {e}. Skipping item for video ID {video_id}"
            )
            failed += 1
            continue

        prompt = f"""
Your task is to accurately answer multiple-choice questions based on the given video.
Select the single most accurate answer from the given choices.
Question: {question}
Choices: {choices}
Your answer should be a capital letter representing your choice: A, B, C, or D. Don't generate any other text.
"""

        system_prompt = "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."
        conversation = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": video_path},  # Video first
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        model_answer = None
        try:
            has_audio = _check_if_video_has_audio(video_path)
            should_process_audio = args.use_audio_in_video and has_audio
            if args.use_audio_in_video and not has_audio:
                pass

            text = processor.apply_chat_template(
                conversation, add_generation_prompt=True, tokenize=False
            )

            audios, images, videos = process_mm_info(
                conversation, use_audio_in_video=should_process_audio
            )

            inputs = processor(
                text=text,
                audio=audios,
                images=images,
                videos=videos,
                return_tensors="pt",
                padding=True,
            )
            inputs = inputs.to(model.device).to(model.dtype)

            gen_out = model.generate(
                **inputs,
                use_audio_in_video=should_process_audio,
                max_new_tokens=4096,
                do_sample=False,
                eos_token_id=processor.tokenizer.eos_token_id,
            )

            input_len = inputs["input_ids"].shape[1]
            text_ids = gen_out[:, input_len:]
            decoded_text = processor.batch_decode(
                text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            print(
                f"\nDecoded text for video {video_id} (Index: {data.index(item)}): '{decoded_text}'"
            )

            model_answer = extract_final_answer(decoded_text)
            if model_answer.upper() not in ["A", "B", "C", "D"]:
                print(
                    f"\nWarning: Could not extract a valid A/B/C/D answer for video {video_id}. Raw output: '{decoded_text}'"
                )

        except Exception as e:
            print(
                f"\nError processing video {video_id} (Index: {data.index(item)}): {e}"
            )
            failed += 1
            continue

        is_correct = evaluate_answer(model_answer, correct_answer)

        if qa_type in qa_type_count:
            qa_type_count[qa_type] += 1
            if is_correct:
                qa_type_correct[qa_type] += 1
        if video_category in video_cat_count:
            video_cat_count[video_category] += 1
            if is_correct:
                video_cat_correct[video_category] += 1
        if video_duration in qa_duration_count:
            qa_duration_count[video_duration] += 1
            if is_correct:
                qa_duration_correct[video_duration] += 1
        if is_correct:
            correct_answers += 1

    print("\n--- Evaluation Summary ---")
    valid_questions = total_questions - failed
    if valid_questions > 0:
        print(
            f"Overall Accuracy: {correct_answers}/{valid_questions} = {correct_answers / valid_questions:.2%}"
        )
    else:
        print("Overall Accuracy: 0/0 = N/A (No questions processed successfully)")
    print(f"(Total items: {total_questions}, Skipped/Failed items: {failed})")

    print("\n--- Accuracy by QA Type ---")
    for qa_type in QA_TYPE:
        count = qa_type_count.get(qa_type, 0)
        correct = qa_type_correct.get(qa_type, 0)
        if count == 0:
            print(f"{qa_type}: 0/0 = N/A")
        else:
            print(f"{qa_type}: {correct}/{count} = {correct / count:.2%}")

    print("\n--- Accuracy by Video Category ---")
    for video_category in VIDEO_CAT:
        count = video_cat_count.get(video_category, 0)
        correct = video_cat_correct.get(video_category, 0)
        if count == 0:
            print(f"{video_category}: 0/0 = N/A")
        else:
            print(f"{video_category}: {correct}/{count} = {correct / count:.2%}")

    print("\n--- Accuracy by Video Duration ---")
    for duration in ["30s", "60s"]:
        count = qa_duration_count.get(duration, 0)
        correct = qa_duration_correct.get(duration, 0)
        if count != 0:
            print(f"{duration} Duration: {correct}/{count} = {correct / count:.2%}")
        else:
            print(f"{duration} Duration: 0/0 = N/A")

    print(f"\nTotal items failed during processing: {failed}")
    print("--- Evaluation Complete ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate Qwen2.5-Omni on a video QA dataset."
    )

    parser.add_argument(
        "--video_base_dir",
        type=str,
        default="",
        help="Base directory containing video folders.",
    )
    parser.add_argument(
        "--json_file_path",
        type=str,
        default="",
        help="Path to the JSON file containing QA pairs.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Hugging Face model name or path to load.",
    )

    parser.add_argument(
        "--use_audio_in_video",
        default=True,
        help="Process audio input from the video along with visual frames.",
    )
    parser.add_argument(
        "--processor_name_or_path",
        type=str,
        default=None,
        help="Hugging Face processor name or path. Defaults to model_name_or_path if not set.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help='Device map for loading the model (e.g., "auto", "cuda:0").',
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="bf16",
        choices=["fp32", "fp16", "bf16"],
        help="Precision for model loading",
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="flash_attention_2",
        choices=["flash_attention_2", "sdpa", "eager", "None"],
        help='Attention implementation (set to "None" to disable or use default).',
    )
    parser.add_argument(
        "--disable_audio_output",
        default=True,
        help="Disable audio output generation capability during model loading.",
    )

    args = parser.parse_args()

    if args.processor_name_or_path is None:
        args.processor_name_or_path = args.model_name_or_path

    dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    torch_dtype = dtype_map.get(args.precision, torch.bfloat16)

    attn_impl = args.attn_implementation if args.attn_implementation != "None" else None

    print(
        f"Loading model: {args.model_name_or_path} with precision {args.precision}..."
    )
    print(f"Attention implementation: {attn_impl}")
    print(f"Device map: {args.device}")
    print(f"Enable audio output: {not args.disable_audio_output}")

    try:
        """model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            args.model_name_or_path,
            device_map=args.device,
            torch_dtype=torch_dtype,
            attn_implementation=attn_impl,
            # For `transformers` the argument is just `enable_audio_output`
            # The generate() function has `return_audio`
            enable_audio_output=False,
        ).eval()"""

        model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            args.model_name_or_path,
            device_map=args.device,
            torch_dtype=torch_dtype,
            attn_implementation=attn_impl,
        ).eval()

        print(f"Loading processor: {args.processor_name_or_path}...")
        processor = Qwen2_5OmniProcessor.from_pretrained(args.processor_name_or_path)

    except Exception as e:
        print(f"Error loading model or processor: {e}")
        sys.exit(1)

    test_all_questions(model, processor, args)
