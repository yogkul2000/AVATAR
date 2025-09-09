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


import sys
import argparse
import json
from tqdm import tqdm
import os
import re


def load_jsonl_data(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]
        return data
    except FileNotFoundError:
        print(f"Error: File not found at '{file_path}'")
        return None
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format in '{file_path}': {e}")
        return None


def evaluate_answer(model_answer, correct_answer_letter):
    if not model_answer:
        return False
    return model_answer.strip().upper() == correct_answer_letter.strip().upper()


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


def run_omnibench_evaluation(model, processor, args):

    data = load_jsonl_data(args.json_file_path)
    if not data:
        print(f"Failed to load data from {args.json_file_path}. Exiting.")
        return

    stats = {
        "total_questions": len(data),
        "correct_answers": 0,
        "failed_processing": 0,
        "by_task_type": {},
        "by_audio_type": {},
    }
    all_results_for_json = []

    for item in data:
        task_type = item.get("task type")
        audio_type = item.get("audio type")
        if task_type and task_type not in stats["by_task_type"]:
            stats["by_task_type"][task_type] = {"count": 0, "correct": 0}
        if audio_type and audio_type not in stats["by_audio_type"]:
            stats["by_audio_type"][audio_type] = {"count": 0, "correct": 0}

    print(f"Starting evaluation on {args.json_file_path}...")
    print(f"Using media base directory: {args.media_base_dir}")

    for item in tqdm(data, desc="Evaluating OmniBench Questions"):
        question = item.get("question")
        options = item.get("options")
        correct_answer_text = item.get("answer")
        audio_filename = item.get("audio_path")
        image_filename = item.get("image_path")
        task_type = item.get("task type")
        audio_type = item.get("audio type")
        index = item.get("index")

        if not all(
            [
                question,
                options,
                correct_answer_text,
                audio_filename,
                image_filename,
                task_type,
                audio_type,
            ]
        ):
            print(f"\nWarning: Skipping item with index {index} due to missing fields.")
            stats["failed_processing"] += 1
            continue

        full_image_path = os.path.join(args.media_base_dir, "image", image_filename)
        full_audio_path = os.path.join(args.media_base_dir, "audio", audio_filename)

        if not os.path.exists(full_image_path):
            print(
                f"\nWarning: Image file not found for index {index} at path {full_image_path}. Skipping."
            )
            stats["failed_processing"] += 1
            continue
        if not os.path.exists(full_audio_path):
            print(
                f"\nWarning: Audio file not found for index {index} at path {full_audio_path}. Skipping."
            )
            stats["failed_processing"] += 1
            continue

        try:
            correct_letter = chr(ord("A") + options.index(correct_answer_text))
        except ValueError:
            print(
                f"\nWarning: Correct answer text '{correct_answer_text}' not found in options for index {index}. Skipping."
            )
            stats["failed_processing"] += 1
            continue

        formatted_choices = "\n".join(
            [f"{chr(ord('A') + i)}. {opt}" for i, opt in enumerate(options)]
        )

        prompt_text = f"""Your task is to answer the multiple-choice question based on the provided image and audio. Select the single most accurate option.

Question: {question}
Choices:
{formatted_choices}

Your answer should be a capital letter representing your choice: A, B, C, or D. Don't generate any other text. Return your choice between the <answer> tags, like this: <answer>A</answer>.
"""
        system_prompt = "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."
        conversation = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": full_image_path},
                    {"type": "audio", "audio": full_audio_path},
                    {"type": "text", "text": prompt_text},
                ],
            },
        ]

        model_answer = None
        decoded_text = ""
        try:
            text = processor.apply_chat_template(
                conversation, add_generation_prompt=True, tokenize=False
            )

            audios, images, videos = process_mm_info(
                conversation, use_audio_in_video=False
            )

            inputs = processor(
                text=text,
                audio=audios,
                images=images,
                videos=videos,
                return_tensors="pt",
                padding=True,
                use_audio_in_video=False,
            )
            inputs = inputs.to(model.device).to(model.dtype)

            gen_out = model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False,
                eos_token_id=processor.tokenizer.eos_token_id,
            )

            input_len = inputs["input_ids"].shape[1]
            text_ids = gen_out[:, input_len:]
            decoded_text = processor.batch_decode(
                text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

            model_answer = extract_final_answer(decoded_text)
            if model_answer.upper() not in ["A", "B", "C", "D"]:
                print(
                    f"\nWarning: Could not extract a valid A/B/C/D answer for index {index}. Raw output: '{decoded_text}'"
                )

        except Exception as e:
            print(f"\nError processing item with index {index}: {e}")
            stats["failed_processing"] += 1
            continue

        is_correct = evaluate_answer(model_answer, correct_letter)

        if is_correct:
            stats["correct_answers"] += 1
            if task_type in stats["by_task_type"]:
                stats["by_task_type"][task_type]["correct"] += 1
            if audio_type in stats["by_audio_type"]:
                stats["by_audio_type"][audio_type]["correct"] += 1

        if task_type in stats["by_task_type"]:
            stats["by_task_type"][task_type]["count"] += 1
        if audio_type in stats["by_audio_type"]:
            stats["by_audio_type"][audio_type]["count"] += 1

        all_results_for_json.append(
            {
                "index": index,
                "question": question,
                "response": decoded_text,
                "parsed_answer": model_answer,
                "is_correct": is_correct,
            }
        )

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(all_results_for_json, f, indent=4)
    print(f"\nInference results saved to {args.output_file}")

    print("\n--- OmniBench Evaluation Summary ---")
    valid_questions = stats["total_questions"] - stats["failed_processing"]
    if valid_questions > 0:
        print(
            f"Overall Accuracy: {stats['correct_answers']}/{valid_questions} = {stats['correct_answers'] / valid_questions:.2%}"
        )
    else:
        print("Overall Accuracy: 0/0 = N/A (No questions processed successfully)")
    print(
        f"(Total items: {stats['total_questions']}, Skipped/Failed items: {stats['failed_processing']})"
    )

    print("\n--- Accuracy by Task Type ---")
    for task_type, values in sorted(stats["by_task_type"].items()):
        if values["count"] == 0:
            print(f"{task_type}: 0/0 = N/A")
        else:
            print(
                f"{task_type}: {values['correct']}/{values['count']} = {values['correct'] / values['count']:.2%}"
            )

    print("\n--- Accuracy by Audio Type ---")
    for audio_type, values in sorted(stats["by_audio_type"].items()):
        if values["count"] == 0:
            print(f"{audio_type}: 0/0 = N/A")
        else:
            print(
                f"{audio_type}: {values['correct']}/{values['count']} = {values['correct'] / values['count']:.2%}"
            )

    print("\n--- Evaluation Complete ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a Qwen Omni model on the OmniBench dataset."
    )

    parser.add_argument(
        "--media_base_dir",
        type=str,
        default="/scratch/ykulka10/OmniBench/mm_data",
        help='Base directory containing the "audio" and "image" subfolders for OmniBench.',
    )
    parser.add_argument(
        "--json_file_path",
        type=str,
        default="/scratch/ykulka10/OmniBench/dataset/batch-5_1142_20240817.jsonl",
        help="Path to the OmniBench JSONL file.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="inference_output_omnibench.json",
        help="Path to save the JSON file with model outputs.",
    )
    parser.add_argument(
        "--model_name_or_path", type=str, help="Hugging Face model name or local path."
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
        help='Attention implementation. "None" to use default.',
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

    try:
        """model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            args.model_name_or_path,
            device_map=args.device,
            torch_dtype=torch_dtype,
            attn_implementation=attn_impl,
            enable_audio_output=False
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
        print(f"Fatal Error: Could not load the model or processor. Details: {e}")
        sys.exit(1)

    run_omnibench_evaluation(model, processor, args)
