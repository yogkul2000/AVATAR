import os
import json
import time
import re
import torch
import logging
import string
from transformers import (
    Qwen2_5OmniForConditionalGeneration,
    Qwen2_5OmniProcessor,
    Qwen2_5OmniThinkerForConditionalGeneration,
)
from qwen_omni_utils import process_mm_info
import torch.multiprocessing as mp
from multiprocessing import Manager
from filelock import FileLock
import numpy as np
from tqdm import tqdm

logging.getLogger().setLevel(logging.ERROR)


MVBENCH_ROOT = "/yourPath/MVBench"
JSON_ROOT = os.path.join(MVBENCH_ROOT, "json")
VIDEO_ROOT = os.path.join(MVBENCH_ROOT, "video")

MODEL_PATH = ""
OUT_DIR = "./exp_results/mvbench_avatar"
NUM_GPUS = 4
USE_AUDIO = True
BASE_MODEL = False

DATA_LIST = {
    "object_interaction": "star/star/Charades_v1_480",
    "action_sequence": "star/star/Charades_v1_480",
    "action_prediction": "star/star/Charades_v1_480",
    "action_localization": "sta/sta/sta_video",
    "moving_count": "clevrer/clevrer/video_validation",
    "fine_grained_pose": "nturgbd",
    "character_order": "perception/perception/videos",
    "object_shuffle": "perception/perception/videos",
    "egocentric_navigation": "vlnqa/vlnqa",
    "moving_direction": "clevrer/clevrer/video_validation",
    "episodic_reasoning": "tvqa/tvqa/frames_fps3_hq",
    "fine_grained_action": "Moments_in_Time_Raw/Moments_in_Time_Raw/videos",
    "scene_transition": "scene_qa/scene_qa/video",
    "state_change": "perception/perception/videos",
    "moving_attribute": "clevrer/clevrer/video_validation",
    "action_antonym": "ssv2_video/ssv2_video",
    "unexpected_action": "FunQA_test/FunQA_test/test",
    "counterfactual_inference": "clevrer/clevrer/video_validation",
    "object_existence": "clevrer/clevrer/video_validation",
    "action_count": "perception/perception/videos",
}


def get_video_path(doc, sub_task):
    dataset_folder = DATA_LIST.get(sub_task)
    if not dataset_folder:
        raise ValueError(f"Task '{sub_task}' not found in DATA_LIST.")

    video_path = os.path.join(VIDEO_ROOT, dataset_folder, doc["video"])
    if os.path.exists(video_path):
        return video_path

    base_folder_name = os.path.basename(dataset_folder)
    if base_folder_name in ["clevrer", "star"]:
        alt_path = os.path.join(VIDEO_ROOT, "data0613", dataset_folder, doc["video"])
        if os.path.exists(alt_path):
            return alt_path

    direct_path = os.path.join(VIDEO_ROOT, base_folder_name, doc["video"])
    if os.path.exists(direct_path):
        return direct_path

    return video_path


def format_question_prompt(doc):
    option_prompt = ""
    option_letters = string.ascii_uppercase
    for char_index, option in enumerate(doc["candidates"]):
        option_prompt += f"({option_letters[char_index]}) {option}\n"

    return f"Question: {doc['question']}\nOptions:\n{option_prompt}Your answer should be a single capital letter representing your choice."


def get_gt_answer(doc):
    try:
        idx = doc["candidates"].index(doc["answer"])
        return string.ascii_uppercase[idx]
    except (ValueError, IndexError):
        return None


def extract_prediction(text):
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


def process_samples_on_gpu(
    gpu_id, task_name, data_subset, result_queue, progress_queue
):
    torch.cuda.set_device(gpu_id)
    device = f"cuda:{gpu_id}"

    print(f"[GPU {gpu_id}] Loading model on {device}...")
    if BASE_MODEL:
        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            torch_dtype="auto",
            device_map=device,
            attn_implementation="flash_attention_2",
            enable_audio_output=False,
        ).eval()
        model.disable_talker()
    else:
        model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            device_map=device,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        ).eval()
    processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_PATH)
    print(
        f"[GPU {gpu_id}] Model ready. Processing {len(data_subset)} samples for task: {task_name}"
    )

    local_results = []
    for sample in data_subset:
        use_audio_flag = True
        try:
            video_path = get_video_path(sample, task_name)
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file not found: {video_path}")

            prompt = format_question_prompt(sample)
            content_items = [
                {"type": "video", "video": video_path},
                {"type": "text", "text": prompt},
            ]
            conv = [{"role": "user", "content": content_items}]

            prompt_text = processor.apply_chat_template(
                conv, add_generation_prompt=True, tokenize=False
            )

            try:
                audios, images, videos = process_mm_info(
                    conv, use_audio_in_video=use_audio_flag
                )
            except Exception:
                use_audio_flag = False
                audios, images, videos = process_mm_info(conv, use_audio_in_video=False)

            inputs = processor(
                text=prompt_text,
                audio=audios,
                images=images,
                videos=videos,
                return_tensors="pt",
                padding=True,
                use_audio_in_video=use_audio_flag,
            ).to(device)

            with torch.no_grad():
                out_ids = model.generate(
                    **inputs,
                    use_audio_in_video=use_audio_flag,
                    do_sample=False,
                    max_new_tokens=4096,
                )

            reply = processor.batch_decode(out_ids, skip_special_tokens=True)[0]

            pred_answer = extract_prediction(reply)
            gt_answer = get_gt_answer(sample)
            is_correct = pred_answer == gt_answer

            result = {
                "task": task_name,
                "video": sample["video"],
                "gt": gt_answer,
                "prediction": pred_answer,
                "correct": is_correct,
            }
            local_results.append(result)

        except Exception as e:
            print(f"[GPU {gpu_id}] ERROR processing sample for task {task_name}: {e}")
            result = {
                "task": task_name,
                "video": sample.get("video"),
                "error": str(e),
                "correct": False,
            }
            local_results.append(result)

        finally:
            progress_queue.put(1)

    result_queue.put(local_results)
    print(f"[GPU {gpu_id}] Completed processing for task: {task_name}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    n_gpus = min(NUM_GPUS, torch.cuda.device_count())
    print(f"🖥️  Using {n_gpus} GPUs for MVBench evaluation.")

    all_task_results = {}
    task_files = sorted([f for f in os.listdir(JSON_ROOT) if f.endswith(".json")])

    for task_file in task_files:
        task_name = task_file.replace(".json", "")
        json_path = os.path.join(JSON_ROOT, task_file)

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        print(
            f"\n{'='*20}\n🚀 Starting evaluation for task: {task_name} ({len(data)} samples)\n{'='*20}"
        )

        out_path_task = os.path.join(OUT_DIR, f"{task_name}_results.jsonl")
        if os.path.exists(out_path_task):
            print(f"⚡️ Found existing results for {task_name}, skipping.")
            task_results = []
            with open(out_path_task, "r", encoding="utf-8") as f_exist:
                for line in f_exist:
                    task_results.append(json.loads(line))
            all_task_results[task_name] = task_results
            continue

        data_splits = np.array_split(data, n_gpus)
        data_splits = [list(split) for split in data_splits if split.size > 0]

        mp.set_start_method("spawn", force=True)
        manager = Manager()
        result_queue = manager.Queue()
        progress_queue = manager.Queue()

        processes = []
        for i in range(len(data_splits)):
            p = mp.Process(
                target=process_samples_on_gpu,
                args=(i, task_name, data_splits[i], result_queue, progress_queue),
            )
            p.start()
            processes.append(p)

        pbar = tqdm(total=len(data), desc=f"Processing {task_name}")
        for _ in range(len(data)):
            progress_queue.get()
            pbar.update(1)
        pbar.close()

        for p in processes:
            p.join()

        task_results = []
        while not result_queue.empty():
            task_results.extend(result_queue.get())

        all_task_results[task_name] = task_results

        with open(out_path_task, "w", encoding="utf-8") as f_out:
            for res in task_results:
                f_out.write(json.dumps(res, ensure_ascii=False) + "\n")
        print(f"💾 Results for {task_name} saved to {out_path_task}")

    print(f"\n\n{'='*20}\n📊 MVBench Final Results Summary\n{'='*20}")

    final_scores = {}
    total_correct = 0
    total_samples = 0

    for task_name, results in sorted(all_task_results.items()):
        correct = sum(1 for r in results if r.get("correct", False))
        total = len(results)
        accuracy = (correct / total * 100) if total > 0 else 0
        final_scores[task_name] = accuracy
        total_correct += correct
        total_samples += total
        print(f"  - {task_name:<25} Accuracy: {accuracy:.2f}% ({correct}/{total})")

    overall_accuracy = (total_correct / total_samples * 100) if total_samples > 0 else 0
    print(f"\n🏆 Overall Average Accuracy: {overall_accuracy:.2f}%")
    print(f"   (Total Correct: {total_correct}, Total Samples: {total_samples})")


if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed = (time.time() - start_time) / 60
    print(f"\n⏱️  Total evaluation time: {elapsed:.1f} minutes")
