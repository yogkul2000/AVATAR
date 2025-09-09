import os, json, time, re, torch, csv
from transformers import (
    Qwen2_5OmniForConditionalGeneration,
    Qwen2_5OmniProcessor,
    Qwen2_5OmniThinkerForConditionalGeneration,
)
from qwen_omni_utils import process_mm_info
import logging
import torch.multiprocessing as mp
from multiprocessing import Manager
from filelock import FileLock
import numpy as np
from tqdm import tqdm
from collections import defaultdict

logging.getLogger().setLevel(logging.ERROR)


DATA_ROOT = ""
JSON_PATH = "worldsense_qa.json"
MODEL_PATH = ""
OUT_PATH = "./exp_results/avatar_worldsense.jsonl"
CSV_PATH = "./exp_results/avatar_worldsense.csv"
USE_AUDIO = True
BASE_MODEL = False
NUM_GPUS = 4

QUESTION_TEMPLATE = (
    "Your task is to accurately answer multiple-choice questions based on the given video. "
    "Select the single most accurate answer from the given choices.\n"
    "Question: {question}\n"
    "Your answer should be a capital letter representing your choice. Don't generate any other text."
)


def extract_answer(text):
    if "assistant\n" in text:
        assistant_response = text.split("assistant\n", 1)[-1]
    else:
        assistant_response = text

    letter_matches = re.findall(r"\b([A-D])\b", assistant_response)
    if letter_matches:
        return letter_matches[-1]

    any_letter_match = re.search(r"([A-D])", assistant_response)
    if any_letter_match:
        return any_letter_match.group(1)

    return ""


def reward_fn(sample, model_output):
    """Calculate reward based on question type."""
    try:
        output_ans = extract_answer(model_output)
        if output_ans == "":
            letter_match = re.search(r"([A-D])", model_output, re.I)
            output_ans = letter_match.group(1).upper() if letter_match else ""

        gt_ans = sample.get("answer", "").upper()

        return 1.0 if output_ans == gt_ans else 0.0

    except Exception as e:
        print(f"Error in reward_fn: {e}")
        return 0.0


def flatten_worldsense_data(data):
    flattened = []
    for video_id, video_data in data.items():
        for task_key, task_data in video_data.items():
            if task_key.startswith("task"):
                sample = {
                    "video_id": video_id,
                    "task_id": task_key,
                    "sample_id": f"{video_id}_{task_key}",
                    "video_duration": video_data.get("video_duration", ""),
                    "domain": video_data.get("domain", ""),
                    "sub_category": video_data.get("sub_category", ""),
                    "task_domain": task_data.get("task_domain", ""),
                    "task_type": task_data.get("task_type", ""),
                    "question": task_data.get("question", ""),
                    "answer": task_data.get("answer", ""),
                    "candidates": task_data.get("candidates", []),
                }
                flattened.append(sample)
    return flattened


def process_samples_on_gpu(
    gpu_id, data_subset, start_idx, existing_ids, result_queue, progress_queue
):
    torch.cuda.set_device(gpu_id)
    device = f"cuda:{gpu_id}"
    use_audio_flag = USE_AUDIO

    print(f"[GPU {gpu_id}] Loading model on {device}...")

    if BASE_MODEL:
        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            device_map=device,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
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
    print(f"[GPU {gpu_id}] Model ready. Processing {len(data_subset)} samples...")

    local_results = []
    for local_idx, sample in enumerate(data_subset):
        sample_id = sample["sample_id"]

        if sample_id in existing_ids:
            continue

        try:
            video_name = sample["video_id"]
            video_path = os.path.join(DATA_ROOT, f"{video_name}.mp4")

            question_text = sample["question"]
            candidates = sample["candidates"]
            formatted_options = "\n".join(candidates)
            full_question = f"{question_text}\n{formatted_options}"

            prompt = QUESTION_TEMPLATE.format(question=full_question)

            content_items = []
            if video_path and os.path.exists(video_path):
                content_items.append({"type": "video", "video": video_path})
            else:
                print(f"[GPU {gpu_id}] Warning: Video file not found: {video_path}")
                continue
            content_items.append({"type": "text", "text": prompt})

            if BASE_MODEL:
                conv = [
                    {
                        "role": "system",
                        "content": [
                            {
                                "type": "text",
                                "text": "You are a helpful assistant that answers multiple-choice questions.",
                            }
                        ],
                    },
                    {"role": "user", "content": content_items},
                ]
            else:
                conv = [{"role": "user", "content": content_items}]

            prompt_text = processor.apply_chat_template(
                conv, add_generation_prompt=True, tokenize=False
            )
            try:
                audios, images, videos = process_mm_info(
                    conv, use_audio_in_video=use_audio_flag
                )
            except Exception as e:
                use_audio_flag = False
                try:
                    audios, images, videos = process_mm_info(
                        conv, use_audio_in_video=False
                    )
                except Exception as e_retry:
                    print(f"process_mm_info failed again: {e_retry}")
                    audios, images, videos = [], [], []

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
                    max_new_tokens=512,
                )

            reply = processor.batch_decode(out_ids, skip_special_tokens=True)[0]

            final_answer = extract_answer(reply)
            if not final_answer:
                letter_match = re.search(r"([A-D])", reply, re.I)
                final_answer = letter_match.group(1).upper() if letter_match else "?"

            reward = reward_fn(sample, reply)
            is_correct = reward == 1.0

            result = {
                "id": sample_id,
                "video_id": sample["video_id"],
                "task_id": sample["task_id"],
                "question": sample["question"],
                "candidates": sample["candidates"],
                "domain": sample["domain"],
                "sub_category": sample["sub_category"],
                "task_domain": sample["task_domain"],
                "task_type": sample["task_type"],
                "video_duration": sample["video_duration"],
                "gt": sample["answer"],
                "prediction": final_answer,
                "full_output": reply,
                "reward": reward,
                "correct": is_correct,
                "video_path": video_path,
                "variant": "base" if BASE_MODEL else "rl",
            }

            local_results.append(result)
            progress_queue.put(1)

        except Exception as e:
            print(f"[GPU {gpu_id}] Error processing sample {sample_id}: {e}")
            result = {"id": sample_id, "error": str(e), "reward": 0.0, "correct": False}
            local_results.append(result)
            progress_queue.put(1)

    result_queue.put((gpu_id, local_results))
    print(f"[GPU {gpu_id}] Completed processing {len(local_results)} samples")
    progress_queue.put("DONE")


def write_results_with_lock(results, out_path):
    lock = FileLock(out_path + ".lock")
    with lock:
        with open(out_path, "a", encoding="utf-8") as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")


def calculate_and_save_metrics(out_path, csv_path, model_name="Qwen2.5-Omni"):
    print("\n📊 Calculating final metrics...")

    task_domains = set()
    task_types = set()
    domains = set()

    correct_counts_domain = defaultdict(int)
    total_counts_domain = defaultdict(int)
    correct_counts_task_domain = defaultdict(int)
    total_counts_task_domain = defaultdict(int)
    correct_counts_task_type = defaultdict(int)
    total_counts_task_type = defaultdict(int)

    all_results_data = []
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    result = json.loads(line)
                    all_results_data.append(result)
                except:
                    continue

    for result in all_results_data:
        domain = result.get("domain", "N/A")
        task_domain = result.get("task_domain", "N/A")
        task_type = result.get("task_type", "N/A")

        domains.add(domain)
        task_domains.add(task_domain)
        task_types.add(task_type)

        total_counts_domain[domain] += 1
        total_counts_task_domain[task_domain] += 1
        total_counts_task_type[task_type] += 1

        if result.get("correct", False):
            correct_counts_domain[domain] += 1
            correct_counts_task_domain[task_domain] += 1
            correct_counts_task_type[task_type] += 1

    accuracies_domain = {}
    for domain in domains:
        if total_counts_domain[domain] > 0:
            accuracies_domain[domain] = (
                correct_counts_domain[domain] / total_counts_domain[domain]
            )
        else:
            accuracies_domain[domain] = 0.0

    accuracies_task_domain = {}
    for task_domain in task_domains:
        if total_counts_task_domain[task_domain] > 0:
            accuracies_task_domain[task_domain] = (
                correct_counts_task_domain[task_domain]
                / total_counts_task_domain[task_domain]
            )
        else:
            accuracies_task_domain[task_domain] = 0.0

    accuracies_task_type = {}
    for task_type in task_types:
        if total_counts_task_type[task_type] > 0:
            accuracies_task_type[task_type] = (
                correct_counts_task_type[task_type] / total_counts_task_type[task_type]
            )
        else:
            accuracies_task_type[task_type] = 0.0

    total_correct = sum(correct_counts_domain.values())
    total_questions = sum(total_counts_domain.values())
    overall_accuracy = total_correct / total_questions if total_questions > 0 else 0

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)

        writer.writerow(["WorldSense Evaluation Results"])
        writer.writerow(["Overall Accuracy", f"{overall_accuracy:.4f}"])
        writer.writerow(["Total Questions", total_questions])
        writer.writerow([])

        writer.writerow(["Domain Results"])
        writer.writerow(["Domain", "Accuracy", "Correct", "Total"])
        for domain in sorted(domains):
            writer.writerow(
                [
                    domain,
                    f"{accuracies_domain[domain]:.4f}",
                    correct_counts_domain[domain],
                    total_counts_domain[domain],
                ]
            )
        writer.writerow([])

        writer.writerow(["Task Domain Results"])
        writer.writerow(["Task Domain", "Accuracy", "Correct", "Total"])
        for task_domain in sorted(task_domains):
            writer.writerow(
                [
                    task_domain,
                    f"{accuracies_task_domain[task_domain]:.4f}",
                    correct_counts_task_domain[task_domain],
                    total_counts_task_domain[task_domain],
                ]
            )
        writer.writerow([])

        writer.writerow(["Task Type Results"])
        writer.writerow(["Task Type", "Accuracy", "Correct", "Total"])
        for task_type in sorted(task_types):
            writer.writerow(
                [
                    task_type,
                    f"{accuracies_task_type[task_type]:.4f}",
                    correct_counts_task_type[task_type],
                    total_counts_task_type[task_type],
                ]
            )

    print(f"\n🎯 {model_name} Model Evaluation Complete")
    print(
        f"Overall Accuracy: {overall_accuracy:.2%} ({total_correct}/{total_questions})"
    )

    print("\n📋 Results by Domain:")
    for domain in sorted(domains):
        if total_counts_domain[domain] > 0:
            print(
                f"  {domain}: {accuracies_domain[domain]:.2%} ({correct_counts_domain[domain]}/{total_counts_domain[domain]})"
            )

    print("\n📋 Results by Task Domain:")
    for task_domain in sorted(task_domains):
        if total_counts_task_domain[task_domain] > 0:
            print(
                f"  {task_domain}: {accuracies_task_domain[task_domain]:.2%} ({correct_counts_task_domain[task_domain]}/{total_counts_task_domain[task_domain]})"
            )

    print("\n📋 Results by Task Type:")
    for task_type in sorted(task_types):
        if total_counts_task_type[task_type] > 0:
            print(
                f"  {task_type}: {accuracies_task_type[task_type]:.2%} ({correct_counts_task_type[task_type]}/{total_counts_task_type[task_type]})"
            )

    print(f"\n💾 Results saved to:")
    print(f"  JSON: {out_path}")
    print(f"  CSV:  {csv_path}")

    return (
        overall_accuracy,
        accuracies_domain,
        accuracies_task_domain,
        accuracies_task_type,
    )


def worker_fn(
    rank,
    world_size,
    data_splits,
    start_indices,
    existing_ids,
    result_queue,
    progress_queue,
):
    if rank < len(data_splits) and len(data_splits[rank]) > 0:
        process_samples_on_gpu(
            rank,
            data_splits[rank],
            start_indices[rank],
            existing_ids,
            result_queue,
            progress_queue,
        )


def main():
    print(f"Loading data from {JSON_PATH}...")
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    data = flatten_worldsense_data(raw_data)
    print(f"📚 {len(data)} samples loaded from {len(raw_data)} videos.")

    existing_ids = set()
    if os.path.exists(OUT_PATH):
        print(f"⚡️ Found existing result file {OUT_PATH}. Resuming...")
        with open(OUT_PATH, "r", encoding="utf-8") as f_exist:
            for line in f_exist:
                try:
                    result = json.loads(line)
                    existing_ids.add(result["id"])
                except:
                    continue
        print(f"⚡️ {len(existing_ids)} samples already completed.")
    else:
        print(f"🆕 No existing result file. Starting fresh.")
        os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    remaining_data = []
    for sample in data:
        sample_id = sample["sample_id"]
        if sample_id not in existing_ids:
            remaining_data.append(sample)

    if not remaining_data:
        print("✅ All samples already processed!")
        model_variant = "BASE" if BASE_MODEL else "RL"
        calculate_and_save_metrics(OUT_PATH, CSV_PATH, f"Qwen2.5-Omni-{model_variant}")
        return

    print(f"📊 {len(remaining_data)} samples remaining to process")

    n_gpus = min(NUM_GPUS, torch.cuda.device_count())
    print(f"🖥️  Using {n_gpus} GPUs for parallel processing")

    data_splits = np.array_split(remaining_data, n_gpus)
    data_splits = [list(split) for split in data_splits]

    start_indices = [sum(len(s) for s in data_splits[:i]) for i in range(n_gpus)]

    mp.set_start_method("spawn", force=True)
    manager = Manager()
    result_queue = manager.Queue()
    progress_queue = manager.Queue()

    print("🚀 Spawning GPU processes...")
    spawn_context = mp.spawn(
        worker_fn,
        args=(
            n_gpus,
            data_splits,
            start_indices,
            existing_ids,
            result_queue,
            progress_queue,
        ),
        nprocs=n_gpus,
        join=False,
    )

    pbar = tqdm(total=len(remaining_data), desc="Processing samples")
    processed = 0
    gpus_done = 0

    while gpus_done < n_gpus:
        try:
            item = progress_queue.get(timeout=10)
            if item == "DONE":
                gpus_done += 1
            else:
                processed += 1
                pbar.update(1)
        except:
            if spawn_context.join(0):
                break
            continue

    pbar.close()

    print("\n⏳ Waiting for all GPU processes to complete...")
    spawn_context.join()
    print("✅ All GPU processes completed")

    all_results = []
    while not result_queue.empty():
        gpu_id, results = result_queue.get()
        all_results.extend(results)

    if all_results:
        print(f"💾 Writing {len(all_results)} new results to file...")
        write_results_with_lock(all_results, OUT_PATH)

    model_variant = "BASE" if BASE_MODEL else "RL"
    calculate_and_save_metrics(OUT_PATH, CSV_PATH, f"Qwen2.5-Omni-{model_variant}")


if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed = (time.time() - start_time) / 60
    print(f"⏱️  Total time: {elapsed:.1f} minutes")
