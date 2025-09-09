import os, json, time, re, torch
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

logging.getLogger().setLevel(logging.ERROR)

DATA_ROOT = ""
JSON_PATH = "tomato.json"
MODEL_PATH = ""
OUT_PATH = "./exp_results/tomato_avatar.jsonl"
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


def reward_fn(sample, model_output):
    try:
        output_ans = extract_answer(model_output)
        if output_ans == "":
            letter_match = re.search(r"([A-E])", model_output, re.I)
            output_ans = letter_match.group(1).upper() if letter_match else ""

        gt_ans = sample.get("answer", "").upper()

        return 1.0 if output_ans == gt_ans else 0.0

    except Exception as e:
        print(f"Error in reward_fn: {e}")
        return 0.0


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
    print(f"[GPU {gpu_id}] Model ready. Processing {len(data_subset)} samples...")

    local_results = []
    for local_idx, sample in enumerate(data_subset):
        global_idx = start_idx + local_idx
        sample_id = sample.get("question_id", f"sample_{global_idx}")

        if sample_id in existing_ids:
            continue

        try:
            video_path = os.path.join(DATA_ROOT, sample["video_path"])

            question_text = sample["processed_question"]

            question_type = "multiple choice"

            prompt = QUESTION_TEMPLATE.format(question=question_text)

            content_items = []
            if video_path and os.path.exists(video_path):
                content_items.append({"type": "video", "video": video_path})
            else:
                print(f"[GPU {gpu_id}] Warning: Video file not found: {video_path}")

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
                    print(
                        f"process_mm_info failed again after setting use_audio_in_video=False: {e_retry}"
                    )
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
                letter_match = re.search(r"([A-E])", reply, re.I)
                final_answer = letter_match.group(1).upper() if letter_match else "?"

            reward = reward_fn(sample, reply)
            is_correct = reward == 1.0

            result = {
                "id": sample_id,
                "question": sample["question"],
                "processed_question": sample["processed_question"],
                "problem_type": question_type,
                "gt": sample.get("answer", ""),
                "prediction": final_answer,
                "full_output": reply,
                "reward": reward,
                "correct": is_correct,
                "video_path": sample["video_path"],
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
    if JSON_PATH.endswith(".jsonl"):
        data = []
        with open(JSON_PATH, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line))
    elif JSON_PATH.endswith(".json"):
        with open(JSON_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        raise ValueError("Input file must be .json or .jsonl")

    print(f"📚 {len(data)} samples loaded.")

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
    for idx, sample in enumerate(data):
        sample_id = sample.get("question_id", f"sample_{idx}")
        if sample_id not in existing_ids:
            remaining_data.append(sample)

    if not remaining_data:
        print("✅ All samples already processed!")
        return

    print(f"📊 {len(remaining_data)} samples remaining to process")

    n_gpus = min(NUM_GPUS, torch.cuda.device_count())
    print(f"🖥️  Using {n_gpus} GPUs for parallel processing")

    data_splits = np.array_split(remaining_data, n_gpus)
    data_splits = [list(split) for split in data_splits]

    start_indices = []
    current_start = 0
    for split in data_splits:
        start_indices.append(current_start)
        current_start += len(split)

    mp.set_start_method("spawn", force=True)
    manager = Manager()
    result_queue = manager.Queue()
    progress_queue = manager.Queue()

    total_to_process = len(remaining_data)

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

    pbar = tqdm(total=total_to_process, desc="Processing samples")
    processed = 0
    gpus_done = 0

    while processed < total_to_process or gpus_done < n_gpus:
        try:
            item = progress_queue.get(timeout=1)
            if item == "DONE":
                gpus_done += 1
                print(f"\n[INFO] GPU finished ({gpus_done}/{n_gpus} done)")
            else:
                processed += 1
                pbar.update(1)
        except:
            continue

    pbar.close()

    print("\n⏳ Waiting for all GPU processes to complete...")
    spawn_context.join()

    print("✅ All GPU processes completed")

    all_results = []
    while not result_queue.empty():
        gpu_id, results = result_queue.get()
        print(f"Collected {len(results)} results from GPU {gpu_id}")
        all_results.extend(results)

    if all_results:
        print(f"💾 Writing {len(all_results)} results to file...")
        write_results_with_lock(all_results, OUT_PATH)

    print("\n📊 Calculating final metrics...")

    all_results_data = []
    with open(OUT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            try:
                all_results_data.append(json.loads(line))
            except:
                continue

    correct = sum(1 for r in all_results_data if r.get("correct", False))
    total = len(all_results_data)

    accuracy = correct / total if total > 0 else 0

    print(f"\n🎯 {'BASE' if BASE_MODEL else 'RL'} Model Evaluation Complete")
    print(f"Overall Accuracy: {accuracy:.2%}")
    print(f"Correct: {correct}/{total}")
    print(f"Results saved to: {OUT_PATH}")


if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed = (time.time() - start_time) / 60
    print(f"⏱️  Total time: {elapsed:.1f} minutes")
