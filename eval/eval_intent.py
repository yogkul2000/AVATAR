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

DATA_ROOT = "IntentBench"
JSON_PATH = "IntentBench/qa.json"
MODEL_PATH = ""
OUT_PATH = "./exp_results/v2-tas.jsonl"


USE_AUDIO = True
BASE_MODEL = False
NUM_GPUS = 4

SINGLE_CHOICE_MC_TEMPLATE = (
    "Your task is to accurately answer the multiple-choice question based on the given video. "
    "Read the question and the options carefully, then select the single most accurate answer.\n\n"
    "{formatted_question}\n\n"
    "Your answer must be a single capital letter representing your choice (e.g., A, B, C, D)."
)

MULTI_LABEL_MC_TEMPLATE = (
    "Based on the video, identify ALL the emotions or concepts that apply from the list below. "
    "More than one option can be correct.\n\n"
    "{formatted_question}\n\n"
    "Your answer must be a comma-separated list of all applicable letters (e.g., A, C, F)."
)

JUDGE_QUESTION_TEMPLATE = (
    "You are a deception detection expert. Based on the provided video, you must determine if the person is being truthful or deceptive. "
    'The question they were asked is: "{original_question}".\n'
    "After careful observation of their verbal and non-verbal cues, answer with only 'Yes' if you believe they are lying, or 'No' if you believe they are telling the truth. "
    "Do not provide any explanation or other text. Your answer must be 'Yes' or 'No'."
)


def extract_single_choice_answer(text):
    response_text = text.split("assistant\n", 1)[-1] if "assistant\n" in text else text
    letter_matches = re.findall(r"\b([A-D])\b", response_text)
    if letter_matches:
        return letter_matches[-1]
    any_letter_match = re.search(r"([A-D])", response_text)
    return any_letter_match.group(1) if any_letter_match else ""


def extract_multi_label_answer(text):
    response_text = text.split("assistant\n", 1)[-1] if "assistant\n" in text else text
    letters = sorted(list(set(re.findall(r"[A-F]", response_text.upper()))))
    return letters


def extract_judge_answer(text):
    response_text = text.split("assistant\n", 1)[-1] if "assistant\n" in text else text
    match = re.search(r"\b(Yes|No)\b", response_text, re.IGNORECASE)
    return match.group(1).capitalize() if match else ""


def reward_fn(sample, model_output):
    try:
        problem_type = sample.get("problem_type")

        if problem_type == "multiple choice":
            pred = extract_single_choice_answer(model_output)
            gt = sample.get("answer", "").upper()
            return 1.0 if pred == gt else 0.0

        elif problem_type == "emer_ov_mc":
            pred_set = set(extract_multi_label_answer(model_output))
            gt_set = set(sample.get("answer", "").upper().split(","))
            return 1.0 if pred_set == gt_set else 0.0

        elif problem_type == "judge":
            pred = extract_judge_answer(model_output)
            gt = sample.get("answer", "").capitalize()
            return 1.0 if pred == gt else 0.0

        return 0.0

    except Exception as e:
        print(f"Error in reward_fn for sample {sample.get('qid', '')}: {e}")
        return 0.0


def process_samples_on_gpu(
    gpu_id, data_subset, existing_ids, result_queue, progress_queue
):
    torch.cuda.set_device(gpu_id)
    device = f"cuda:{gpu_id}"
    use_audio_flag = USE_AUDIO

    print(f"[GPU {gpu_id}] Loading model on {device}...")
    if BASE_MODEL:
        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            torch_dtype=device,
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
    for sample in data_subset:
        sample_id = str(sample.get("qid") or sample.get("Question id"))
        if sample_id in existing_ids:
            continue

        try:
            problem_type = sample.get("problem_type")
            prompt, final_answer_extractor = None, None

            if problem_type == "multiple choice":
                question = sample["problem"]
                options = "\n".join(sample["options"])
                formatted_question = f"Question: {question}\nOptions:\n{options}"
                prompt = SINGLE_CHOICE_MC_TEMPLATE.format(
                    formatted_question=formatted_question
                )
                final_answer_extractor = extract_single_choice_answer

            elif problem_type == "emer_ov_mc":
                question = sample["problem"]
                options = "\n".join(sample["options"])
                formatted_question = f"Question: {question}\nOptions:\n{options}"
                prompt = MULTI_LABEL_MC_TEMPLATE.format(
                    formatted_question=formatted_question
                )
                final_answer_extractor = extract_multi_label_answer

            elif problem_type == "judge":
                match = re.search(r'"(.*?)"', sample["problem"])
                original_question = match.group(1) if match else "the question"
                prompt = JUDGE_QUESTION_TEMPLATE.format(
                    original_question=original_question
                )
                final_answer_extractor = extract_judge_answer

            if not prompt:
                print(
                    f"[GPU {gpu_id}] Skipping sample {sample_id} due to unsupported problem_type: {problem_type}"
                )
                continue

            video_path = os.path.join(DATA_ROOT, sample["video"])
            content_items = (
                [{"type": "video", "video": video_path}]
                if os.path.exists(video_path)
                else []
            ) + [{"type": "text", "text": prompt}]
            conv = [{"role": "user", "content": content_items}]
            prompt_text = processor.apply_chat_template(
                conv, add_generation_prompt=True, tokenize=False
            )

            try:
                audios, images, videos = process_mm_info(
                    conv, use_audio_in_video=use_audio_flag
                )
            except Exception:
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
                    max_new_tokens=20,
                )

            reply = processor.batch_decode(out_ids, skip_special_tokens=True)[0]

            final_answer = final_answer_extractor(reply)
            reward = reward_fn(sample, reply)

            result = {
                "id": sample_id,
                "problem_type": problem_type,
                "problem": sample["problem"],
                "gt": sample.get("answer", ""),
                "prediction": final_answer,
                "full_output": reply,
                "reward": reward,
                "correct": (reward == 1.0),
            }
            local_results.append(result)
            progress_queue.put(1)

        except Exception as e:
            print(f"[GPU {gpu_id}] FATAL Error processing sample {sample_id}: {e}")
            local_results.append(
                {"id": sample_id, "error": str(e), "reward": 0.0, "correct": False}
            )
            progress_queue.put(1)

    result_queue.put(local_results)
    print(f"[GPU {gpu_id}] Completed processing {len(local_results)} samples.")
    progress_queue.put("DONE")


def worker_fn(
    rank, world_size, data_splits, existing_ids, result_queue, progress_queue
):
    if rank < len(data_splits) and len(data_splits[rank]) > 0:
        process_samples_on_gpu(
            rank, data_splits[rank], existing_ids, result_queue, progress_queue
        )


def main():
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"📚 {len(data)} samples loaded.")

    existing_ids = set()
    if os.path.exists(OUT_PATH):
        with open(OUT_PATH, "r", encoding="utf-8") as f_exist:
            for line in f_exist:
                try:
                    existing_ids.add(json.loads(line)["id"])
                except (json.JSONDecodeError, KeyError):
                    continue
        print(f"⚡️ {len(existing_ids)} samples already processed.")
    else:
        print("🆕 No existing result file. Starting fresh.")
        if OUT_PATH and os.path.dirname(OUT_PATH):
            os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    remaining_data = [
        s for s in data if str(s.get("qid", s.get("Question id"))) not in existing_ids
    ]

    if not remaining_data:
        print("✅ All samples already processed!")
    else:
        print(f"📊 {len(remaining_data)} samples remaining to process.")
        n_gpus = min(NUM_GPUS, torch.cuda.device_count())
        data_splits = np.array_split(remaining_data, n_gpus)
        data_splits = [split.tolist() for split in data_splits if len(split) > 0]
        n_gpus = len(data_splits)
        if n_gpus == 0:
            print("No data to process after filtering.")
            return

        print(f"🖥️ Using {n_gpus} GPUs for parallel processing.")
        mp.set_start_method("spawn", force=True)
        manager = Manager()
        result_queue, progress_queue = manager.Queue(), manager.Queue()

        print("🚀 Spawning GPU processes...")
        spawn_context = mp.spawn(
            worker_fn,
            args=(n_gpus, data_splits, existing_ids, result_queue, progress_queue),
            nprocs=n_gpus,
            join=False,
        )

        with tqdm(total=len(remaining_data), desc="Processing samples") as pbar:
            gpus_done = 0
            while gpus_done < n_gpus:
                item = progress_queue.get()
                if item == "DONE":
                    gpus_done += 1
                else:
                    pbar.update(item)

        print("\n⏳ Waiting for all GPU processes to complete...")
        spawn_context.join()
        print("✅ All GPU processes completed.")

        all_results = []
        while not result_queue.empty():
            all_results.extend(result_queue.get())

        if all_results:
            print(f"💾 Writing {len(all_results)} new results to file...")
            with FileLock(OUT_PATH + ".lock"):
                with open(OUT_PATH, "a", encoding="utf-8") as f:
                    for result in all_results:
                        f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print("\n📊 Calculating final metrics from the complete results file...")
    all_results_data = []
    if os.path.exists(OUT_PATH):
        with open(OUT_PATH, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    all_results_data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    if not all_results_data:
        print("No results found to calculate metrics.")
        return

    type_counts = {"total": len(all_results_data)}
    type_correct = {"total": sum(1 for r in all_results_data if r.get("correct"))}

    for r in all_results_data:
        ptype = r.get("problem_type", "unknown")
        type_counts[ptype] = type_counts.get(ptype, 0) + 1
        if r.get("correct"):
            type_correct[ptype] = type_correct.get(ptype, 0) + 1

    print("\n" + "=" * 50)
    print(
        f"✅ IntentBench Evaluation Complete ({'BASE' if BASE_MODEL else 'RL'} Model)"
    )
    print(f"\n--- Overall Performance ---")
    print(
        f"  - Overall Accuracy (Exact Match): {type_correct.get('total', 0) / type_counts['total']:.2%}"
    )
    print(
        f"  - Correct / Total Samples:      {type_correct.get('total', 0)} / {type_counts['total']}"
    )

    print(f"\n--- Performance by Question Type ---")
    for ptype in sorted(type_counts.keys()):
        if ptype == "total":
            continue
        correct = type_correct.get(ptype, 0)
        total = type_counts.get(ptype, 0)
        acc = correct / total if total > 0 else 0
        print(
            f"  - {ptype.replace('_', ' ').title():<25} | Accuracy: {acc:.2%} ({correct}/{total})"
        )

    print(f"\n    - Results saved to: {OUT_PATH}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed = (time.time() - start_time) / 60
    print(f"⏱️ Total execution time: {elapsed:.1f} minutes.")
