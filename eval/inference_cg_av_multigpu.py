import os
import json
import time
import torch
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
from pathlib import Path

import decord
from transformers import (
    Qwen2_5OmniForConditionalGeneration,
    Qwen2_5OmniProcessor,
    Qwen2_5OmniThinkerForConditionalGeneration,
)
from qwen_omni_utils import process_mm_info

import torch.multiprocessing as mp
from multiprocessing import Manager
from filelock import FileLock

import logging

logging.getLogger().setLevel(logging.ERROR)

PROMPTS = {
    "long_acc": "Watch the video and answer the question '{question}' with a number. Just output the number itself, don't output anything else.",
    "ref_acc": "Watch the video and answer the question '{question}' with a number. Just output the number itself, don't output anything else.",
    "event": 'Watch the video and provide your answer to the question \'{question}\', including the start and end timestamps for each event. Format your answer in JSON, enclosed in <answer> and </answer> tags. The output should look like this: <answer>[["start_time", "end_time"], ...]</answer>. Ensure each timestamp is in seconds (e.g., xx.xx ).',
    "object": 'According to the given video frames, answer the question \'{question}\', including the bounding box for the query object in the first frame where it appears. For subsequent frames where the object appears, do not provide the bounding box again. Format your answer in JSON, enclosed within <answer> and </answer> tags. The output should look like this: <answer>{{"Frame1": [[xmin, ymin, x_max, y_max]], "Frame2": []}}</answer>.',
    "attribute": 'According to the given video frames, answer the question \'{question}\', clustering the objects based on the question. For each unique cluster, assign a unique label and return the bounding box for each object in the first frame where it appears. Format your answer in JSON, enclosed within <answer> and </answer> tags. The output should look like this: <answer>{{"Frame 1": [{{"bbox": [x_min, y_min, x_max, y_max], \'label\': "Label 1"}}], "Frame 2": [...] }}</answer>.',
}


def get_video_frames(video_path, timestamps):
    try:
        vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
        duration = len(vr)
        fps = vr.get_avg_fps()
        frame_indices = sorted(
            list(set([min(int(t * fps), duration - 1) for t in timestamps]))
        )
        frames = vr.get_batch(frame_indices).asnumpy()
        return [Image.fromarray(frame) for frame in frames]
    except Exception as e:
        print(f"Error extracting frames from {video_path}: {e}")
        return []


def write_results_with_lock(results, out_dir):
    for result in results:
        file_path = os.path.join(out_dir, f"{result['id']}.txt")
        lock = FileLock(file_path + ".lock")
        with lock:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(result["prediction"])


def process_samples_on_gpu(gpu_id, data_subset, args, result_queue, progress_queue):
    torch.cuda.set_device(gpu_id)
    device = f"cuda:{gpu_id}"
    print(f"[GPU {gpu_id}] Loading model on {device}...")

    if args.base_model:
        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            args.model_path,
            torch_dtype="auto",
            device_map=device,
            attn_implementation="flash_attention_2",
            enable_audio_output=False,
        ).eval()
        model.disable_talker()
    else:
        model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            args.model_path,
            device_map=device,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        ).eval()
    processor = Qwen2_5OmniProcessor.from_pretrained(args.model_path)

    print(
        f"[GPU {gpu_id}] Model ready. Processing {len(data_subset)} samples for '{args.eval_mode}' mode..."
    )

    local_results = []
    for sample in data_subset:
        sample_id = str(sample["index"])
        question = sample["question"]

        try:
            content_items = []

            if args.eval_mode == "long_acc":
                prompt = PROMPTS["long_acc"].format(question=question)
                video_path = os.path.join(args.data_root, sample["video"])
                if os.path.exists(video_path):
                    content_items.append({"type": "video", "video": video_path})
                else:
                    print(f"[GPU {gpu_id}] Warning: Video not found: {video_path}")
                    continue

            elif args.eval_mode == "ref_acc":
                prompt = PROMPTS["ref_acc"].format(question=question)

                start, end = sample["query_interval"]
                video_id = sample["video"].replace(".mp4", "")
                start_str = f"{start:.2f}"
                end_str = f"{end:.2f}"
                clip_filename = f"{video_id}_{start_str}_{end_str}.mp4"

                video_path = os.path.join(args.clips_path, clip_filename)
                if os.path.exists(video_path):
                    content_items.append({"type": "video", "video": video_path})
                else:
                    print(f"[GPU {gpu_id}] Warning: Clip not found: {video_path}")
                    continue

            elif args.eval_mode == "clue_acc":
                item_category = sample["category"]
                prompt = PROMPTS.get(item_category, PROMPTS["event"]).format(
                    question=question
                )
                video_path = os.path.join(args.data_root, sample["video"])
                if not os.path.exists(video_path):
                    print(f"[GPU {gpu_id}] Warning: Video not found: {video_path}")
                    continue
                if item_category in ["object", "attribute"]:
                    clue_timestamps = [c["timestamp"] for c in sample["clue"]]
                    frames = get_video_frames(video_path, clue_timestamps)
                    if not frames:
                        print(
                            f"[GPU {gpu_id}] Warning: Could not extract frames for {sample_id}. Skipping."
                        )
                        continue
                    for frame in frames:
                        content_items.append({"type": "image", "image": frame})
                else:
                    content_items.append({"type": "video", "video": video_path})

            content_items.append({"type": "text", "text": prompt})

            if args.base_model:
                conv = [
                    {
                        "role": "system",
                        "content": [
                            {"type": "text", "text": "You are a helpful assistant."}
                        ],
                    },
                    {"role": "user", "content": content_items},
                ]
            else:
                conv = [{"role": "user", "content": content_items}]

            prompt_text = processor.apply_chat_template(
                conv, add_generation_prompt=True, tokenize=False
            )
            audios, images, videos = process_mm_info(conv, use_audio_in_video=True)
            inputs = processor(
                text=prompt_text,
                audio=audios,
                images=images,
                videos=videos,
                return_tensors="pt",
                padding=True,
                use_audio_in_video=True,
            ).to(device)

            with torch.no_grad():
                out_ids = model.generate(
                    **inputs,
                    use_audio_in_video=True,
                    do_sample=False,
                    max_new_tokens=1024,
                )
            reply = processor.batch_decode(out_ids, skip_special_tokens=True)[0]
            assistant_response = reply.split("assistant\n")[-1].strip()

            local_results.append({"id": sample_id, "prediction": assistant_response})
            progress_queue.put(1)

        except Exception as e:
            print(f"[GPU {gpu_id}] ERROR processing sample {sample_id}: {e}")
            progress_queue.put(1)

    result_queue.put(local_results)
    print(f"[GPU {gpu_id}] Completed processing its batch.")
    progress_queue.put("DONE")


def worker_fn(rank, data_splits, args, result_queue, progress_queue):
    if rank < len(data_splits) and len(data_splits[rank]) > 0:
        process_samples_on_gpu(
            rank, data_splits[rank], args, result_queue, progress_queue
        )


def main(args):
    print(f"Loading data from {args.json_path}...")
    with open(args.json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"📚 {len(data)} samples loaded.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    existing_ids = {f.stem for f in output_dir.glob("*.txt")}
    if existing_ids:
        print(
            f"⚡️ Found {len(existing_ids)} existing results in {output_dir}. Resuming..."
        )
        remaining_data = [s for s in data if str(s["index"]) not in existing_ids]
    else:
        print(f"🆕 No existing results found. Starting fresh.")
        remaining_data = data

    if not remaining_data:
        print("✅ All samples already processed!")
        return
    print(f"📊 {len(remaining_data)} samples remaining to process.")

    n_gpus = min(args.num_gpus, torch.cuda.device_count())
    print(f"🖥️  Using {n_gpus} GPUs for parallel processing.")

    data_splits = np.array_split(remaining_data, n_gpus)
    data_splits = [list(split) for split in data_splits if len(split) > 0]
    n_gpus = len(data_splits)

    mp.set_start_method("spawn", force=True)
    manager = Manager()
    result_queue = manager.Queue()
    progress_queue = manager.Queue()

    print("🚀 Spawning GPU processes...")
    spawn_context = mp.spawn(
        worker_fn,
        args=(data_splits, args, result_queue, progress_queue),
        nprocs=n_gpus,
        join=False,
    )

    with tqdm(total=len(remaining_data), desc=f"Processing '{args.eval_mode}'") as pbar:
        gpus_done = 0
        while gpus_done < n_gpus:
            item = progress_queue.get()
            if item == "DONE":
                gpus_done += 1
            else:
                pbar.update(1)

    print("\n⏳ Waiting for all GPU processes to join...")
    spawn_context.join()
    print("✅ All GPU processes completed.")

    print("💾 Collecting and writing results...")
    all_results = []
    while not result_queue.empty():
        all_results.extend(result_queue.get())

    write_results_with_lock(all_results, str(output_dir))
    print(f"🎉 Inference complete. Predictions saved to '{output_dir}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run inference for CG-AV-Counting using the Qwen Omni template."
    )
    parser.add_argument(
        "--eval_mode",
        type=str,
        required=True,
        choices=["long_acc", "ref_acc", "clue_acc"],
    )
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument(
        "--data_root", type=str, default="/scratch/ykulka10/data/DATA_ROOT/CG-Bench"
    )
    parser.add_argument(
        "--json_path",
        type=str,
        default="/scratch/ykulka10/data/DATA_ROOT/CG-AV-Counting/cg-av-counting.json",
    )
    parser.add_argument(
        "--clips_path",
        type=str,
        default="/scratch/ykulka10/data/DATA_ROOT/CG-AV-Counting/",
        help="Directory of pre-clipped videos (required for 'ref_acc' mode).",
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_gpus", type=int, default=4)
    parser.add_argument(
        "--base_model",
        action="store_true",
        help="Set this flag if using the base model instead of the GRPO/Thinker model.",
    )
    args = parser.parse_args()

    if args.eval_mode == "ref_acc" and not args.clips_path:
        raise ValueError("--clips_path is required for 'ref_acc' mode.")

    start_time = time.time()
    main(args)
    elapsed = (time.time() - start_time) / 60
    print(f"⏱️  Total time: {elapsed:.2f} minutes.")
