import os
import torch
from transformers import (
    Qwen2_5OmniThinkerForConditionalGeneration,
    Qwen2_5OmniProcessor,
)
from qwen_omni_utils import process_mm_info


def run_inference():
    MODEL_PATH = ""
    VIDEO_PATH = ""
    QUESTION = "Use available audio and video to answer: Why the person is doing what they are doing? Give reasoning between <think> and </think> tags."

    device = "cuda:0"
    use_audio_flag = True

    print("Loading model...")
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        device_map=device,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).eval()
    processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_PATH)
    print("Model loaded.")

    content_items = [
        {"type": "video", "video": VIDEO_PATH},
        {"type": "text", "text": QUESTION},
    ]

    conv = [{"role": "user", "content": content_items}]

    prompt_text = processor.apply_chat_template(
        conv, add_generation_prompt=True, tokenize=False
    )

    try:
        audios, images, videos = process_mm_info(
            conv, use_audio_in_video=use_audio_flag
        )
    except Exception as e:
        print(f"Failed to process with audio, retrying without: {e}")
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

    print("Generating response...")
    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            use_audio_in_video=use_audio_flag,
            do_sample=False,
            max_new_tokens=512,
        )

    reply = processor.batch_decode(out_ids, skip_special_tokens=True)[0]

    print("\n" + "=" * 20 + " MODEL OUTPUT " + "=" * 20)
    print(reply)
    print("=" * 54 + "\n")


if __name__ == "__main__":
    run_inference()
