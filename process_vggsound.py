#!/usr/bin/env python
import os
import sys
import torch
import numpy as np
import pandas as pd
import argparse
from tqdm.auto import tqdm

from chatbridge.common.config import Config
from chatbridge.common.dist_utils import get_rank
from chatbridge.common.registry import registry
from chatbridge.conversation.conversation import (
    Chat,
    CONV_VISION,
    CONV_AUDIO,
    CONV_VIDEO,
)
from chatbridge.processors.blip_processors import BlipAudioEvalProcessor
from chatbridge.processors.alpro_processors import AlproVideoEvalProcessor

# imports modules for registration
from chatbridge.datasets.builders import *
from chatbridge.models import *
from chatbridge.processors import *
from chatbridge.runners import *
from chatbridge.tasks import *


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process VGGSound dataset with ChatBridge"
    )
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument(
        "--gpu-id", type=int, default=0, help="specify the gpu to load the model."
    )
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/storage/slurm/zverev/datasets/vggsound",
        help="Path to the directory containing video files",
    )
    parser.add_argument(
        "--video_csv",
        type=str,
        default="../../data/train.csv",
        help="CSV file that contains the list of video IDs",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="../../data/chatbridge_predictions.csv",
        help="Output CSV file for writing predictions",
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Temperature for generation"
    )
    parser.add_argument(
        "--num_beams", type=int, default=1, help="Number of beams for beam search"
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=300, help="Maximum new tokens to generate"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=2000,
        help="Maximum total length of the conversation",
    )
    parser.add_argument(
        "--page",
        type=int,
        default=1,
        help="Page number to process",
    )
    parser.add_argument(
        "--per_page",
        type=int,
        default=1000,
        help="Number of videos to process per page",
    )
    parser.add_argument(
        "--modality",
        type=str,
        default="video",
        choices=["video", "audio", "image"],
        help="Modality to use for processing",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Do you hear or see '{cl}' class in this video? Answer only with yes or no.",
        help="Prompt to use",
    )
    parser.add_argument(
        "--prompt_mode",
        type=str,
        default="single",
        choices=["single", "multi"],
        help="Prompt mode to use: 'single' for one prompt with all classes, 'multi' for individual prompts per class",
    )

    args = parser.parse_args()
    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    total_size = (param_size + buffer_size) / 1024**2  # Convert to MB
    return total_size


def get_video_list(csv_path):
    """
    Reads video IDs from a CSV file.
    Assumes CSV with two columns: video_id and label. If the video_id does not
    end with '.mp4', it appends '.mp4'.
    """
    if not os.path.exists(csv_path):
        print(f"CSV file {csv_path} not found.")
        return []
    df = pd.read_csv(csv_path, names=["video_id", "label"], header=None)
    video_ids = df["video_id"].tolist()
    video_ids = [vid if vid.endswith(".mp4") else vid + ".mp4" for vid in video_ids]
    return video_ids


def write_predictions_csv(predictions, responses, output_csv):
    """
    Writes the predictions dictionary to a CSV file.
    The CSV will have two columns: video_id and suggestions.
    """
    df_table = {
        i: {
            "video_id": vid,
            "suggestions": list(predictions[vid]),
            "response": responses[vid],
        }
        for i, vid in enumerate(predictions.keys())
    }
    df = pd.DataFrame.from_dict(df_table, orient="index")
    df.to_csv(output_csv, index=False)
    print(f"Predictions CSV saved to {output_csv}")


# Read audio classes from CSV
CLASSES = pd.read_csv("../../data/audio_classes.csv")["display_name"].tolist()


@torch.inference_mode()
def process_video(
    chat,
    dataset_path,
    video_id,
    temperature,
    num_beams,
    max_new_tokens,
    max_length,
    modality="video",
    prompt="Do you hear or see '{cl}' class in this video? Answer only with yes or no.",
    prompt_mode="single",
):
    """
    Process a single video file and detect classes.
    Returns a list of detected classes based on whether the generated response
    contains the word "yes".
    """
    video_path = os.path.join(dataset_path, "video", video_id)

    # Select appropriate conversation template based on modality
    if modality == "video":
        chat_state = CONV_VIDEO.copy()
    elif modality == "audio":
        chat_state = CONV_AUDIO.copy()
    else:  # image
        chat_state = CONV_VISION.copy()

    # Initialize chat with video
    img_list = []
    llm_message = chat.upload_img(video_path, chat_state, img_list, type=modality)

    detected = []
    response = ""

    # Process detection classes based on prompt mode
    if prompt_mode == "single":
        prompt_text = prompt.format(cl=", ".join(CLASSES))

        # Ask the question
        chat.ask(prompt_text, chat_state)

        # Get the answer
        response = chat.answer(
            conv=chat_state,
            img_list=img_list,
            num_beams=num_beams,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            max_length=max_length,
        )[0]

        # Check for detected classes in the response
        for cl in CLASSES:
            if cl.lower() in response.lower():
                detected.append(cl)

    elif prompt_mode == "multi":
        all_responses = []
        for cl in tqdm(CLASSES, desc="Processing classes", leave=False):
            # Reset the chat state for each class
            if modality == "video":
                class_chat_state = CONV_VIDEO.copy()
            elif modality == "audio":
                class_chat_state = CONV_AUDIO.copy()
            else:  # image
                class_chat_state = CONV_VISION.copy()

            # Initialize chat with video for each class
            class_img_list = []
            chat.upload_img(video_path, class_chat_state, class_img_list, type=modality)

            # Format prompt for this specific class
            prompt_text = prompt.format(cl=cl)

            # Ask the question
            chat.ask(prompt_text, class_chat_state)

            # Get the answer
            class_response = chat.answer(
                conv=class_chat_state,
                img_list=class_img_list,
                num_beams=num_beams,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                max_length=max_length,
            )[0]

            if "yes" in class_response.lower():
                detected.append(cl)

            all_responses.append(f"{cl}: {class_response}")

        response = ",".join(all_responses)

    else:
        raise ValueError(
            f"Invalid prompt mode: {prompt_mode}. Supported modes: 'single', 'multi'"
        )

    # Return the unique set of detected classes and the response
    return list(set(detected)), response


def main():
    # Parse arguments
    args = parse_args()

    # Load configuration
    cfg = Config(args)
    model_config = cfg.model_cfg
    model_config.device_8bit = args.gpu_id

    # Set up seeds for reproducibility
    setup_seeds(cfg)

    # Initialize model
    print("Initializing ChatBridge model")
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config)

    # Get model size
    model_size_mb = get_model_size(model)
    print(f"Model size: {model_size_mb:.2f} MB")

    # Move model to GPU
    model.to(f"cuda:{args.gpu_id}")
    model.eval()

    # Initialize processors
    vis_processor = AlproVideoEvalProcessor(image_size=224, n_frms=4)
    aud_processor = BlipAudioEvalProcessor()

    # Initialize chat
    chat = Chat(model, vis_processor, aud_processor, device=f"cuda:{args.gpu_id}")
    print("Initialization Finished")

    # Get video list
    video_list = get_video_list(args.video_csv)
    if not video_list:
        print("No videos found to process.")
        return

    # Process a subset (page) of videos
    page_videos = video_list[
        args.page * args.per_page : (args.page + 1) * args.per_page
    ]

    # Update output filename to include page and prompt mode
    args.output_csv = args.output_csv.replace(
        ".csv", f"_{args.prompt_mode}_page_{args.page}.csv"
    )

    predictions = {}
    responses = {}

    # Process each video
    for video_id in tqdm(page_videos, desc="Processing Videos"):
        try:
            detected_classes, response = process_video(
                chat=chat,
                dataset_path=args.dataset_path,
                video_id=video_id,
                temperature=args.temperature,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                max_length=args.max_length,
                modality=args.modality,
                prompt=args.prompt,
                prompt_mode=args.prompt_mode,
            )

            predictions[video_id] = detected_classes
            responses[video_id] = response

            # Periodically write predictions to CSV
            write_predictions_csv(predictions, responses, args.output_csv)

        except Exception as e:
            print(f"Error processing video {video_id}: {e}")
            continue

    print(f"Completed processing {len(page_videos)} videos.")
    print(f"Results saved to {args.output_csv}")


if __name__ == "__main__":
    main()
