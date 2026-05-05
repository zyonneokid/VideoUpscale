import json
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import gradio as gr


REAL_ESRGAN_DIR = Path(os.environ.get("REAL_ESRGAN_DIR", "/content/Real-ESRGAN")).resolve()
FFMPEG_BIN = os.environ.get("FFMPEG_BIN", "ffmpeg")
FFPROBE_BIN = os.environ.get("FFPROBE_BIN", "ffprobe")
MAX_TILE_SIZE = 256


def ensure_runtime_ready() -> None:
    script_path = REAL_ESRGAN_DIR / "inference_realesrgan_video.py"
    if not script_path.exists():
        raise gr.Error(
            "Real-ESRGAN is not installed in this session. Run the Colab setup cells again."
        )


def probe_video(input_path: Path) -> dict:
    command = [
        FFPROBE_BIN,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "json",
        str(input_path),
    ]
    process = subprocess.run(command, capture_output=True, text=True, check=False)
    if process.returncode != 0:
        raise gr.Error(f"Could not read video metadata.\n\n{process.stderr.strip()}")

    payload = json.loads(process.stdout)
    stream = payload["streams"][0]
    return {"width": int(stream["width"]), "height": int(stream["height"])}


def choose_model(preset: str) -> str:
    if preset == "general":
        return "realesr-general-x4v3"
    if preset == "anime":
        return "realesr-animevideov3"
    return "RealESRGAN_x4plus"


def calculate_outscale(width: int, height: int, target_mode: str) -> float:
    if target_mode == "4x":
        return 4.0

    scale = min(3840 / width, 2160 / height)
    scale = max(1.0, min(scale, 4.0))
    return round(scale, 3)


def build_inference_command(
    input_path: Path,
    output_dir: Path,
    model_name: str,
    outscale: float,
    tile_size: int,
) -> list[str]:
    return [
        "python",
        "inference_realesrgan_video.py",
        "-i",
        str(input_path),
        "-o",
        str(output_dir),
        "-n",
        model_name,
        "-s",
        str(outscale),
        "--suffix",
        "upscaled",
        "-t",
        str(tile_size),
    ]


def finalize_to_4k(source_path: Path, final_path: Path) -> str:
    command = [
        FFMPEG_BIN,
        "-y",
        "-i",
        str(source_path),
        "-vf",
        "scale=3840:2160:force_original_aspect_ratio=decrease,pad=3840:2160:(ow-iw)/2:(oh-ih)/2",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-c:a",
        "copy",
        str(final_path),
    ]
    process = subprocess.run(command, capture_output=True, text=True, check=False)
    if process.returncode != 0:
        raise gr.Error(f"4K packaging failed.\n\n{process.stderr.strip()[-3000:]}")
    return process.stderr.strip()


def upscale_video(
    input_video: str | None,
    preset: str,
    target_mode: str,
    progress: gr.Progress = gr.Progress(),
):
    if not input_video:
        raise gr.Error("Upload a video first.")

    ensure_runtime_ready()
    input_path = Path(input_video).resolve()
    metadata = probe_video(input_path)
    width = metadata["width"]
    height = metadata["height"]
    outscale = calculate_outscale(width, height, target_mode)
    model_name = choose_model(preset)

    output_dir = Path(tempfile.mkdtemp(prefix="realesrgan-output-"))
    final_path = output_dir / f"{input_path.stem}_{target_mode}.mp4"
    tile_size = MAX_TILE_SIZE if max(width, height) >= 1280 else 0

    progress(0.1, desc="Preparing Real-ESRGAN")
    command = build_inference_command(input_path, output_dir, model_name, outscale, tile_size)
    process = subprocess.run(
        command,
        cwd=str(REAL_ESRGAN_DIR),
        capture_output=True,
        text=True,
        check=False,
    )

    logs = (process.stdout or "") + "\n" + (process.stderr or "")
    logs = logs.strip()
    if process.returncode != 0:
        raise gr.Error(f"Upscaling failed.\n\n{logs[-3000:]}")

    raw_output = output_dir / f"{input_path.stem}_upscaled.mp4"
    if not raw_output.exists():
        candidates = sorted(output_dir.glob("*.mp4"))
        if not candidates:
            raise gr.Error("Real-ESRGAN finished but no output file was produced.")
        raw_output = candidates[0]

    progress(0.85, desc="Finalizing output")
    final_logs = logs
    if target_mode == "4k":
        ffmpeg_logs = finalize_to_4k(raw_output, final_path)
        final_logs = f"{logs}\n\n{ffmpeg_logs}".strip()
    else:
        shutil.move(str(raw_output), str(final_path))

    progress(1.0, desc="Finished")
    return str(final_path), str(final_path), final_logs or "Completed successfully."


def build_app() -> gr.Blocks:
    with gr.Blocks(title="4K Video Upscaler") as demo:
        gr.Markdown(
            """
            # 4K Video Upscaler
            Upload a normal video, process it with Real-ESRGAN on Colab GPU,
            then preview and download the upscaled result.
            """
        )

        with gr.Row():
            input_video = gr.Video(label="Input Video")
            preview_video = gr.Video(label="Upscaled Preview")

        with gr.Row():
            preset = gr.Dropdown(
                choices=[
                    ("General video", "general"),
                    ("Anime video", "anime"),
                    ("Sharp detail boost", "detail"),
                ],
                value="general",
                label="Upscaling Preset",
            )
            target_mode = gr.Radio(
                choices=[
                    ("True 4K output (3840x2160)", "4k"),
                    ("4x upscale", "4x"),
                ],
                value="4k",
                label="Output Mode",
            )

        run_button = gr.Button("Upscale Video", variant="primary")
        download_file = gr.File(label="Download Output")
        logs = gr.Textbox(label="Processing Logs", lines=16)

        run_button.click(
            fn=upscale_video,
            inputs=[input_video, preset, target_mode],
            outputs=[preview_video, download_file, logs],
        )

    return demo


if __name__ == "__main__":
    demo = build_app()
    demo.queue(default_concurrency_limit=1).launch(share=True)
