import json
import os
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

import gradio as gr


RUNTIME_DIR = Path(os.environ.get("VIDEO2X_RUNTIME_DIR", ".video2x_runtime")).resolve()
PINNED_VERSION = "6.4.0"


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, destination.open("wb") as output_file:
        shutil.copyfileobj(response, output_file)


def _get_latest_tag() -> str:
    api_url = "https://api.github.com/repos/k4yt3x/video2x/releases/latest"
    request = urllib.request.Request(
        api_url,
        headers={"Accept": "application/vnd.github+json", "User-Agent": "video2x-colab-app"},
    )
    try:
        with urllib.request.urlopen(request) as response:
            data = json.load(response)
            return str(data["tag_name"]).lstrip("v")
    except (urllib.error.URLError, KeyError, json.JSONDecodeError):
        return PINNED_VERSION


def ensure_video2x_binary() -> Path:
    runtime_dir = RUNTIME_DIR
    binary_path = runtime_dir / "Video2X-x86_64.AppImage"
    if binary_path.exists():
        return binary_path

    version = _get_latest_tag()
    download_url = (
        f"https://github.com/k4yt3x/video2x/releases/download/{version}/Video2X-x86_64.AppImage"
    )
    _download(download_url, binary_path)
    binary_path.chmod(0o755)
    return binary_path


def build_command(
    binary_path: Path,
    input_path: Path,
    output_path: Path,
    preset: str,
    target_mode: str,
) -> list[str]:
    command = [str(binary_path), "-i", str(input_path), "-o", str(output_path)]

    if preset == "general":
        command.extend(
            ["-p", "realesrgan", "--realesrgan-model", "realesrgan-plus"]
        )
    elif preset == "anime":
        command.extend(
            ["-p", "realesrgan", "--realesrgan-model", "realesr-animevideov3"]
        )
    else:
        command.extend(
            ["-p", "libplacebo", "--libplacebo-shader", "anime4k-v4-a+a"]
        )

    if target_mode == "4k":
        command.extend(["-w", "3840", "-h", "2160"])
    else:
        command.extend(["-s", "4"])

    return command


def upscale_video(
    input_video: str | None,
    preset: str,
    target_mode: str,
    progress: gr.Progress = gr.Progress(),
):
    if not input_video:
        raise gr.Error("Upload a video first.")

    progress(0.05, desc="Preparing Video2X runtime")
    binary_path = ensure_video2x_binary()

    input_path = Path(input_video).resolve()
    output_dir = Path(tempfile.mkdtemp(prefix="video2x-output-"))
    output_path = output_dir / f"{input_path.stem}_4k.mp4"

    command = build_command(binary_path, input_path, output_path, preset, target_mode)
    env = os.environ.copy()
    env["APPIMAGE_EXTRACT_AND_RUN"] = "1"

    progress(0.15, desc="Running video upscaling")
    process = subprocess.run(
        command,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    logs = (process.stdout or "") + "\n" + (process.stderr or "")
    logs = logs.strip()

    if process.returncode != 0:
        raise gr.Error(f"Video2X failed.\n\n{logs[-3000:]}")

    progress(1.0, desc="Finished")
    if not output_path.exists():
        raise gr.Error("Video2X finished but no output file was produced.")

    return str(output_path), str(output_path), logs or "Completed successfully."


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Video2X 4K Upscaler") as demo:
        gr.Markdown(
            """
            # Video2X 4K Upscaler
            Upload a normal video, process it with Video2X on the Colab GPU,
            then preview and download the 4K result.
            """
        )

        with gr.Row():
            input_video = gr.Video(label="Input Video")
            preview_video = gr.Video(label="Upscaled Preview")

        with gr.Row():
            preset = gr.Dropdown(
                choices=[
                    ("General video (RealESRGAN)", "general"),
                    ("Anime video (RealESRGAN AnimeVideo)", "anime"),
                    ("Anime4K shader (fast)", "anime4k"),
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

        run_button = gr.Button("Upscale To 4K", variant="primary")
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
