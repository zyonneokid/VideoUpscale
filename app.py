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

MODEL_CONFIGS = {
    "RealESRGAN_x4plus": {
        "arch": "rrdb",
        "scale": 4,
        "urls": [
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        ],
        "rrdb": {"num_block": 23, "scale": 4},
    },
    "realesr-animevideov3": {
        "arch": "srvgg",
        "scale": 4,
        "urls": [
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth",
        ],
        "srvgg": {"num_conv": 16, "upscale": 4},
    },
    "realesr-general-x4v3": {
        "arch": "srvgg_dni",
        "scale": 4,
        "urls": [
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth",
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
        ],
        "srvgg": {"num_conv": 32, "upscale": 4},
    },
}


def ensure_runtime_ready() -> None:
    if not REAL_ESRGAN_DIR.exists():
        raise gr.Error("Real-ESRGAN is not installed in this session. Run the Colab setup cells again.")


def run_command(command: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        check=False,
    )


def probe_video(input_path: Path) -> dict:
    command = [
        FFPROBE_BIN,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(input_path),
    ]
    process = run_command(command)
    if process.returncode != 0:
        raise gr.Error(f"Could not read video metadata.\n\n{process.stderr.strip()}")

    payload = json.loads(process.stdout)
    stream = payload["streams"][0]
    fps_numerator, fps_denominator = stream["r_frame_rate"].split("/")
    fps = float(fps_numerator) / float(fps_denominator)
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": fps,
        "duration": float(payload["format"].get("duration", 0.0)),
    }


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
    return round(max(1.0, min(scale, 4.0)), 3)


def create_upsampler(model_name: str, tile_size: int):
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from basicsr.utils.download_util import load_file_from_url
    from realesrgan import RealESRGANer
    from realesrgan.archs.srvgg_arch import SRVGGNetCompact

    config = MODEL_CONFIGS[model_name]
    if config["arch"] == "rrdb":
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=config["rrdb"]["num_block"],
            num_grow_ch=32,
            scale=config["rrdb"]["scale"],
        )
    else:
        model = SRVGGNetCompact(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_conv=config["srvgg"]["num_conv"],
            upscale=config["srvgg"]["upscale"],
            act_type="prelu",
        )

    weights_dir = REAL_ESRGAN_DIR / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    model_path: str | list[str] = str(weights_dir / f"{model_name}.pth")
    for url in config["urls"]:
        model_path = load_file_from_url(url=url, model_dir=str(weights_dir), progress=True, file_name=None)

    dni_weight = None
    if config["arch"] == "srvgg_dni":
        model_path = [
            str(weights_dir / "realesr-general-x4v3.pth"),
            str(weights_dir / "realesr-general-wdn-x4v3.pth"),
        ]
        dni_weight = [0.5, 0.5]

    return RealESRGANer(
        scale=config["scale"],
        model_path=model_path,
        dni_weight=dni_weight,
        model=model,
        tile=tile_size,
        tile_pad=10,
        pre_pad=0,
        half=True,
    )


def extract_frames(input_path: Path, frames_dir: Path) -> str:
    frames_dir.mkdir(parents=True, exist_ok=True)
    command = [
        FFMPEG_BIN,
        "-y",
        "-i",
        str(input_path),
        "-vsync",
        "0",
        str(frames_dir / "frame%08d.png"),
    ]
    process = run_command(command)
    if process.returncode != 0:
        raise gr.Error(f"Frame extraction failed.\n\n{process.stderr.strip()[-3000:]}")
    return process.stderr.strip()


def upscale_frames(
    frames_dir: Path,
    output_frames_dir: Path,
    upsampler,
    outscale: float,
    progress: gr.Progress,
    progress_start: float,
    progress_end: float,
) -> str:
    import cv2

    frame_paths = sorted(frames_dir.glob("frame*.png"))
    if not frame_paths:
        raise gr.Error("No frames were extracted from the input video.")

    output_frames_dir.mkdir(parents=True, exist_ok=True)
    logs: list[str] = []
    total_frames = len(frame_paths)

    for index, frame_path in enumerate(frame_paths, start=1):
        image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if image is None:
            raise gr.Error(f"Could not read extracted frame: {frame_path.name}")
        try:
            output, _ = upsampler.enhance(image, outscale=outscale)
        except RuntimeError as error:
            raise gr.Error(f"Frame upscaling failed on {frame_path.name}.\n\n{error}") from error

        output_path = output_frames_dir / frame_path.name
        if not cv2.imwrite(str(output_path), output):
            raise gr.Error(f"Could not write upscaled frame: {output_path.name}")

        ratio = index / total_frames
        progress(
            progress_start + (progress_end - progress_start) * ratio,
            desc=f"Upscaling frames ({index}/{total_frames})",
        )
        if index == 1 or index == total_frames or index % 30 == 0:
            logs.append(f"Processed frame {index}/{total_frames}")

    return "\n".join(logs)


def encode_video(
    output_frames_dir: Path,
    input_path: Path,
    final_path: Path,
    fps: float,
    target_mode: str,
) -> str:
    if target_mode == "4k":
        video_filter = "pad=3840:2160:(ow-iw)/2:(oh-ih)/2:color=black"
    else:
        video_filter = "null"

    command = [
        FFMPEG_BIN,
        "-y",
        "-framerate",
        f"{fps}",
        "-i",
        str(output_frames_dir / "frame%08d.png"),
        "-i",
        str(input_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a?",
        "-vf",
        video_filter,
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-shortest",
        str(final_path),
    ]
    process = run_command(command)
    if process.returncode != 0:
        raise gr.Error(f"Final video encoding failed.\n\n{process.stderr.strip()[-3000:]}")
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
    outscale = calculate_outscale(metadata["width"], metadata["height"], target_mode)
    model_name = choose_model(preset)
    tile_size = 128 if max(metadata["width"], metadata["height"]) >= 1280 else 0

    workspace_dir = Path(tempfile.mkdtemp(prefix="realesrgan-work-"))
    frames_dir = workspace_dir / "frames"
    output_frames_dir = workspace_dir / "upscaled_frames"
    final_path = workspace_dir / f"{input_path.stem}_{target_mode}.mp4"

    progress(0.05, desc="Extracting video frames")
    logs = [extract_frames(input_path, frames_dir)]

    progress(0.2, desc="Loading Real-ESRGAN model")
    upsampler = create_upsampler(model_name, tile_size)

    frame_logs = upscale_frames(
        frames_dir=frames_dir,
        output_frames_dir=output_frames_dir,
        upsampler=upsampler,
        outscale=outscale,
        progress=progress,
        progress_start=0.25,
        progress_end=0.9,
    )
    if frame_logs:
        logs.append(frame_logs)

    progress(0.92, desc="Encoding final video")
    logs.append(
        encode_video(
            output_frames_dir=output_frames_dir,
            input_path=input_path,
            final_path=final_path,
            fps=metadata["fps"],
            target_mode=target_mode,
        )
    )

    progress(1.0, desc="Finished")
    return str(final_path), str(final_path), "\n\n".join(part for part in logs if part).strip()


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
