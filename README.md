# 4K Video Upscaler for Colab

This project gives you a browser-based app for turning a normal video into a 4K video on a Google Colab GPU session.

It started from the goal of using [Video2X](https://github.com/k4yt3x/video2x), but the current Video2X Linux AppImage requires a newer `glibc` than Google Colab provides. To keep the workflow working in Colab, this app now uses the official [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) video pipeline directly.

## What this app does

- Upload a source video in the browser
- Run Real-ESRGAN in Colab with GPU acceleration
- Create either:
  - true 4K output at `3840x2160`, or
  - a `4x` upscale
- Preview the result in the browser
- Download the processed file

## Files

- `app.py`: Gradio browser app
- `requirements.txt`: Python dependency list
- `video2x_colab_app.ipynb`: Colab notebook that launches the browser app

## How to use

1. Open [video2x_colab_app.ipynb](https://colab.research.google.com/github/zyonneokid/VideoUpscale/blob/main/video2x_colab_app.ipynb) in Google Colab.
2. Run the notebook cells from top to bottom.
3. Open the Gradio public link shown by Colab.
4. Upload your video.
5. Choose a preset.
6. Keep `True 4K output (3840x2160)` selected if your goal is a 4K file.
7. Download the final output after processing finishes.

## Notes

- Free Colab sessions are temporary. Large videos can exceed memory, disk, or session-time limits.
- `General video` is the best starting point for most live-action clips.
- `Anime video` is tuned for anime and stylized content.
- `Sharp detail boost` can look crisper on some sources, but it can also exaggerate noise.

## Upstream References

- Video2X project: [k4yt3x/video2x](https://github.com/k4yt3x/video2x)
- Real-ESRGAN project: [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)
- Official video inference script: [inference_realesrgan_video.py](https://github.com/xinntao/Real-ESRGAN/blob/master/inference_realesrgan_video.py)
