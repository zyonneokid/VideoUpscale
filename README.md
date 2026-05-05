# Video2X Colab 4K Upscaler

This project gives you a browser-based app for turning a normal video into a 4K video with [Video2X](https://github.com/k4yt3x/video2x) on a Google Colab GPU session.

## What this app does

- Upload a source video in the browser
- Run Video2X in Colab
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

1. Upload this project folder to Google Drive or a GitHub repo that Colab can open from.
2. Open [video2x_colab_app.ipynb](/D:/Hari/MyProjects/VideoUpscaler/video2x_colab_app.ipynb) in Google Colab from that folder.
3. Make sure `app.py` is in the same Colab working directory as the notebook.
4. Run the notebook cells from top to bottom.
5. Open the Gradio public link shown by Colab.
6. Upload your video.
7. Choose an upscaling preset.
8. Keep `True 4K output (3840x2160)` selected if your goal is a 4K file.
9. Download the final output after processing finishes.

## Notes

- The app downloads the latest Video2X Linux AppImage at runtime and falls back to `6.4.0` if the latest-release lookup fails.
- Free Colab sessions are temporary. Large videos can exceed memory, disk, or session-time limits.
- RealESRGAN is usually the best starting point for live-action and mixed content.
- Anime4K is faster, but it is more stylized and usually better for anime-like sources.
- Video2X is licensed under AGPL-3.0. See the upstream project for licensing details.

## Upstream References

- Project: [k4yt3x/video2x](https://github.com/k4yt3x/video2x)
- Docs: [docs.video2x.org](https://docs.video2x.org/)
- Command-line examples: [running/command-line.html](https://docs.video2x.org/running/command-line.html)
