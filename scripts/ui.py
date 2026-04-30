"""Launch the MouseHash chat UI (Gradio).

Usage:
    python scripts/ui.py
    python scripts/ui.py --model gpt-4o
    python scripts/ui.py --share          # public Gradio link

Requires ANTHROPIC_API_KEY (default model) or the appropriate key for the
chosen model.  Install the agents extra first:
    pip install -e ".[agents]"
"""
from __future__ import annotations

import argparse


def create_app(model_id: str):
    import gradio as gr

    from mousehash.agents.smolagents_adapter import make_coordinator
    from mousehash.agents.ui_helpers import (
        extract_plot_path,
        extract_plot_png_path,
        normalize_image_path,
        render_plot_iframe,
    )

    coordinator = make_coordinator(model_id=model_id)

    def submit_message(message: str, history: list[dict]):
        if not message.strip():
            return history, None, render_plot_iframe(None), ""

        response = coordinator.run(message, reset=False)
        response_text = response if isinstance(response, str) else str(response)
        plot_path = extract_plot_path(response_text)
        plot_png_path = extract_plot_png_path(response_text)
        history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": response_text},
        ]
        return history, normalize_image_path(plot_png_path), render_plot_iframe(plot_path), ""

    def clear_chat():
        coordinator.memory.reset()
        coordinator.monitor.reset()
        return [], None, render_plot_iframe(None), ""

    with gr.Blocks(title="MouseHash UI") as app:
        gr.Markdown("# MouseHash UI")
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(label="Agent", height=620)
                prompt = gr.Textbox(
                    label="Prompt",
                    placeholder=(
                        "Ask for a vanilla cell trace or an animate/inanimate plot, "
                        "for example: Plot the vanilla dF/F trace for cell_specimen_id 662275084"
                    ),
                )
                with gr.Row():
                    submit = gr.Button("Send", variant="primary")
                    clear = gr.Button("Clear")
            with gr.Column(scale=2):
                plot_image = gr.Image(
                    value=None,
                    label="Latest Plot Preview",
                    height=320,
                )
                plot_html = gr.HTML(
                    value=render_plot_iframe(None),
                    label="Latest Plotly Report",
                )

        submit.click(submit_message, inputs=[prompt, chatbot], outputs=[chatbot, plot_image, plot_html, prompt])
        prompt.submit(submit_message, inputs=[prompt, chatbot], outputs=[chatbot, plot_image, plot_html, prompt])
        clear.click(clear_chat, outputs=[chatbot, plot_image, plot_html, prompt])

    return app


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--model",
        default="claude-sonnet-4-6",
        help="LiteLLM model string (default: claude-sonnet-4-6)",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public Gradio share link",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Local port to serve the UI on (default: 7860)",
    )
    args = parser.parse_args()

    print(f"\nStarting MouseHash UI (model={args.model}) on http://localhost:{args.port}")
    print("Ask the coordinator to run the pipeline or query your analysis results.\n")

    create_app(args.model).launch(
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
