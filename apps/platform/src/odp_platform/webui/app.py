from __future__ import annotations

import logging
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

import gradio as gr

from odp_platform.common.logging_utils import get_logger
from odp_platform.common.paths import LOGGING_DIR
from odp_platform.webui.config_tab import create_config_ui
from odp_platform.webui.dashboard import create_dashboard_ui
from odp_platform.webui.dataset_analysis import create_dataset_analysis_ui
from odp_platform.webui.dataset_browser import create_dataset_browser_ui
from odp_platform.webui.model_demo import create_model_demo_ui
from odp_platform.webui.training_tab import create_training_ui
from odp_platform.webui.validation_tab import create_validation_ui

logger = logging.getLogger(__name__)

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
WALLPAPER_PATH = ASSETS_DIR / "wallpaper-prism.png"
WALLPAPER_URL = f"/gradio_api/file={WALLPAPER_PATH.as_posix()}"

APP_CSS = """
:root {
  --odp-ink: #111827;
  --odp-white: #f8fbff;
  --odp-muted: #3b4a5c;
  --odp-line: rgba(255, 255, 255, 0.46);
  --odp-glass: rgba(255, 255, 255, 0.18);
  --odp-glass-strong: rgba(255, 255, 255, 0.28);
  --odp-tint-blue: #3a79f7;
  --odp-tint-teal: #13a6a1;
  --odp-tint-coral: #d76c55;
  --odp-text-shadow: 0 1px 1px rgba(0, 0, 0, 0.48), 0 2px 3px rgba(0, 0, 0, 0.22);
  --odp-text-shadow-soft: 0 1px 1px rgba(0, 0, 0, 0.36), 0 2px 2px rgba(0, 0, 0, 0.14);
}

html,
body,
gradio-app,
.gradio-container {
  min-height: 100vh;
  color: var(--odp-ink) !important;
  background:
    linear-gradient(180deg, rgba(255, 255, 255, 0.18), rgba(255, 255, 255, 0.04) 42%, rgba(16, 24, 36, 0.18)),
    url("__WALLPAPER_URL__") center center / cover fixed no-repeat !important;
  font-family: Inter, -apple-system, BlinkMacSystemFont, "SF Pro Display", "SF Pro Text", "Segoe UI", sans-serif !important;
}

gradio-app {
  background: transparent !important;
}

.gradio-container::before {
  content: "";
  position: fixed;
  inset: 0;
  pointer-events: none;
  background:
    radial-gradient(circle at 24% 12%, rgba(255, 255, 255, 0.48), transparent 28%),
    radial-gradient(circle at 80% 8%, rgba(255, 255, 255, 0.34), transparent 24%),
    linear-gradient(180deg, rgba(255, 255, 255, 0.1), rgba(255, 255, 255, 0.03));
  mix-blend-mode: screen;
}

.odp-shell {
  width: min(1680px, calc(100vw - 88px));
  max-width: none;
  margin: 0 auto;
  padding: 24px 0 48px;
}

.block.odp-title {
  position: relative;
  display: flex !important;
  justify-content: center !important;
  padding: 34px 24px 58px !important;
  margin-bottom: 4px !important;
  border: 0 !important;
  border-radius: 0 !important;
  background: transparent !important;
  box-shadow: none !important;
  backdrop-filter: none !important;
  -webkit-backdrop-filter: none !important;
  overflow: visible !important;
}

.block.odp-title > div {
  width: 100%;
}

.block.odp-title .prose,
.block.odp-title .odp-title-art {
  position: relative;
  z-index: 1;
  text-align: center !important;
}

.odp-title::after {
  content: "";
  position: absolute;
  inset: 1px;
  border-radius: 29px;
  display: none;
  background: transparent;
  pointer-events: none;
  z-index: 0;
}

.odp-title .odp-glass-word {
  display: block;
  width: min(74vw, 900px);
  max-width: 100%;
  height: auto;
  margin: 0 auto;
  overflow: visible;
  color: transparent !important;
  background: none !important;
  filter:
    drop-shadow(0 -1px 0 rgba(255, 255, 255, 0.16))
    drop-shadow(0 1px 2px rgba(0, 0, 0, 0.12));
  isolation: isolate;
  mix-blend-mode: screen;
}

.odp-title .odp-glass-word text {
  font-family: Inter, -apple-system, BlinkMacSystemFont, "SF Pro Display", "SF Pro Text", "Segoe UI", sans-serif;
  font-size: 122px;
  font-weight: 900;
  letter-spacing: 0;
}

.odp-title .odp-title-fill {
  fill: #f8fcff;
  stroke: none;
  opacity: 0.29;
}

.odp-title .odp-title-stroke {
  fill: none;
  stroke: #ffffff;
  stroke-width: 1.5px;
  stroke-linejoin: round;
  stroke-linecap: round;
  opacity: 0.48;
}

.odp-title .odp-glass-word::before,
.odp-title .odp-glass-word::after {
  content: attr(data-text);
  position: absolute;
  inset: 0;
  pointer-events: none;
  display: none !important;
}

.odp-title .odp-glass-word::before {
  z-index: -1;
  color: transparent;
  -webkit-text-stroke: 0;
  filter: none;
}

.odp-title .odp-glass-word::after {
  color: transparent;
  background: none !important;
  background-clip: text !important;
  -webkit-background-clip: text !important;
  -webkit-text-fill-color: transparent !important;
  opacity: 0;
  mix-blend-mode: normal;
}

.odp-title .odp-title-caption,
.odp-title p {
  display: block;
  width: fit-content;
  max-width: 100%;
  margin: 14px auto 0;
  color: rgba(248, 251, 255, 0.88) !important;
  font-size: clamp(18px, 1.8vw, 28px);
  font-weight: 650;
  text-shadow: var(--odp-text-shadow-soft) !important;
  filter: none !important;
}

.odp-title * {
  opacity: 1 !important;
}

.tabs,
.tab-nav,
.tabitem,
.block,
.form,
.panel,
.gap,
.contain {
  border-color: rgba(255, 255, 255, 0.5) !important;
}

.tabs {
  border-bottom: 0 !important;
  box-shadow: none !important;
  background: transparent !important;
}

.tabs::before,
.tabs::after,
.tab-container::before,
.tab-container::after,
.tab-wrapper::before,
.tab-wrapper::after {
  display: none !important;
  content: none !important;
}

.tab-wrapper,
.tab-container {
  overflow: visible !important;
}

.tab-container.visually-hidden {
  display: none !important;
}

.tab-nav {
  display: inline-flex !important;
  gap: 8px;
  overflow: visible !important;
  padding: 10px 16px 16px !important;
  margin: 0 0 8px -16px !important;
  border: 0 !important;
  border-radius: 999px !important;
  background: transparent !important;
  box-shadow: none !important;
  backdrop-filter: none !important;
  -webkit-backdrop-filter: none !important;
}

.tab-container:not(.visually-hidden) {
  display: inline-flex !important;
  gap: 8px;
  overflow: visible !important;
  padding: 10px 16px 16px !important;
  margin: 0 0 18px -16px !important;
  border: 0 !important;
  background: transparent !important;
  box-shadow: none !important;
}

.tabitem > .column,
.tabs > .column {
  gap: 20px !important;
}

.odp-row {
  display: grid !important;
  width: 100%;
  gap: 20px !important;
  align-items: stretch !important;
  margin-bottom: 20px !important;
}

.odp-row > * {
  width: 100% !important;
  min-width: 0 !important;
  height: 100% !important;
}

.odp-row > .form {
  display: contents !important;
}

.odp-row > .form > * {
  width: 100% !important;
  min-width: 0 !important;
  height: 100% !important;
}

.odp-row > .form,
.odp-row > .block,
.odp-row > button,
.odp-row > .form > .block,
.odp-row > .form > label,
.odp-row > .form > div {
  min-height: 96px !important;
}

.odp-row > button {
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
}

.odp-row-action {
  grid-template-columns: minmax(240px, 0.7fr) minmax(420px, 2fr);
}

.odp-row-two {
  grid-template-columns: repeat(2, minmax(0, 1fr));
}

.odp-row-three {
  grid-template-columns: repeat(3, minmax(0, 1fr));
}

.odp-row-four {
  grid-template-columns: repeat(4, minmax(0, 1fr));
}

.odp-row-five {
  grid-template-columns: repeat(5, minmax(0, 1fr));
}

.tabitem > .column > .block:not(.odp-title),
.tabitem > .column > button {
  width: 100% !important;
}

.tabitem > .column > .block:has(.cm-editor),
.tabitem > .column > .block:has(.json-holder),
.tabitem > .column > .block:has(.image-container),
.tabitem > .column > .block:has(.upload-container) {
  min-height: 320px !important;
}

@media (max-width: 1180px) {
  .odp-row-action,
  .odp-row-three,
  .odp-row-four,
  .odp-row-five {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}

@media (max-width: 760px) {
  .odp-shell {
    width: min(100% - 32px, 1680px);
  }

  .odp-row,
  .odp-row-action,
  .odp-row-two,
  .odp-row-three,
  .odp-row-four,
  .odp-row-five {
    grid-template-columns: 1fr;
  }
}

.tab-nav button,
.tab-container:not(.visually-hidden) button[role="tab"] {
  position: relative !important;
  z-index: 1;
  min-height: 38px !important;
  padding: 0 18px !important;
  border: 1px solid rgba(255, 255, 255, 0.36) !important;
  border-radius: 999px !important;
  color: var(--odp-white) !important;
  background:
    linear-gradient(145deg, rgba(255, 255, 255, 0.12), rgba(255, 255, 255, 0.035)) !important;
  box-shadow:
    inset 0 1px 0 rgba(255, 255, 255, 0.34),
    0 10px 24px rgba(5, 10, 18, 0.1) !important;
  backdrop-filter: blur(20px) saturate(1.45);
  -webkit-backdrop-filter: blur(20px) saturate(1.45);
  font-weight: 650 !important;
  transform-origin: center center;
  transition: transform 160ms ease, background 160ms ease, border-color 160ms ease, box-shadow 160ms ease;
}

.tab-nav button::before,
.tab-nav button::after,
.tab-nav button.selected::before,
.tab-nav button.selected::after,
.tab-nav button[aria-selected="true"]::before,
.tab-nav button[aria-selected="true"]::after,
.tab-container:not(.visually-hidden) button[role="tab"]::before,
.tab-container:not(.visually-hidden) button[role="tab"]::after,
.tab-container:not(.visually-hidden) button[role="tab"].selected::before,
.tab-container:not(.visually-hidden) button[role="tab"].selected::after,
.tab-container:not(.visually-hidden) button[role="tab"][aria-selected="true"]::before,
.tab-container:not(.visually-hidden) button[role="tab"][aria-selected="true"]::after {
  display: none !important;
  content: none !important;
}

.tab-nav button.selected,
.tab-nav button[aria-selected="true"],
.tab-container:not(.visually-hidden) button[role="tab"].selected,
.tab-container:not(.visually-hidden) button[role="tab"][aria-selected="true"] {
  z-index: 2;
  color: var(--odp-white) !important;
  background:
    linear-gradient(145deg, rgba(255, 255, 255, 0.18), rgba(255, 255, 255, 0.055)) !important;
  border-color: rgba(255, 255, 255, 0.56) !important;
  box-shadow:
    inset 0 1px 0 rgba(255, 255, 255, 0.42),
    0 12px 28px rgba(5, 10, 18, 0.14) !important;
  transform: scale(1.12);
}

.block,
.panel {
  border: 1px solid rgba(255, 255, 255, 0.36) !important;
  border-radius: 20px !important;
  background:
    linear-gradient(145deg, rgba(255, 255, 255, 0.12), rgba(255, 255, 255, 0.035)) !important;
  box-shadow:
    inset 0 1px 0 rgba(255, 255, 255, 0.34),
    0 16px 42px rgba(5, 10, 18, 0.12) !important;
  backdrop-filter: blur(20px) saturate(1.45);
  -webkit-backdrop-filter: blur(20px) saturate(1.45);
  overflow: hidden !important;
}

.form {
  border: 0 !important;
  background: transparent !important;
  box-shadow: none !important;
  backdrop-filter: none !important;
  -webkit-backdrop-filter: none !important;
  overflow: visible !important;
}

.block.odp-title,
.odp-title {
  border-radius: 0 !important;
}

.block.odp-title > div > .prose.odp-title {
  margin: 0 !important;
  padding: 0 !important;
  border: 0 !important;
  border-radius: 0 !important;
  background: transparent !important;
  box-shadow: none !important;
  backdrop-filter: none;
  -webkit-backdrop-filter: none;
}

.block.odp-title::after,
.odp-title::after {
  border-radius: 29px !important;
}

.block > .label-wrap,
.form > .label-wrap,
.panel > .label-wrap,
.label-wrap {
  position: relative !important;
  inset: auto !important;
  z-index: 2 !important;
  display: flex !important;
  min-height: 30px !important;
  align-items: center !important;
  margin: 0 !important;
  padding: 10px 14px 5px !important;
  border: 0 !important;
  background: transparent !important;
  color: var(--odp-white) !important;
  text-shadow: var(--odp-text-shadow);
}

button {
  border-radius: 16px !important;
  border: 1px solid rgba(255, 255, 255, 0.36) !important;
  background:
    linear-gradient(145deg, rgba(255, 255, 255, 0.12), rgba(255, 255, 255, 0.035)) !important;
  color: var(--odp-white) !important;
  box-shadow:
    inset 0 1px 0 rgba(255, 255, 255, 0.34),
    0 16px 42px rgba(5, 10, 18, 0.12) !important;
  backdrop-filter: blur(20px) saturate(1.45);
  -webkit-backdrop-filter: blur(20px) saturate(1.45);
  text-shadow: var(--odp-text-shadow-soft);
  transition: transform 160ms ease, box-shadow 160ms ease, background 160ms ease;
}

button:hover {
  transform: translateY(-1px);
  background:
    linear-gradient(145deg, rgba(255, 255, 255, 0.15), rgba(255, 255, 255, 0.045)) !important;
}

button:active {
  transform: translateY(1px) scale(0.99);
}

button.primary,
button.primary:hover {
  color: var(--odp-white) !important;
  border-color: rgba(255, 255, 255, 0.38) !important;
  background:
    linear-gradient(145deg, rgba(255, 255, 255, 0.13), rgba(255, 255, 255, 0.04)) !important;
  box-shadow:
    inset 0 1px 0 rgba(255, 255, 255, 0.34),
    0 16px 42px rgba(5, 10, 18, 0.12) !important;
}

input,
textarea,
select,
.wrap,
.wrap-inner,
.input-container,
.dataframe,
.table-wrap {
  border-radius: 16px !important;
}

input,
textarea,
select,
.input-container {
  border-color: rgba(255, 255, 255, 0.38) !important;
  background: transparent !important;
  color: var(--odp-white) !important;
  box-shadow: none !important;
  backdrop-filter: none !important;
  -webkit-backdrop-filter: none !important;
  text-shadow: var(--odp-text-shadow-soft);
}

textarea {
  min-height: 34px !important;
  max-height: 38px !important;
  resize: none !important;
  overflow: hidden !important;
  white-space: pre !important;
  overflow-wrap: normal !important;
}

.block > label.block,
.block > label.container,
.block > .container.block {
  display: flex !important;
  flex-direction: column !important;
  justify-content: center !important;
  width: 100% !important;
  min-height: 0 !important;
  margin: 0 !important;
  padding: 12px 14px !important;
  border: 0 !important;
  border-radius: 0 !important;
  background: transparent !important;
  box-shadow: none !important;
  backdrop-filter: none !important;
  -webkit-backdrop-filter: none !important;
  overflow: visible !important;
}

.block > label.block > span:first-child,
.block > label.container > span:first-child,
.block > .container.block > span:first-child {
  position: static !important;
  z-index: 1 !important;
  display: block !important;
  margin: 0 0 10px !important;
  padding: 0 !important;
  border: 0 !important;
  border-radius: 0 !important;
  background: transparent !important;
}

.block > label.block > input,
.block > label.container > input,
.block > .container.block > input {
  width: 100% !important;
  min-height: 34px !important;
  padding: 0 !important;
  border: 0 !important;
}

label,
.label-wrap,
.prose,
.markdown {
  color: var(--odp-ink) !important;
}

.label-wrap,
.block > label,
.block > span,
.panel > label,
.form > label,
.markdown > p:first-child,
.prose > p:first-child {
  color: var(--odp-white) !important;
  text-shadow: var(--odp-text-shadow);
}

.label-wrap *,
.block > label *,
.block > span *,
.panel > label *,
.form > label * {
  background: transparent !important;
  color: var(--odp-white) !important;
  text-shadow: var(--odp-text-shadow);
}

.block > .container > span,
.form > .container > span,
.panel > .container > span,
.block span.svelte-g2oxp3,
.form span.svelte-g2oxp3,
.panel span.svelte-g2oxp3,
.block > label > span,
.form > label > span,
.panel > label > span,
.block label.float,
.form label.float,
.panel label.float {
  background: transparent !important;
  border: 0 !important;
  color: var(--odp-white) !important;
  text-shadow: var(--odp-text-shadow) !important;
}

.block label:has(input[type="radio"]),
.block label.svelte-1bx8sav {
  border: 1px solid rgba(255, 255, 255, 0.48) !important;
  border-radius: 12px !important;
  background: rgba(255, 255, 255, 0.12) !important;
  color: var(--odp-white) !important;
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.36) !important;
  text-shadow: var(--odp-text-shadow-soft) !important;
}

.block label:has(input[type="radio"]:checked),
.block label.svelte-1bx8sav.selected {
  background:
    linear-gradient(135deg, rgba(58, 121, 247, 0.72), rgba(19, 166, 161, 0.52)) !important;
  color: var(--odp-white) !important;
}

.dataframe,
.table-wrap,
table {
  background: transparent !important;
  border-color: rgba(255, 255, 255, 0.26) !important;
  box-shadow: none !important;
  backdrop-filter: none !important;
  -webkit-backdrop-filter: none !important;
}

.dataframe tbody,
.dataframe thead,
.dataframe tr,
.dataframe td,
.dataframe th,
.table-wrap tbody,
.table-wrap thead,
.table-wrap tr,
.table-wrap td,
.table-wrap th {
  background: transparent !important;
  box-shadow: none !important;
}

.table-container .header-row {
  min-height: 36px !important;
  align-items: center !important;
  padding: 10px 14px 4px !important;
  border: 0 !important;
  background: transparent !important;
}

.table-container .label,
.table-container .label p {
  margin: 0 !important;
  padding: 0 !important;
  background: transparent !important;
  color: var(--odp-white) !important;
  font-weight: 650 !important;
  text-shadow: var(--odp-text-shadow) !important;
}

.table-wrap th .cell-wrap,
.table-wrap th .header-content,
.table-wrap th button,
.table-wrap th .header-button,
.table-wrap th .cell-menu-button {
  border: 0 !important;
  border-radius: 0 !important;
  background: transparent !important;
  box-shadow: none !important;
  backdrop-filter: none !important;
  -webkit-backdrop-filter: none !important;
}

.table-wrap th button:hover {
  transform: none !important;
  background: transparent !important;
}

.table-container button,
.table-wrap button,
.dataframe button {
  padding: 0 !important;
  border: 0 !important;
  border-radius: 0 !important;
  background: transparent !important;
  box-shadow: none !important;
  backdrop-filter: none !important;
  -webkit-backdrop-filter: none !important;
  text-shadow: inherit !important;
}

.table-container button:hover,
.table-wrap button:hover,
.dataframe button:hover {
  transform: none !important;
  background: transparent !important;
}

th {
  border-radius: 0 !important;
  background: transparent !important;
  color: var(--odp-white) !important;
  font-weight: 700 !important;
  box-shadow: none !important;
  text-shadow: var(--odp-text-shadow-soft);
}

td {
  background: transparent !important;
  color: var(--odp-white) !important;
  box-shadow: none !important;
  text-shadow: var(--odp-text-shadow-soft);
}

.wrap,
.wrap-inner,
.contain,
.input-container > *,
.table-wrap > *,
.dataframe > * {
  background: transparent !important;
  box-shadow: none !important;
}

.caption,
caption,
.table-wrap caption {
  display: none !important;
}

.table-wrap {
  overflow: hidden !important;
}

code,
pre,
.cm-editor,
.cm-scroller {
  border-radius: 16px !important;
  background: rgba(15, 23, 35, 0.88) !important;
  color: #eef7ff !important;
}

footer { display: none !important; }
""".replace("__WALLPAPER_URL__", WALLPAPER_URL)


def create_app() -> gr.Blocks:
    get_logger(
        base_path=LOGGING_DIR,
        log_type="webui",
        log_level=logging.INFO,
        logger_name="odp-webui",
    )
    logger.info("创建 ODPlatform Gradio UI")

    with gr.Blocks(
        title="ODPlatform",
        theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
        css=APP_CSS,
    ) as app:
        with gr.Column(elem_classes=["odp-shell"]):
            gr.HTML(
                """
                <section class="odp-title-art" aria-label="ODPlatform 目标检测开发平台">
                    <svg class="odp-glass-word" viewBox="0 0 940 160" role="img" aria-label="ODPlatform">
                        <text class="odp-title-fill" x="470" y="90" text-anchor="middle" dominant-baseline="middle">ODPlatform</text>
                        <text class="odp-title-stroke" x="470" y="90" text-anchor="middle" dominant-baseline="middle">ODPlatform</text>
                    </svg>
                    <div class="odp-title-caption">目标检测开发平台</div>
                </section>
                """,
                elem_classes=["odp-title"],
            )
            with gr.Tabs():
                with gr.TabItem("Dashboard"):
                    create_dashboard_ui()
                with gr.TabItem("数据集浏览"):
                    create_dataset_browser_ui()
                with gr.TabItem("数据集分析"):
                    create_dataset_analysis_ui()
                with gr.TabItem("训练"):
                    create_training_ui()
                with gr.TabItem("模型演示"):
                    create_model_demo_ui()
                with gr.TabItem("数据校验"):
                    create_validation_ui()
                with gr.TabItem("配置管理"):
                    create_config_ui()
    return app


def main() -> None:
    create_app().launch(
        server_name="0.0.0.0",
        server_port=7860,
        allowed_paths=[str(ASSETS_DIR)],
    )


if __name__ == "__main__":
    main()
