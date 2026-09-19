# -*- coding: utf-8 -*-
"""用 gradio 托管自包含的全双工自动回放页，--share 暴露公网链接。
gradio 6 的 gr.HTML 通过 innerHTML 注入，不会执行 <script>，
因此播放器 JS 必须经 head 参数注入；body 只保留结构与数据。
"""
import os, re
import gradio as gr

DIR = os.path.dirname(os.path.abspath(__file__))
HTML = open(os.path.join(DIR, "full_duplex_demo.html"), encoding="utf-8").read()

# 提取执行逻辑 JS（body 里那个"function boot()"脚本），不包含 appdata 数据脚本
m = re.search(r"<script>\s*function boot\(\)\{(?:.|\n)*?\}\n?if \(document\.readyState[\s\S]*?</script>", HTML)
if m:
    exec_js = m.group(0).replace("<script>", "").replace("</script>", "")
    body = HTML.replace(m.group(0), "")
    head = "<script>" + exec_js + "</script>"
else:
    body, head = HTML, ""

with gr.Blocks(title="全双工自动回放", theme=gr.themes.Base(primary_hue="blue")) as demo:
    gr.HTML(body, head=head, full_width=True)

demo.launch(share=True, server_name="0.0.0.0", server_port=7860, show_error=True)