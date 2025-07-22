import os
import tempfile
import asyncio
import base64
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
import edge_tts
from bs4 import BeautifulSoup
import nest_asyncio

# 修复异步问题（Render 等云服务中必须）
nest_asyncio.apply()

# 初始化 OpenAI 客户端（环境变量中需设置 OPENAI_API_KEY）
client = OpenAI()

app = Flask(__name__)
CORS(app)

# 🔊 文本转语音（英国英语 Libby 女声）
async def text_to_speech(text):
    communicate = edge_tts.Communicate(text, "en-GB-LibbyNeural")
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        await communicate.save(f.name)
        with open(f.name, "rb") as audio_file:
            return base64.b64encode(audio_file.read()).decode("utf-8")

# 🌐 DuckDuckGo 网页搜索
def search_web(query):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        url = f"https://html.duckduckgo.com/html/?q={query}"
        res = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(res.text, "html.parser")
        results = soup.select("a.result__snippet")[:3]
        return "\n".join([r.get_text(strip=True) for r in results]) or "未能找到相关信息"
    except Exception as e:
        return f"搜索失败：{e}"

@app.route("/")
def home():
    return "联网语音助手 API 已启动", 200

# 🔁 语音对话接口
@app.route("/chat", methods=["POST"])
def chat():
    try:
        if 'audio' not in request.files:
            return jsonify({"error": "未上传音频"}), 400

        audio_file = request.files['audio']

        # 保存为临时文件（假设是 .wav 格式）
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio:
            audio_file.save(tmp_audio.name)

            # 语音识别：使用 Whisper
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=open(tmp_audio.name, "rb")
            )
            question = transcript.text
            print("🎤 用户问题：", question)

        # 执行联网搜索
        web_info = search_web(question)
        print("🌐 搜索内容：", web_info)

        # GPT 综合回答
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "你是一个结合网络搜索信息的语音助手"},
                {"role": "user", "content": f"问题：{question}\n\n搜索结果：\n{web_info}"}
            ]
        )
        reply = response.choices[0].message.content
        print("🤖 回复：", reply)

        # 合成语音
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        audio_base64 = loop.run_until_complete(text_to_speech(reply))
        loop.close()

        return jsonify({
            "text": reply,
            "audio_base64": audio_base64
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"服务器错误：{str(e)}"}), 500

# ✅ 入口
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
