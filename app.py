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
from pydub import AudioSegment
import nest_asyncio

# 修复异步问题（Render 上用）
nest_asyncio.apply()

# 初始化 OpenAI 客户端（需设置 OPENAI_API_KEY 环境变量）
client = OpenAI()
app = Flask(__name__)
CORS(app)

# 文本转语音，返回 base64 音频字符串
async def text_to_speech(text):
    communicate = edge_tts.Communicate(text, "zh-CN-XiaoxiaoNeural")
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        await communicate.save(f.name)
        with open(f.name, "rb") as audio_file:
            return base64.b64encode(audio_file.read()).decode("utf-8")

# 联网搜索，默认用百度
def search_web(query):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        url = f"https://www.google.com/s?wd={query}"
        r = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(r.text, "html.parser")
        snippets = soup.select("div.c-abstract")[:3]
        results = "\n".join([s.get_text(strip=True) for s in snippets if s.get_text()])
        return results or "未能找到相关信息"
    except Exception as e:
        return f"搜索失败：{str(e)}"

@app.route("/")
def home():
    return "联网语音助手 API 已启动", 200

@app.route('/chat', methods=['POST'])
def chat():
    try:
        if 'audio' not in request.files:
            return jsonify({"error": "未上传音频"}), 400

        # 保存 m4a 文件
        audio_file = request.files['audio']
        with tempfile.NamedTemporaryFile(suffix=".m4a", delete=False) as tmp_audio:
            audio_file.save(tmp_audio.name)

        # 转换成 wav 格式（OpenAI 只接受 wav）
        audio = AudioSegment.from_file(tmp_audio.name, format="m4a")
        wav_path = tmp_audio.name.replace(".m4a", ".wav")
        audio.export(wav_path, format="wav")

        # Whisper 语音识别
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=open(wav_path, "rb")
        )
        question = transcript.text
        print("🎤 识别内容：", question)

        # 联网搜索内容
        web_result = search_web(question)
        print("🌐 搜索结果：", web_result)

        # ChatGPT 综合搜索结果生成回答
        chat_response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # 你也可以用 gpt-3.5-turbo
            messages=[
                {"role": "system", "content": "你是一个结合网络信息的语音助手。"},
                {"role": "user", "content": f"用户提问：{question}\n\n以下是搜索结果：\n{web_result}"}
            ]
        )
        reply = chat_response.choices[0].message.content
        print("🤖 GPT 回复：", reply)

        # 语音合成
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

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
