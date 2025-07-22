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

nest_asyncio.apply()
client = OpenAI()

app = Flask(__name__)
CORS(app)

async def text_to_speech(text):
    communicate = edge_tts.Communicate(text, "en-GB-LibbyNeural")
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        await communicate.save(f.name)
        with open(f.name, "rb") as audio_file:
            return base64.b64encode(audio_file.read()).decode("utf-8")

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

def get_weather(lat, lon):
    try:
        url = f"https://wttr.in/{lat},{lon}?format=3"
        r = requests.get(url, timeout=5)
        return r.text.strip()
    except:
        return "无法获取天气信息"

@app.route("/")
def home():
    return "语音助手 API 已启动", 200

@app.route("/chat", methods=["POST"])
def chat():
    try:
        if 'audio' not in request.files:
            return jsonify({"error": "未上传音频"}), 400

        audio_file = request.files['audio']
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio:
            audio_file.save(tmp_audio.name)

            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=open(tmp_audio.name, "rb"),
                language="en"
            )
            question = transcript.text
            print("🎤 识别内容：", question)

        # 获取位置信息
        lat = request.form.get("lat")
        lon = request.form.get("lon")
        location_info = f"纬度：{lat}, 经度：{lon}" if lat and lon else "未知位置"

        # 天气信息
        weather_info = get_weather(lat, lon) if lat and lon else "未提供位置信息，无法查询天气"

        # 网络搜索
        web_info = search_web(question)

        # 生成回复
        chat = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "你是一个能使用实时天气与网页搜索信息的语音助手。"},
                {"role": "user", "content": f"用户问题：{question}\n\n当前位置：{location_info}\n天气：{weather_info}\n网络搜索结果：{web_info}"}
            ]
        )
        reply = chat.choices[0].message.content
        print("🤖 GPT 回复：", reply)

        # 语音回复
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        audio_base64 = loop.run_until_complete(text_to_speech(reply))
        loop.close()

        return jsonify({"text": reply, "audio_base64": audio_base64})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
