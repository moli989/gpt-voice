from flask import Flask, request, jsonify
from flask_cors import CORS
import openai, asyncio, tempfile, base64
import edge_tts
import os
import traceback

# ✅ 配置 OpenAI API Key（Render 中设置环境变量）
client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

app = Flask(__name__)
CORS(app)

# ✅ edge-tts 合成语音为 base64 MP3
async def text_to_speech(text):
    communicate = edge_tts.Communicate(text, "zh-CN-XiaoxiaoNeural")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
        await communicate.save(f.name)
        with open(f.name, "rb") as audio_file:
            return base64.b64encode(audio_file.read()).decode("utf-8")

# ✅ 主接口：接收音频 → Whisper识别 → GPT回答 → TTS语音返回
@app.route("/chat", methods=["POST"])
def chat():
    try:
        if 'audio' not in request.files:
            return jsonify({"error": "缺少语音文件"}), 400

        # 保存上传音频
        audio_file = request.files['audio']
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio:
            audio_file.save(tmp_audio.name)

        # ✅ 使用 Whisper V2 语音识别
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=open(tmp_audio.name, "rb")
        )
        question = transcript.text
        print("🧠 识别语音内容：", question)

        # ✅ GPT 生成回答
        chat_response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "你是一个语音助手"},
                {"role": "user", "content": question}
            ]
        )
        answer = chat_response.choices[0].message.content
        print("🤖 GPT 回复：", answer)

        # ✅ TTS 合成语音
        audio_base64 = asyncio.run(text_to_speech(answer))
        return jsonify({"text": answer, "audio_base64": audio_base64})

    except Exception as e:
        print("❌ 出错：", str(e))
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# ✅ 启动 Flask（Render 专用）
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
