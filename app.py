from flask import Flask, request, jsonify
from flask_cors import CORS
import openai, asyncio, tempfile, base64, os, traceback
import edge_tts
import nest_asyncio

nest_asyncio.apply()

# ✅ 正确初始化 OpenAI 客户端（v1.0+）
client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

app = Flask(__name__)
CORS(app)

@app.route('/')
def health_check():
    return "Voice Assistant API is running", 200

# ✅ 文本转语音（edge-tts → base64）
async def text_to_speech(text):
    communicate = edge_tts.Communicate(text, "zh-CN-XiaoxiaoNeural")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
        await communicate.save(f.name)
        with open(f.name, "rb") as audio_file:
            return base64.b64encode(audio_file.read()).decode("utf-8")

@app.route("/chat", methods=["POST"])
def chat():
    try:
        print("📥 收到语音上传请求")

        if 'audio' not in request.files:
            return jsonify({"error": "No audio uploaded"}), 400

        # 保存上传音频
        audio_file = request.files['audio']
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio:
            audio_file.save(tmp_audio.name)

        # ✅ 语音识别（新版 OpenAI v1.x）
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=open(tmp_audio.name, "rb")
        )
        question = transcript.text
        print("🧠 Whisper识别结果：", question)

        # ✅ GPT 回答（新版调用）
        chat_response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "你是一个语音助手"},
                {"role": "user", "content": question}
            ]
        )
        answer = chat_response.choices[0].message.content
        print("🤖 GPT 回复：", answer)

        # ✅ TTS 合成
        audio_base64 = asyncio.run(text_to_speech(answer))

        return jsonify({
            "text": answer,
            "audio_base64": audio_base64
        })

    except Exception as e:
        print("❌ 出错：", str(e))
        traceback.print_exc()
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

# ✅ Render 自动适配端口（或本地默认 10000）
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
