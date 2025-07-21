from flask import Flask, request, jsonify
from flask_cors import CORS
import openai, asyncio, tempfile, base64
import edge_tts
import os

openai.api_key = os.environ.get("OPENAI_API_KEY")

app = Flask(__name__)
CORS(app)

async def text_to_speech(text):
    communicate = edge_tts.Communicate(text, "zh-CN-XiaoxiaoNeural")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
        await communicate.save(f.name)
        with open(f.name, "rb") as audio_file:
            return base64.b64encode(audio_file.read()).decode("utf-8")

@app.route("/chat", methods=["POST"])
def chat():
    print("📥 收到语音上传请求")

    if 'audio' not in request.files:
        return jsonify({"error": "No audio uploaded"}), 400

    audio_file = request.files['audio']
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        audio_file.save(f.name)
        audio_path = f.name

    try:
        transcript = openai.Audio.transcribe("whisper-1", open(audio_path, "rb"))
        question = transcript["text"]
        print("🧠 Whisper 识别内容：", question)

        chat_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "你是一个语音助手"},
                {"role": "user", "content": question}
            ]
        )
        answer = chat_response["choices"][0]["message"]["content"]
        print("🤖 GPT 回复：", answer)

        audio_base64 = asyncio.run(text_to_speech(answer))
        return jsonify({"text": answer, "audio_base64": audio_base64})

    except Exception as e:
        print("❌ 出错：", str(e))
        return jsonify({"error": str(e)}), 500

# 前面是你已有的完整路由逻辑...
# 最后一行添加：

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
