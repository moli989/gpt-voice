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
    print("📥 收到上传请求")

    # ✅ 检查是否上传了音频文件
    if 'audio' not in request.files:
        return jsonify({"error": "No audio uploaded"}), 400

    audio_file = request.files['audio']

    # ✅ 保存临时音频文件
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        audio_file.save(f.name)
        audio_path = f.name

    try:
        # ✅ Whisper 识别文字
        transcript = openai.Audio.transcribe("whisper-1", open(audio_path, "rb"))
        question = transcript["text"]
        print("🧠 识别内容：", question)

        # ✅ ChatGPT 回答
        chat_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "你是一个语音助手"},
                {"role": "user", "content": question}
            ]
        )
        answer = chat_response["choices"][0]["message"]["content"]
        print("🤖 GPT 回复：", answer)

        # ✅ 生成语音
        audio_base64 = asyncio.run(text_to_speech(answer))

        return jsonify({"text": answer, "audio_base64": audio_base64})

    except Exception as e:
        print("❌ 错误：", str(e))
        return jsonify({"error": str(e)}), 500
