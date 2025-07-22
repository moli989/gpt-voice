from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
import tempfile
import base64
import os
import asyncio
from edge_tts import Communicate

client = OpenAI()
app = Flask(__name__)
CORS(app)

@app.route("/chat", methods=["POST"])
def chat():
    try:
        if 'audio' not in request.files:
            return jsonify({"error": "未上传音频"}), 400

        audio_file = request.files['audio']
        with tempfile.NamedTemporaryFile(suffix=".m4a", delete=False) as tmp_audio:
            audio_file.save(tmp_audio.name)

            # 语音转文字
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=open(tmp_audio.name, "rb")
            )
            question = transcript.text
            print("🎤 识别内容：", question)

        # 与 ChatGPT 对话
        chat_response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": question}]
        )
        answer = chat_response.choices[0].message.content
        print("🤖 GPT 回复：", answer)

        # 文字转语音
        async def tts():
            communicate = Communicate(text=answer, voice="zh-CN-XiaoxiaoNeural")
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_audio_out:
                await communicate.save(tmp_audio_out.name)
                with open(tmp_audio_out.name, "rb") as f:
                    return base64.b64encode(f.read()).decode("utf-8")

        audio_base64 = asyncio.run(tts())

        return jsonify({"text": answer, "audio_base64": audio_base64})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    import os
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
