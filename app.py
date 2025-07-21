from flask import Flask, request, jsonify
from flask_cors import CORS
import openai, asyncio, tempfile, base64, os
import edge_tts
import nest_asyncio  # 新增

# 修复异步事件循环冲突
nest_asyncio.apply()

openai.api_key = os.environ.get("OPENAI_API_KEY")

app = Flask(__name__)
CORS(app)

# 新增健康检查路由
@app.route('/')
def health_check():
    return "Voice Assistant API is running", 200

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
    
    # 优化文件处理
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp_audio:
        audio_file.save(tmp_audio.name)
        try:
            transcript = openai.Audio.transcribe("whisper-1", open(tmp_audio.name, "rb"))
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

            # 修复异步调用
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                audio_base64 = loop.run_until_complete(text_to_speech(answer))
            finally:
                loop.close()
                
            return jsonify({"text": answer, "audio_base64": audio_base64})

        except Exception as e:
            print("❌ 出错：", str(e))
            return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # 读取环境变量
    app.run(host="0.0.0.0", port=port)
