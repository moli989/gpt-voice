import os
import uuid
import asyncio
from flask import Flask, request, jsonify, send_file
from dotenv import load_dotenv
from openai import OpenAI
import edge_tts

# 加载 .env 文件中的环境变量
load_dotenv()
client = OpenAI()  # 自动从 .env 中读取 OPENAI_API_KEY

app = Flask(__name__)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_input = data.get("message", "")
    print("👉 User input:", user_input)

    if not user_input:
        return jsonify({"error": "Missing 'message'"}), 400

    try:
        # GPT 回复
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # 使用你当前有权限的模型
            messages=[
                {"role": "system", "content": "You are an English-speaking assistant."},
                {"role": "user", "content": user_input}
            ]
        )
        reply_text = response.choices[0].message.content
        print("🤖 GPT reply:", reply_text)

        # 使用英文语音生成 MP3
        output_file = f"reply_{uuid.uuid4().hex}.mp3"
        print("🔊 Generating speech...")
        asyncio.run(text_to_speech(reply_text, output_file))
        print("✅ MP3 saved:", output_file)

        return send_file(output_file, mimetype="audio/mpeg")

    except Exception as e:
        print("❌ Error occurred:", str(e))
        return jsonify({"error": str(e)}), 500

# TTS 函数：使用英文女声
async def text_to_speech(text, output_path):
    communicate = edge_tts.Communicate(text, voice="en-US-AriaNeural")
    await communicate.save(output_path)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
