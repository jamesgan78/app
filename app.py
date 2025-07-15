import streamlit as st
import os
import json
import re
import requests
from huggingface_hub import InferenceClient
from datetime import datetime
from opencc import OpenCC
cc = OpenCC('s2t')  # s2t: simplified to traditional



def hf_translate(api_url, text, src_lang=None, tgt_lang=None):
    payload = {
        "inputs": text,
        "parameters": {},
        "options": {"wait_for_model": True}
    }
    if src_lang and tgt_lang:
        payload["parameters"]["src_lang"] = src_lang
        payload["parameters"]["tgt_lang"] = tgt_lang

    response = requests.post(api_url, headers=HEADERS, json=payload)
    if response.status_code != 200:
        raise Exception(f"API error {response.status_code}: {response.text}")
    result = response.json()
    return result[0]["translation_text"]

def get_word_info(word):
    resp = requests.get(f"https://api.dictionaryapi.dev/api/v2/entries/en/{word}")
    if resp.status_code != 200:
        return None
    data = resp.json()[0]
    # 只取第一种词性、定义和例句
    meaning = data["meanings"][0]
    return {
        "partOfSpeech": meaning["partOfSpeech"],
        "definition": meaning["definitions"][0]["definition"],
        "example": meaning["definitions"][0].get("example", "")
    }

def query_jisho(word):
    url = f"https://jisho.org/api/v1/search/words?keyword={word}"
    r = requests.get(url)
    if r.status_code != 200:
        return None
    data = r.json()
    if not data["data"]:
        return None
    entry = data["data"][0]
    japanese = entry["japanese"][0]
    senses = entry["senses"][0]

    # 改為嘗試拿第一個例句（日文），如果沒有就空字串
    example_sentence = ""
    if "sentences" in entry and len(entry["sentences"]) > 0:
        example_sentence = entry["sentences"][0].get("ja", "")
    else:
        # 有些字典結果會放在 senses 的 "links" 或其它欄位，視需求再擴展
        example_sentence = ""

    return {
        "word": japanese.get("word", ""),
        "reading": japanese.get("reading", ""),
        "part_of_speech": senses.get("parts_of_speech", []),
        "english_definitions": senses.get("english_definitions", []),
        "example_sentence": example_sentence
    }

def generate_example_sentence(word):
    prompt = (
        f"請用單詞 '{word}' 造一個簡單的日文例句，並附上中文翻譯，格式如下：\n"
        "例句（日文）：...\n"
        "例句（中文）：..."
    )
    try:
        response = client.chat.completions.create(
            model="Qwen/Qwen2.5-1.5B-Instruct",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.7,
        )
        text = response.choices[0].message.content.strip()
        return text
    except Exception as e:
        return f"生成例句時出錯：{e}"


# === Hugging Face Token ===
HF_TOKEN = st.secrets["HF_TOKEN"]
API_JA_EN = "https://api-inference.huggingface.co/models/facebook/mbart-large-50-many-to-many-mmt"
API_EN_ZH = "https://api-inference.huggingface.co/models/Helsinki-NLP/opus-mt-en-zh"
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

client = InferenceClient(token=HF_TOKEN)


# === 初始化狀態 ===

if "article_text" not in st.session_state:
    st.session_state.article_text = ""
if "questions" not in st.session_state:
    st.session_state.questions = []
if "user_answers" not in st.session_state:
    st.session_state.user_answers = []
if "submitted" not in st.session_state:
    st.session_state.submitted = False
if "explanation" not in st.session_state:
    st.session_state.explanation = None
if "show_explanation" not in st.session_state:
    st.session_state.show_explanation = False



explanation=""
# === 頁面標題 ===
st.title("📖 日文閱讀理解練習平台")

# === 輸入文章 ===
st.text_area("請貼入日文文章：", height=200, key="article_text")

# === 題目生成 ===
if st.button("生成題目"):
    prompt = (
        "あなたは日本語の先生です。以下の文章の理解力を試す質問を3問作成してください。質問内容は文章内の情報のみを使うようにしてください。それぞれの質問に対する答えは1つだけになるようにしてください。\n"
        "出力は必ず以下のJSON形式のみで返答してください：\n"
        "{\n"
        '  "questions": [\n'
        '    {"question":"...", "options":["...","...","...","..."], "answer":"...", "explanation":"..."},\n'
        "    ... 合計3問 ...\n"
        "  ]\n"
        "}\n\n"
        f"文章：{st.session_state.article_text}"
    )

    with st.spinner("題目生成中..."):
        try:
            response = client.chat.completions.create(
                model="Qwen/Qwen2.5-1.5B-Instruct",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0.7,
            )
        except Exception as e:
            st.error(f"模型呼叫錯誤：{e}")
            st.stop()

        raw_output = response.choices[0].message.content
        #st.markdown("##### 📦 模型輸出（可查看格式）")
        #st.code(raw_output)

        json_match = re.search(r"\{.*\}", raw_output, re.DOTALL)
        if not json_match:
            st.error("❌ 找不到 JSON 格式題目，請嘗試重新生成")
            st.stop()

        json_text = json_match.group(0)
        try:
            data = json.loads(json_text)
        except json.JSONDecodeError as e:
            st.error(f"JSON 解碼失敗：{e}")
            st.code(json_text)
            st.stop()

        st.session_state.questions = data.get("questions", [])
        st.session_state.user_answers = ["" for _ in st.session_state.questions]
        st.session_state.submitted = False
        st.session_state.explanation = None
        
        # === 額外：生成解說內容並暫存到 session_state
        explain_prompt = (
            f"你是一位擅長語言教學的專家，請用繁體中文詳細解釋以下日文文章的內容，包括：\n"
            f"1. 文章主旨是什麼？\n"
            f"2. 有哪些重要語法？\n"
            f"3. 文章中的關鍵詞彙有哪些意思？\n"
            f"4. 如果有難句，如何理解？\n\n"
            f"日文文章如下：{st.session_state.article_text}"
        )

        try:
            explain_response = client.chat.completions.create(
                model="Qwen/Qwen2.5-1.5B-Instruct",
                messages=[{"role": "user", "content": explain_prompt}],
                max_tokens=800,
                temperature=0.7,
            )
            st.session_state.explanation = explain_response.choices[0].message.content
            st.session_state.explanation = cc.convert(st.session_state.explanation)
        except Exception as e:
            st.session_state.explanation = f"⚠️ 無法生成中文解說：{e}"
    
    st.session_state.show_explanation = False
        

# === 顯示題目與作答 ===
if st.session_state.questions:
    st.subheader("📝 作答區")
    for i, q in enumerate(st.session_state.questions):
        st.markdown(f"#### Q{i+1}: {q['question']}")
        st.session_state.user_answers[i] = st.radio(
            f"選擇答案 {i+1}",
            options=q["options"],
            index=q["options"].index(st.session_state.user_answers[i]) if st.session_state.user_answers[i] in q["options"] else 0,
            key=f"q_{i}"
        )

    if st.button("提交答案"):
        st.session_state.show_explanation = True
        score = sum(
            1 for i, q in enumerate(st.session_state.questions)
            if st.session_state.user_answers[i] == q["answer"]
        )
        st.success(f"🎉 你總共答對 {score}/{len(st.session_state.questions)} 題！")

        for i, q in enumerate(st.session_state.questions):
            is_correct = st.session_state.user_answers[i] == q["answer"]
            color = "✅" if is_correct else "❌"
            st.markdown(f"**{color} Q{i+1} 正解：** {q['answer']} — {q['explanation']}")        
        

        # 儲存成績到 JSON
        history = []
        record_file = "score_history.json"
        if os.path.exists(record_file):
            with open(record_file, "r", encoding="utf-8") as f:
                history = json.load(f)

        new_record = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "score": score,
        "total": len(st.session_state.questions),
        "questions": [q["question"] for q in st.session_state.questions],
        "answers": st.session_state.user_answers,
        "correct_answers": [q["answer"] for q in st.session_state.questions],
        "explanations": [q["explanation"] for q in st.session_state.questions],
        "article_explanation": st.session_state.explanation
    }
        

        history.append(new_record)

        with open(record_file, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

        st.info("📁 成績已儲存到 `score_history.json`")
        
if st.session_state.show_explanation and st.session_state.explanation:
    st.markdown("### 🧠 整篇文章的中文講解")
    st.write(st.session_state.explanation)


# --- 單字翻譯功能區塊 ---
st.markdown("---")
st.subheader("🌐 日文 ➜ 中文 翻譯器")

input_text = st.text_input("請輸入日文單字或短語：", key="translate_input")

if st.button("翻譯"):
    with st.spinner("翻譯中...請稍候"):
        try:
            # 1️⃣ 日 ➜ 英（mbart 需要指定語言代碼）
            en_result = hf_translate(API_JA_EN, input_text, src_lang="ja_XX", tgt_lang="en_XX")

            # 2️⃣ 英 ➜ 中文（Helsinki 模型不需要語言代碼）
            zh_result = hf_translate(API_EN_ZH, en_result)
            zh_result = cc.convert(zh_result)  # 簡轉繁

            # 顯示結果
            st.success("✅ 翻譯成功！")
            st.markdown(f"""
            - 原始日文：`{input_text}`
            - 翻譯中文：`{zh_result}`
            """)

            # 單字詞彙資訊
            info = query_jisho(input_text)
            if info:
                st.write(f"單詞：{info['word']}（讀音：{info['reading']}）")
                st.write(f"詞性：{', '.join(info['part_of_speech'])}")
                st.write(f"英文定義：{', '.join(info['english_definitions'])}")
                if info['example_sentence']:
                    st.write(f"例句（日文）：{info['example_sentence']}")
                else:
                    # 用 Qwen 生成例句
                    example = generate_example_sentence(input_text)
                    st.markdown("🔹 **單詞使用例句**")
                    lines = example.splitlines()
                    ja_line = ""
                    zh_line = ""

                    for line in lines:
                        if "例句（日文）" in line:
                            ja_line = line.replace("例句（日文）：", "").strip()
                        elif "例句（中文）" in line:
                            zh_line = line.replace("例句（中文）：", "").strip()
                    zh_line = cc.convert(zh_line)  # 加這行即可

                    # 顯示
                    if ja_line:
                        st.write(f"日文：{ja_line}")
                    if zh_line:
                        st.write(f"中文：{zh_line}")
            else:
                st.write("找不到該單詞的詞彙資訊")
                example = generate_example_sentence(input_text)
                st.markdown("🔹 **Qwen生成的例句**")
                st.write(example)

        except Exception as e:
            st.error("❌ 發生錯誤：")
            st.exception(e)

   # === 顯示歷史紀錄 ===
st.markdown("---")
st.subheader("📊 歷史成績紀錄")

if os.path.exists("score_history.json"):
    with open("score_history.json", "r", encoding="utf-8") as f:
        history_data = json.load(f)
    if history_data:
        for idx, record in enumerate(reversed(history_data[-10:])):
            with st.expander(f"🗓️ {record['date']} - 分數：{record['score']}/{record['total']}"):
                for i in range(len(record["questions"])):
                    st.markdown(f"**Q{i+1}:** {record['questions'][i]}")
                    if record['answers'][i]!=record['correct_answers'][i]:
                        st.markdown(f"❌ 你的答案：{record['answers'][i]}")
                        st.markdown(f"✅ 正確答案：{record['correct_answers'][i]}")
                    else:
                        st.markdown(f"👉 你的答案：{record['answers'][i]}")
                        st.markdown(f"✅ 正確答案：{record['correct_answers'][i]}")
                    st.markdown(f"📘 解說：{record['explanations'][i]}")
                    st.markdown("---")
    else:
        st.write("目前尚無成績紀錄。")
else:
    st.write("尚未產生任何成績紀錄。")