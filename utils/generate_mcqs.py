import os
import re
import traceback
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from utils.summarizer import Summarizer

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class MCQGenerator:

    @staticmethod
    def extract_text(file_path: str) -> str:
        return Summarizer.extract_text_from_file(file_path)

   
    @staticmethod
    @st.cache_data(show_spinner=False)
    def gpt_generate_mcqs_cached(prompt: str) -> str:


        """Cached GPT call to reduce regeneration delay."""
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a smart tutor. Generate high-quality multiple-choice questions from the content provided. "
                            "Use a clear academic tone suitable for students preparing for exams."
                        )
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1200
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[❌ GPT API Error] {e}")
            traceback.print_exc()
            return ""

    @staticmethod
    def generate_mcqs_from_file(file_path: str, max_questions: int = 10) -> list:
        text = MCQGenerator.extract_text(file_path)
        if not text or len(text.split()) < 50:
            print("[⚠️ Warning] Insufficient content for MCQ generation.")
            return []

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_text(text)

        all_mcqs = []
        question_count = 0

        for chunk in chunks:
            if question_count >= max_questions:
                break

            prompt = (
                f"Generate {max_questions - question_count} multiple-choice questions from the following academic content:\n\n"
                f"{chunk}\n\n"
                "For each question, use the EXACT format below:\n"
                "Q: <question>\n"
                "A. <option A>\n"
                "B. <option B>\n"
                "C. <option C>\n"
                "D. <option D>\n"
                "Answer: <A/B/C/D>\n\n"
                "Do not add explanations or section titles."
            )

            output = MCQGenerator.gpt_generate_mcqs_cached(prompt)
            if not output or not output.strip().startswith("Q:"):
                print("[⚠️ GPT output malformed or empty]", output[:300])
                continue

            try:
                parsed = MCQGenerator.parse_mcqs(output)
                for q in parsed:
                    if question_count < max_questions:
                        all_mcqs.append(q)
                        question_count += 1
                    else:
                        break
            except Exception as e:
                print(f"[❌ Failed to parse MCQs]: {e}")
                traceback.print_exc()

        return all_mcqs

    @staticmethod
    def parse_mcqs(gpt_output: str) -> list:
        mcqs = []
        if not gpt_output:
            print("[⚠️ Empty GPT output for MCQ parsing]")
            return mcqs

        blocks = re.split(r"\n*Q:\s*", gpt_output.strip())
        for block in blocks:
            if not block.strip():
                continue

            block = "Q: " + block.strip()
            try:
                question = re.search(r"Q:\s*(.*)", block)
                a_opt = re.search(r"A[\.:]?\s+(.*)", block)
                b_opt = re.search(r"B[\.:]?\s+(.*)", block)
                c_opt = re.search(r"C[\.:]?\s+(.*)", block)
                d_opt = re.search(r"D[\.:]?\s+(.*)", block)
                answer = re.search(r"Answer:?\s*([ABCD])", block, re.IGNORECASE)

                if not all([question, a_opt, b_opt, c_opt, d_opt, answer]):
                    print("[🚨 Malformed MCQ Block]")
                    print(block[:300])
                    raise ValueError("Missing one or more parts")

                options = {
                    "A": a_opt.group(1).strip(),
                    "B": b_opt.group(1).strip(),
                    "C": c_opt.group(1).strip(),
                    "D": d_opt.group(1).strip(),
                }


                correct_letter = answer.group(1).upper()
                correct_answer = options[correct_letter]

                mcqs.append({
                    "question": question.group(1).strip(),
                    "options": list(options.values()),
                    "correct": correct_answer,
                    "explanation": f"The correct answer is {correct_letter}: {correct_answer}"
                })

            except Exception as e:
                print(f"[⚠️ Error parsing block]: {e}")
                print("🔍 Problematic block:")
                print(block[:500])
                continue

        return mcqs
