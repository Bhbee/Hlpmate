import os
from typing import List, Tuple
from utils.prepare_vectordb import PrepareVectorDB
from utils.summarizer import Summarizer
from utils.generate_mcqs import MCQGenerator
from utils.load_config import LoadConfig

# Load app configuration
APPCFG = LoadConfig()

SUPPORTED_FORMATS = ["pdf", "docx", "pptx", "txt", "xlsx"]

class UploadFile:
    @staticmethod
    def process_uploaded_file(
        file_path: str,
        chatbot: List[Tuple[str, str]],
        selected_action: str
    ) -> Tuple[str, List[Tuple[str, str]]]:
        """
        Handle user-uploaded file and selected action.

        Args:
            file_path: Path to uploaded file.
            chatbot: Current chatbot conversation state.
            selected_action: User's selected action.

        Returns:
            Tuple of blank string (reserved) and updated chatbot history.
        """
        file_ext = file_path.split(".")[-1].lower()

        if file_ext not in SUPPORTED_FORMATS:
            chatbot.append((" ", f"❌ Unsupported file type: .{file_ext}"))
            return "", chatbot

        try:
            if selected_action == "Upload doc: Process for RAG":
                processor = PrepareVectorDB(
                    data_directory=[file_path],
                    persist_directory=APPCFG.custom_persist_directory,
                    openai_api_key=APPCFG.openai_api_key,
                    chunk_size=APPCFG.chunk_size,
                    chunk_overlap=APPCFG.chunk_overlap
                )
                processor.prepare_and_save_vectordb()
                chatbot.append((" ", "✅ Vector database created. You can now chat with your file."))

            elif selected_action == "Upload doc: Give Full summary":
                summary = Summarizer.summarize(file_path)
                chatbot.append((" ", summary))

            elif selected_action == "Upload doc: Generate MCQs":
                text = Summarizer.extract_text_from_file(file_path)
                if not text.strip():
                    chatbot.append((" ", "⚠️ Could not extract text from the file for question generation."))
                    return "", chatbot

                questions = MCQGenerator(text, max_questions=5)
                if not questions:
                    chatbot.append((" ", "❌ MCQ generation failed or returned no questions."))
                else:
                    for idx, q in enumerate(questions, 1):
                        q_display = (
                            f"**Q{idx}. {q['question']}**\n\n"
                            f"A. {q['options'][0]}\n"
                            f"B. {q['options'][1]}\n"
                            f"C. {q['options'][2]}\n"
                            f"D. {q['options'][3]}\n\n"
                            f"✅ Correct Answer: **{q['correct']}**\n"
                            f"📝 Explanation: {q['explanation']}\n\n"
                        )
                        chatbot.append((" ", q_display))

            else:
                chatbot.append((" ", "❗ Please select a valid action from the dropdown."))

        except Exception as e:
            chatbot.append((" ", f"❌ Operation failed: {e}"))

        return "", chatbot
