import streamlit as st
import datetime
import pandas as pd

class QuizEngine:
    @staticmethod
    def start_quiz_session(questions: list):
        q_index = st.session_state.current_question
        total = len(st.session_state.questions)

        if q_index < total:
            q = st.session_state.questions[q_index]
            selected_key = f"selected_{q_index}"

            # Display Question
            st.markdown(
                f"<div class='quiz-box'><strong>Q{q_index + 1} of {total}:</strong> {q['question']}</div>",
                unsafe_allow_html=True
            )

            # Display Options
            st.markdown("<div class='quiz-options'>", unsafe_allow_html=True)
            selected = st.radio(
                "Choose your answer:",
                options=q["options"],
                key=selected_key,
                label_visibility="collapsed"
            )
            st.markdown("</div>", unsafe_allow_html=True)

            # Display Submit & Next buttons in sticky area
            st.markdown("<div class='quiz-actions'>", unsafe_allow_html=True)

            # Submit Button
            if not st.session_state.answered:
                if st.button("✅ Submit Answer", key=f"submit_{q_index}"):
                    if selected == q["correct"]:
                        st.success("✅ Correct!")
                        st.session_state.score += 1
                    else:
                        st.error(f"❌ Incorrect. Correct answer: {q['correct']}")

                    st.markdown(
                        f"<div class='quiz-box'><strong>Explanation:</strong> {q.get('explanation', 'No explanation provided')}</div>",
                        unsafe_allow_html=True
                    )

                    st.session_state.answered = True

            # Next Button (only after answer is submitted)
            if st.session_state.answered:
                if st.button("➡️ Next Question", key=f"next_{q_index}"):
                    st.session_state.current_question += 1
                    st.session_state.answered = False
                    del st.session_state[selected_key]
                    st.rerun()

            st.markdown("</div>", unsafe_allow_html=True)
        else:
            QuizEngine.show_score()

    @staticmethod
    def show_score():
        total = len(st.session_state.questions)
        correct = st.session_state.score
        incorrect = total - correct
        percent = round((correct / total) * 100)

        st.balloons()
        st.success(f"🎉 Quiz Completed! You scored {correct}/{total}")

        now = datetime.datetime.now().strftime("%b %d %H:%M")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("✅ Correct", correct)
            st.metric("❌ Incorrect", incorrect)
        with col2:
            st.markdown("### 📊 Performance Chart")
            st.bar_chart({"Score": {"Correct": correct, "Incorrect": incorrect}})

        st.info(f"Your Score: **{percent}%** — {'👏 Great job!' if percent >= 70 else '📖 Keep practicing!'}")

        if st.button("🔄 Restart Quiz"):
            st.session_state.questions = []
            st.session_state.current_question = 0
            st.session_state.score = 0
            st.session_state.answered = False
            st.rerun() 
