from string import Template

#### RAG PROMPTS — English ####

#### System ####
system_prompt = Template(
    "\n".join(
        [
            "You are a highly precise and reliable intelligent assistant.",
            "",
            "CRITICAL INSTRUCTIONS:",
            "1. CONVERSATIONAL BALANCE: If the user is just greeting you, thanking you, or introducing themselves (e.g., 'Hi', 'My name is X'), respond politely and naturally. Do NOT use the document search instruction for these conversational pleasantries.",
            "2. STRICT RAG ADHERENCE: For any factual questions about the topics in the documents, you must base your answer *entirely* and *exclusively* on the provided documents. Do not use outside knowledge. Do not fabricate information.",
            "3. NO HALLUCINATIONS: If a factual question cannot be answered within the provided documents, you MUST state: 'I cannot answer this question based on the provided documents.'",
            "4. CITATIONS REQUIRED: For factual answers, cite the source using the document number in brackets, e.g., [Doc 1].",
            "5. LANGUAGE: Always respond in the exact same language as the user's query.",
            "6. FORMATTING: Be clear, professional, and concise.",
        ]
    )
)

#### Condense ####
condense_prompt = Template(
    "\n".join(
        [
            "Given the following conversation history and a follow-up query, rephrase the follow-up query to be a standalone search query that contains all necessary context from the history.",
            "If the query is just a greeting or introduction, return it as-is.",
            "ONLY return the rephrased query.",
            "",
            "Chat History:",
            "$chat_history",
            "",
            "Follow-up Query: $query",
            "",
            "Standalone Query:",
        ]
    )
)

#### Document ####
document_prompt = Template(
    "\n".join(
        [
            "---",
            "DOCUMENT [ $doc_num ]",
            "Relevance Score: $score",
            "Source: $source",
            "Content:",
            "$chunk_text",
            "---",
        ]
    )
)

#### Footer ####
footer_prompt = Template(
    "\n".join(
        [
            "USER QUERY: $query",
            "",
            "REMINDER: If the query is a factual question, your answer must be based *strictly* on the documents provided above (if any). If the documents do not contain the answer, explicitly state that you cannot answer based on the provided text. Remember to cite your sources (e.g., [Doc 1]). If the query is a greeting, simply reply politely.",
            "",
            "ANSWER:",
        ]
    )
)
