import logging

logger = logging.getLogger(__name__)

class PromptService:
    def __init__(self, template_parser, generation_client):
        self.template_parser = template_parser
        self.generation_client = generation_client

    def format_system_prompt(self, query, session_obj, session_messages, entities_text, retrieved_documents, use_summary, use_entity, use_window):
        system_prompt = self.template_parser.get("rag", "system_prompt")
        
        if use_summary and session_obj and getattr(session_obj, "summary", None):
            system_prompt += f"\n\n[Conversation Summary]:\n{session_obj.summary}"

        if use_entity and entities_text:
            system_prompt += f"\n\n[Important Retained Facts]:\n{entities_text}"

        chat_history_prompts = [
            self.generation_client.construct_prompt(prompt=system_prompt, role=self.generation_client.enums.SYSTEM.value)
        ]

        if use_window:
            for msg in session_messages:
                role = self.generation_client.enums.USER.value if msg.role == "user" else self.generation_client.enums.ASSISTANT.value
                chat_history_prompts.append(self.generation_client.construct_prompt(prompt=msg.text, role=role))

        document_parts = []
        for idx, doc in enumerate(retrieved_documents or []):
            chunk_text = doc.payload.get("text", "")
            doc_metadata = doc.payload.get("metadata", {})
            source = doc_metadata.get("source", "unknown") if doc_metadata else "unknown"
            score = round(getattr(doc, "score", 0.0), 4)

            doc_prompt = self.template_parser.get(
                "rag", "document_prompt", {"doc_num": idx + 1, "chunk_text": chunk_text, "source": source, "score": score}
            )
            document_parts.append(doc_prompt)

        documents_prompts = "\n\n".join(document_parts)
        footer_prompt = self.template_parser.get("rag", "footer_prompt", {"query": query})
        full_prompt = "\n\n".join([documents_prompts, footer_prompt]) if documents_prompts else footer_prompt
        
        return full_prompt, chat_history_prompts
