from .BaseDataModel import BaseDataModel
from .db_schemes import ChatMessage
from .enums.DataBaseEnum import DataBaseEnum





class MessageModel(BaseDataModel):
    def __init__(self, db_client: object):
        super().__init__(db_client=db_client)
        self.collection = self.db_client[DataBaseEnum.COLLECTION_CHAT_MESSAGE_NAME.value]

    @classmethod
    async def create_instance(cls, db_client: object):
        instance = cls(db_client)
        await instance.init_collection()
        return instance

    async def init_collection(self):
        self.collection = self.db_client[DataBaseEnum.COLLECTION_CHAT_MESSAGE_NAME.value]
        indexes = ChatMessage.get_indexes()
        for index in indexes:
            await self.collection.create_index(
                index["key"], name=index["name"], unique=index["unique"], background=True
            )

    async def create_message(self, message: ChatMessage):
        result = await self.collection.insert_one(
            message.dict(by_alias=True, exclude_unset=True)
        )
        message.id = result.inserted_id
        return message

    async def get_messages_by_session(self, session_id: str, skip: int = 0, limit: int = 5):
        cursor = self.collection.find({"session_id": session_id}).sort("created_at", -1).skip(skip).limit(limit)
        messages = []
        async for document in cursor:
            messages.append(ChatMessage(**document))
        # Return chronologically (oldest to newest among the limit)
        messages.reverse()
        return messages

    # Phase 6: Session management methods
    async def delete_messages_by_session(self, session_id: str) -> int:
        """Delete all messages belonging to a session. Returns deleted count."""
        result = await self.collection.delete_many({"session_id": session_id})
        return result.deleted_count

    async def add_message_to_session(self, session_id: str, role: str, content: str):
        """Convenience method: create and save a single message."""
        from datetime import datetime, timezone
        message = ChatMessage(
            session_id=session_id,
            role=role,
            text=content,
            created_at=datetime.now(timezone.utc),
        )
        return await self.create_message(message=message)

    async def get_message_count(self, session_id: str) -> int:
        """Return the total number of messages in a session."""
        return await self.collection.count_documents({"session_id": session_id})
