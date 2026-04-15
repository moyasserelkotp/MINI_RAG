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
        all_collections = await self.db_client.list_collection_names()
        if DataBaseEnum.COLLECTION_CHAT_MESSAGE_NAME.value not in all_collections:
            self.collection = self.db_client[DataBaseEnum.COLLECTION_CHAT_MESSAGE_NAME.value]
            await self.collection.create_index("session_id", name="session_id_idx")

    async def create_message(self, message: ChatMessage):
        result = await self.collection.insert_one(
            message.dict(by_alias=True, exclude_unset=True)
        )
        message.id = result.inserted_id
        return message

    async def get_messages_by_session(self, session_id: str, limit: int = 5):
        cursor = self.collection.find({"session_id": session_id}).sort("created_at", -1).limit(limit)
        messages = []
        async for document in cursor:
            messages.append(ChatMessage(**document))
        # Return chronologically (oldest to newest among the limit)
        messages.reverse()
        return messages
