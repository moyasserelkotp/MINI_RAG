from .BaseDataModel import BaseDataModel
from .db_schemes import ChatSession
from .enums.DataBaseEnum import DataBaseEnum
from bson.objectid import ObjectId

class SessionModel(BaseDataModel):

    def __init__(self, db_client: object):
        super().__init__(db_client=db_client)
        self.collection = self.db_client[DataBaseEnum.COLLECTION_CHAT_SESSION_NAME.value]

    @classmethod
    async def create_instance(cls, db_client: object):
        instance = cls(db_client)
        await instance.init_collection()
        return instance

    async def init_collection(self):
        all_collections = await self.db_client.list_collection_names()
        if DataBaseEnum.COLLECTION_CHAT_SESSION_NAME.value not in all_collections:
            self.collection = self.db_client[DataBaseEnum.COLLECTION_CHAT_SESSION_NAME.value]
            await self.collection.create_index("session_id", name="session_id_idx", unique=True)
            await self.collection.create_index("project_id", name="project_id_idx")

    async def create_session(self, session: ChatSession):
        result = await self.collection.insert_one(
            session.dict(by_alias=True, exclude_unset=True)
        )
        session.id = result.inserted_id
        return session

    async def get_session_or_create_one(self, session_id: str, project_id: ObjectId):
        record = await self.collection.find_one({"session_id": session_id})
        if record is None:
            session = ChatSession(session_id=session_id, project_id=project_id)
            session = await self.create_session(session=session)
            return session
        return ChatSession(**record)

    async def update_summary(self, session_id: str, new_summary: str):
        from datetime import datetime
        await self.collection.update_one(
            {"session_id": session_id},
            {"$set": {"summary": new_summary, "updated_at": datetime.utcnow()}}
        )

    async def increment_message_count(self, session_id: str, amount: int = 1):
        from datetime import datetime
        await self.collection.update_one(
            {"session_id": session_id},
            {"$inc": {"message_count": amount}, "$set": {"updated_at": datetime.utcnow()}}
        )
