from .BaseDataModel import BaseDataModel
from .db_schemes import ChatSession
from .enums.DataBaseEnum import DataBaseEnum
from bson.objectid import ObjectId
from datetime import datetime, timezone



class SessionModel(BaseDataModel):
    def __init__(self, db_client: object):
        super().__init__(db_client=db_client)
        self.collection = self.db_client[DataBaseEnum.COLLECTION_CHAT_SESSION_NAME.value]

    @classmethod
    async def create_instance(cls, db_client: object):
        instance = cls(db_client)
        if not getattr(cls, "_indexes_created", False):
            await instance.init_collection()
            cls._indexes_created = True
        return instance

    async def init_collection(self):
        self.collection = self.db_client[DataBaseEnum.COLLECTION_CHAT_SESSION_NAME.value]
        indexes = ChatSession.get_indexes()
        for index in indexes:
            await self.collection.create_index(
                index["key"], name=index["name"], unique=index["unique"], background=True
            )

    async def create_session(self, session: ChatSession):
        result = await self.collection.insert_one(
            session.dict(by_alias=True, exclude_unset=True)
        )
        session.id = result.inserted_id
        return session

    async def get_session_or_create_one(self, session_id: str, project_id: str):
        record = await self.collection.find_one({"session_id": session_id})
        if record is None:
            session = ChatSession(session_id=session_id, project_id=str(project_id))
            session = await self.create_session(session=session)
            return session
        
        if str(record.get("project_id")) != str(project_id):
            raise ValueError("Session isolation violation: session_id belongs to a different project")
            
        return ChatSession(**record)

    async def update_summary(self, session_id: str, new_summary: str):
        await self.collection.update_one(
            {"session_id": session_id},
            {"$set": {"summary": new_summary, "updated_at": datetime.now(timezone.utc)}}
        )

    async def increment_message_count(self, session_id: str, amount: int = 1):
        await self.collection.update_one(
            {"session_id": session_id},
            {"$inc": {"message_count": amount}, "$set": {"updated_at": datetime.now(timezone.utc)}}
        )

    async def update_summary_and_reset_count(self, session_id: str, new_summary: str):
        """SVC-01: Atomic combined update — sets new summary and resets message_count to 0.

        This replaces the pattern of calling update_summary() followed by a
        separate update_one({message_count: 0}), which had a race window between
        the two MongoDB operations.
        """
        await self.collection.update_one(
            {"session_id": session_id},
            {
                "$set": {
                    "summary": new_summary,
                    "message_count": 0,
                    "updated_at": datetime.now(timezone.utc),
                }
            }
        )

    # Phase 6: Session management methods
    async def get_all_sessions(self, project_id, skip: int = 0, limit: int = 500):
        """Return all sessions for a given project_id (ObjectId or str)."""
        cursor = self.collection.find({"project_id": project_id}).sort("created_at", -1).skip(skip).limit(limit)
        sessions = []
        async for doc in cursor:
            sessions.append(ChatSession(**doc))
        total = await self.collection.count_documents({"project_id": project_id})
        return sessions, total

    async def delete_session(self, session_id: str):
        """Delete a session document by session_id string."""
        result = await self.collection.delete_one({"session_id": session_id})
        return result.deleted_count > 0

    async def get_session_by_id(self, session_id: str):
        """Return a single session by its session_id string."""
        record = await self.collection.find_one({"session_id": session_id})
        if record:
            return ChatSession(**record)
        return None

    async def get_session_scoped(self, session_id: str, project_id):
        """Return a session only if it belongs to the given project.

        Prevents cross-project session access.  ``project_id`` is the MongoDB
        ObjectId (or str equivalent) stored on the session document.

        Returns None if not found OR if it belongs to a different project —
        callers must not distinguish between the two cases to avoid leaking
        session existence across projects.
        """
        record = await self.collection.find_one(
            {"session_id": session_id, "project_id": project_id}
        )
        if record:
            return ChatSession(**record)
        return None
