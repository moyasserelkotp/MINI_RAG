from .BaseDataModel import BaseDataModel
from .db_schemes.agent_run import AgentRun
from .enums.DataBaseEnum import DataBaseEnum
from bson import ObjectId


class AgentRunModel(BaseDataModel):

    def __init__(self, db_client: object):
        super().__init__(db_client=db_client)
        self.collection = self.db_client[DataBaseEnum.COLLECTION_AGENT_RUNS_NAME.value]

    @classmethod
    async def create_instance(cls, db_client: object):
        instance = cls(db_client)
        if not getattr(cls, "_indexes_created", False):
            await instance.init_collection()
            cls._indexes_created = True
        return instance

    async def init_collection(self):
        self.collection = self.db_client[DataBaseEnum.COLLECTION_AGENT_RUNS_NAME.value]
        indexes = AgentRun.get_indexes()
        for index in indexes:
            await self.collection.create_index(
                index["key"], name=index["name"], unique=index["unique"], background=True
            )

    def _resolve_project_id(self, project_id):
        return (
            ObjectId(project_id)
            if isinstance(project_id, str)
            else project_id
        )

    async def create_agent_run(self, agent_run: AgentRun):
        result = await self.collection.insert_one(
            agent_run.dict(by_alias=True, exclude_unset=True)
        )
        agent_run.id = result.inserted_id
        return agent_run

    async def get_agent_run(self, run_id: str):
        record = await self.collection.find_one({"run_id": run_id})
        return AgentRun(**record) if record else None

    async def update_agent_run(self, run_id: str, update_data: dict):
        result = await self.collection.update_one(
            {"run_id": run_id},
            {"$set": update_data}
        )
        return result.modified_count > 0

    async def get_project_agent_runs(self, project_id: str, limit: int = 100):
        records = await self.collection.find(
            {"project_id": self._resolve_project_id(project_id)}
        ).sort("started_at", -1).limit(limit).to_list(length=limit)
        return [AgentRun(**record) for record in records]
