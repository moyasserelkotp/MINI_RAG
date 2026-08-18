from .BaseDataModel import BaseDataModel
from .db_schemes import Project
from .enums.DataBaseEnum import DataBaseEnum


class ProjectModel(BaseDataModel):

    def __init__(self, db_client: object):
        super().__init__(db_client=db_client)
        self.collection = self.db_client[DataBaseEnum.COLLECTION_PROJECT_NAME.value]

    @classmethod
    async def create_instance(cls, db_client: object):
        instance = cls(db_client)
        if not getattr(cls, "_indexes_created", False):
            await instance.init_collection()
            cls._indexes_created = True
        return instance

    async def init_collection(self):
        self.collection = self.db_client[DataBaseEnum.COLLECTION_PROJECT_NAME.value]
        indexes = Project.get_indexes()
        for index in indexes:
            await self.collection.create_index(
                index["key"], name=index["name"], unique=index["unique"], background=True
            )

    async def create_project(self, project: Project):
        result = await self.collection.insert_one(
            project.dict(by_alias=True, exclude_unset=True)
        )
        project.id = result.inserted_id
        return project

    async def get_project_or_create_one(self, project_id: str):
        record = await self.collection.find_one({"project_id": project_id})
        if record is None:
            project = Project(project_id=project_id)
            project = await self.create_project(project=project)
            return project
        return Project(**record)

    async def get_all_projects(self, page: int = 1, page_size: int = 10):
        total_documents = await self.collection.count_documents({})
        total_pages = max(1, (total_documents + page_size - 1) // page_size)

        cursor = self.collection.find().skip((page - 1) * page_size).limit(page_size)
        projects = []
        async for document in cursor:
            projects.append(Project(**document))

        return projects, total_pages

    async def get_project_by_id(self, project_id: str):
        """Fetch a project by its project_id string. Returns None if not found.

        Use this for every READ or DELETE operation.  Never auto-creates a project.
        """
        record = await self.collection.find_one({"project_id": project_id})
        if record is None:
            return None
        return Project(**record)

    # Convenience alias used in routes for clarity.
    get_project_or_404 = get_project_by_id

    async def delete_project(self, project_id: str) -> bool:
        result = await self.collection.delete_one({"project_id": project_id})
        return result.deleted_count > 0
