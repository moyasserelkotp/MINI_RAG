from .BaseDataModel import BaseDataModel
from .db_schemes import Asset
from .enums.DataBaseEnum import DataBaseEnum
from bson import ObjectId


class AssetModel(BaseDataModel):

    def __init__(self, db_client: object):
        super().__init__(db_client=db_client)
        self.collection = self.db_client[DataBaseEnum.COLLECTION_ASSET_NAME.value]

    @classmethod
    async def create_instance(cls, db_client: object):
        instance = cls(db_client)
        await instance.init_collection()
        return instance

    async def init_collection(self):
        self.collection = self.db_client[DataBaseEnum.COLLECTION_ASSET_NAME.value]
        indexes = Asset.get_indexes()
        for index in indexes:
            await self.collection.create_index(
                index["key"], name=index["name"], unique=index["unique"], background=True
            )

    def _resolve_project_id(self, asset_project_id):
        return (
            ObjectId(asset_project_id)
            if isinstance(asset_project_id, str)
            else asset_project_id
        )

    async def create_asset(self, asset: Asset):
        result = await self.collection.insert_one(
            asset.dict(by_alias=True, exclude_unset=True)
        )
        asset.id = result.inserted_id
        return asset

    async def get_all_project_assets(self, asset_project_id, asset_type: str):
        records = await self.collection.find(
            {
                "asset_project_id": self._resolve_project_id(asset_project_id),
                "asset_type": asset_type,
            }
        ).to_list(length=None)
        return [Asset(**record) for record in records]

    async def get_asset_record(
        self, asset_project_id, asset_name: str, asset_type: str = None
    ):
        query = {
            "asset_project_id": self._resolve_project_id(asset_project_id),
            "asset_name": asset_name,
        }
        if asset_type:
            query["asset_type"] = asset_type

        record = await self.collection.find_one(query)
        return Asset(**record) if record else None

    async def delete_all_project_assets(self, asset_project_id) -> int:
        result = await self.collection.delete_many(
            {"asset_project_id": self._resolve_project_id(asset_project_id)}
        )
        return result.deleted_count

    async def get_asset_by_id(self, asset_id: str):
        """Fetch a single asset by its MongoDB ObjectId string."""
        try:
            record = await self.collection.find_one({"_id": ObjectId(asset_id)})
            return Asset(**record) if record else None
        except Exception:
            return None

    async def delete_asset_by_id(self, asset_id: str) -> bool:
        """Delete a single asset document by its ObjectId. Returns True if deleted."""
        try:
            result = await self.collection.delete_one({"_id": ObjectId(asset_id)})
            return result.deleted_count == 1
        except Exception:
            return False

