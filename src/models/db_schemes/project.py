import re
from pydantic import BaseModel, Field, validator
from typing import Optional
from bson.objectid import ObjectId

# Single source of truth for project_id format.
# Routes use:  Path(..., pattern=PROJECT_ID_PATTERN)
# Schema uses: same pattern in the validator below.
PROJECT_ID_PATTERN = r"^[a-zA-Z0-9_-]{1,64}$"
_PROJECT_ID_RE = re.compile(PROJECT_ID_PATTERN)


class Project(BaseModel):
    id: Optional[ObjectId] = Field(None, alias="_id")
    project_id: str = Field(..., min_length=1, max_length=64)

    @validator("project_id")
    def validate_project_id(cls, value: str) -> str:
        if not _PROJECT_ID_RE.match(value):
            raise ValueError(
                "project_id must be 1-64 characters and contain only "
                "alphanumeric characters, underscores, or hyphens."
            )
        return value

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def get_indexes(cls):
        return [
            {"key": [("project_id", 1)], "name": "project_id_index_1", "unique": True}
        ]
