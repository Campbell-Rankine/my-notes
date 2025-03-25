from pydantic import BaseModel
from typing import List, Optional


class ServerSchema(BaseModel):
    x: float
