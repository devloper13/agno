import asyncio
import csv
import io
import os
from pathlib import Path
from typing import IO, Any, List, Optional, Union
from urllib.parse import urlparse
from uuid import uuid4

from agno.utils.http import async_fetch_with_retry, fetch_with_retry

try:
    import pandas as pd
except ImportError:
    raise ImportError("`pandas not installed. Please install it with `uv pip install pandas``")

try:
    import aiofiles
except ImportError:
    raise ImportError("`aiofiles` not installed. Please install it with `pip install aiofiles`")

from agno.document.base import Document
from agno.document.reader.base import Reader
from agno.utils.log import logger


class SingleCSVReader(Reader):
    """Reader for CSV files"""

    def read(self, file: Union[Path, IO[Any]], delimiter: str = ",", quotechar: str = '"') -> List[Document]:
        try:

            """
            Loads Quora dataset and drops nulls.
            """
            df = pd.read_csv(file).dropna(subset=["question1", "question2"])
            df = df.head(100)  # Remove later - Only for testing

            documents = [Document(row.question1) for row in df.itertuples(index=False)]

            return documents

                
        except Exception as e:
            logger.error(f"Error reading: {file.name if isinstance(file, IO) else file}: {e}")
            return []

    async def async_read(
        self, file: Union[Path, IO[Any]], delimiter: str = ",", quotechar: str = '"', page_size: int = 1000
    ) -> List[Document]:
        pass
