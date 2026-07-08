from abc import ABC, abstractmethod
from typing import Tuple, List
from core.models.document import Document
from core.models.node import Node
from core.models.edge import Edge

class BaseParser(ABC):
    """
    Abstract base class for all document parsers.
    Ensures that any parser (Docling, Video, Audio) standardizes 
    to the same Node/Edge output structure.
    """
    @abstractmethod
    def parse(self, file_path: str, tenant_id: str = "default") -> Tuple[Document, List[Node], List[Edge]]:
        """
        Parses a document and returns the Document metadata, 
        along with extracted Nodes and Edges.
        """
        pass