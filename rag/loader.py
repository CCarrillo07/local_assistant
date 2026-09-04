from pathlib import Path

import pymupdf

from logger import get_logger
from rag.models import Document

logger = get_logger(__name__)

def load_pdf(path:Path) -> list[Document]:

    documents = []

    with pymupdf.open(path) as pdf:

        for page_number, page in enumerate(
            pdf,
            start=1
        ):

            text = page.get_text(
                "text",
                sort=True
            ).strip()

            if not text:
                continue

            document = Document(
                text=text,
                source=path.name,
                page=page_number
            )

            documents.append(document)
    
    return documents

def load_txt(path: Path) -> list[Document]:

    text = path.read_text(
        encoding="utf-8"
    ).strip()

    if not text:
        return []
    
    return [
        Document(
            text=text,
            source=path.name
        )
    ]

def load_documents(
    directory: str | Path
) -> list[Document]:

    directory = Path(directory)

    documents = []

    for path in directory.rglob("*"):

        if not path.is_file():
            continue
        
        logger.info(
            "Loading document: %s",
            path.name
        )

        if path.suffix.lower() == ".pdf":
            
            loaded = load_pdf(path)

        elif path.suffix.lower() == ".txt":
            loaded = load_txt(path)

        else:
            logger.info(
                "Skipping unsupported file: %s",
                path.name
            )

            continue

        documents.extend(loaded)

    logger.info(
        "Loaded %s document sections",
        len(documents)
    )

    return documents