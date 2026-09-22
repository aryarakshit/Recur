"""Knowledge Base Indexer for RAG pipeline.

Parses Markdown files in `knowledge/`, splits them into structured chunks by section,
computes vector embeddings, and builds a FAISS index with persisted metadata.
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING

import faiss
import numpy as np

from rag.models import DocumentChunk
from storage.memory_store import MEMORY_FILE_NAME

if TYPE_CHECKING:
    from ai.embeddings import EmbeddingProvider
    from storage.database import Database

logger = logging.getLogger(__name__)


class KnowledgeIndexer:
    def __init__(
        self,
        knowledge_dir: Path | str,
        faiss_index_path: Path | str,
        metadata_path: Path | str,
        embedding_provider: EmbeddingProvider,
        db: Database | None = None,
    ) -> None:
        self.knowledge_dir = Path(knowledge_dir)
        self.faiss_index_path = Path(faiss_index_path)
        self.metadata_path = Path(metadata_path)
        self.embedding_provider = embedding_provider
        self.db = db

    def _parse_markdown_file(self, file_path: Path) -> list[DocumentChunk]:
        """Parses a single Markdown file into section-based chunks."""
        text = file_path.read_text(encoding="utf-8")
        source_name = file_path.name

        # Extract updated_at date if present
        date_match = re.search(r"\*Updated:\s*([0-9\-]+)\*", text)
        updated_at = date_match.group(1) if date_match else ""

        # Extract top-level title
        title_match = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
        doc_title = title_match.group(1).strip() if title_match else source_name

        # Split into sections based on ## or ### headers
        lines = text.splitlines()
        chunks: list[DocumentChunk] = []

        current_section = ""
        current_lines: list[str] = []
        chunk_idx = 0

        for line in lines:
            header_match = re.match(r"^(#{2,3})\s+(.+)$", line)
            if header_match:
                if current_lines:
                    section_body = "\n".join(current_lines).strip()
                    if not current_section:
                        # Preamble before first header
                        pass
                    elif len(section_body) > 20:
                        chunk_text = f"Document: {doc_title} ({source_name})\nSection: {current_section}\n\n{section_body}"
                        chunks.append(
                            DocumentChunk(
                                chunk_id=f"{source_name}#{chunk_idx}",
                                source=source_name,
                                section=current_section,
                                updated_at=updated_at,
                                text=chunk_text,
                                metadata={"doc_title": doc_title},
                            )
                        )
                        chunk_idx += 1
                    current_lines = []
                current_section = header_match.group(2).strip()
            else:
                current_lines.append(line)

        # Append last section or single document if no headers
        if current_lines:
            section_body = "\n".join(current_lines).strip()
            if not current_section:
                # Document has no subheaders, index as whole
                if len(section_body) > 20:
                    chunks.append(
                        DocumentChunk(
                            chunk_id=f"{source_name}#0",
                            source=source_name,
                            section=doc_title,
                            updated_at=updated_at,
                            text=f"Document: {doc_title} ({source_name})\n\n{section_body}",
                            metadata={"doc_title": doc_title},
                        )
                    )
            elif len(section_body) > 20:
                chunk_text = f"Document: {doc_title} ({source_name})\nSection: {current_section}\n\n{section_body}"
                chunks.append(
                    DocumentChunk(
                        chunk_id=f"{source_name}#{chunk_idx}",
                        source=source_name,
                        section=current_section,
                        updated_at=updated_at,
                        text=chunk_text,
                        metadata={"doc_title": doc_title},
                    )
                )

        return chunks

    def _parse_pdf_file(self, file_path: Path) -> list[DocumentChunk]:
        """Parses a PDF file into page-based chunks."""
        try:
            import pypdf
            reader = pypdf.PdfReader(str(file_path))
        except Exception as e:
            logger.error("Failed to load PDF reader for %s: %s", file_path, e)
            return []

        chunks: list[DocumentChunk] = []
        source_name = file_path.name
        doc_title = file_path.stem.replace("-", " ").replace("_", " ").title()

        for page_idx, page in enumerate(reader.pages, start=1):
            page_text = (page.extract_text() or "").strip()
            if len(page_text) > 20:
                chunk_text = f"Document: {doc_title} ({source_name})\nPage: {page_idx}\n\n{page_text}"
                chunks.append(
                    DocumentChunk(
                        chunk_id=f"{source_name}#page{page_idx}",
                        source=source_name,
                        section=f"Page {page_idx}",
                        updated_at="",
                        text=chunk_text,
                        metadata={"doc_title": doc_title, "page": page_idx},
                    )
                )
        return chunks

    def build_index(self) -> dict[str, int | str | float]:
        """Indexes all markdown and PDF files in knowledge directory and writes index + metadata."""
        start_time = time.time()
        # Organizer memory is injected into every prompt directly (see MemoryStore),
        # so it is kept out of similarity search.
        md_files = sorted(p for p in self.knowledge_dir.glob("*.md") if p.name != MEMORY_FILE_NAME)
        pdf_files = sorted(list(self.knowledge_dir.glob("*.pdf")))
        all_files = sorted(md_files + pdf_files)

        if not all_files:
            logger.warning("No markdown or PDF files found in %s", self.knowledge_dir)
            return {"file_count": 0, "chunk_count": 0, "time_taken": 0.0}

        all_chunks: list[DocumentChunk] = []
        for file_path in all_files:
            try:
                if file_path.suffix.lower() == ".pdf":
                    chunks = self._parse_pdf_file(file_path)
                else:
                    chunks = self._parse_markdown_file(file_path)
                all_chunks.extend(chunks)
            except Exception as e:
                logger.error("Error reading %s: %s", file_path, e)

        if not all_chunks:
            logger.warning("No chunks created from knowledge documents.")
            return {"file_count": len(all_files), "chunk_count": 0, "time_taken": 0.0}

        # Embed all chunks
        texts = [chunk.text for chunk in all_chunks]
        embeddings = self.embedding_provider.embed_texts(texts)
        emb_matrix = np.array(embeddings, dtype=np.float32)

        dimension = emb_matrix.shape[1]
        # FAISS IndexFlatIP computes inner product (which equals cosine similarity for normalized vectors)
        index = faiss.IndexFlatIP(dimension)
        index.add(emb_matrix)

        # Save FAISS index
        self.faiss_index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(self.faiss_index_path))

        # Save metadata
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_payload = {
            "dimension": dimension,
            "updated_at": time.time(),
            "chunks": [chunk.to_dict() for chunk in all_chunks],
        }
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata_payload, f, indent=2)

        elapsed = time.time() - start_time

        # Update metrics in database if available
        if self.db:
            self.db.set_metric("last_rebuild_time", time.time())
            self.db.set_metric("kb_file_count", len(md_files))
            self.db.set_metric("kb_chunk_count", len(all_chunks))
            self.db.set_metric("embedding_dimension", dimension)

        logger.info(
            "Indexed %d files (%d chunks) in %.2fs. Saved index to %s",
            len(md_files),
            len(all_chunks),
            elapsed,
            self.faiss_index_path,
        )

        return {
            "file_count": len(md_files),
            "chunk_count": len(all_chunks),
            "dimension": dimension,
            "time_taken": round(elapsed, 2),
        }
