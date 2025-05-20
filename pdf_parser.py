import os
from typing import List

from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader


class TariffProcessor:
    def __init__(self, pdf_path: str, output_dir: str = "vectorstore"):
        self.pdf_path = pdf_path
        self.output_dir = output_dir
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})

    @staticmethod
    def _convert_to_markdown(text: str) -> str:
        """
        Convert plain text to Markdown format
        :param text: Plain text.
        :return: Markdown format of the text.
        """
        lines = text.split("\n")
        result = []

        # Process each line
        for line in lines:
            # Clean up extra spaces without regex
            words = line.split()
            cleaned_line = " ".join(words)

            # Detect section headers (lines starting with "SECTION")
            if cleaned_line.upper().startswith("SECTION"):
                # Split section number and title
                parts = cleaned_line.split(maxsplit=1)
                if len(parts) > 1:
                    section_number = parts[0]
                    title = parts[1]
                    result.append(f"### {section_number} {title}")
                    continue

            # Detect table-like lines (lines containing | characters)
            if "|" in cleaned_line:
                # Convert to Markdown table format
                cells = [cell.strip() for cell in cleaned_line.split("|") if cell.strip()]
                if cells:
                    result.append("| " + " | ".join(cells) + " |")
                    # Add separator line for table headers
                    if all(cell.replace("-", "").strip() == "" for cell in cells):
                        result.append("|---" * len(cells) + "|")
                    continue

            # Add regular lines as-is
            result.append(cleaned_line)

        return "\n".join(result)

    def _create_document_chunks(self, text: str, source: str) -> List[Document]:
        """
        Split text into semantic chunks with metadata
        :param text: extracted text from pdf
        :param source: sources of the text
        :return: List of semantic chunks with metadata
        """
        # Convert to markdown
        md_text = self._convert_to_markdown(text)

        # Create base document
        doc = Document(page_content=md_text, metadata={"source": source})

        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
            length_function=len,
        )

        return splitter.split_documents([doc])

    def process_pdf(self) -> str:
        """
        Main processing pipeline
        :return: message that the pdf is processed and chunks are saved.
        """
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

        # Load PDF
        loader = PyPDFLoader(self.pdf_path)
        pages = loader.load()

        # Combine all pages
        full_text = "\n".join([page.page_content for page in pages])

        # Create chunks
        chunks = self._create_document_chunks(full_text, "Port Tariff.pdf")

        # Create vector store
        vectorstore = FAISS.from_documents(chunks, self.embeddings)

        # Save locally
        vectorstore.save_local(self.output_dir)

        return f"Processed {len(chunks)} chunks and saved to {self.output_dir}"


# Usage
if __name__ == "__main__":
    processor = TariffProcessor(pdf_path="Port Tariff.pdf")
    print(processor.process_pdf())
