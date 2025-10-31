"""
PDF processing module for extracting text, tables, and images.
"""
import os
import re
import base64
import uuid
import time
from typing import List, Tuple
from unstructured.partition.pdf import partition_pdf
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage


class PDFProcessor:
    """Process PDF files to extract text, tables, and images."""
    
    def __init__(self, config):
        """
        Initialize PDF processor.
        
        Args:
            config: Config instance with LLM settings
        """
        self.config = config
        self.llm = config.get_llm_model(temperature=0, max_tokens=1024)
        self.batch_size = 10  # Process in batches to avoid memory issues
        self.delay_between_batches = 1  # Seconds to wait between batches
        
    def extract_elements(self, pdf_path: str, output_image_dir: str) -> List:
        """
        Extract raw elements from PDF.
        
        Args:
            pdf_path: Path to PDF file
            output_image_dir: Directory to save extracted images
            
        Returns:
            List of raw PDF elements
        """
        print(f"Extracting elements from PDF: {pdf_path}")
        
        raw_elements = partition_pdf(
            filename=pdf_path,
            extract_images_in_pdf=True,
            infer_table_structure=True,
            chunking_strategy="by_title",
            max_characters=2000,
            new_after_n_chars=1500,
            extract_image_block_output_dir=output_image_dir,
        )
        
        print(f"Extracted {len(raw_elements)} elements")
        return raw_elements
    
    def _summarize_element(self, element_type: str, content: str) -> str:
        """
        Summarize a text or table element using LLM.
        
        Args:
            element_type: "text" or "table"
            content: Content to summarize
            
        Returns:
            Summary string
        """
        prompt_text = f"""Summarize the following {element_type}:
{content}

Provide a concise but complete summary. Be careful with numbers and don't make things up."""
        
        response = self.llm.invoke([HumanMessage(content=prompt_text)])
        return response.content
    
    def _summarize_image(self, encoded_image: str) -> str:
        """
        Summarize an image using vision-capable LLM.
        
        Args:
            encoded_image: Base64 encoded image
            
        Returns:
            Image description string
        """
        messages = [
            SystemMessage(content="You are an AI assistant that analyzes images and extracts information."),
            HumanMessage(content=[
                {
                    "type": "text",
                    "text": "Describe the text and information shown in this image. If there is no meaningful content, say 'No data found'."
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encoded_image}"
                    },
                },
            ])
        ]
        
        response = self.llm.invoke(messages)
        return response.content
    
    @staticmethod
    def _encode_image(image_path: str) -> str:
        """Encode image to base64."""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')
    
    @staticmethod
    def _has_no_data(text: str) -> bool:
        """Check if text indicates no meaningful data."""
        pattern = r"no data found"
        return bool(re.search(pattern, text, re.IGNORECASE))
    
    def _get_page_index_map(self, raw_elements: List) -> dict:
        """
        Create a mapping of page numbers to the last element index on that page.
        
        Args:
            raw_elements: List of extracted PDF elements
            
        Returns:
            Dictionary mapping page numbers to last element indices
        """
        last_indices = {}
        for i, element in enumerate(raw_elements):
            if element is not None and hasattr(element, 'metadata') and hasattr(element.metadata, 'page_number'):
                page_number = element.metadata.page_number
                last_indices[page_number] = i
        return last_indices
    
    def _process_text_tables_batched(
        self,
        raw_elements: List,
        max_elements: int = None
    ) -> Tuple[List[Document], List[str]]:
        """
        Process text and tables in batches to prevent memory issues.
        
        Args:
            raw_elements: List of extracted PDF elements
            max_elements: Maximum number to process (None = all)
            
        Returns:
            Tuple of (documents list, raw texts list)
        """
        print("\n📝 Processing text and tables in batches...")
        documents = []
        raw_texts = []
        
        # Determine elements to process
        elements_to_process = raw_elements if max_elements is None else raw_elements[:max_elements]
        total = len(elements_to_process)
        
        # Process in batches
        for batch_start in range(0, total, self.batch_size):
            batch_end = min(batch_start + self.batch_size, total)
            batch = elements_to_process[batch_start:batch_end]
            
            print(f"  📦 Batch {batch_start//self.batch_size + 1}: Processing elements {batch_start+1}-{batch_end} of {total}")
            
            for i, element in enumerate(batch):
                global_idx = batch_start + i
                
                try:
                    # Skip None or invalid elements
                    if element is None:
                        continue
                    
                    element_type_str = str(type(element).__name__)
                    
                    if 'CompositeElement' in element_type_str:
                        # Text element
                        text_content = element.text
                        summary = self._summarize_element("text", text_content)
                        
                        doc = Document(
                            page_content=summary,
                            metadata={
                                'id': str(uuid.uuid4()),
                                'type': 'text',
                                'original_content': text_content
                            }
                        )
                        documents.append(doc)
                        raw_texts.append(text_content)
                        
                    elif 'Table' in element_type_str:
                        # Table element
                        table_content = element.text
                        summary = self._summarize_element("table", table_content)
                        
                        doc = Document(
                            page_content=summary,
                            metadata={
                                'id': str(uuid.uuid4()),
                                'type': 'table',
                                'original_content': table_content
                            }
                        )
                        documents.append(doc)
                        raw_texts.append(table_content)
                        
                except Exception as e:
                    print(f"    Warning: Error processing element {global_idx+1}: {str(e)[:50]}")
                    continue
            
            # Small delay between batches to prevent API rate limits
            if batch_end < total:
                time.sleep(self.delay_between_batches)
        
        print(f"Processed {len(documents)} text/table elements")
        return documents, raw_texts
    
    def _process_images_batched(
        self,
        raw_elements: List,
        documents: List[Document],
        raw_texts: List[str],
        output_image_dir: str
    ):
        """
        Process images in batches to prevent memory issues.
        
        Args:
            raw_elements: List of extracted PDF elements
            documents: List to append image documents to
            raw_texts: List to append image texts to
            output_image_dir: Directory containing extracted images
        """
        print("\nProcessing images in batches...")
        page_map = self._get_page_index_map(raw_elements)
        image_count = 0
        
        if os.path.exists(output_image_dir):
            image_files = [f for f in os.listdir(output_image_dir) 
                          if f.endswith(('.png', '.jpg', '.jpeg'))]
            
            total_images = len(image_files)
            if total_images == 0:
                print("  No images found")
                return
            
            print(f"  Found {total_images} images to process")
            
            # Process images in batches
            for batch_start in range(0, total_images, self.batch_size):
                batch_end = min(batch_start + self.batch_size, total_images)
                batch_files = image_files[batch_start:batch_end]
                
                print(f"  Batch {batch_start//self.batch_size + 1}: Processing images {batch_start+1}-{batch_end} of {total_images}")
                
                for img_file in batch_files:
                    try:
                        image_path = os.path.join(output_image_dir, img_file)
                        
                        # Extract page number from filename
                        try:
                            page_num = int(img_file.split("-")[1])
                        except (IndexError, ValueError):
                            print(f"    Warning: Skipping {img_file} (couldn't parse page number)")
                            continue
                        
                        # Encode and summarize image
                        encoded_image = self._encode_image(image_path)
                        img_summary = self._summarize_image(encoded_image)
                        
                        # Skip if no meaningful content
                        if self._has_no_data(img_summary):
                            continue
                        
                        # Try to find related text
                        context_text = ""
                        for offset in range(0, 3):
                            check_page = page_num - offset
                            if check_page in page_map:
                                element_idx = page_map[check_page]
                                element = raw_elements[element_idx]
                                if hasattr(element, 'text'):
                                    context_text = self._summarize_element("text", element.text)
                                    break
                        
                        # Combine image summary with context
                        full_summary = img_summary
                        if context_text:
                            full_summary = f"{img_summary}\n\nContext: {context_text}"
                        
                        doc = Document(
                            page_content=full_summary,
                            metadata={
                                'id': str(uuid.uuid4()),
                                'type': 'image',
                                'original_content': encoded_image,
                                'page': page_num
                            }
                        )
                        documents.append(doc)
                        raw_texts.append(full_summary)
                        image_count += 1
                        
                    except Exception as e:
                        print(f"    Warning: Error processing {img_file}: {str(e)[:50]}")
                        continue
                
                # Delay between batches
                if batch_end < total_images:
                    time.sleep(self.delay_between_batches)
            
            print(f"Processed {image_count} images")
        else:
            print("  No image directory found")
    
    def process_pdf(
        self,
        pdf_path: str,
        output_image_dir: str = None,
        max_elements: int = None,
        skip_images: bool = False,
        batch_size: int = None
    ) -> Tuple[List[Document], List[str]]:
        """
        Process PDF and extract all elements with summaries.
        Automatically chunks processing to prevent crashes.
        
        Args:
            pdf_path: Path to PDF file
            output_image_dir: Directory for extracted images (default: data/images)
            max_elements: Maximum number of text/table elements to process (None = all)
            skip_images: Skip image processing to save time/resources
            batch_size: Number of elements to process at once (default: 10)
            
        Returns:
            Tuple of (documents list, raw texts list)
        """
        if batch_size:
            self.batch_size = batch_size
            
        if output_image_dir is None:
            output_image_dir = os.path.join(
                os.path.dirname(pdf_path), "images"
            )
        
        os.makedirs(output_image_dir, exist_ok=True)
        
        # Extract raw elements
        raw_elements = self.extract_elements(pdf_path, output_image_dir)
        
        # Inform user about what will be processed
        total_elements = len(raw_elements)
        elements_to_process = total_elements if max_elements is None else min(max_elements, total_elements)
        
        print(f"\nProcessing Plan:")
        print(f"  Total elements found: {total_elements}")
        print(f"  Will process: {elements_to_process}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Estimated batches: {(elements_to_process + self.batch_size - 1) // self.batch_size}")
        print(f"  Skip images: {skip_images}")
        
        if elements_to_process > 100:
            est_minutes = elements_to_process // 10  # Rough estimate: 10 elements per minute
            print(f"  Estimated time: ~{est_minutes} minutes")
            print(f"  Estimated cost: ~${elements_to_process * 0.015:.2f}")
        
        # Process text and tables in batches
        documents, raw_texts = self._process_text_tables_batched(raw_elements, max_elements)
        
        # Process images if requested
        if not skip_images:
            self._process_images_batched(raw_elements, documents, raw_texts, output_image_dir)
        else:
            print("\nSkipping image processing (skip_images=True)")
        
        print(f"\nTotal: {len(documents)} documents extracted")
        return documents, raw_texts
