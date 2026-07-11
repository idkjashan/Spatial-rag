import os
import re
import json
import base64
import asyncio
import logging
from typing import List, Dict, Tuple, Optional

from openai import AsyncOpenAI, RateLimitError, APIConnectionError, APITimeoutError
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type

from core.models.node import Node, ModalityCategory
from core.models.edge import Edge
from core.models.document import Document

try:
    from json_repair import repair_json
    HAS_JSON_REPAIR = True
except ImportError:
    HAS_JSON_REPAIR = False

logger = logging.getLogger(__name__)

class StrictRateLimiter:
    def __init__(self, requests_per_minute: int):
        self.min_interval = 60.0 / requests_per_minute if requests_per_minute > 0 else 0
        self.lock = asyncio.Lock()
        self.next_allowed_time = 0.0

    async def acquire(self):
        if self.min_interval == 0: return
        async with self.lock:
            now = asyncio.get_event_loop().time()
            if now < self.next_allowed_time:
                wait_time = self.next_allowed_time - now
                await asyncio.sleep(wait_time)
            self.next_allowed_time = max(now, self.next_allowed_time) + self.min_interval


class ContextualEnricher:
    def __init__(
        self, 
        llm_base_url: str = "http://localhost:11434/v1", 
        llm_api_key: str = "ollama",
        slm_model: str = "llama3.1:8b-instruct-q4_K_M",
        vlm_model: str = "llava:13b",
        max_concurrent_requests: int = 3,
        requests_per_minute: int = 0,
        request_timeout: int = 120
    ):
        self.client = AsyncOpenAI(
            base_url=llm_base_url, 
            api_key=llm_api_key, 
            timeout=request_timeout,
            max_retries=0
        )
        self.slm_model = slm_model
        self.vlm_model = vlm_model
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)
        self.rate_limiter = StrictRateLimiter(requests_per_minute)

    def process(self, document: Document, nodes: List[Node], edges: List[Edge]) -> Tuple[Document, List[Node], List[Edge]]:
        logger.info(f"Starting Phase 2 Contextual Enrichment for {len(nodes)} nodes...")
        try:
            loop = asyncio.get_running_loop()
            raise RuntimeError("Async event loop already running. Use `await enricher.aprocess(...)` instead.")
        except RuntimeError:
            return asyncio.run(self.aprocess(document, nodes, edges))

    async def aprocess(self, document: Document, nodes: List[Node], edges: List[Edge]) -> Tuple[Document, List[Node], List[Edge]]:
        node_map = {n.id: n for n in nodes}
        tasks = []
        
        for node in nodes:
            if "[VLM_SUMMARY]" in node.content or "[SLM_SUMMARY]" in node.content or "[SLM_EXPLANATION]" in node.content:
                continue

            if node.modality_category == ModalityCategory.IMAGE and node.image_path:
                tasks.append(self._enrich_image(node))
            elif node.modality_category == ModalityCategory.TABLE_CONTAINER:
                tasks.append(self._enrich_table(node))
            elif node.modality_category == ModalityCategory.EQUATION:
                tasks.append(self._enrich_formula(node, node_map, edges))
            elif node.modality_category == ModalityCategory.DOCUMENT_STRUCTURE and node.modality == "list":
                tasks.append(self._enrich_list(node, node_map, edges))
                
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for res in results:
            if isinstance(res, Exception):
                logger.error(f"Enrichment task failed permanently: {res}")
                
        document.touch()
        return document, nodes, edges

    def _parse_json_response(self, text: Optional[str]) -> Dict:
        if not text: return {}
        text = re.sub(r'^```(?:json)?\s*', '', text.strip())
        text = re.sub(r'\s*```$', '', text.strip())
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match: text = match.group(0)
            
        if HAS_JSON_REPAIR:
            return repair_json(text, return_objects=True)
        else:
            try: return json.loads(text)
            except json.JSONDecodeError: return {}

    @retry(
        retry=retry_if_exception_type((RateLimitError, APIConnectionError, APITimeoutError)),
        stop=stop_after_attempt(5),
        wait=wait_fixed(2),
        before_sleep=lambda retry_state: logger.warning(f"API limit/network error. Retrying...")
    )
    async def _call_llm(self, messages: List[Dict], json_mode: bool = False) -> Optional[str]:
        await self.rate_limiter.acquire()
        async with self.semaphore:
            kwargs = {"model": self.slm_model, "messages": messages, "temperature": 0.1}
            if json_mode: kwargs["response_format"] = {"type": "json_object"}
            response = await self.client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content
            if not content: logger.warning(f"LLM ({self.slm_model}) returned empty content.")
            return content

    @retry(
        retry=retry_if_exception_type((RateLimitError, APIConnectionError, APITimeoutError)),
        stop=stop_after_attempt(5),
        wait=wait_fixed(2),
        before_sleep=lambda retry_state: logger.warning(f"API limit/network error. Retrying...")
    )
    async def _call_vlm(self, messages: List[Dict]) -> Optional[str]:
        await self.rate_limiter.acquire()
        async with self.semaphore:
            response = await self.client.chat.completions.create(model=self.vlm_model, messages=messages, temperature=0.1)
            content = response.choices[0].message.content
            if not content: logger.warning(f"VLM ({self.vlm_model}) returned empty content.")
            return content

    def _encode_image(self, image_path: str) -> Optional[str]:
        if not image_path: return None
        
        actual_path = image_path
        if not os.path.exists(actual_path):
            abs_path = os.path.abspath(image_path)
            if os.path.exists(abs_path):
                actual_path = abs_path
            else:
                logger.warning(f"    Image file does not exist: {image_path} (also tried {abs_path})")
                return None
                
        try:
            with open(actual_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            logger.error(f"    Error encoding image {actual_path}: {e}")
            return None

    async def _enrich_image(self, node: Node):
        base64_img = self._encode_image(node.image_path)
        if not base64_img: return
        
        prompt = (
            "You are an expert engineering assistant analyzing a diagram or figure from a technical document. "
            "Provide a dense, 2-sentence summary of what the image depicts. "
            "Then, identify any distinct components, labels, or flow directions visible. "
            "Return your response strictly as JSON: {\"summary\": \"...\", \"components\": [\"...\", \"...\"]}"
        )
        messages = [
            {"role": "system", "content": "You are a helpful vision assistant designed to output JSON."},
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_img}"}}
            ]}
        ]
        response_str = await self._call_vlm(messages)
        data = self._parse_json_response(response_str)
        summary = data.get("summary", "")
        components = data.get("components", [])
        if summary:
            node.content = f"[VLM_SUMMARY]: {summary}\n\n{node.content}"
            node.node_meta["vlm_components"] = components
            logger.info(f"  [VLM] Enriched Image {node.id}: {summary[:50]}...")

    async def _enrich_table(self, node: Node):
        lines = node.content.split('\n')
        snippet = '\n'.join(lines[:5]) 
        prompt = (
            "Analyze the following markdown table snippet from a technical document. "
            "Do NOT repeat the table data. Instead, provide a 1-sentence summary of what entities or data this table catalogs. "
            "Return strictly as JSON: {\"summary\": \"...\"}\n\n"
            f"Table Snippet:\n{snippet}"
        )
        messages = [{"role": "system", "content": "You are a helpful assistant designed to output JSON."}, {"role": "user", "content": prompt}]
        response_str = await self._call_llm(messages, json_mode=True)
        data = self._parse_json_response(response_str)
        summary = data.get("summary", "")
        if summary:
            node.content = f"[SLM_SUMMARY]: {summary}\n\n{node.content}"
            logger.info(f"  [SLM] Enriched Table {node.id}: {summary[:50]}...")

    async def _enrich_formula(self, node: Node, node_map: Dict[str, Node], edges: List[Edge]):
        latex_text = ""
        image_path = None
        child_node = None
        
        logger.info(f"  [FORMULA] Processing equation node {node.id}...")
        
        for e in edges:
            if e.source_id == node.id and e.type == "contains_latex":
                child_node = node_map.get(e.target_id)
                if child_node:
                    content = child_node.content or ""
                    clean_content = re.sub(r'^\[SECTION_CONTEXT\].*?>\s*', '', content).strip()
                    if clean_content:
                        latex_text = clean_content
                        
                    if child_node.image_path: 
                        image_path = child_node.image_path
                    break
                else:
                    logger.warning(f"  [WARN] Formula {node.id} has 'contains_latex' edge to missing node {e.target_id}")
        
        if latex_text:
            logger.info(f"    Path: TEXT SLM. Text: {latex_text[:80]}...")
            prompt = (
                "You are an expert engineer. Translate the following LaTeX or mathematical formula into plain English. "
                "Explain what it calculates or represents in 1-2 sentences. Do not output the raw LaTeX. "
                "Return strictly as JSON: {\"explanation\": \"...\"}\n\n"
                f"Formula:\n{latex_text}"
            )
            messages = [{"role": "system", "content": "You are a helpful assistant designed to output JSON."}, {"role": "user", "content": prompt}]
            try:
                response_str = await self._call_llm(messages, json_mode=True)
                data = self._parse_json_response(response_str)
                explanation = data.get("explanation", "")
                if not explanation and response_str: explanation = response_str.strip().strip('"')
                if explanation:
                    node.content = f"[SLM_EXPLANATION]: {explanation}"
                    logger.info(f"  [SLM] Enriched Formula {node.id}: {explanation[:50]}...")
            except Exception as e:
                logger.error(f"  [SLM] Error enriching Formula {node.id} from text: {e}")

        elif image_path:
            logger.info(f"    Path: IMAGE VLM. Image: {image_path}")
            base64_img = self._encode_image(image_path)
            if not base64_img: 
                logger.warning(f"    Skipping VLM due to missing/invalid image for {node.id}")
                return
                
            prompt = (
                "You are an expert engineer. Look at this mathematical formula. "
                "First, extract the formula in LaTeX format. Second, explain what it calculates in 1-2 sentences. "
                "Return strictly as JSON: {\"latex\": \"...\", \"explanation\": \"...\"}"
            )
            messages = [
                {"role": "system", "content": "You are a helpful vision assistant designed to output JSON."},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_img}"}}
                ]}
            ]
            try:
                response_str = await self._call_vlm(messages)
                data = self._parse_json_response(response_str)
                extracted_latex = data.get("latex", "")
                explanation = data.get("explanation", "")
                
                if extracted_latex and child_node:
                    child_node.content = extracted_latex
                    child_node.node_meta["latex"] = extracted_latex
                    logger.info(f"    Extracted LaTeX and saved to child node: {extracted_latex[:80]}...")
                
                if explanation:
                    node.content = f"[SLM_EXPLANATION]: {explanation}"
                    logger.info(f"  [VLM] Enriched Formula {node.id}: {explanation[:50]}...")
                else:
                    logger.warning(f"    VLM returned no explanation for {node.id}. Response: {response_str[:100] if response_str else 'None'}")
            except Exception as e:
                logger.error(f"  [VLM] Error enriching Formula {node.id} from image: {e}")
        
        else:
            logger.warning(f"  [SKIP] Formula {node.id}: No LaTeX text or image found in subgraph.")

    async def _enrich_list(self, node: Node, node_map: Dict[str, Node], edges: List[Edge]):
        items = []
        for e in edges:
            if e.source_id == node.id and e.type == "contains_item":
                child = node_map.get(e.target_id)
                if child: 
                    clean_content = re.sub(r'^\[SECTION_CONTEXT\].*?>\s*', '', child.content).strip()
                    if clean_content:
                        items.append(clean_content)
                
        if not items: return
        
        list_snippet = '\n'.join(f"- {i}" for i in items[:5])
        prompt = (
            "Analyze the following list items from a technical document. "
            "Provide a 1-sentence summary of what this list enumerates or describes. "
            "Return strictly as JSON: {\"summary\": \"...\"}\n\n"
            f"List Items:\n{list_snippet}"
        )
        messages = [{"role": "system", "content": "You are a helpful assistant designed to output JSON."}, {"role": "user", "content": prompt}]
        response_str = await self._call_llm(messages, json_mode=True)
        data = self._parse_json_response(response_str)
        summary = data.get("summary", "")
        if summary:
            node.content = f"[SLM_SUMMARY]: {summary}\n\nList Group"
            logger.info(f"  [SLM] Enriched List {node.id}: {summary[:50]}...")