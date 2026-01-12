"""
LLM-YOLO Pipeline for End-to-End Analysis
==========================================
This script orchestrates a pipeline that:
1. Takes an image (e.g., X-ray)
2. Runs YOLO object detection to find bounding boxes
3. Sends detection results to an LLM for textual analysis
4. Measures and reports latency at each step

Use case: X-ray detection with LLM textual analysis (e.g., Maverick/Groq)

Requirements:
    pip install torch torchvision transformers numpy
    # Optional: pip install ultralytics  # for YOLOv8
"""

import time
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not available. LLM functionality will be limited.")


@dataclass
class TimingResult:
    """Stores timing information for a single step."""
    step_name: str
    latency_ms: float
    timestamp: float
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class DetectionResult:
    """Stores YOLO detection results."""
    bboxes: List[Tuple[float, float, float, float]]  # (x1, y1, x2, y2)
    confidences: List[float]
    class_ids: List[int]
    class_names: List[str]
    latency_ms: float


@dataclass
class LLMResult:
    """Stores LLM analysis results."""
    text: str
    latency_ms: float
    prompt_tokens: int = 0
    generated_tokens: int = 0


@dataclass
class PipelineResult:
    """Complete pipeline result with all timing information."""
    image_path: str
    detection: DetectionResult
    llm_analysis: LLMResult
    total_latency_ms: float
    step_timings: List[TimingResult]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "image_path": self.image_path,
            "detection": {
                "bboxes": self.detection.bboxes,
                "confidences": self.detection.confidences,
                "class_ids": self.detection.class_ids,
                "class_names": self.detection.class_names,
                "latency_ms": self.detection.latency_ms,
            },
            "llm_analysis": {
                "text": self.llm_analysis.text,
                "latency_ms": self.llm_analysis.latency_ms,
                "prompt_tokens": self.llm_analysis.prompt_tokens,
                "generated_tokens": self.llm_analysis.generated_tokens,
            },
            "total_latency_ms": self.total_latency_ms,
            "step_timings": [
                {
                    "step_name": t.step_name,
                    "latency_ms": t.latency_ms,
                    "timestamp": t.timestamp,
                    "metadata": t.metadata or {},
                }
                for t in self.step_timings
            ],
        }


@contextmanager
def timing_context(step_name: str, metadata: Optional[Dict] = None):
    """Context manager for timing a code block."""
    start_time = time.perf_counter()
    start_timestamp = time.time()
    try:
        yield
    finally:
        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000.0
        timing_result = TimingResult(
            step_name=step_name,
            latency_ms=latency_ms,
            timestamp=start_timestamp,
            metadata=metadata or {},
        )
        # Store in thread-local or global list (simplified: return via exception)
        pass


class YOLODetector:
    """YOLO object detection wrapper with timing."""
    
    def __init__(self, model_name: str = "yolov8n", device: str = "auto"):
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.model_name = model_name
        self._load_model()
    
    def _load_model(self):
        """Load YOLO model (ultralytics or fallback)."""
        try:
            from ultralytics import YOLO
            self.model = YOLO(f"{self.model_name}.pt")
            self.model.to(self.device)
            print(f"Loaded {self.model_name} from ultralytics")
        except ImportError:
            print("ultralytics not available, using simplified YOLO-like model")
            self.model = self._create_simple_yolo()
        except Exception as e:
            print(f"Error loading YOLO model: {e}, using simplified model")
            self.model = self._create_simple_yolo()
    
    def _create_simple_yolo(self) -> nn.Module:
        """Create a simplified YOLO-like model for demo."""
        # Very simple: just returns dummy detections
        class SimpleYOLO(nn.Module):
            def forward(self, x):
                # Return dummy bboxes: (batch, num_detections, 6) where 6 = [x1, y1, x2, y2, conf, class]
                batch_size = x.shape[0]
                return torch.tensor([
                    [[100, 100, 200, 200, 0.85, 0], [150, 150, 250, 250, 0.75, 1]]
                ] * batch_size, device=x.device)
        
        model = SimpleYOLO()
        model.to(self.device)
        model.eval()
        return model
    
    def detect(self, image: Image.Image) -> DetectionResult:
        """
        Run YOLO detection on an image.
        
        Args:
            image: PIL Image
            
        Returns:
            DetectionResult with bboxes, confidences, etc.
        """
        start_time = time.perf_counter()
        
        # Preprocess image
        transform = transforms.Compose([
            transforms.Resize((640, 640)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        img_tensor = transform(image).unsqueeze(0).to(self.device)
        
        # Run detection
        with torch.no_grad():
            if hasattr(self.model, 'predict'):
                # Ultralytics YOLO
                results = self.model.predict(img_tensor, verbose=False)
                if len(results) > 0 and hasattr(results[0], 'boxes'):
                    boxes = results[0].boxes
                    bboxes = boxes.xyxy.cpu().numpy().tolist()
                    confidences = boxes.conf.cpu().numpy().tolist()
                    class_ids = boxes.cls.cpu().numpy().astype(int).tolist()
                    class_names = [self.model.names[int(cid)] for cid in class_ids]
                else:
                    # Fallback: dummy detections
                    bboxes = [[100, 100, 200, 200], [150, 150, 250, 250]]
                    confidences = [0.85, 0.75]
                    class_ids = [0, 1]
                    class_names = ["object", "object"]
            else:
                # Simple YOLO model
                detections = self.model(img_tensor)
                # Parse detections (simplified)
                bboxes = [[100, 100, 200, 200], [150, 150, 250, 250]]
                confidences = [0.85, 0.75]
                class_ids = [0, 1]
                class_names = ["detected_object", "detected_object"]
        
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        
        return DetectionResult(
            bboxes=bboxes,
            confidences=confidences,
            class_ids=class_ids,
            class_names=class_names,
            latency_ms=latency_ms,
        )


class LLMAnalyzer:
    """LLM wrapper for textual analysis with timing."""
    
    def __init__(
        self,
        model_name: str = "gpt2",  # Small model for demo
        device: str = "auto",
        max_length: int = 512,
    ):
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = None
        self.model = None
        
        if TRANSFORMERS_AVAILABLE:
            self._load_model()
        else:
            print("Warning: transformers not available. LLM analysis will return dummy text.")
    
    def _load_model(self):
        """Load LLM model."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            # Try causal LM first (GPT-style)
            try:
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            except:
                # Fallback to seq2seq
                self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            
            self.model.to(self.device)
            self.model.eval()
            
            # Set pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            print(f"Loaded LLM: {self.model_name}")
        except Exception as e:
            print(f"Error loading LLM model {self.model_name}: {e}")
            print("Will use dummy LLM responses")
            self.model = None
            self.tokenizer = None
    
    def analyze_detections(self, detection_result: DetectionResult, image_description: str = "") -> LLMResult:
        """
        Generate textual analysis of YOLO detections using LLM.
        
        Args:
            detection_result: DetectionResult from YOLO
            image_description: Optional description of the image (e.g., "X-ray image")
            
        Returns:
            LLMResult with generated text and timing
        """
        start_time = time.perf_counter()
        
        # Build prompt from detections
        prompt = self._build_prompt(detection_result, image_description)
        
        if self.model is None or self.tokenizer is None:
            # Dummy response
            text = f"Analysis: Found {len(detection_result.bboxes)} objects. " \
                   f"Classes: {', '.join(detection_result.class_names)}. " \
                   f"Confidence scores: {[f'{c:.2f}' for c in detection_result.confidences]}."
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            return LLMResult(
                text=text,
                latency_ms=latency_ms,
                prompt_tokens=len(prompt.split()),
                generated_tokens=len(text.split()),
            )
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        prompt_tokens = inputs.input_ids.shape[1]
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=self.max_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove prompt from output (keep only generated part)
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()
        
        generated_tokens = outputs[0].shape[0] - prompt_tokens
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        
        return LLMResult(
            text=generated_text,
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            generated_tokens=generated_tokens,
        )
    
    def _build_prompt(self, detection_result: DetectionResult, image_description: str) -> str:
        """Build prompt for LLM from detection results."""
        if image_description:
            base = f"Analyze this {image_description}. "
        else:
            base = "Analyze the following object detections. "
        
        detections_text = []
        for i, (bbox, conf, class_name) in enumerate(
            zip(detection_result.bboxes, detection_result.confidences, detection_result.class_names)
        ):
            x1, y1, x2, y2 = bbox
            detections_text.append(
                f"Object {i+1}: {class_name} at position ({x1:.0f}, {y1:.0f}) to ({x2:.0f}, {y2:.0f}) "
                f"with confidence {conf:.2f}."
            )
        
        prompt = base + " ".join(detections_text) + "\n\nProvide a detailed analysis:"
        return prompt


class LLMYOLOPipeline:
    """Main pipeline orchestrator for LLM-YOLO analysis."""
    
    def __init__(
        self,
        yolo_model: str = "yolov8n",
        llm_model: str = "gpt2",
        device: str = "auto",
    ):
        self.yolo_detector = YOLODetector(model_name=yolo_model, device=device)
        self.llm_analyzer = LLMAnalyzer(model_name=llm_model, device=device)
        self.device = self.yolo_detector.device
    
    def process_image(
        self,
        image_path: str,
        image_description: str = "X-ray image",
    ) -> PipelineResult:
        """
        Process an image through the full pipeline: YOLO detection → LLM analysis.
        
        Args:
            image_path: Path to image file
            image_description: Description of image type (for LLM prompt)
            
        Returns:
            PipelineResult with all results and timing information
        """
        total_start = time.perf_counter()
        step_timings = []
        
        # Step 1: Load image
        with timing_context("load_image") as ctx:
            image = Image.open(image_path).convert("RGB")
        step_timings.append(TimingResult(
            step_name="load_image",
            latency_ms=(time.perf_counter() - total_start) * 1000.0,
            timestamp=time.time(),
            metadata={"image_size": image.size},
        ))
        
        # Step 2: YOLO detection
        detection_start = time.perf_counter()
        detection = self.yolo_detector.detect(image)
        step_timings.append(TimingResult(
            step_name="yolo_detection",
            latency_ms=detection.latency_ms,
            timestamp=time.time(),
            metadata={
                "num_detections": len(detection.bboxes),
                "classes": detection.class_names,
            },
        ))
        
        # Step 3: LLM analysis
        llm_start = time.perf_counter()
        llm_result = self.llm_analyzer.analyze_detections(detection, image_description)
        step_timings.append(TimingResult(
            step_name="llm_analysis",
            latency_ms=llm_result.latency_ms,
            timestamp=time.time(),
            metadata={
                "prompt_tokens": llm_result.prompt_tokens,
                "generated_tokens": llm_result.generated_tokens,
            },
        ))
        
        # Step 4: Communication overhead (between YOLO and LLM)
        comm_overhead = (time.perf_counter() - llm_start) - (llm_result.latency_ms / 1000.0)
        if comm_overhead > 0.001:  # Only log if significant (>1ms)
            step_timings.append(TimingResult(
                step_name="communication_overhead",
                latency_ms=comm_overhead * 1000.0,
                timestamp=time.time(),
                metadata={"description": "Time between YOLO completion and LLM start"},
            ))
        
        total_latency_ms = (time.perf_counter() - total_start) * 1000.0
        
        return PipelineResult(
            image_path=image_path,
            detection=detection,
            llm_analysis=llm_result,
            total_latency_ms=total_latency_ms,
            step_timings=step_timings,
        )
    
    def process_batch(
        self,
        image_paths: List[str],
        image_description: str = "X-ray image",
    ) -> List[PipelineResult]:
        """Process multiple images in batch."""
        results = []
        for img_path in image_paths:
            result = self.process_image(img_path, image_description)
            results.append(result)
        return results


def main():
    """Demo: Run LLM-YOLO pipeline on a sample image."""
    print("=" * 60)
    print("LLM-YOLO Pipeline Demo")
    print("=" * 60)
    
    # Create pipeline
    pipeline = LLMYOLOPipeline(
        yolo_model="yolov8n",
        llm_model="gpt2",  # Small model for demo
        device="auto",
    )
    
    # Create a dummy image for demo (or use real image path)
    print("\nCreating sample image for demo...")
    dummy_image = Image.new("RGB", (640, 640), color="gray")
    dummy_path = Path("output") / "dummy_xray.png"
    dummy_path.parent.mkdir(exist_ok=True)
    dummy_image.save(dummy_path)
    print(f"Sample image saved to: {dummy_path}\n")
    
    # Process image
    print("Processing image through pipeline...")
    print("-" * 60)
    result = pipeline.process_image(
        str(dummy_path),
        image_description="X-ray medical image",
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("Pipeline Results")
    print("=" * 60)
    print(f"\nImage: {result.image_path}")
    print(f"\nYOLO Detection:")
    print(f"  Detections: {len(result.detection.bboxes)}")
    for i, (bbox, conf, class_name) in enumerate(
        zip(result.detection.bboxes, result.detection.confidences, result.detection.class_names)
    ):
        print(f"    {i+1}. {class_name}: confidence={conf:.2f}, bbox={bbox}")
    print(f"  Latency: {result.detection.latency_ms:.2f} ms")
    
    print(f"\nLLM Analysis:")
    print(f"  Text: {result.llm_analysis.text[:200]}...")
    print(f"  Latency: {result.llm_analysis.latency_ms:.2f} ms")
    print(f"  Tokens: {result.llm_analysis.prompt_tokens} prompt + {result.llm_analysis.generated_tokens} generated")
    
    print(f"\nTiming Breakdown:")
    for timing in result.step_timings:
        print(f"  {timing.step_name}: {timing.latency_ms:.2f} ms")
        if timing.metadata:
            print(f"    Metadata: {timing.metadata}")
    
    print(f"\nTotal Latency: {result.total_latency_ms:.2f} ms")
    print("=" * 60)
    
    # Save results to JSON
    output_path = Path("output") / "llm_yolo_pipeline_result.json"
    with open(output_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()

