import json
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from pathlib import Path
import csv

from .pattern_matcher import MedicalReading, ReadingType
from .number_extractor import ExtractedNumber

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ProcessingStatus(Enum):
    """Status of image processing."""
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"

@dataclass
class ProcessingResult:
    """Result of processing a single image."""
    image_path: str
    timestamp: str
    status: ProcessingStatus
    readings: List[MedicalReading]
    confidence: float
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None

@dataclass
class BatchResult:
    """Result of processing a batch of images."""
    total_images: int
    successful_images: int
    failed_images: int
    results: List[ProcessingResult]
    summary: Dict[str, Any]

class DataProcessor:
    """Processes and exports medical readings data."""
    
    def __init__(self, output_dir: str):
        """Initialize the data processor.
        
        Args:
            output_dir: Directory to save processed results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def process_single_image(
        self,
        image_path: str,
        readings: List[MedicalReading],
        confidence: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ProcessingResult:
        """Process a single image's readings.
        
        Args:
            image_path: Path to the processed image
            readings: List of medical readings extracted from the image
            confidence: Confidence score of the OCR process
            metadata: Additional metadata about the image/device
            
        Returns:
            ProcessingResult containing the processing status and results
        """
        if not readings:
            return ProcessingResult(
                image_path=image_path,
                timestamp=datetime.now().isoformat(),
                status=ProcessingStatus.FAILED,
                readings=[],
                confidence=confidence,
                error_message="No readings found",
                metadata=metadata or {}
            )
        
        # Count valid readings
        valid_readings = [r for r in readings if r.is_valid]
        
        # Determine status
        if len(valid_readings) == len(readings):
            status = ProcessingStatus.SUCCESS
            error_message = None
        elif len(valid_readings) == 0:
            status = ProcessingStatus.FAILED
            error_message = "No valid readings found"
        else:
            status = ProcessingStatus.PARTIAL
            error_message = f"Only {len(valid_readings)} of {len(readings)} readings are valid"
        
        result = ProcessingResult(
            image_path=image_path,
            timestamp=datetime.now().isoformat(),
            status=status,
            readings=readings,
            confidence=confidence,
            error_message=error_message,
            metadata=metadata or {}
        )
        
        # Save result to JSON
        self._save_result(result)
        
        return result
    
    def process_batch(
        self,
        image_results: List[Dict[str, Any]]
    ) -> BatchResult:
        """Process a batch of image results.
        
        Args:
            image_results: List of dictionaries containing image processing results
            
        Returns:
            BatchResult containing summary statistics and individual results
        """
        results = []
        successful_images = 0
        failed_images = 0
        total_readings = 0
        valid_readings = 0
        total_confidence = 0.0
        reading_type_counts = {rt.name: 0 for rt in ReadingType}
        
        for img_result in image_results:
            result = self.process_single_image(
                image_path=img_result['image_path'],
                readings=img_result['readings'],
                confidence=img_result['confidence'],
                metadata=img_result.get('metadata')
            )
            
            results.append(result)
            
            # Update statistics
            if result.status == ProcessingStatus.SUCCESS:
                successful_images += 1
            elif result.status == ProcessingStatus.FAILED:
                failed_images += 1
            
            total_readings += len(result.readings)
            valid_readings += sum(1 for r in result.readings if r.is_valid)
            total_confidence += result.confidence
            
            # Count reading types
            for reading in result.readings:
                if reading.is_valid:
                    reading_type_counts[reading.type.name] += 1
        
        # Calculate summary
        summary = {
            'total_readings': total_readings,
            'valid_readings': valid_readings,
            'average_confidence': total_confidence / len(image_results) if image_results else 0.0,
            'reading_types': reading_type_counts
        }
        
        batch_result = BatchResult(
            total_images=len(image_results),
            successful_images=successful_images,
            failed_images=failed_images,
            results=results,
            summary=summary
        )
        
        # Save batch result
        self._save_batch_result(batch_result)
        
        return batch_result
    
    def export_to_csv(self, batch_result: BatchResult) -> str:
        """Export batch results to CSV format.
        
        Args:
            batch_result: BatchResult to export
            
        Returns:
            Path to the created CSV file
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_path = self.output_dir / f'results_{timestamp}.csv'
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow([
                'Image Path',
                'Timestamp',
                'Status',
                'Reading Type',
                'Value',
                'Unit',
                'Is Valid',
                'Confidence'
            ])
            
            # Write data rows
            for result in batch_result.results:
                for reading in result.readings:
                    writer.writerow([
                        result.image_path,
                        result.timestamp,
                        result.status.value,
                        reading.type.name,
                        reading.value,
                        reading.unit,
                        reading.is_valid,
                        result.confidence
                    ])
        
        return str(csv_path)
    
    def _save_result(self, result: ProcessingResult) -> None:
        """Save a single processing result to JSON.
        
        Args:
            result: ProcessingResult to save
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'result_{timestamp}.json'
        
        data = {
            'image_path': result.image_path,
            'timestamp': result.timestamp,
            'status': result.status.value,
            'readings': [
                {
                    'type': r.type.name,
                    'value': r.value,
                    'unit': r.unit,
                    'is_valid': r.is_valid
                }
                for r in result.readings
            ],
            'confidence': result.confidence,
            'error_message': result.error_message,
            'metadata': result.metadata
        }
        
        with open(self.output_dir / filename, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _save_batch_result(self, batch_result: BatchResult) -> None:
        """Save a batch result to JSON.
        
        Args:
            batch_result: BatchResult to save
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'batch_result_{timestamp}.json'
        
        data = {
            'total_images': batch_result.total_images,
            'successful_images': batch_result.successful_images,
            'failed_images': batch_result.failed_images,
            'summary': batch_result.summary,
            'results': [
                {
                    'image_path': r.image_path,
                    'timestamp': r.timestamp,
                    'status': r.status.value,
                    'readings': [
                        {
                            'type': reading.type.name,
                            'value': reading.value,
                            'unit': reading.unit,
                            'is_valid': reading.is_valid
                        }
                        for reading in r.readings
                    ],
                    'confidence': r.confidence,
                    'error_message': r.error_message,
                    'metadata': r.metadata
                }
                for r in batch_result.results
            ]
        }
        
        with open(self.output_dir / filename, 'w') as f:
            json.dump(data, f, indent=2) 