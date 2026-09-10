"""Experiment Run Management System for STADS

Provides object-oriented experiment run representation with JSON persistence,
status tracking, and thread-safe updates for parallel execution.

This module solves the maintainability and resumability issues with the
original tuple-based approach in experiments_main.py.
"""

import os
import json
import threading
import hashlib
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple, List
from enum import Enum


class ExperimentStatus(Enum):
    """Status enumeration for experiment runs."""
    NOT_STARTED = "not_started"
    RUNNING = "running"  # Added for granularity
    FINISHED = "finished"
    ERROR = "error"


@dataclass
class ExperimentRun:
    """Object representation of a single experiment run.
    
    This class encapsulates all parameters needed to run a STADS experiment
    and provides equality based on parameter values (not object identity).
    
    Attributes:
        gt_name: Ground truth video name
        scanned_pixel_percent: Percentage of pixels to scan
        sampler_type: Type of sampler ('adaptive', 'stratified', 'random')
        interpol_method: Interpolation method ('linear', 'cubic')
        has_temporal_sampler: Whether temporal sampling is enabled
        has_temporal_reconstruction: Whether temporal reconstruction is enabled
        alpha: Temporal scaling factor
        adaptive_fraction: Fraction of budget for adaptive refinement
        min_density_gamma: Minimum density floor
        temporal_method: Temporal method ('temporal_variance', 'optical_flow')
        temporal_residual_cutoff: Cutoff for temporal variance
        temporal_residual_confidence_scale: Confidence scaling for temporal signal
        sample_sequence: Sample sequence type ('uniform', 'stratified', 'halton')
        status: Current status of the experiment
        result_path: Path to results (if any)
        error_message: Error message if status is ERROR
    """
    
    # Core parameters (used for equality)
    gt_name: str
    scanned_pixel_percent: float
    sampler_type: str = "adaptive"
    interpol_method: str = "linear"
    has_temporal_sampler: bool = True
    has_temporal_reconstruction: bool = True
    alpha: Optional[float] = None
    adaptive_fraction: float = 0.0
    min_density_gamma: float = 0.0
    temporal_method: str = "optical_flow"
    temporal_residual_cutoff: float = 12.0
    temporal_residual_confidence_scale: float = 100.0
    sample_sequence: str = "stratified"
    
    # Runtime/state parameters (NOT used for equality)
    status: ExperimentStatus = ExperimentStatus.NOT_STARTED
    result_path: Optional[str] = None
    error_message: Optional[str] = None
    experiment_id: str = field(default="", compare=False)
    
    def __post_init__(self):
        """Initialize experiment_id based on parameter hash if not provided."""
        if not self.experiment_id:
            self.experiment_id = self._generate_id()
    
    def _generate_id(self) -> str:
        """Generate a unique ID based on the experiment parameters."""
        # Create a string representation of all equality-relevant parameters
        params_str = (
            f"{self.gt_name}|{self.scanned_pixel_percent}|{self.sampler_type}|"
            f"{self.interpol_method}|{self.has_temporal_sampler}|{self.has_temporal_reconstruction}|"
            f"{self.alpha}|{self.adaptive_fraction}|{self.min_density_gamma}|"
            f"{self.temporal_method}|{self.temporal_residual_cutoff}|"
            f"{self.temporal_residual_confidence_scale}|{self.sample_sequence}"
        )
        return hashlib.md5(params_str.encode()).hexdigest()[:12]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "gt_name": self.gt_name,
            "scanned_pixel_percent": self.scanned_pixel_percent,
            "sampler_type": self.sampler_type,
            "interpol_method": self.interpol_method,
            "has_temporal_sampler": self.has_temporal_sampler,
            "has_temporal_reconstruction": self.has_temporal_reconstruction,
            "alpha": self.alpha,
            "adaptive_fraction": self.adaptive_fraction,
            "min_density_gamma": self.min_density_gamma,
            "temporal_method": self.temporal_method,
            "temporal_residual_cutoff": self.temporal_residual_cutoff,
            "temporal_residual_confidence_scale": self.temporal_residual_confidence_scale,
            "sample_sequence": self.sample_sequence,
            "status": self.status.value,
            "result_path": self.result_path,
            "error_message": self.error_message,
            "experiment_id": self.experiment_id,
        }
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentRun":
        """Create ExperimentRun from dictionary (e.g., from JSON)."""
        data = data.copy()
        # Convert status string to enum
        if "status" in data and isinstance(data["status"], str):
            data["status"] = ExperimentStatus(data["status"])
        return cls(**data)
    
    def to_tuple(self) -> Tuple:
        """Convert to tuple format for backward compatibility with run_sampler."""
        return (
            self.gt_name,
            self.scanned_pixel_percent,
            self.sampler_type,
            self.interpol_method,
            self.has_temporal_sampler,
            self.has_temporal_reconstruction,
            self.alpha,
            self.adaptive_fraction,
            self.min_density_gamma,
            self.temporal_method,
            self.temporal_residual_cutoff,
            self.temporal_residual_confidence_scale,
            self.sample_sequence,
        )
    
    def mark_started(self):
        """Mark experiment as running."""
        self.status = ExperimentStatus.RUNNING
    
    def mark_finished(self, result_path: Optional[str] = None):
        """Mark experiment as finished."""
        self.status = ExperimentStatus.FINISHED
        self.result_path = result_path
    
    def mark_error(self, error_message: str):
        """Mark experiment as errored."""
        self.status = ExperimentStatus.ERROR
        self.error_message = error_message
    
    def is_complete(self) -> bool:
        """Check if experiment is complete (finished or error)."""
        return self.status in (ExperimentStatus.FINISHED, ExperimentStatus.ERROR)
    
    def is_ready_to_run(self) -> bool:
        """Check if experiment is ready to run (not started or error)."""
        return self.status in (ExperimentStatus.NOT_STARTED, ExperimentStatus.ERROR)
    
    @classmethod
    def from_task_tuple(cls, task: Tuple) -> "ExperimentRun":
        """Create ExperimentRun from task tuple (backward compatibility).
        
        The tuple order matches run_sampler's signature:
        (gt_name, scanned_pixel_percent, sampler_type, interpol_method,
         has_temporal_sampler, has_temporal_reconstruction, alpha,
         adaptive_fraction, min_density_gamma, temporal_method,
         temporal_residual_cutoff, temporal_residual_confidence_scale,
         sample_sequence)
        """
        if len(task) != 13:
            raise ValueError(f"Expected 13-element tuple, got {len(task)}")
        
        return cls(
            gt_name=task[0],
            scanned_pixel_percent=task[1],
            sampler_type=task[2],
            interpol_method=task[3],
            has_temporal_sampler=task[4],
            has_temporal_reconstruction=task[5],
            alpha=task[6],
            adaptive_fraction=task[7],
            min_density_gamma=task[8],
            temporal_method=task[9],
            temporal_residual_cutoff=task[10],
            temporal_residual_confidence_scale=task[11],
            sample_sequence=task[12],
        )


class ExperimentRunManager:
    """Manages a collection of experiment runs with JSON persistence.
    
    Provides three modes for JSON usage:
    - NO_JSON: No JSON persistence (original behavior)
    - USE_ONLY: Use only JSON file, skip assembly
    - USE_AND_UPDATE: Merge assembly with JSON, filter finished, add new
    
    Thread-safe for parallel execution with ProcessPoolExecutor.
    """
    
    NO_JSON = "no_json"
    USE_ONLY = "use_only"
    USE_AND_UPDATE = "use_and_update"
    
    def __init__(self, json_path: Optional[str] = None, mode: str = NO_JSON):
        """Initialize the experiment run manager.
        
        Args:
            json_path: Path to JSON file for persistence
            mode: One of NO_JSON, USE_ONLY, USE_AND_UPDATE
        """
        self.json_path = json_path
        self.mode = mode
        self.experiments: Dict[str, ExperimentRun] = {}
        self._lock = threading.Lock()
        self._dirty = False  # Track if JSON needs to be written
    
    def _load_from_json(self) -> Dict[str, ExperimentRun]:
        """Load experiments from JSON file."""
        if not self.json_path or not os.path.exists(self.json_path):
            return {}
        
        try:
            with open(self.json_path, 'r') as f:
                data = json.load(f)
            
            experiments = {}
            for exp_id, exp_data in data.get("experiments", {}).items():
                try:
                    experiment = ExperimentRun.from_dict(exp_data)
                    experiments[exp_id] = experiment
                except Exception as e:
                    print(f"Warning: Could not load experiment {exp_id}: {e}")
            
            return experiments
        except Exception as e:
            print(f"Error loading JSON: {e}")
            return {}
    
    def _save_to_json(self):
        """Save experiments to JSON file."""
        if not self.json_path:
            print("[JSON DEBUG] No json_path specified, skipping save")
            return
        
        try:
            os.makedirs(os.path.dirname(self.json_path) or ".", exist_ok=True)
            
            # Count statuses for debug output
            started_count = sum(1 for e in self.experiments.values() if e.status == ExperimentStatus.RUNNING)
            finished_count = sum(1 for e in self.experiments.values() if e.status == ExperimentStatus.FINISHED)
            error_count = sum(1 for e in self.experiments.values() if e.status == ExperimentStatus.ERROR)
            not_started_count = sum(1 for e in self.experiments.values() if e.status == ExperimentStatus.NOT_STARTED)
            
            print(f"[JSON DEBUG] Saving {len(self.experiments)} experiments to {self.json_path}")
            print(f"[JSON DEBUG] Status counts: NOT_STARTED={not_started_count}, RUNNING={started_count}, FINISHED={finished_count}, ERROR={error_count}")
            
            data = {
                "experiments": {
                    exp_id: exp.to_dict() 
                    for exp_id, exp in self.experiments.items()
                },
                "metadata": {
                    "total_count": len(self.experiments),
                    "completed_count": sum(1 for e in self.experiments.values() if e.is_complete()),
                    "not_started_count": not_started_count,
                    "error_count": error_count,
                    "running_count": started_count,
                }
            }
            
            # Atomic write using temp file
            temp_path = self.json_path + ".tmp"
            with open(temp_path, 'w') as f:
                json.dump(data, f, indent=2)
            os.replace(temp_path, self.json_path)
            self._dirty = False
            print(f"[JSON DEBUG] Successfully saved to {self.json_path}")
        except Exception as e:
            import traceback
            print(f"[JSON DEBUG] Error saving JSON: {e}")
            print(traceback.format_exc())
    
    def _mark_dirty(self):
        """Mark that JSON needs to be saved."""
        self._dirty = True
    
    def add_experiment(self, experiment: ExperimentRun):
        """Add an experiment to the collection."""
        with self._lock:
            if experiment.experiment_id not in self.experiments:
                self.experiments[experiment.experiment_id] = experiment
                self._mark_dirty()
    
    def add_experiments(self, experiments: List[ExperimentRun]):
        """Add multiple experiments to the collection."""
        with self._lock:
            for experiment in experiments:
                if experiment.experiment_id not in self.experiments:
                    self.experiments[experiment.experiment_id] = experiment
                    self._mark_dirty()
    
    def get_ready_experiments(self) -> List[ExperimentRun]:
        """Get experiments that are ready to run (not started or error)."""
        with self._lock:
            return [e for e in self.experiments.values() if e.is_ready_to_run()]
    
    def get_all_experiments(self) -> List[ExperimentRun]:
        """Get all experiments."""
        with self._lock:
            return list(self.experiments.values())
    
    def initialize(self, assembled_experiments: Optional[List[ExperimentRun]] = None):
        """Initialize based on mode and assembled experiments.
        
        Args:
            assembled_experiments: List of experiments from for-loop assembly
            
        Returns:
            List of ExperimentRun objects to execute
        """
        if self.mode == self.NO_JSON:
            # Mode a: No JSON usage - use only assembled experiments
            if assembled_experiments:
                self.experiments = {e.experiment_id: e for e in assembled_experiments}
                print(f"[JSON DEBUG] NO_JSON: created {len(self.experiments)} experiments, no JSON persistence")
            return assembled_experiments or []
        
        elif self.mode == self.USE_ONLY:
            # Mode b: Use only JSON file, skip assembly
            self.experiments = self._load_from_json()
            ready_experiments = self.get_ready_experiments()
            print(f"[JSON DEBUG] USE_ONLY: loaded {len(self.experiments)} experiments from JSON, {len(ready_experiments)} ready to run")
            return ready_experiments
        
        elif self.mode == self.USE_AND_UPDATE:
            # Mode c: Use and update - merge assembly with JSON
            # First load existing from JSON
            existing_experiments = self._load_from_json()
            print(f"[JSON DEBUG] USE_AND_UPDATE: loaded {len(existing_experiments)} existing experiments from JSON")
            
            if assembled_experiments:
                # Create a dict of existing by experiment_id
                existing_by_id = {e.experiment_id: e for e in existing_experiments.values()}
                
                # Merge: keep existing experiments (with their status), add new ones
                new_count = 0
                for exp in assembled_experiments:
                    if exp.experiment_id not in existing_by_id:
                        # Add new experiment with NOT_STARTED status
                        exp.status = ExperimentStatus.NOT_STARTED
                        existing_by_id[exp.experiment_id] = exp
                        new_count += 1
                
                self.experiments = existing_by_id
                existing_count = len(assembled_experiments) - new_count
                print(f"[JSON DEBUG] USE_AND_UPDATE: {existing_count} existing, {new_count} new, {len(self.experiments)} total")
                self._save_to_json()  # Save the merged set
            
            # Return only experiments that are not finished
            ready_experiments = self.get_ready_experiments()
            print(f"[JSON DEBUG] USE_AND_UPDATE: {len(self.experiments)} total, {len(ready_experiments)} ready to run")
            return ready_experiments
        
        return []
    
    def mark_experiment_started(self, experiment_id: str):
        """Mark an experiment as started. Thread-safe."""
        with self._lock:
            if experiment_id in self.experiments:
                self.experiments[experiment_id].mark_started()
                self._mark_dirty()
                print(f"[JSON DEBUG] Marked {experiment_id[:8]}... as STARTED")
            else:
                print(f"[JSON DEBUG] WARNING: experiment {experiment_id[:8]}... not found in manager")
    
    def mark_experiment_finished(self, experiment_id: str, result_path: Optional[str] = None):
        """Mark an experiment as finished. Thread-safe."""
        with self._lock:
            if experiment_id in self.experiments:
                self.experiments[experiment_id].mark_finished(result_path)
                self._mark_dirty()
                print(f"[JSON DEBUG] Marked {experiment_id[:8]}... as FINISHED")
            else:
                print(f"[JSON DEBUG] WARNING: experiment {experiment_id[:8]}... not found in manager")
    
    def mark_experiment_error(self, experiment_id: str, error_message: str):
        """Mark an experiment as errored. Thread-safe."""
        with self._lock:
            if experiment_id in self.experiments:
                self.experiments[experiment_id].mark_error(error_message)
                self._mark_dirty()
                print(f"[JSON DEBUG] Marked {experiment_id[:8]}... as ERROR: {error_message[:50] if error_message else 'None'}")
            else:
                print(f"[JSON DEBUG] WARNING: experiment {experiment_id[:8]}... not found in manager")
    
    def get_experiment_by_id(self, experiment_id: str) -> Optional[ExperimentRun]:
        """Get experiment by its ID."""
        with self._lock:
            return self.experiments.get(experiment_id)

    def get_experiment_by_task(self, task: Tuple) -> Optional[ExperimentRun]:
        """Find experiment by task tuple (for backward compatibility)."""
        # Create a temporary ExperimentRun from the task tuple
        temp_experiment = ExperimentRun.from_task_tuple(task)
        with self._lock:
            return self.experiments.get(temp_experiment.experiment_id)
    
    def save_if_dirty(self):
        """Save JSON if it has been modified."""
        if self._dirty:
            self._save_to_json()
    
    def finalize(self):
        """Finalize and save any pending changes."""
        self.save_if_dirty()


# Factory methods for creating ExperimentRun objects
def create_experiment_from_parameters(
    gt_name: str,
    scanned_pixel_percent: float,
    sampler_type: str = "adaptive",
    interpol_method: str = "linear",
    has_temporal_sampler: bool = True,
    has_temporal_reconstruction: bool = True,
    alpha: Optional[float] = None,
    adaptive_fraction: float = 0.0,
    min_density_gamma: float = 0.0,
    temporal_method: str = "optical_flow",
    temporal_residual_cutoff: float = 12.0,
    temporal_residual_confidence_scale: float = 100.0,
    sample_sequence: str = "stratified",
) -> ExperimentRun:
    """Factory method to create an ExperimentRun from individual parameters."""
    return ExperimentRun(
        gt_name=gt_name,
        scanned_pixel_percent=scanned_pixel_percent,
        sampler_type=sampler_type,
        interpol_method=interpol_method,
        has_temporal_sampler=has_temporal_sampler,
        has_temporal_reconstruction=has_temporal_reconstruction,
        alpha=alpha,
        adaptive_fraction=adaptive_fraction,
        min_density_gamma=min_density_gamma,
        temporal_method=temporal_method,
        temporal_residual_cutoff=temporal_residual_cutoff,
        temporal_residual_confidence_scale=temporal_residual_confidence_scale,
        sample_sequence=sample_sequence,
    )


def create_experiments_from_parameter_lists(
    gt_names: List[str],
    scanned_pixel_percentages: List[float],
    sampler_types: List[str] = ["adaptive"],
    interpol_methods: List[str] = ["linear"],
    has_temporal_samplers: List[bool] = [True],
    has_temporal_reconstructions: List[bool] = [True],
    alphas: List[Optional[float]] = [1.0],
    adaptive_fractions: List[float] = [0.0],
    min_density_gammas: List[float] = [0.0],
    temporal_methods: List[str] = ["optical_flow"],
    temporal_residual_cutoffs: List[float] = [12.0],
    temporal_residual_confidence_scales: List[float] = [100.0],
    sample_sequences: List[str] = ["stratified"],
) -> List[ExperimentRun]:
    """Factory method to create multiple ExperimentRun objects from parameter lists.
    
    This replaces the nested for-loop approach with a cleaner functional approach.
    """
    experiments = []
    
    for gt_name in gt_names:
        for interpol_method in interpol_methods:
            for use_temporal_sampler in has_temporal_samplers:
                for use_temporal_reconstruction in has_temporal_reconstructions:
                    # temporalMethod only affects behaviour when use_temporal_sampler is True
                    methods = temporal_methods if use_temporal_sampler else temporal_methods[:1]
                    for temporal_method in methods:
                        # temporalResidualCutoff/ConfidenceScale only affect temporal_variance
                        cutoffs = temporal_residual_cutoffs if temporal_method == "temporal_variance" else temporal_residual_cutoffs[:1]
                        scales = temporal_residual_confidence_scales if temporal_method == "temporal_variance" else temporal_residual_confidence_scales[:1]
                        
                        for temporal_residual_cutoff in cutoffs:
                            for temporal_residual_confidence_scale in scales:
                                for scanned_pixel_percent in scanned_pixel_percentages:
                                    for adaptive_fraction in adaptive_fractions:
                                        for min_density_gamma in min_density_gammas:
                                            for sample_sequence in sample_sequences:
                                                if use_temporal_reconstruction:
                                                    for alpha in alphas:
                                                        experiments.append(create_experiment_from_parameters(
                                                            gt_name=gt_name,
                                                            scanned_pixel_percent=scanned_pixel_percent,
                                                            sampler_type="adaptive",
                                                            interpol_method=interpol_method,
                                                            has_temporal_sampler=use_temporal_sampler,
                                                            has_temporal_reconstruction=use_temporal_reconstruction,
                                                            alpha=alpha,
                                                            adaptive_fraction=adaptive_fraction,
                                                            min_density_gamma=min_density_gamma,
                                                            temporal_method=temporal_method,
                                                            temporal_residual_cutoff=temporal_residual_cutoff,
                                                            temporal_residual_confidence_scale=temporal_residual_confidence_scale,
                                                            sample_sequence=sample_sequence,
                                                        ))
                                                else:
                                                    experiments.append(create_experiment_from_parameters(
                                                        gt_name=gt_name,
                                                        scanned_pixel_percent=scanned_pixel_percent,
                                                        sampler_type="adaptive",
                                                        interpol_method=interpol_method,
                                                        has_temporal_sampler=use_temporal_sampler,
                                                        has_temporal_reconstruction=use_temporal_reconstruction,
                                                        alpha=1.0,  # alpha irrelevant when temporal reconstruction disabled
                                                        adaptive_fraction=adaptive_fraction,
                                                        min_density_gamma=min_density_gamma,
                                                        temporal_method=temporal_method,
                                                        temporal_residual_cutoff=temporal_residual_cutoff,
                                                        temporal_residual_confidence_scale=temporal_residual_confidence_scale,
                                                        sample_sequence=sample_sequence,
                                                    ))
    
    return experiments


def get_json_mode_from_env() -> str:
    """Get JSON mode from environment variable."""
    mode = os.environ.get("STADS_JSON_MODE", ExperimentRunManager.NO_JSON)
    if mode not in [ExperimentRunManager.NO_JSON, ExperimentRunManager.USE_ONLY, ExperimentRunManager.USE_AND_UPDATE]:
        print(f"Warning: Invalid JSON mode '{mode}', defaulting to '{ExperimentRunManager.NO_JSON}'")
        return ExperimentRunManager.NO_JSON
    return mode


def get_json_path_from_env(default_path: str = "experiments_state.json") -> str:
    """Get JSON path from environment variable or use default."""
    return os.environ.get("STADS_JSON_PATH", default_path)
