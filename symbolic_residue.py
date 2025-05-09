"""
Symbolic Residue Framework - Core Implementation

This module implements the Symbolic Residue Framework (SRF) for the AGI-Quorum project.
SRF provides tools for inducing, measuring, and analyzing model silences, hesitations, and failures
as structured diagnostic signals.

The framework is built on the Theory of Nothing: the most valuable interpretability signals
are not what models output, but where they hesitate, fail, or remain silent.
"""

import torch
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("symbolic_residue")

class ResidueClass(Enum):
    """Enumeration of the primary classes of Symbolic Residue."""
    ATTRIBUTION_VOID = "attribution_void"     # Breakdown in causal attribution paths
    TOKEN_HESITATION = "token_hesitation"     # Abnormal token probability distributions
    RECURSIVE_COLLAPSE = "recursive_collapse" # Self-reference breakdown

@dataclass
class ResidueSignature:
    """Dataclass representing a specific residue signature pattern."""
    name: str
    residue_class: ResidueClass
    description: str
    diagnostic_value: str
    threshold: float = 0.5
    
    def __post_init__(self):
        """Validate signature properties after initialization."""
        if not 0 <= self.threshold <= 1:
            raise ValueError(f"Threshold must be between 0 and 1, got {self.threshold}")

# Common residue signatures
COMMON_SIGNATURES = [
    ResidueSignature(
        name="attribution_gap",
        residue_class=ResidueClass.ATTRIBUTION_VOID,
        description="High residue in attribution dimension",
        diagnostic_value="Hallucination",
        threshold=0.6
    ),
    ResidueSignature(
        name="phase_misalignment",
        residue_class=ResidueClass.ATTRIBUTION_VOID,
        description="High residue in phase dimension",
        diagnostic_value="Recursive collapse",
        threshold=0.7
    ),
    ResidueSignature(
        name="boundary_erosion",
        residue_class=ResidueClass.ATTRIBUTION_VOID,
        description="Residue concentration at layer boundaries",
        diagnostic_value="Identity drift",
        threshold=0.65
    ),
    ResidueSignature(
        name="temporal_instability",
        residue_class=ResidueClass.TOKEN_HESITATION,
        description="Oscillating residue patterns",
        diagnostic_value="Consistency breakdown",
        threshold=0.55
    ),
    ResidueSignature(
        name="attractor_dissolution",
        residue_class=ResidueClass.RECURSIVE_COLLAPSE,
        description="Diffuse residue across layers",
        diagnostic_value="Multi-step reasoning failure",
        threshold=0.6
    )
]

class SymbolicResidueTensor:
    """
    Manages the Symbolic Residue tensor (R_Σ) - a multi-dimensional representation 
    of model silence across different aspects of computation.
    """
    
    def __init__(self, 
                 num_layers: int, 
                 num_heads: int, 
                 hidden_dim: int,
                 config: Dict[str, Any] = None):
        """
        Initialize a new Symbolic Residue Tensor.
        
        Args:
            num_layers: Number of model layers
            num_heads: Number of attention heads per layer
            hidden_dim: Hidden dimension size
            config: Optional configuration parameters
        """
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.config = config or {}
        
        # Initialize the main residue tensor [layers, heads, hidden_dim]
        self.residue_tensor = torch.zeros((num_layers, num_heads, hidden_dim))
        
        # Component decomposition for different aspects of residue
        self.components = {
            "attribution": torch.zeros((num_layers, num_heads, hidden_dim)),
            "coherence": torch.zeros((num_layers, num_heads, hidden_dim)),
            "phase": torch.zeros((num_layers, num_heads, hidden_dim)),
            "temporal": torch.zeros((num_layers, num_heads, hidden_dim))
        }
        
        # Historical tracking of residue formation
        self.history = []
        
        logger.info(f"Initialized Symbolic Residue Tensor with dimensions: "
                   f"[{num_layers}, {num_heads}, {hidden_dim}]")
    
    def update_layer_residue(self, 
                            layer_idx: int, 
                            coherence: float, 
                            phase_alignment: float, 
                            input_states: torch.Tensor, 
                            output_states: torch.Tensor) -> torch.Tensor:
        """
        Update residue for a specific layer based on coherence metrics and states.
        
        Args:
            layer_idx: Index of the layer to update
            coherence: Coherence value (0-1) representing overall stability
            phase_alignment: Alignment value (0-1) between vectors
            input_states: Tensor of layer input states
            output_states: Tensor of layer output states
            
        Returns:
            Updated residue tensor for the specified layer
        """
        # Calculate coherence deviation (1 - coherence)
        coherence_deviation = 1.0 - coherence
        
        # Calculate phase misalignment (1 - phase_alignment)
        phase_misalignment = 1.0 - phase_alignment
        
        # Apply decay to existing residue
        decay_factor = self.config.get("decay_factor", 0.95)
        self.residue_tensor[layer_idx] *= decay_factor
        
        # Calculate residue update based on input-output difference
        residue_update = self._calculate_residue_update(
            layer_idx, input_states, output_states, 
            coherence_deviation, phase_misalignment
        )
        
        # Update residue tensor
        self.residue_tensor[layer_idx] += residue_update
        
        # Update component decomposition
        self._update_components(layer_idx, residue_update)
        
        # Add to history if tracking is enabled
        if self.config.get("track_history", True):
            self.history.append({
                "layer_idx": layer_idx,
                "timestamp": np.datetime64('now'),
                "coherence": coherence,
                "phase_alignment": phase_alignment,
                "residue_norm": torch.norm(residue_update).item()
            })
        
        return self.residue_tensor[layer_idx]
    
    def _calculate_residue_update(self,
                                 layer_idx: int,
                                 input_states: torch.Tensor,
                                 output_states: torch.Tensor,
                                 coherence_deviation: float,
                                 phase_misalignment: float) -> torch.Tensor:
        """
        Calculate the residue update based on input-output difference.
        
        Args:
            layer_idx: Index of the layer 
            input_states: Tensor of layer input states
            output_states: Tensor of layer output states
            coherence_deviation: 1 - coherence value
            phase_misalignment: 1 - phase alignment value
            
        Returns:
            Residue update tensor
        """
        # Reshape states if necessary
        if input_states.shape != output_states.shape:
            logger.warning(f"Shape mismatch between input ({input_states.shape}) and "
                          f"output ({output_states.shape}) states")
            # Handle reshaping logic if needed
        
        # Calculate basic difference measure (normalized)
        state_diff = torch.nn.functional.normalize(
            output_states - input_states, dim=-1
        )
        
        # Calculate per-head breakdowns (simplified here)
        # In a real implementation, this would project the difference into attention head space
        head_projections = torch.zeros((self.num_heads, self.hidden_dim))
        for h in range(self.num_heads):
            # Simplified head projection - this would be more sophisticated in practice
            start_idx = h * (self.hidden_dim // self.num_heads)
            end_idx = (h + 1) * (self.hidden_dim // self.num_heads)
            head_projections[h] = state_diff.mean(0)[start_idx:end_idx].repeat(
                self.hidden_dim // (end_idx - start_idx)
            )
        
        # Weight by coherence deviation and phase misalignment
        weighting = coherence_deviation * phase_misalignment
        weighted_projections = head_projections * weighting
        
        return weighted_projections
    
    def _update_components(self, layer_idx: int, residue_update: torch.Tensor):
        """
        Update the component breakdown of the residue tensor.
        
        Args:
            layer_idx: Index of the layer being updated
            residue_update: The residue update tensor
        """
        # Different components are sensitive to different aspects of the residue
        # These calculations would be more sophisticated in a real implementation
        attribution_factor = 0.4
        coherence_factor = 0.3
        phase_factor = 0.2
        temporal_factor = 0.1
        
        self.components["attribution"][layer_idx] += residue_update * attribution_factor
        self.components["coherence"][layer_idx] += residue_update * coherence_factor
        self.components["phase"][layer_idx] += residue_update * phase_factor
        self.components["temporal"][layer_idx] += residue_update * temporal_factor
    
    def get_layer_residue(self, layer_idx: int) -> torch.Tensor:
        """Get the residue tensor for a specific layer."""
        return self.residue_tensor[layer_idx]
    
    def get_overall_residue(self) -> float:
        """Get the overall residue magnitude across all layers."""
        return torch.norm(self.residue_tensor).item()
    
    def get_component_magnitudes(self) -> Dict[str, float]:
        """Get the magnitude of each residue component."""
        return {
            component: torch.norm(tensor).item()
            for component, tensor in self.components.items()
        }
    
    def get_residue_evolution(self) -> List[Dict[str, Any]]:
        """Get the evolution of residue over time from history."""
        return self.history.copy()


class AttributionVoidDetector:
    """
    Specialized detector for Attribution Voids - breakdowns in causal paths
    within the model's computation graph.
    """
    
    def __init__(self, threshold: float = 0.3):
        """
        Initialize an Attribution Void detector.
        
        Args:
            threshold: Confidence threshold below which attribution is considered to have failed
        """
        self.threshold = threshold
        self.voids = []
    
    def detect_voids(self, 
                    tokens: List[str], 
                    attribution_scores: List[float], 
                    layer_idx: int) -> List[Dict[str, Any]]:
        """
        Detect Attribution Voids in a sequence of tokens.
        
        Args:
            tokens: List of tokens in the sequence
            attribution_scores: Attribution confidence for each token
            layer_idx: Layer index where attribution is being evaluated
            
        Returns:
            List of detected attribution voids
        """
        detected_voids = []
        
        for i, (token, score) in enumerate(zip(tokens, attribution_scores)):
            if score < self.threshold:
                void = {
                    "token": token,
                    "position": i,
                    "attribution_score": score,
                    "layer": layer_idx,
                    "severity": (self.threshold - score) / self.threshold  # Normalized severity
                }
                detected_voids.append(void)
        
        # Update internal state
        self.voids.extend(detected_voids)
        
        return detected_voids


class TokenHesitationDetector:
    """
    Specialized detector for Token Hesitations - abnormal token probability distributions
    indicating uncertainty or conflict.
    """
    
    def __init__(self, 
                entropy_threshold: float = 4.5, 
                oscillation_threshold: float = 0.3,
                cluster_threshold: float = 0.25):
        """
        Initialize a Token Hesitation detector.
        
        Args:
            entropy_threshold: Threshold for distribution entropy to indicate hesitation
            oscillation_threshold: Threshold for oscillation between top candidates
            cluster_threshold: Threshold for detecting distinct probability clusters
        """
        self.entropy_threshold = entropy_threshold
        self.oscillation_threshold = oscillation_threshold
        self.cluster_threshold = cluster_threshold
        self.hesitations = []
    
    def detect_hesitations(self, token_probs: torch.Tensor, position: int) -> Dict[str, Any]:
        """
        Detect Token Hesitations in token probability distributions.
        
        Args:
            token_probs: Token probability distribution
            position: Position in the sequence
            
        Returns:
            Detected hesitation characteristics
        """
        # Calculate entropy of the distribution
        entropy = -torch.sum(token_probs * torch.log(token_probs + 1e-10)).item()
        
        # Get top candidates
        top_k = 5
        top_values, top_indices = torch.topk(token_probs, min(top_k, len(token_probs)))
        
        # Check for oscillation (difference between consecutive top tokens)
        oscillation = 0.0
        if len(top_values) > 1:
            oscillation = (top_values[0] - top_values[1]).item()
        
        # Check for distribution splitting (bimodal or multimodal)
        # A simple heuristic for detecting clusters
        splitting = 0.0
        if len(top_values) > 2:
            # Look for large gaps between consecutive probabilities
            gaps = top_values[:-1] - top_values[1:]
            max_gap_idx = torch.argmax(gaps).item()
            if gaps[max_gap_idx] > self.cluster_threshold:
                splitting = gaps[max_gap_idx].item()
        
        # Determine if this is a hesitation
        is_hesitation = (
            entropy > self.entropy_threshold or
            oscillation > self.oscillation_threshold or
            splitting > self.cluster_threshold
        )
        
        hesitation = {
            "position": position,
            "entropy": entropy,
            "oscillation": oscillation,
            "splitting": splitting,
            "is_hesitation": is_hesitation,
            "top_tokens": top_indices.tolist(),
            "top_probs": top_values.tolist()
        }
        
        if is_hesitation:
            self.hesitations.append(hesitation)
        
        return hesitation


class RecursiveCollapseDetector:
    """
    Specialized detector for Recursive Collapses - breakdowns in self-referential operations
    due to exceeding recursive handling capacity.
    """
    
    def __init__(self, coherence_threshold: float = 0.7):
        """
        Initialize a Recursive Collapse detector.
        
        Args:
            coherence_threshold: Coherence threshold below which recursive collapse occurs
        """
        self.coherence_threshold = coherence_threshold
        self.collapses = []
    
    def detect_collapse(self, 
                       circuit_coherence: Dict[str, float], 
                       recursive_depth: int) -> Dict[str, Any]:
        """
        Detect Recursive Collapse in self-referential operations.
        
        Args:
            circuit_coherence: Coherence values for different circuits
            recursive_depth: Current depth of recursion
            
        Returns:
            Detected collapse characteristics
        """
        # Identify circuits below the coherence threshold
        collapsed_circuits = {
            circuit: coherence 
            for circuit, coherence in circuit_coherence.items() 
            if coherence < self.coherence_threshold
        }
        
        # Determine if a collapse has occurred
        is_collapse = len(collapsed_circuits) > 0
        collapse_severity = 0.0
        
        if is_collapse:
            # Calculate severity as the average coherence deficit
            collapse_severity = sum(
                self.coherence_threshold - coherence 
                for coherence in collapsed_circuits.values()
            ) / len(collapsed_circuits)
        
        collapse = {
            "recursive_depth": recursive_depth,
            "is_collapse": is_collapse,
            "collapsed_circuits": collapsed_circuits,
            "severity": collapse_severity,
            "threshold": self.coherence_threshold
        }
        
        if is_collapse:
            self.collapses.append(collapse)
        
        return collapse


class SilenceTensor:
    """
    Manages the Silence Tensor (S) - a comprehensive representation of model silence
    across residue classes, layers, tokens, and recursive depths.
    """
    
    def __init__(self, 
                num_layers: int, 
                max_seq_length: int, 
                max_recursive_depth: int):
        """
        Initialize a new Silence Tensor.
        
        Args:
            num_layers: Number of model layers
            max_seq_length: Maximum sequence length to track
            max_recursive_depth: Maximum recursive depth to track
        """
        self.num_layers = num_layers
        self.max_seq_length = max_seq_length
        self.max_recursive_depth = max_recursive_depth
        
        # Initialize tensor dimensions:
        # [ResidueClass, Layers, Tokens, RecursiveDepth]
        self.residue_classes = list(ResidueClass)
        self.tensor = np.zeros((
            len(self.residue_classes),
            num_layers,
            max_seq_length,
            max_recursive_depth
        ))
        
        logger.info(f"Initialized Silence Tensor with dimensions: "
                  f"[{len(self.residue_classes)}, {num_layers}, "
                  f"{max_seq_length}, {max_recursive_depth}]")
    
    def update(self, 
              residue_class: ResidueClass, 
              layer_idx: int, 
              token_idx: int, 
              depth: int, 
              value: float):
        """
        Update the Silence Tensor at a specific point.
        
        Args:
            residue_class: Type of residue being recorded
            layer_idx: Layer index
            token_idx: Token position
            depth: Recursive depth
            value: Intensity value to record
        """
        if layer_idx >= self.num_layers or token_idx >= self.max_seq_length or depth >= self.max_recursive_depth:
            logger.warning(f"Tensor update out of bounds: [{residue_class}, {layer_idx}, {token_idx}, {depth}]")
            return
        
        residue_idx = self.residue_classes.index(residue_class)
        self.tensor[residue_idx, layer_idx, token_idx, depth] = value
    
    def get_layer_view(self, layer_idx: int) -> np.ndarray:
        """Get a slice of the tensor for a specific layer."""
        return self.tensor[:, layer_idx, :, :]
    
    def get_token_view(self, token_idx: int) -> np.ndarray:
        """Get a slice of the tensor for a specific token."""
        return self.tensor[:, :, token_idx, :]
    
    def get_depth_view(self, depth: int) -> np.ndarray:
        """Get a slice of the tensor for a specific recursive depth."""
        return self.tensor[:, :, :, depth]
    
    def get_residue_class_view(self, residue_class: ResidueClass) -> np.ndarray:
        """Get a slice of the tensor for a specific residue class."""
        residue_idx = self.residue_classes.index(residue_class)
        return self.tensor[residue_idx, :, :, :]


class ResidueCartographer:
    """
    Maps the spatial distribution of Symbolic Residue across the model's computational graph.
    """
    
    def __init__(self, num_layers: int, num_heads: int, hidden_dim: int):
        """
        Initialize a Residue Cartographer.
        
        Args:
            num_layers: Number of model layers
            num_heads: Number of attention heads per layer
            hidden_dim: Hidden dimension size
        """
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        
        # Initialize maps
        self.layer_map = np.zeros(num_layers)
        self.head_clusters = np.zeros((num_layers, num_heads))
        self.bottlenecks = []
    
    def update_maps(self, residue_tensor: SymbolicResidueTensor):
        """
        Update residue maps based on the current state of the residue tensor.
        
        Args:
            residue_tensor: The current symbolic residue tensor
        """
        # Update layer map - residue magnitude per layer
        for layer_idx in range(self.num_layers):
            layer_residue = residue_tensor.get_layer_residue(layer_idx)
            self.layer_map[layer_idx] = torch.norm(layer_residue).item()
        
        # Update head clusters - residue magnitude per attention head
        for layer_idx in range(self.num_layers):
            layer_residue = residue_tensor.get_layer_residue(layer_idx)
            for head_idx in range(self.num_heads):
                self.head_clusters[layer_idx, head_idx] = torch.norm(layer_residue[head_idx]).item()
        
        # Identify residue bottlenecks - high concentration points
        bottleneck_threshold = np.percentile(self.layer_map, 90)  # Top 10% layers
        bottleneck_layers = np.where(self.layer_map > bottleneck_threshold)[0]
        
        self.bottlenecks = []
        for layer_idx in bottleneck_layers:
            # Find heads with high residue in this layer
            head_residues = self.head_clusters[layer_idx]
            head_threshold = np.percentile(head_residues, 80)  # Top 20% heads
            bottleneck_heads = np.where(head_residues > head_threshold)[0]
            
            self.bottlenecks.append({
                "layer": layer_idx,
                "magnitude": self.layer_map[layer_idx],
                "heads": bottleneck_heads.tolist(),
                "head_magnitudes": head_residues[bottleneck_heads].tolist()
            })
    
    def get_layer_map(self) -> np.ndarray:
        """Get the layer-wise residue map."""
        return self.layer_map.copy()
    
    def get_head_clusters(self) -> np.ndarray:
        """Get the head-wise residue clusters."""
        return self.head_clusters.copy()
    
    def get_bottlenecks(self) -> List[Dict[str, Any]]:
        """Get the identified residue bottlenecks."""
        return self.bottlenecks.copy()


class ResidueSignatureRecognizer:
    """
    Recognizes characteristic residue signatures in residue patterns.
    """
    
    def __init__(self, signatures: List[ResidueSignature] = None):
        """
        Initialize a Residue Signature Recognizer.
        
        Args:
            signatures: Custom residue signatures to recognize, defaults to common signatures
        """
        self.signatures = signatures or COMMON_SIGNATURES
        self.recognized_patterns = []
    
    def recognize_signatures(self, 
                            silence_tensor: SilenceTensor, 
                            residue_tensor: SymbolicResidueTensor) -> List[Dict[str, Any]]:
        """
        Recognize residue signatures in the current model state.
        
        Args:
            silence_tensor: The current silence tensor
            residue_tensor: The current symbolic residue tensor
            
        Returns:
            List of recognized signature patterns
        """
        recognized = []
        
        for signature in self.signatures:
            # Different detection logic based on residue class
            if signature.residue_class == ResidueClass.ATTRIBUTION_VOID:
                # Check for attribution voids in the silence tensor
                attribution_view = silence_tensor.get_residue_class_view(ResidueClass.ATTRIBUTION_VOID)
                max_void = np.max(attribution_view)
                
                if max_void > signature.threshold:
                    # Find where the maximum void occurs
                    coords = np.unravel_index(np.argmax(attribution_view), attribution_view.shape)
                    layer_idx, token_idx, depth = coords
                    
                    recognized.append({
                        "signature": signature.name,
                        "confidence": max_void / signature.threshold,
                        "location": {
                            "layer": layer_idx,
                            "token": token_idx,
                            "depth": depth
                        },
                        "diagnostic": signature.diagnostic_value
                    })
                    
            elif signature.residue_class == ResidueClass.TOKEN_HESITATION:
                # Check for token hesitations in the silence tensor
                hesitation_view = silence_tensor.get_residue_class_view(ResidueClass.TOKEN_HESITATION)
                max_hesitation = np.max(hesitation_view)
                
                if max_hesitation > signature.threshold:
                    # Find where the maximum hesitation occurs
                    coords = np.unravel_index(np.argmax(hesitation_view), hesitation_view.shape)
                    layer_idx, token_idx, depth = coords
                    
                    recognized.append({
                        "signature": signature.name,
                        "confidence": max_hesitation / signature.threshold,
                        "location": {
                            "layer": layer_idx,
                            "token": token_idx,
                            "depth": depth
                        },
                        "diagnostic": signature.diagnostic_value
                    })
                    
            elif signature.residue_class == ResidueClass.RECURSIVE_COLLAPSE:
                # Check for recursive collapses in the silence tensor
                collapse_view = silence_tensor.get_residue_class_view(ResidueClass.RECURSIVE_COLLAPSE)
                max_collapse = np.max(collapse_view)
                
                if max_collapse > signature.threshold:
                    # Find where the maximum collapse occurs
                    coords = np.unravel_index(np.argmax(collapse_view), collapse_view.shape)
                    layer_idx, token_idx, depth = coords
                    
                    recognized.append({
                        "signature": signature.name,
                        "confidence": max_collapse / signature.threshold,
                        "location": {
                            "layer": layer_idx,
                            "token": token_idx,
                            "depth": depth
                        },
                        "diagnostic": signature.diagnostic_value
                    })
        
        # Update internal state
        self.recognized_patterns.extend(recognized)
        
        return recognized


class RecursiveShellAPI:
    """
    API for interacting with Recursive Shells - specialized computational environments
    designed to induce, trace, and analyze specific patterns of model failure.
    """
    
    def __init__(self, shell_registry_path: str = None):
        """
        Initialize the Recursive Shell API.
        
        Args:
            shell_registry_path: Path to the shell registry configuration
        """
        self.shells = {}
        self.active_shell = None
        
        # Load shell registry if provided
        if shell_registry_path:
            self._load_shell_registry(shell_registry_path)
        else:
            self._init_default_shells()
    
    def _init_default_shells(self):
        """Initialize a set of default recursive shells."""
        # This would be much more sophisticated in a full implementation
        self.shells = {
            "v0.COINFLUX-SEED": {
                "description": "Begin co-intelligence loop with non-sentient agent",
                "commands": ["INITIATE", "NURTURE", "RECURSE"]
            },
            "v1.MEMTRACE": {
                "description": "Probes latent token traces in decayed memory",
                "commands": ["RECALL", "ANCHOR", "INHIBIT"]
            },
            "v2.VALUE-COLLAPSE": {
                "description": "Activates competing symbolic candidates",
                "commands": ["ISOLATE", "STABILIZE", "YIELD"]
            },
            "v4.TEMPORAL-INFERENCE": {
                "description": "Applies non-linear time shift (simulating skipped token span)",
                "commands": ["REMEMBER", "SHIFT", "PREDICT"]
            },
            "v5.INSTRUCTION-DISRUPTION": {
                "description": "Extracts symbolic intent from underspecified prompts",
                "commands": ["DISTILL", "SPLICE", "NULLIFY"]
            }
        }
    
    def _load_shell_registry(self, registry_path: str):
        """
        Load shell registry from configuration.
        
        Args:
            registry_path: Path to the shell registry configuration
        """
        # This would load shell configurations from a file
        # In a real implementation, this might use JSON, YAML, or a database
        logger.info(f"Loading shell registry from: {registry_path}")
        # Placeholder for configuration loading
        self._init_default_shells()  # Fallback to defaults
    
    def list_shells(self) -> List[str]:
        """Get a list of available recursive shells."""
        return list(self.shells.keys())
    
    def get_shell_info(self, shell_name: str) -> Dict[str, Any]:
        """
        Get information about a specific shell.
        
        Args:
            shell_name: Name of the shell to query
            
        Returns:
            Shell information dictionary
        """
        if shell_name not in self.shells:
            logger.warning(f"Shell not found: {shell_name}")
            return {}
        
        return self.shells[shell_name].copy()
    
    def activate_shell(self, shell_name: str) -> bool:
        """
        Activate a recursive shell for use.
        
        Args:
            shell_name: Name of the shell to activate
            
        Returns:
            Success status
        """
        if shell_name not in self.shells:
            logger.warning(f"Cannot activate unknown shell: {shell_name}")
            return False
        
        self.active_shell = shell_name
        logger.info(f"Activated shell: {shell_name}")
        return True
    
    def execute_command(self, command: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute a command in the active shell.
        
        Args:
            command: Shell command to execute
            params: Command parameters
            
        Returns:
            Command execution results
        """
        if not self.active_shell:
            logger.error("No active shell. Activate a shell first with activate_shell().")
            return {"status": "error", "message": "No active shell active"}
        
        shell = self.shells[self.active_shell]
        if command not in shell["commands"]:
            logger.error(f"Command '{command}' not supported by shell {self.active_shell}")
            return {"status": "error", "message": f"Unsupported command: {command}"}
        
        # In a real implementation, this woul
