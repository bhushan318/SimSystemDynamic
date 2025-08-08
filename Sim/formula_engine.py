# -*- coding: utf-8 -*-
"""
Advanced Formula Evaluation Engine for System Dynamics
Implements secure formula evaluation with system dynamics functions and caching
"""

import ast
import operator
import numpy as np
import hashlib
import time
from typing import Dict, Any, Set, Callable, List, Union, Optional, Tuple
from functools import lru_cache
from dataclasses import dataclass
from enum import Enum
import warnings
import json

# ===============================================================================
# Core Data Structures
# ===============================================================================

@dataclass
class ModelContext:
    """Comprehensive context for formula evaluation"""
    current_time: float = 0.0
    dt: float = 1.0
    start_time: float = 0.0
    end_time: float = 100.0
    step: int = 0
    
    # Stock values (name -> current values)
    stocks: Dict[str, Any] = None
    
    # Parameters (name -> value)
    parameters: Dict[str, Any] = None
    
    # Auxiliary variables
    auxiliaries: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.stocks is None:
            self.stocks = {}
        if self.parameters is None:
            self.parameters = {}
        if self.auxiliaries is None:
            self.auxiliaries = {}
    
    def get_variable(self, name: str) -> Any:
        """Get variable value by name with fallback chain"""
        # Check stocks first
        if name in self.stocks:
            return self.stocks[name]
        
        # Check parameters
        if name in self.parameters:
            return self.parameters[name]
        
        # Check auxiliaries
        if name in self.auxiliaries:
            return self.auxiliaries[name]
        
        # Check time variables
        time_vars = {
            'TIME': self.current_time,
            'STARTTIME': self.start_time,
            'STOPTIME': self.end_time,
            'DT': self.dt
        }
        
        if name.upper() in time_vars:
            return time_vars[name.upper()]
        
        raise NameError(f"Variable '{name}' not found in context")

class FormulaType(Enum):
    """Types of formulas supported"""
    RATE = "rate"           # Flow rate formula
    AUXILIARY = "auxiliary" # Auxiliary calculation
    INITIAL = "initial"     # Initial value
    CONSTANT = "constant"   # Parameter value
    LOOKUP = "lookup"       # Lookup table
    CONSTRAINT = "constraint" # Constraint formula

# ===============================================================================
# Exception Classes
# ===============================================================================

class FormulaEvaluationError(Exception):
    """Formula evaluation failed"""
    def __init__(self, formula: str, error: str, context: str = ""):
        self.formula = formula
        self.error = error
        self.context = context
        super().__init__(f"Formula evaluation failed: '{formula}' - {error} {context}")

class SecurityError(Exception):
    """Security violation in formula"""
    pass

class CompilationError(Exception):
    """Formula compilation failed"""
    pass

# ===============================================================================
# Compiled Formula Container
# ===============================================================================

@dataclass
class CompiledFormula:
    """Container for compiled formula with metadata"""
    formula: str
    compiled_code: Any
    dependencies: Set[str]
    hash: str
    formula_type: FormulaType = FormulaType.RATE
    last_used: float = 0.0
    use_count: int = 0
    compile_time: float = 0.0
    
    def __post_init__(self):
        self.last_used = time.time()

# ===============================================================================
# System Dynamics Functions
# ===============================================================================

class SystemDynamicsFunctions:
    """Collection of system dynamics specific functions"""
    
    @staticmethod
    def step_function(height: Union[float, np.ndarray], 
                     step_time: float, 
                     context: ModelContext) -> Union[float, np.ndarray]:
        """STEP function: returns height if time >= step_time, else 0"""
        if context.current_time >= step_time:
            return height
        else:
            if isinstance(height, np.ndarray):
                return np.zeros_like(height)
            else:
                return 0.0
    
    @staticmethod
    def pulse_function(height: Union[float, np.ndarray], 
                      start_time: float, 
                      duration: float,
                      context: ModelContext) -> Union[float, np.ndarray]:
        """PULSE function: returns height during pulse period"""
        t = context.current_time
        if start_time <= t <= start_time + duration:
            return height
        else:
            if isinstance(height, np.ndarray):
                return np.zeros_like(height)
            else:
                return 0.0
    
    @staticmethod
    def ramp_function(slope: float, 
                     start_time: float, 
                     end_time: float,
                     context: ModelContext) -> float:
        """RAMP function: linear ramp between start and end time"""
        t = context.current_time
        if t < start_time:
            return 0.0
        elif t > end_time:
            return slope * (end_time - start_time)
        else:
            return slope * (t - start_time)
    
    @staticmethod
    def lookup_function(input_value: Union[float, np.ndarray], 
                       lookup_table: List[List[float]]) -> Union[float, np.ndarray]:
        """LOOKUP function: linear interpolation lookup"""
        if not lookup_table:
            raise ValueError("Lookup table cannot be empty")
        
        # Extract x and y values
        x_values = [row[0] for row in lookup_table]
        y_values = [row[1] for row in lookup_table]
        
        # Ensure sorted order
        sorted_pairs = sorted(zip(x_values, y_values))
        x_values, y_values = zip(*sorted_pairs)
        
        return np.interp(input_value, x_values, y_values)
    
    @staticmethod
    def if_then_else(condition: Union[bool, np.ndarray], 
                    then_value: Any, 
                    else_value: Any) -> Any:
        """IF_THEN_ELSE function"""
        if isinstance(condition, np.ndarray):
            return np.where(condition, then_value, else_value)
        else:
            return then_value if condition else else_value
    @staticmethod
    def min_function(*args) -> Union[float, np.ndarray]:
        """MIN function for multiple arguments"""
        if len(args) == 1 and hasattr(args[0], '__iter__') and not isinstance(args[0], str):
            return np.min(args[0])
        return np.minimum.reduce(args)
    
    @staticmethod
    def max_function(*args) -> Union[float, np.ndarray]:
        """MAX function for multiple arguments"""
        if len(args) == 1 and hasattr(args[0], '__iter__') and not isinstance(args[0], str):
            return np.max(args[0])
        return np.maximum.reduce(args)
    
    @staticmethod
    def delay_function(input_value: Union[float, np.ndarray],
                      delay_time: float,
                      initial_value: float = 0.0,
                      context: ModelContext = None) -> Union[float, np.ndarray]:
        """DELAY function: simple first-order delay (simplified implementation)"""
        # In a full implementation, this would maintain delay history
        # For now, return a simple approximation
        if context and hasattr(context, 'delay_values'):
            # Would implement proper delay logic here
            pass
        
        # Simplified: return input_value (no delay)
        return input_value

# ===============================================================================
# Formula Security Validator
# ===============================================================================

class FormulaSecurityValidator:
    """Validates formula security using AST analysis"""
    
    def __init__(self):
        # Allowed names in formulas
        self.allowed_names = {
            # Mathematical operations
            'abs', 'min', 'max', 'sum', 'len',
            # NumPy functions  
            'np', 'sin', 'cos', 'tan', 'exp', 'log', 'sqrt', 'log10',
            'mean', 'std', 'median', 'percentile',
            # System dynamics functions
            'STEP', 'PULSE', 'RAMP', 'LOOKUP', 'IF_THEN_ELSE',
            'MIN', 'MAX', 'DELAY', 'SMOOTH', 'TREND',
            # Time functions
            'TIME', 'STARTTIME', 'STOPTIME', 'DT',
            # Mathematical constants
            'pi', 'e', 'inf', 'nan',
            # Logical operators
            'and', 'or', 'not'
        }
        
        # Allowed numpy attributes
        self.allowed_numpy_attrs = {
            'sin', 'cos', 'tan', 'exp', 'log', 'log10', 'sqrt',
            'abs', 'min', 'max', 'mean', 'std', 'median',
            'sum', 'prod', 'any', 'all', 'where',
            'pi', 'e', 'inf', 'nan', 'isnan', 'isfinite',
            'clip', 'maximum', 'minimum'
        }
        
        # Allowed operators
        self.allowed_operators = {
            ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
            ast.LShift, ast.RShift, ast.BitOr, ast.BitXor, ast.BitAnd,
            ast.USub, ast.UAdd, ast.Not, ast.Invert,
            ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
            ast.Is, ast.IsNot, ast.In, ast.NotIn,
            ast.And, ast.Or
        }
    
    def validate_formula(self, formula: str, allowed_variables: Set[str] = None) -> Dict[str, Any]:
        """Validate formula for security and correctness"""
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'dependencies': set(),
            'complexity_score': 0
        }
        
        try:
            # Parse formula into AST
            tree = ast.parse(formula, mode='eval')
            
            # Security validation
            self._validate_ast_security(tree, validation_result, allowed_variables or set())
            
            # Extract dependencies
            validation_result['dependencies'] = self._extract_dependencies(tree)
            
            # Calculate complexity
            validation_result['complexity_score'] = self._calculate_complexity(tree)
            
        except SyntaxError as e:
            validation_result['valid'] = False
            validation_result['errors'].append(f"Syntax error: {str(e)}")
        except Exception as e:
            validation_result['valid'] = False
            validation_result['errors'].append(f"Validation error: {str(e)}")
        
        return validation_result
    
    def _validate_ast_security(self, node: ast.AST, result: Dict, allowed_variables: Set[str]):
        """Recursively validate AST for security"""
        if isinstance(node, ast.Name):
            # Check if variable name is allowed
            if (node.id not in self.allowed_names and 
                node.id not in allowed_variables):
                result['warnings'].append(f"Unknown variable: {node.id}")
            
        elif isinstance(node, ast.Call):
            # Check function calls
            if isinstance(node.func, ast.Name):
                if node.func.id not in self.allowed_names:
                    result['errors'].append(f"Forbidden function: {node.func.id}")
                    result['valid'] = False
            elif isinstance(node.func, ast.Attribute):
                # Check attribute access (e.g., np.sin)
                if (isinstance(node.func.value, ast.Name) and 
                    node.func.value.id == 'np'):
                    if node.func.attr not in self.allowed_numpy_attrs:
                        result['errors'].append(f"Forbidden numpy function: {node.func.attr}")
                        result['valid'] = False
        
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            result['errors'].append("Import statements not allowed")
            result['valid'] = False
            
        elif isinstance(node, ast.Attribute):
            # Allow numpy attributes but validate them
            if isinstance(node.value, ast.Name) and node.value.id == 'np':
                if node.attr not in self.allowed_numpy_attrs:
                    result['errors'].append(f"Forbidden numpy attribute: {node.attr}")
                    result['valid'] = False
        
        elif hasattr(ast, 'Exec') and isinstance(node, ast.Exec):
            result['errors'].append("Exec statements not allowed")
            result['valid'] = False
        
        elif hasattr(ast, 'Eval') and isinstance(node, ast.Eval):
            result['errors'].append("Eval statements not allowed")
            result['valid'] = False
        
        # Recursively validate child nodes
        for child in ast.iter_child_nodes(node):
            self._validate_ast_security(child, result, allowed_variables)
    
    def _extract_dependencies(self, node: ast.AST) -> Set[str]:
        """Extract all variable names referenced in formula"""
        dependencies = set()
        
        for child in ast.walk(node):
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                # Only add if it's not a built-in function
                if child.id not in self.allowed_names:
                    dependencies.add(child.id)
        
        return dependencies
    
    def _calculate_complexity(self, node: ast.AST) -> int:
        """Calculate formula complexity score"""
        complexity = 0
        
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                complexity += 2  # Function calls add complexity
            elif isinstance(child, (ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare)):
                complexity += 1  # Operations add complexity
            elif isinstance(child, ast.IfExp):
                complexity += 3  # Conditional expressions are complex
        
        return complexity

# ===============================================================================
# Formula Compiler with Caching
# ===============================================================================

class FormulaCompiler:
    """Compiles and caches formulas for optimized evaluation"""
    
    def __init__(self, max_cache_size: int = 1000, cache_timeout: float = 3600):
        self.formula_cache: Dict[str, CompiledFormula] = {}
        self.max_cache_size = max_cache_size
        self.cache_timeout = cache_timeout
        self.validator = FormulaSecurityValidator()
        self.compilation_stats = {
            'total_compilations': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'compilation_time': 0.0
        }
    
    @lru_cache(maxsize=1000)
    def compile_formula(self, formula: str, 
                       formula_type: FormulaType = FormulaType.RATE,
                       allowed_variables: Set[str] = None) -> CompiledFormula:
        """Compile formula to optimized representation with caching"""
        
        start_time = time.time()
        
        # Create cache key
        cache_key = self._create_cache_key(formula, formula_type, allowed_variables)
        
        # Check cache first
        if cache_key in self.formula_cache:
            cached_formula = self.formula_cache[cache_key]
            
            # Check if cached formula is still valid (not expired)
            if time.time() - cached_formula.last_used < self.cache_timeout:
                cached_formula.last_used = time.time()
                cached_formula.use_count += 1
                self.compilation_stats['cache_hits'] += 1
                return cached_formula
            else:
                # Remove expired formula
                del self.formula_cache[cache_key]
        
        # Cache miss - compile formula
        self.compilation_stats['cache_misses'] += 1
        self.compilation_stats['total_compilations'] += 1
        
        try:
            # Validate formula
            if allowed_variables is None:
                allowed_variables = set()
            
            validation = self.validator.validate_formula(formula, allowed_variables)
            
            if not validation['valid']:
                raise CompilationError(f"Formula validation failed: {validation['errors']}")
            
            # Parse and compile
            tree = ast.parse(formula, mode='eval')
            compiled_code = compile(tree, '<formula>', 'eval')
            
            # Create compiled formula object
            compiled_formula = CompiledFormula(
                formula=formula,
                compiled_code=compiled_code,
                dependencies=validation['dependencies'],
                hash=hashlib.md5(formula.encode()).hexdigest(),
                formula_type=formula_type,
                compile_time=time.time() - start_time
            )
            
            # Cache the compiled formula
            self._cache_formula(cache_key, compiled_formula)
            
            compilation_time = time.time() - start_time
            self.compilation_stats['compilation_time'] += compilation_time
            
            return compiled_formula
            
        except Exception as e:
            raise CompilationError(f"Formula compilation failed: {str(e)}")
    
    def _create_cache_key(self, formula: str, formula_type: FormulaType, 
                         allowed_variables: Set[str]) -> str:
        """Create cache key from formula and metadata"""
        var_hash = hashlib.md5(str(sorted(allowed_variables or [])).encode()).hexdigest()[:8]
        formula_hash = hashlib.md5(formula.encode()).hexdigest()[:16]
        return f"{formula_type.value}:{var_hash}:{formula_hash}"
    
    def _cache_formula(self, cache_key: str, compiled_formula: CompiledFormula):
        """Cache compiled formula with size management"""
        # Check cache size limit
        if len(self.formula_cache) >= self.max_cache_size:
            # Remove oldest unused formulas
            self._cleanup_cache()
        
        self.formula_cache[cache_key] = compiled_formula
    
    def _cleanup_cache(self, target_size: Optional[int] = None):
        """Clean up cache by removing old/unused formulas"""
        if target_size is None:
            target_size = self.max_cache_size // 2
        
        # Sort by last used time and use count
        cache_items = list(self.formula_cache.items())
        cache_items.sort(key=lambda x: (x[1].last_used, x[1].use_count))
        
        # Remove oldest items
        items_to_remove = len(cache_items) - target_size
        for i in range(max(0, items_to_remove)):
            del self.formula_cache[cache_items[i][0]]
    
    def get_compilation_stats(self) -> Dict[str, Any]:
        """Get compilation and cache statistics"""
        cache_hit_rate = (self.compilation_stats['cache_hits'] / 
                         max(1, self.compilation_stats['cache_hits'] + self.compilation_stats['cache_misses']))
        
        return {
            **self.compilation_stats,
            'cache_size': len(self.formula_cache),
            'cache_hit_rate': cache_hit_rate,
            'avg_compilation_time': (self.compilation_stats['compilation_time'] / 
                                   max(1, self.compilation_stats['total_compilations']))
        }
    
    def clear_cache(self):
        """Clear all cached formulas"""
        self.formula_cache.clear()

# ===============================================================================
# Main Formula Engine
# ===============================================================================

class FormulaEngine:
    """Main formula evaluation engine with security and optimization"""
    
    def __init__(self, max_cache_size: int = 1000, enable_numpy: bool = True):
        self.compiler = FormulaCompiler(max_cache_size)
        self.enable_numpy = enable_numpy
        
        # System dynamics functions
        self.system_functions = {
            'STEP': SystemDynamicsFunctions.step_function,
            'PULSE': SystemDynamicsFunctions.pulse_function,
            'RAMP': SystemDynamicsFunctions.ramp_function,
            'LOOKUP': SystemDynamicsFunctions.lookup_function,
            'IF_THEN_ELSE': SystemDynamicsFunctions.if_then_else,
            'MIN': SystemDynamicsFunctions.min_function,
            'MAX': SystemDynamicsFunctions.max_function,
            'DELAY': SystemDynamicsFunctions.delay_function,
            'TIME': lambda context: context.current_time,
            'STARTTIME': lambda context: context.start_time,
            'STOPTIME': lambda context: context.end_time,
            'DT': lambda context: context.dt
        }
        
        # Evaluation statistics
        self.evaluation_stats = {
            'total_evaluations': 0,
            'successful_evaluations': 0,
            'failed_evaluations': 0,
            'total_evaluation_time': 0.0,
            'errors': []
        }
    
    def evaluate(self, formula: str, context: ModelContext,
                formula_type: FormulaType = FormulaType.RATE) -> Union[float, np.ndarray]:
        """Safely evaluate a formula string with comprehensive error handling"""
        
        start_time = time.time()
        self.evaluation_stats['total_evaluations'] += 1
        
        try:
            # Get allowed variables from context
            allowed_variables = set()
            allowed_variables.update(context.stocks.keys())
            allowed_variables.update(context.parameters.keys())
            allowed_variables.update(context.auxiliaries.keys())
            
            # Compile formula
            compiled_formula = self.compiler.compile_formula(
                formula, formula_type, frozenset(allowed_variables)
            )
            
            # Create evaluation namespace
            namespace = self._create_namespace(context)
            
            # Evaluate with restricted namespace
            result = eval(compiled_formula.compiled_code, {"__builtins__": {}}, namespace)
            
            # Ensure result is numeric
            if isinstance(result, (list, tuple)):
                result = np.array(result, dtype=float)
            elif not isinstance(result, (int, float, np.ndarray)):
                result = float(result)
            
            # Post-processing: ensure finite values
            if isinstance(result, np.ndarray):
                if not np.all(np.isfinite(result)):
                    warnings.warn(f"Formula '{formula}' produced non-finite values")
                    result = np.nan_to_num(result, nan=0.0, posinf=1e12, neginf=-1e12)
            elif not np.isfinite(result):
                warnings.warn(f"Formula '{formula}' produced non-finite value: {result}")
                result = 0.0
            
            self.evaluation_stats['successful_evaluations'] += 1
            
            evaluation_time = time.time() - start_time
            self.evaluation_stats['total_evaluation_time'] += evaluation_time
            
            return result
            
        except Exception as e:
            self.evaluation_stats['failed_evaluations'] += 1
            
            error_info = {
                'formula': formula,
                'error': str(e),
                'context_time': context.current_time,
                'timestamp': time.time()
            }
            self.evaluation_stats['errors'].append(error_info)
            
            # Keep only last 100 errors
            if len(self.evaluation_stats['errors']) > 100:
                self.evaluation_stats['errors'] = self.evaluation_stats['errors'][-100:]
            
            raise FormulaEvaluationError(
                formula, str(e), f"at t={context.current_time}"
            )
    
    def _create_namespace(self, context: ModelContext) -> Dict[str, Any]:
        """Create safe evaluation namespace"""
        
        # Base namespace with mathematical functions
        namespace = {
            # Mathematical functions
            'abs': abs,
            'min': min,
            'max': max,
            'sum': sum,
            'len': len,
            'round': round,
            'int': int,
            'float': float,
            # Mathematical constants
            'pi': np.pi,
            'e': np.e,
            'inf': np.inf,
            'nan': np.nan,
            'sqrt': np.sqrt
        }
        
        # Add NumPy if enabled
        if self.enable_numpy:
            # Safe subset of numpy functions
            numpy_functions = {
                'np': type('numpy', (), {
                    'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
                    'exp': np.exp, 'log': np.log, 'log10': np.log10, 'sqrt': np.sqrt,
                    'abs': np.abs, 'min': np.min, 'max': np.max,
                    'mean': np.mean, 'std': np.std, 'median': np.median,
                    'sum': np.sum, 'prod': np.prod,
                    'any': np.any, 'all': np.all, 'where': np.where,
                    'pi': np.pi, 'e': np.e, 'inf': np.inf, 'nan': np.nan,
                    'isnan': np.isnan, 'isfinite': np.isfinite,
                    'clip': np.clip, 'maximum': np.maximum, 'minimum': np.minimum
                })()
            }
            namespace.update(numpy_functions)
        
        # Add system dynamics functions
        for name, func in self.system_functions.items():
            if name in ['TIME', 'STARTTIME', 'STOPTIME', 'DT']:
                # Time functions need context
                namespace[name] = func(context)
            elif name in ['STEP', 'PULSE', 'RAMP', 'DELAY']:
                # Functions that need context as last parameter
                namespace[name] = lambda *args, func=func, ctx=context: func(*args, ctx)
            else:
                # Other functions need context as parameter
                # namespace[name] = lambda *args, func=func: func(*args, context)
                namespace[name] = func
        
        # Add context variables
        # Stock values
        for name, value in context.stocks.items():
            namespace[name] = value
        
        # Parameters
        for name, value in context.parameters.items():
            namespace[name] = value
        
        # Auxiliaries
        for name, value in context.auxiliaries.items():
            namespace[name] = value
        
        # Context object for advanced functions
        namespace['context'] = context
        
        return namespace
    
    def validate_formula(self, formula: str, context: ModelContext = None) -> Dict[str, Any]:
        """Validate formula without evaluating it"""
        if context is None:
            context = ModelContext()
        
        allowed_variables = set()
        allowed_variables.update(context.stocks.keys())
        allowed_variables.update(context.parameters.keys())
        allowed_variables.update(context.auxiliaries.keys())
        
        return self.compiler.validator.validate_formula(formula, allowed_variables)
    
    def get_formula_dependencies(self, formula: str) -> Set[str]:
        """Get formula dependencies without compilation"""
        try:
            tree = ast.parse(formula, mode='eval')
            return self.compiler.validator._extract_dependencies(tree)
        except Exception:
            return set()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive engine statistics"""
        compiler_stats = self.compiler.get_compilation_stats()
        
        success_rate = (self.evaluation_stats['successful_evaluations'] / 
                       max(1, self.evaluation_stats['total_evaluations']))
        
        avg_evaluation_time = (self.evaluation_stats['total_evaluation_time'] /
                             max(1, self.evaluation_stats['total_evaluations']))
        
        return {
            'evaluation': {
                **self.evaluation_stats,
                'success_rate': success_rate,
                'avg_evaluation_time': avg_evaluation_time
            },
            'compilation': compiler_stats,
            'recent_errors': self.evaluation_stats['errors'][-10:]  # Last 10 errors
        }
    
    def clear_caches(self):
        """Clear all caches"""
        self.compiler.clear_cache()
        self.compiler.compile_formula.cache_clear()
    
    def export_statistics(self, filepath: str):
        """Export statistics to JSON file"""
        stats = self.get_statistics()
        
        # Make statistics JSON serializable
        serializable_stats = json.loads(json.dumps(stats, default=str))
        
        with open(filepath, 'w') as f:
            json.dump(serializable_stats, f, indent=2)

# ===============================================================================
# Integration with Existing Simulation System
# ===============================================================================

def enhance_stock_with_formulas(stock_class):
    """Enhance Stock class with formula support"""
    
    def set_initial_formula(self, formula: str, engine: FormulaEngine):
        """Set formula for initial value calculation"""
        self._initial_formula = formula
        self._formula_engine = engine
    
    def evaluate_initial_value(self, context: ModelContext):
        """Evaluate initial value using formula"""
        if hasattr(self, '_initial_formula') and hasattr(self, '_formula_engine'):
            try:
                result = self._formula_engine.evaluate(
                    self._initial_formula, context, FormulaType.INITIAL
                )
                if isinstance(result, np.ndarray):
                    self.values = result.reshape(self.values.shape)
                else:
                    self.values = np.full_like(self.values, result)
            except Exception as e:
                warnings.warn(f"Failed to evaluate initial formula for {self.name}: {e}")
    
    # Add methods to stock class
    stock_class.set_initial_formula = set_initial_formula
    stock_class.evaluate_initial_value = evaluate_initial_value
    
    return stock_class

def enhance_flow_with_formulas(flow_class):
    """Enhance Flow class with advanced formula support"""
    
    def set_rate_formula(self, formula: str, engine: FormulaEngine):
        """Set formula for rate calculation"""
        self._rate_formula = formula
        self._formula_engine = engine
        
        # Replace rate_expression with formula evaluation
        def formula_rate_expression():
            if hasattr(self, '_current_context') and self._current_context is not None:
                return self._formula_engine.evaluate(
                    self._rate_formula, self._current_context, FormulaType.RATE
                )
            else:
                # Create a basic context if none is available
                try:
                    # Try to get context from the engine's parent model
                    if hasattr(engine, '_last_context'):
                        context = engine._last_context
                    else:
                        # Create minimal context
                        context = ModelContext(
                            current_time=0.0,
                            stocks={},
                            parameters={}
                        )
                    
                    return self._formula_engine.evaluate(
                        self._rate_formula, context, FormulaType.RATE
                    )
                except:
                    # Last resort - return 0
                    return 0.0
        
        self.rate_expression = formula_rate_expression
        # Initialize context storage
        self._current_context = None

    
    def set_context(self, context: ModelContext):
        """Set current context for formula evaluation"""
        self._current_context = context
    
    def get_formula_dependencies(self) -> Set[str]:
        """Get formula dependencies"""
        if hasattr(self, '_rate_formula') and hasattr(self, '_formula_engine'):
            return self._formula_engine.get_formula_dependencies(self._rate_formula)
        return set()
    
    def validate_formula(self, context: ModelContext = None) -> Dict[str, Any]:
        """Validate flow formula"""
        if hasattr(self, '_rate_formula') and hasattr(self, '_formula_engine'):
            return self._formula_engine.validate_formula(self._rate_formula, context)
        return {'valid': True, 'errors': [], 'warnings': []}
    
    # Add methods to flow class
    flow_class.set_rate_formula = set_rate_formula
    flow_class.set_context = set_context
    flow_class.get_formula_dependencies = get_formula_dependencies
    flow_class.validate_formula = validate_formula
    
    return flow_class

# ===============================================================================
# Testing and Example Usage
# ===============================================================================

def test_formula_engine():
    """Comprehensive test of the formula engine"""
    
    print("🧪 Testing Formula Evaluation Engine")
    print("=" * 60)
    
    # Create engine
    engine = FormulaEngine()
    
    # Create test context
    context = ModelContext(
        current_time=5.0,
        dt=1.0,
        start_time=0.0,
        end_time=10.0,
        stocks={'Population': 1000, 'Resources': 500},
        parameters={'growth_rate': 0.02, 'capacity': 2000}
    )
    
    # Test cases
    test_cases = [
        # Basic arithmetic
        ("2 + 3 * 4", 14),
        ("Population * growth_rate", 20),
        
        # System dynamics functions
        ("STEP(100, 3)", 100),  # Current time is 5, step at 3
        ("PULSE(50, 2, 4)", 50),  # Pulse from 2 to 6, current time is 5
        ("IF_THEN_ELSE(Population > 800, 1, 0)", 1),
        
        # Mathematical functions
        ("np.sin(TIME)", np.sin(5.0)),
        ("sqrt(Population)", np.sqrt(1000)),
        ("MIN(Population, capacity)", 1000),
        
        # Complex expressions
        ("Population * (1 - Population / capacity) * growth_rate", 1000 * (1 - 1000/2000) * 0.02),
        
        # Lookup table test
        ("LOOKUP(TIME, [[0, 10], [5, 50], [10, 100]])", 50),
    ]
    
    print("\n1. Testing Formula Evaluation:")
    for formula, expected in test_cases:
        try:
            result = engine.evaluate(formula, context)
            status = "✓" if abs(result - expected) < 1e-10 else "✗"
            print(f"   {status} {formula} = {result} (expected: {expected})")
        except Exception as e:
            print(f"   ✗ {formula} = ERROR: {e}")
    
    # Test validation
    print("\n2. Testing Formula Validation:")
    validation_cases = [
        ("Population * growth_rate", True),
        ("import os", False),  # Security violation
        ("eval('2+2')", False),  # Security violation
        ("unknown_variable * 2", True),  # Warning but not error
    ]
    
    for formula, should_be_valid in validation_cases:
        validation = engine.validate_formula(formula, context)
        status = "✓" if validation['valid'] == should_be_valid else "✗"
        print(f"   {status} {formula} - Valid: {validation['valid']}")
        if validation['errors']:
            print(f"       Errors: {validation['errors']}")
        if validation['warnings']:
            print(f"       Warnings: {validation['warnings']}")
    
    # Test caching performance
    print("\n3. Testing Caching Performance:")
    formula = "Population * growth_rate * (1 - Population / capacity)"
    
    # First evaluation (cache miss)
    start = time.time()
    result1 = engine.evaluate(formula, context)
    time1 = time.time() - start
    
    # Second evaluation (cache hit)
    start = time.time()
    result2 = engine.evaluate(formula, context)
    time2 = time.time() - start
    
    print(f"   First evaluation: {time1:.6f}s")
    print(f"   Second evaluation: {time2:.6f}s")
    # print(f"   Speedup: {time1/time2:.1f}x")
    
    # Statistics
    print("\n4. Engine Statistics:")
    stats = engine.get_statistics()
    print(f"   Total evaluations: {stats['evaluation']['total_evaluations']}")
    print(f"   Success rate: {stats['evaluation']['success_rate']:.1%}")
    print(f"   Cache hit rate: {stats['compilation']['cache_hit_rate']:.1%}")
    print(f"   Avg evaluation time: {stats['evaluation']['avg_evaluation_time']:.6f}s")
    
    print("\n✅ Formula Engine test completed!")

if __name__ == "__main__":
    test_formula_engine()