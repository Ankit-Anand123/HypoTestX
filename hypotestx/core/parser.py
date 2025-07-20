"""
Advanced natural language hypothesis parsing using NLP libraries
"""
import re
import spacy
import nltk
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

@dataclass
class ParsedHypothesis:
    """Structured representation of parsed hypothesis"""
    test_type: str
    group_column: Optional[str]
    value_column: Optional[str]
    group_values: Optional[Tuple[str, str]]
    comparison_type: str  # "greater", "less", "different", "equal"
    tail: str  # "two-sided", "greater", "less"
    confidence_level: float
    variables: List[str]
    raw_text: str
    entities: Dict[str, Any]
    intent: str
    confidence_score: float

class AdvancedHypothesisParser:
    """Advanced natural language hypothesis parser using NLP libraries"""
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        # Initialize spaCy
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            print(f"SpaCy model '{model_name}' not found. Please install it with:")
            print(f"python -m spacy download {model_name}")
            # Fallback to basic parser
            self.nlp = None
        
        # Initialize NLTK components
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Statistical test patterns with confidence scores
        self.test_patterns = {
            "two_sample_ttest": {
                "keywords": ["compare", "difference", "between", "groups", "mean", "average"],
                "patterns": [
                    r"(?:compare|difference between)\s+(.+?)\s+(?:and|vs|versus)\s+(.+?)(?:\s+in terms of|\s+for|\s+on)\s+(.+)",
                    r"(?:do|are)\s+(.+?)\s+(?:different from|higher than|lower than|more than|less than)\s+(.+?)(?:\s+in|\s+for|\s+on)\s+(.+)",
                    r"(?:is there a difference in)\s+(.+?)\s+(?:between)\s+(.+?)\s+(?:and)\s+(.+)"
                ],
                "confidence": 0.8
            },
            "one_sample_ttest": {
                "keywords": ["test", "against", "equals", "different from", "value"],
                "patterns": [
                    r"(?:is|are)\s+(.+?)\s+(?:equal to|different from|greater than|less than)\s+(\d+\.?\d*)",
                    r"(?:test|check)\s+(?:if|whether)\s+(.+?)\s+(?:equals?|=)\s+(\d+\.?\d*)"
                ],
                "confidence": 0.7
            },
            "chi_square": {
                "keywords": ["association", "relationship", "independent", "categorical", "frequency"],
                "patterns": [
                    r"(?:is there an? association between)\s+(.+?)\s+(?:and)\s+(.+)",
                    r"(?:are)\s+(.+?)\s+(?:and)\s+(.+?)\s+(?:independent|associated|related)",
                    r"(?:relationship between)\s+(.+?)\s+(?:and)\s+(.+)"
                ],
                "confidence": 0.8
            },
            "anova": {
                "keywords": ["multiple", "groups", "more than", "several", "across"],
                "patterns": [
                    r"(?:compare|difference)\s+(.+?)\s+(?:across|among|between)\s+(?:multiple|several|three or more|more than two)\s+(.+)",
                    r"(?:is there a difference in)\s+(.+?)\s+(?:across|among)\s+(.+?)\s+(?:groups|categories)"
                ],
                "confidence": 0.8
            },
            "correlation": {
                "keywords": ["correlation", "relationship", "associated", "related", "linked"],
                "patterns": [
                    r"(?:correlation between)\s+(.+?)\s+(?:and)\s+(.+)",
                    r"(?:are)\s+(.+?)\s+(?:and)\s+(.+?)\s+(?:correlated|related|associated)",
                    r"(?:relationship between)\s+(.+?)\s+(?:and)\s+(.+)"
                ],
                "confidence": 0.7
            }
        }
        
        # Comparison type patterns
        self.comparison_patterns = {
            "greater": {
                "keywords": ["more", "higher", "greater", "larger", "above", "exceed"],
                "patterns": [
                    r"(.+?)\s+(?:spend|earn|score|have|get|are)\s+(?:more|higher|greater)\s+than\s+(.+)",
                    r"(?:do|are)\s+(.+?)\s+(?:spend|earn|score)\s+more\s+than\s+(.+)",
                    r"(.+?)\s+(?:>|greater than|higher than)\s+(.+)"
                ],
                "confidence": 0.9
            },
            "less": {
                "keywords": ["less", "lower", "smaller", "below", "under"],
                "patterns": [
                    r"(.+?)\s+(?:spend|earn|score|have|get|are)\s+(?:less|lower|smaller)\s+than\s+(.+)",
                    r"(?:do|are)\s+(.+?)\s+(?:spend|earn|score)\s+less\s+than\s+(.+)",
                    r"(.+?)\s+(?:<|less than|lower than)\s+(.+)"
                ],
                "confidence": 0.9
            },
            "different": {
                "keywords": ["different", "differ", "difference", "compare", "versus", "vs"],
                "patterns": [
                    r"(?:is there a )?(?:significant )?difference between\s+(.+?)\s+and\s+(.+)",
                    r"(?:do|are)\s+(.+?)\s+(?:and\s+)?(.+?)\s+different",
                    r"compare\s+(.+?)\s+(?:and|vs|versus)\s+(.+)"
                ],
                "confidence": 0.8
            },
            "equal": {
                "keywords": ["equal", "same", "identical", "equivalent"],
                "patterns": [
                    r"(?:are|is)\s+(.+?)\s+(?:equal to|same as|identical to)\s+(.+)",
                    r"(.+?)\s+(?:=|equals?)\s+(.+)",
                    r"no difference between\s+(.+?)\s+and\s+(.+)"
                ],
                "confidence": 0.8
            }
        }
        
        # Statistical concepts vocabulary
        self.stats_vocabulary = {
            "measures": ["mean", "average", "median", "mode", "variance", "std", "deviation"],
            "comparisons": ["higher", "lower", "greater", "less", "more", "fewer", "same", "different"],
            "groups": ["group", "category", "type", "class", "segment", "cohort"],
            "variables": ["variable", "factor", "feature", "attribute", "measure", "metric"],
            "tests": ["test", "check", "analyze", "examine", "compare", "evaluate"]
        }
    
    def parse(self, hypothesis_text: str, data=None) -> ParsedHypothesis:
        """Parse hypothesis text using advanced NLP"""
        # Preprocess text
        cleaned_text = self._preprocess_text(hypothesis_text)
        
        # Extract entities and relationships using spaCy
        entities = self._extract_entities(cleaned_text)
        
        # Determine intent and test type
        intent, test_type, confidence = self._classify_intent(cleaned_text, entities)
        
        # Extract comparison information
        comparison_type, group_values = self._extract_comparison_advanced(cleaned_text, entities)
        
        # Extract confidence level
        confidence_level = self._extract_confidence_level(cleaned_text)
        
        # Extract variables using NLP
        variables = self._extract_variables_nlp(cleaned_text, entities, data)
        
        # Infer columns using advanced matching
        group_column, value_column = self._infer_columns_advanced(
            cleaned_text, variables, entities, data
        )
        
        # Refine test type based on data analysis
        if data is not None:
            test_type = self._refine_test_type(test_type, group_column, value_column, data)
        
        # Determine tail
        tail = self._determine_tail(comparison_type)
        
        return ParsedHypothesis(
            test_type=test_type,
            group_column=group_column,
            value_column=value_column,
            group_values=group_values,
            comparison_type=comparison_type,
            tail=tail,
            confidence_level=confidence_level,
            variables=variables,
            raw_text=hypothesis_text,
            entities=entities,
            intent=intent,
            confidence_score=confidence
        )
    
    def _preprocess_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Convert to lowercase
        text = text.lower().strip()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Expand contractions
        contractions = {
            "don't": "do not",
            "doesn't": "does not",
            "isn't": "is not",
            "aren't": "are not",
            "won't": "will not",
            "can't": "cannot"
        }
        
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)
        
        return text
    
    def _extract_entities(self, text: str) -> Dict[str, Any]:
        """Extract entities using spaCy NER"""
        entities = {
            "variables": [],
            "groups": [],
            "numbers": [],
            "comparisons": [],
            "statistical_terms": []
        }
        
        if self.nlp is None:
            return entities
        
        doc = self.nlp(text)
        
        # Extract named entities
        for ent in doc.ents:
            if ent.label_ in ["CARDINAL", "QUANTITY", "PERCENT"]:
                entities["numbers"].append(ent.text)
            elif ent.label_ in ["PERSON", "ORG", "GPE"]:
                entities["groups"].append(ent.text)
        
        # Extract statistical terms
        for token in doc:
            lemma = token.lemma_.lower()
            
            # Check if it's a statistical concept
            for category, terms in self.stats_vocabulary.items():
                if lemma in terms:
                    entities["statistical_terms"].append({
                        "term": lemma,
                        "category": category,
                        "pos": token.pos_
                    })
            
            # Extract comparison words
            if token.pos_ in ["ADJ", "ADV"] and lemma in ["more", "less", "higher", "lower", "greater", "smaller"]:
                entities["comparisons"].append(lemma)
        
        # Extract noun phrases as potential variables
        for chunk in doc.noun_chunks:
            if len(chunk.text.split()) <= 3:  # Avoid very long phrases
                entities["variables"].append(chunk.text)
        
        return entities
    
    def _classify_intent(self, text: str, entities: Dict) -> Tuple[str, str, float]:
        """Classify the intent and determine test type"""
        best_match = None
        best_score = 0
        best_test = "unknown"
        
        for test_type, config in self.test_patterns.items():
            score = 0
            
            # Check keyword matches
            keyword_matches = sum(1 for keyword in config["keywords"] if keyword in text)
            score += (keyword_matches / len(config["keywords"])) * 0.5
            
            # Check pattern matches
            pattern_match = False
            for pattern in config["patterns"]:
                if re.search(pattern, text):
                    pattern_match = True
                    break
            
            if pattern_match:
                score += 0.4
            
            # Check statistical terms
            stats_terms = [term["term"] for term in entities.get("statistical_terms", [])]
            relevant_terms = set(config["keywords"]) & set(stats_terms)
            if relevant_terms:
                score += 0.1
            
            # Apply base confidence
            score *= config["confidence"]
            
            if score > best_score:
                best_score = score
                best_test = test_type
                best_match = config
        
        intent = f"hypothesis_testing_{best_test}"
        return intent, best_test, best_score
    
    def _extract_comparison_advanced(self, text: str, entities: Dict) -> Tuple[str, Optional[Tuple[str, str]]]:
        """Extract comparison type using NLP"""
        comparison_scores = {}
        
        for comp_type, config in self.comparison_patterns.items():
            score = 0
            
            # Check keywords
            keyword_matches = sum(1 for keyword in config["keywords"] if keyword in text)
            score += keyword_matches
            
            # Check patterns
            for pattern in config["patterns"]:
                match = re.search(pattern, text)
                if match:
                    score += 2
                    groups = match.groups()
                    if len(groups) >= 2:
                        comparison_scores[comp_type] = (score, (groups[0].strip(), groups[1].strip()))
                    else:
                        comparison_scores[comp_type] = (score, None)
            
            if comp_type not in comparison_scores:
                comparison_scores[comp_type] = (score, None)
        
        # Find best match
        if comparison_scores:
            best_comp = max(comparison_scores.keys(), key=lambda x: comparison_scores[x][0])
            if comparison_scores[best_comp][0] > 0:
                return best_comp, comparison_scores[best_comp][1]
        
        return "different", None
    
    def _extract_confidence_level(self, text: str) -> float:
        """Extract confidence level with enhanced patterns"""
        patterns = [
            r"(?:at|with)\s+(\d+)%?\s+confidence",
            r"alpha\s*=\s*([0-9.]+)",
            r"α\s*=\s*([0-9.]+)",
            r"significance level\s+([0-9.]+)",
            r"p\s*<\s*([0-9.]+)",
            r"(\d+)%\s+confidence",
            r"0\.0([1-9])\s+level"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                value = float(match.group(1))
                if "confidence" in pattern or "%" in text:
                    return 1 - (value / 100) if value > 1 else 1 - value
                else:
                    return value if value <= 1 else value / 100
        
        return 0.05  # Default alpha
    
    def _extract_variables_nlp(self, text: str, entities: Dict, data=None) -> List[str]:
        """Extract variables using NLP techniques"""
        variables = []
        
        # From entities
        variables.extend(entities.get("variables", []))
        
        # From statistical terms
        for term_info in entities.get("statistical_terms", []):
            if term_info["category"] == "variables":
                variables.append(term_info["term"])
        
        # Match against data columns if available
        if data is not None and hasattr(data, 'columns'):
            column_names = list(data.columns)
            
            # Direct matches
            for col in column_names:
                if col.lower() in text:
                    variables.append(col)
            
            # Fuzzy matching using lemmatization
            text_tokens = set(self.lemmatizer.lemmatize(word.lower()) 
                            for word in word_tokenize(text) 
                            if word.lower() not in self.stop_words)
            
            for col in column_names:
                col_tokens = set(self.lemmatizer.lemmatize(word.lower()) 
                               for word in word_tokenize(col))
                
                if col_tokens & text_tokens:  # If there's any overlap
                    if col not in variables:
                        variables.append(col)
        
        # Remove duplicates and clean
        variables = list(set(variables))
        variables = [var for var in variables if len(var.strip()) > 1]
        
        return variables
    
    def _infer_columns_advanced(self, text: str, variables: List[str], 
                              entities: Dict, data=None) -> Tuple[Optional[str], Optional[str]]:
        """Advanced column inference using NLP and data analysis"""
        if data is None:
            return None, None
        
        group_column = None
        value_column = None
        
        # Analyze data types
        if hasattr(data, 'dtypes'):
            categorical_cols = []
            numerical_cols = []
            
            for col in data.columns:
                if data[col].dtype in ['object', 'category', 'bool']:
                    categorical_cols.append(col)
                elif data[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                    numerical_cols.append(col)
            
            # Enhanced keyword matching
            context_clues = {
                "value_keywords": [
                    "spend", "cost", "price", "amount", "value", "money", "income", "salary",
                    "score", "rating", "measure", "count", "number", "quantity", "time",
                    "age", "height", "weight", "distance", "revenue", "profit"
                ],
                "group_keywords": [
                    "gender", "sex", "group", "type", "category", "class", "segment",
                    "department", "region", "country", "state", "city", "team", "status",
                    "level", "grade", "condition", "treatment", "method"
                ]
            }
            
            # Score columns based on context
            value_scores = {}
            group_scores = {}
            
            for col in numerical_cols:
                score = 0
                col_lower = col.lower()
                
                # Direct keyword matches
                for keyword in context_clues["value_keywords"]:
                    if keyword in col_lower or keyword in text:
                        score += 2
                
                # Token overlap
                col_tokens = set(word.lower() for word in word_tokenize(col))
                text_tokens = set(word.lower() for word in word_tokenize(text))
                overlap = len(col_tokens & text_tokens)
                score += overlap
                
                value_scores[col] = score
            
            for col in categorical_cols:
                score = 0
                col_lower = col.lower()
                
                # Direct keyword matches
                for keyword in context_clues["group_keywords"]:
                    if keyword in col_lower or keyword in text:
                        score += 2
                
                # Token overlap
                col_tokens = set(word.lower() for word in word_tokenize(col))
                text_tokens = set(word.lower() for word in word_tokenize(text))
                overlap = len(col_tokens & text_tokens)
                score += overlap
                
                # Check unique value count (good grouping variables have 2-10 unique values)
                unique_count = data[col].nunique()
                if 2 <= unique_count <= 10:
                    score += 1
                
                group_scores[col] = score
            
            # Select best matches
            if value_scores:
                value_column = max(value_scores.keys(), key=lambda x: value_scores[x])
                if value_scores[value_column] == 0 and numerical_cols:
                    value_column = numerical_cols[0]  # Fallback
            
            if group_scores:
                group_column = max(group_scores.keys(), key=lambda x: group_scores[x])
                if group_scores[group_column] == 0 and categorical_cols:
                    group_column = categorical_cols[0]  # Fallback
        
        return group_column, value_column
    
    def _refine_test_type(self, initial_test: str, group_column: Optional[str], 
                         value_column: Optional[str], data) -> str:
        """Refine test type based on actual data characteristics"""
        if group_column is None or value_column is None:
            return initial_test
        
        if not hasattr(data, 'dtypes'):
            return initial_test
        
        group_dtype = str(data[group_column].dtype)
        value_dtype = str(data[value_column].dtype)
        
        # Check if our assumptions are correct
        if (group_dtype in ['object', 'category', 'bool'] and 
            value_dtype in ['int64', 'float64', 'int32', 'float32']):
            
            unique_groups = data[group_column].nunique()
            
            if unique_groups == 2:
                return "two_sample_ttest"
            elif unique_groups > 2:
                return "anova"
            else:
                return "one_sample_ttest"
        
        elif (group_dtype in ['object', 'category', 'bool'] and 
              value_dtype in ['object', 'category', 'bool']):
            return "chi_square"
        
        elif (group_dtype in ['int64', 'float64', 'int32', 'float32'] and 
              value_dtype in ['int64', 'float64', 'int32', 'float32']):
            return "correlation"
        
        return initial_test
    
    def _determine_tail(self, comparison_type: str) -> str:
        """Determine test tail from comparison type"""
        if comparison_type == "greater":
            return "greater"
        elif comparison_type == "less":
            return "less"
        else:
            return "two-sided"

# Backward compatibility - simple parser for basic use cases
class SimpleHypothesisParser:
    """Simple regex-based parser for basic hypothesis parsing"""
    
    def __init__(self):
        self.comparison_patterns = {
            "greater": [
                r"(\w+)\s+(?:spend|earn|score|have|get|are)\s+more\s+than\s+(\w+)",
                r"(\w+)\s+(?:>|greater than|higher than)\s+(\w+)",
                r"do\s+(\w+)\s+(?:spend|earn|score)\s+more\s+than\s+(\w+)",
            ],
            "less": [
                r"(\w+)\s+(?:spend|earn|score|have|get|are)\s+less\s+than\s+(\w+)",
                r"(\w+)\s+(?:<|less than|lower than)\s+(\w+)",
                r"do\s+(\w+)\s+(?:spend|earn|score)\s+less\s+than\s+(\w+)",
            ],
            "different": [
                r"(?:is there a )?(?:significant )?difference between\s+(\w+)\s+and\s+(\w+)",
                r"(?:do|are)\s+(\w+)\s+(?:and\s+)?(\w+)\s+different",
                r"compare\s+(\w+)\s+(?:and|vs|versus)\s+(\w+)"
            ]
        }
    
    def parse(self, hypothesis_text: str, data=None) -> ParsedHypothesis:
        """Simple parsing for basic cases"""
        # Implementation similar to original but returning ParsedHypothesis
        # This serves as a fallback when advanced NLP fails
        pass

# Factory function to choose parser
def create_parser(advanced: bool = True) -> Union[AdvancedHypothesisParser, SimpleHypothesisParser]:
    """Factory function to create appropriate parser"""
    if advanced:
        try:
            return AdvancedHypothesisParser()
        except Exception as e:
            print(f"Advanced parser failed to initialize: {e}")
            print("Falling back to simple parser...")
            return SimpleHypothesisParser()
    else:
        return SimpleHypothesisParser()