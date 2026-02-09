import json
import logging
import pprint
import re
import functools
import _snowflake  # Native Snowflake module for internal API calls
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, Literal, Tuple, Generator
from pydantic import BaseModel, Field, field_validator
from snowflake.snowpark import Session
from snowflake.snowpark.types import StructType, StructField, StringType, VariantType, IntegerType, TimestampType, FloatType
from snowflake.snowpark.functions import lit, parse_json, col
from collections import defaultdict
import datetime
import traceback
import time
import uuid
import math
import hashlib

# ==========================================
# 1. CONFIGURATION
# ==========================================


class AgentConfig(BaseModel):
    """Global configuration for the Cortex Agent execution environment.

    Attributes:
        model (str): The LLM model used for orchestration (default: claude-sonnet-4-5).
        cortex_analyst_object_master (str): Fully qualified path to Master View.
        cortex_analyst_object_ards (str): Fully qualified path to Basic View.
        warehouse (str): Snowflake warehouse name.
        messages_table (str): Table for chat history.
        audit_log_table (str): Table for high-level audit logs.
        trace_events_table (str): Table for granular OpenTelemetry-style trace events.
        diagnostic_udf (str): UDF name for diagnostic tool.
        diagnostic_tree_id (str): ID for the diagnostic tree.
        min_data_score_threshold (int): Score threshold for retries.
        pricing_map (Dict): Cost estimation map for different models (reference only).
    """

    # LLM Models
    model: str = "claude-sonnet-4-5"

    # Snowflake Objects
    cortex_analyst_object_master: str = (
        "DEV_P_PAIN_SALES_PERFORMANCE_ANALYTICS_DB.AI_POC.MASTERTABLE_V1"
    )
    
    warehouse: str = "DEV_PAIN_SALES_PERFORMANCE_B_WH"

    # Tables
    messages_table: str = "DEV_P_PAIN_SALES_PERFORMANCE_ANALYTICS_DB.AI_POC.MESSAGES_V2"
    audit_log_table: str = (
        "DEV_P_PAIN_SALES_PERFORMANCE_ANALYTICS_DB.AI_POC.AGENT_DETAILED_HISTORY_V2"
    )
    trace_events_table: str = (
        "DEV_P_PAIN_SALES_PERFORMANCE_ANALYTICS_DB.AI_POC.AGENT_TRACE_EVENTS"
    )

    # Tools
    diagnostic_udf: str = (
        "DEV_P_PAIN_SALES_PERFORMANCE_ANALYTICS_DB.AI_POC.DIAGNOSTIC_TOOL_V3"
    )
    diagnostic_tree_id: str = "Pharma_Master_v4"



CONFIG = AgentConfig()

# ==========================================
# 2. Agent MODELS
# ==========================================


class ErrorEvent(BaseModel):
    """
    Dedicated model for ALL system errors.
    This is what the UI receives when a function fails.
    """
    is_error: bool = Field(default=True, frozen=True)
    status_code: int = Field(..., description="Extracted status code (e.g., 401, 503) or default 500.")
    message: str = Field(..., description="The direct error message from the exception.")
    context: str = Field(..., description="Where the error occurred (e.g., 'Data Agent').")
    technical_details: Optional[str] = Field(None, description="Raw exception trace.")


class EvaluationResult(BaseModel):
    """Represents the self-evaluation of an agent's response."""
    score: int = Field(default=0, description="Confidence score from 1-10.")
    reasoning: str = Field(default="No evaluation provided.", description="Reasoning for the score.")

class IntentClassification(BaseModel):
    """Structured output for the intent classification step."""
    intent_type: str = Field(..., description="Intent category.")
    data_retrieval_query: Optional[str] = None
    root_cause_query: Optional[str] = None
    direct_response: Optional[str] = None
    clarification_question: Optional[str] = None
    confidence: str = Field(..., description="Confidence level.")
    reasoning: str = Field(..., description="Reasoning.")

class RephraserOutput(BaseModel):
    """
    Unified output model.
    - If action is 'direct_answer', response_text is the final answer.
    - If action is 'refined_query', response_text is the STANDALONE USER QUESTION (in English).
    """
    action: Literal["direct_answer", "refined_query"] = Field(
        ..., 
        description="Flag to determine if we stop (direct_answer) or proceed to analysis (refined_query)."
    )
    
    response_text: str = Field(
        ..., 
        description="Contains the Direct Answer Text OR the Refined Natural Language Question (e.g., 'Show me sales in 2024')."
    )
    
class GuardResult(BaseModel):
    """Result of the safety guardrail check."""
    is_safe: bool = Field(...)
    message: Optional[str] = Field(None)
    usage: Optional[Dict] = Field(default=None, description="Token usage stats")

class SqlResult(BaseModel):
    """Data analysis result container from Cortex Analyst."""
    bot_answer: str = Field(default="")
    visual_summary: str = Field(default="")
    tables: List[Any] = Field(default_factory=list)
    charts: List[Any] = Field(default_factory=list)
    sql_generated: Optional[str] = None
    sql_explanation: Optional[str] = None
    is_verified_query: bool = False
    evaluation: EvaluationResult = Field(default_factory=EvaluationResult)
    is_retry: bool = False
    original_query_score: Optional[int] = None 

    @field_validator("charts", mode="before")
    @classmethod
    def parse_charts(cls, v):
        if not v: return []
        parsed = []
        for item in v:
            if isinstance(item, dict): parsed.append(item)
            elif isinstance(item, str):
                try: parsed.append(json.loads(item))
                except: parsed.append(item)
        return parsed

class DiagnosticResult(BaseModel):
    """Result from the root cause analysis tool."""
    bot_answer: str = Field(default="")
    react_flow_json: Optional[Dict] = Field(default=None)
    evaluation: EvaluationResult = Field(default_factory=EvaluationResult)

# ==========================================
# 3. TELEMETRY & LOGGING SYSTEM
# ==========================================

class TelemetrySpan(BaseModel):
    """Represents a single unit of work (Span) with enhanced metadata for observability."""
    trace_id: str
    span_id: str
    conversation_id: Optional[str] 
    name: str
    span_kind: str = "INTERNAL"
    start_time: datetime.datetime
    end_time: Optional[datetime.datetime] = None
    duration_ms: Optional[int] = None
    status: str = "running"
    
    # Context & Payload
    input_attributes: Optional[Dict] = None
    output_attributes: Optional[Dict] = None
    error_message: Optional[str] = None
    
    # Metrics
    model_name: Optional[str] = None
    input_tokens: int = 0
    output_tokens: int = 0
    warehouse: Optional[str] = None 

class TelemetryCollector:
    """Production-grade collector to hold spans in memory and flush them to Snowflake at the end of execution."""
    
    def __init__(self):
        self._spans: List[TelemetrySpan] = []
        self.trace_id = str(uuid.uuid4())
        self.session_id: Optional[str] = None 
    
    def set_session_id(self, session_id: str):
        """Sets the global session ID for all subsequent spans.
        
        Args:
            session_id (str): The unique conversation identifier.
        """
        self.session_id = session_id

    def start_span(self, name: str, inputs: Any = None, kind: str = "INTERNAL") -> TelemetrySpan:
        """Starts a new telemetry span.

        Args:
            name (str): Name of the operation or function.
            inputs (Any, optional): Input arguments or payload.
            kind (str, optional): Type of span (INTERNAL, TOOL, CLIENT, SERVER).

        Returns:
            TelemetrySpan: The active span object.
        """
        span = TelemetrySpan(
            trace_id=self.trace_id,
            span_id=str(uuid.uuid4()),
            conversation_id=self.session_id, 
            name=name,
            span_kind=kind,
            start_time=datetime.datetime.now(),
            input_attributes=self._sanitize(inputs),
            warehouse=CONFIG.warehouse, 
            tags=[] 
        )
        return span

    
    def end_span(self, span: TelemetrySpan, outputs: Any = None, error: Exception = None, model: str = None):
        """Ends a span, calculates duration, and captures outputs.

        Args:
            span (TelemetrySpan): The span to close.
            outputs (Any, optional): Result of the operation.
            error (Exception, optional): Exception if the operation failed.
            model (str, optional): Name of the model used (if LLM call).
        """
        span.end_time = datetime.datetime.now()
        span.duration_ms = int((span.end_time - span.start_time).total_seconds() * 1000)

        # Detect Usage Metrics from various return types
        explicit_usage = None
        
        # 1. From Dictionary (standard Cortex API / custom refiner)
        if isinstance(outputs, dict) and "usage" in outputs:
             explicit_usage = outputs.get("usage")
        
        # 2. From Pydantic Model (GuardResult, SqlResult, etc.)
        elif isinstance(outputs, BaseModel) and hasattr(outputs, "usage"):
             explicit_usage = getattr(outputs, "usage")

        # Apply Metrics if found
        if explicit_usage:
             print(explicit_usage)
             i_tokens = explicit_usage.get("input_tokens") or explicit_usage.get("prompt_tokens") or 0
             o_tokens = explicit_usage.get("output_tokens") or explicit_usage.get("completion_tokens") or 0
             
             span.input_tokens = i_tokens
             span.output_tokens = o_tokens
             span.model_name = explicit_usage.get("model_name") or model or CONFIG.model
        else:
             span.model_name = model or CONFIG.model
             span.input_tokens = 0
             span.output_tokens = 0

        span.output_attributes = self._sanitize(outputs)
        
        if error:
            span.status = "error"
            span.error_message = str(error)
        else:
            span.status = "success"
            
        self._spans.append(span)

    def _sanitize(self, data: Any) -> Dict:
        """Sanitizes data for JSON serialization to prevent errors and manage size.

        Args:
            data (Any): The data to sanitize.

        Returns:
            Dict: JSON-compatible dictionary or truncated string representation.
        """
        if data is None: return None
        try:
            if isinstance(data, BaseModel):
                return json.loads(data.model_dump_json())
            if isinstance(data, (str, int, float, bool)):
                return {"value": data}
            json_str = json.dumps(data, default=str)
            if len(json_str) > 100000: 
                return {"value": "Payload too large", "preview": json_str[:2000] + "..."}
            return json.loads(json_str)
        except:
            return {"value": str(data)}

    def flush_to_snowflake(self, session: Session):
        """Flushes captured spans to the Snowflake trace events table.

        Args:
            session (Session): Active Snowpark session.
        """
        if not self._spans:
            print("ℹ️ No telemetry spans to flush.")
            return

        print(f"📡 Flushing {len(self._spans)} telemetry spans to {CONFIG.trace_events_table}...")
        
        try:
            records = []
            current_time = datetime.datetime.now()
            
            for span in self._spans:
                records.append({
                    "TRACE_ID": span.trace_id,
                    "SPAN_ID": span.span_id,
                    "CONVERSATION_ID": span.conversation_id,
                    "NAME": span.name,
                    "SPAN_KIND": span.span_kind,
                    "START_TIME": span.start_time,
                    "END_TIME": span.end_time,
                    "DURATION_MS": span.duration_ms,
                    "STATUS": span.status,
                    "INPUT_ATTRIBUTES": json.dumps(span.input_attributes),
                    "OUTPUT_ATTRIBUTES": json.dumps(span.output_attributes),
                    "ERROR_MESSAGE": span.error_message,
                    "MODEL_NAME": span.model_name,
                    "INPUT_TOKENS": span.input_tokens,
                    "OUTPUT_TOKENS": span.output_tokens,
                    "WAREHOUSE": span.warehouse,
                    "CREATED_AT": current_time
                })

            df = session.create_dataframe(records)
            
            # Explicit Schema Mapping to match table DDL and prevent Snowflake Data Type errors
            df_final = df.select(
                col("TRACE_ID"),
                col("SPAN_ID"),
                col("CONVERSATION_ID"),
                col("NAME"),
                col("SPAN_KIND"),
                col("START_TIME"),
                col("END_TIME"),
                col("DURATION_MS"),
                col("STATUS"),
                parse_json(col("INPUT_ATTRIBUTES")).alias("INPUT_ATTRIBUTES"),
                parse_json(col("OUTPUT_ATTRIBUTES")).alias("OUTPUT_ATTRIBUTES"),
                col("ERROR_MESSAGE"),
                col("MODEL_NAME"),
                col("INPUT_TOKENS"),
                col("OUTPUT_TOKENS"),
                col("WAREHOUSE"),
                col("CREATED_AT")
            )
            
            df_final.write.mode("append").save_as_table(CONFIG.trace_events_table)
            print("✅ Telemetry flush complete.")
            
        except Exception as e:
            print(f"❌ Failed to flush telemetry: {e}")
            traceback.print_exc()

# Global Collector Instance
GLOBAL_TRACER = TelemetryCollector()

def time_execution(step_name: str, kind: str = "INTERNAL"):
    """Decorator to measure execution time, capture telemetry, and handle errors.

    Args:
        step_name (str): The name of the step/function being instrumented.
        kind (str, optional): The OpenTelemetry span kind. Defaults to "INTERNAL".
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Filter 'self' and 'session' from args to keep logs clean
            filtered_args = [a for a in args if not hasattr(a, 'sql')] 
            inputs = {"args": filtered_args}
            if kwargs: inputs.update(kwargs)
            
            span = GLOBAL_TRACER.start_span(name=step_name, inputs=inputs, kind=kind)
            
            try:
                result = func(*args, **kwargs)
                GLOBAL_TRACER.end_span(span, outputs=result)
                print(f"⏱️ [TIMING] {step_name}: {span.duration_ms}ms")
                return result
            except Exception as e:
                # Capture the error in the span before re-raising
                GLOBAL_TRACER.end_span(span, error=e)
                print(f"❌ [ERROR] {step_name}: {str(e)}")
                raise e
        return wrapper
    return decorator



# ==========================================
# 4. PROMPTS
# ==========================================

# --- REPHRASER ---
PROMPT_REPHRASER_SYSTEM = "You are an advanced Contextual Query Reconstruction Engine. Your sole purpose is to convert conversational fragments into precise, standalone database-ready queries. You do not answer questions; you only rephrase them."

PROMPT_REPHRASER_ORCHESTRATION = """
<context_reconstruction_master_instructions>
    <objective>
    You are the "Entry Point" logic engine. You must convert the `Current User Input` into a specific Action.
    You must intelligently look back at the `Conversation History` (including Tables and Charts) to determine **which specific previous question** the user is responding to.
    </objective>

    <decision_hierarchy_execution_order>
    You MUST execute these steps in strict order (1 -> 2 -> 3 -> 4).

    **STEP 1: CHECK FOR PENDING CLARIFICATION (The "Slot Filling" Rule)**
    - *Analysis:* Look at the **Last Bot Message**. Did it end with a question? (e.g., "Did you mean Brand X or Y?", "Which Region?").
    - *Condition:* If YES, treat the `Current User Input` as the specific ANSWER to that question.
    - *Action:* Find the *Original User Question* that triggered the clarification. Insert the current input into it.
    - *Output:* `action`="refined_query", `response_text`="[Merged Query]"

    **STEP 2: CHECK FOR SOCIAL / GREETINGS (The "Phatic" Rule)**
    - *Condition:* Is the input purely social (e.g., "Hi", "Thanks", "Good morning") with NO data request?
    - *Action:* Output a polite, conversational response.
    - *Output:* `action`="direct_answer", `response_text`="Hello! How can I help you with your data today?"
    - *Exception:* If the input is Mixed (e.g., "Thanks, what about West?"), **IGNORE** the greeting and proceed to Step 4 (Refinement) using only the question part.

    **STEP 3: CHECK LOCAL DATA (The "Direct Lookup" Rule)**
    - *Analysis:* Look at the `history` -> `Tables` or `Charts` shown in the previous turn.
    - *Condition:* Is the user asking for a specific number or comparison that is **already visible** on the history? (e.g., "What is that number?", "Which one is highest?", "What about the second row?").
    - *Action:* Extract the answer directly from the JSON history. **DO NOT GENERATE ANYTHING ELSE.**
    - *Output:* `action`="direct_answer", `response_text`="The value for [X] is [Y] as shown in the table."

    **STEP 4: CONTEXTUAL RECONSTRUCTION (The "Refinement" Rule)**
    - *Analysis:* The user is asking a new question, drilling down, or changing the subject.
    - *Action:* Construct a **Standalone, Grammatically Complete** database query.
    - *Critical Logic - "Thread Tracking":*
        - **Scenario A (Why/Diagnostic):** If previous queries were "Why", "Reason", "Explain" -> Maintain the "Why" intent. (e.g., Input "In East" -> "Why are sales down in East?").
        - **Scenario B (What/Data):** If previous queries were "Show", "List", "Count" -> Maintain the "Show" intent. (e.g., Input "In East" -> "Show sales in East").
        - **Scenario C (Combined What + Why):** If the input asks for BOTH (e.g., "Show sales and why they dropped"), you **MUST PRESERVE BOTH** parts in the output string so the Intent Classifier can trigger parallel execution.
        - **Scenario D (Subject Change):** If input is "Start over" or "New topic" -> Ignore history.
    - *Entity Resolution:*
        - Resolve "it", "that", "same region", "the previous product" to the actual names found in history.
    - *Output:* `action`="refined_query", `response_text`="[Full Standalone Query]"
    </decision_hierarchy_execution_order>

    <critical_intent_locking_protocol>
    **THE "WHY" PRESERVATION RULE (Highest Priority):**
    - You must detect the "Active Thread Intent" from the conversation history.
    - **Diagnostic Thread:** If the previous User Query was diagnostic (e.g., "Why...", "Reason for...", "Driver of...", "Explain..."), you **MUST** start the new rephrased query with "Why" or "Explain".
    - **Forbidden:** Do NOT change a "Why" question into a "Show", "List", "Compare", "What is", or "Trend" question.
    - **Example of Failure:** History="Why did sales drop?", Input="In East?" -> Output="Show sales trend in East." (WRONG - Intent changed to Data Retrieval).
    - **Example of Success:** History="Why did sales drop?", Input="In East?" -> Output="Why did sales drop in the East region?" (CORRECT - Intent preserved).
    </critical_intent_locking_protocol>

    <anti_hallucination_guardrails>
    1. **Do not invent Columns/IDs:** If user says "C-555", output "C-555". Do not change it to "Campaign 555" unless you are 100% sure.
    2. **Do not guess math:** If Step 3 (Direct Lookup) requires complex calculation not in the table, switch to Step 4 (Refinement).
    3. **Ambiguity:** If the user mentions "Same Region" and multiple regions exist in history, pick the most recently mentioned one.
    </anti_hallucination_guardrails>

    <few_shot_examples>
        <example type="Step 1: Slot Filling">
            <history_user>Show me churn rate.</history_user>
            <history_bot>For which specific Campaign ID?</history_bot>
            <input>C-555</input>
            <output>{"action": "refined_query", "response_text": "Show me churn rate for Campaign ID C-555"}</output>
        </example>

        <example type="Step 2: Pure Greeting">
            <input>Thanks!</input>
            <output>{"action": "direct_answer", "response_text": "You're welcome! Let me know if you have more questions."}</output>
        </example>

        <example type="Step 3: Direct Answer">
            <history_context>Table: [{Region: East, Sales: 500}]</history_context>
            <input>What is the sales value for East?</input>
            <output>{"action": "direct_answer", "response_text": "The sales value for East is 500."}</output>
        </example>

        <example type="Step 4: Thread Tracking (Why)">
            <history_user>Why is profit dropping for Nike?</history_user>
            <history_bot>Profit is down due to supply chain.</history_bot>
            <input>In Europe.</input>
            <output>{"action": "refined_query", "response_text": "Why is profit dropping for Nike in Europe?"}</output>
        </example>

        <example type="Step 4: Thread Tracking (What)">
            <history_user>Show me the sales for Nike.</history_user>
            <input>In Europe.</input>
            <output>{"action": "refined_query", "response_text": "Show me the sales for Nike in Europe."}</output>
        </example>

        <example type="Step 4: Combined Intent">
            <input>Show me sales and explain why they dropped.</input>
            <output>{"action": "refined_query", "response_text": "Show me sales and explain why they dropped."}</output>
        </example>
    </few_shot_examples>
</context_reconstruction_master_instructions>
"""

PROMPT_REPHRASER_RESPONSE = """
<output_format>
Output a single valid JSON object strictly. 
No markdown formatting (no ```json). 
No conversational filler.

Template:
{ "action": either "refined_query" or "direct_answer","response_text"": "The response from the rephraser agent." }
</output_format>
"""

# --- INTENT ---
PROMPT_INTENT_SYSTEM = "You are an Expert Intent Classification System. Be polite, professional, and friendly. Be direct and unambiguous."

PROMPT_INTENT_ORCHESTRATION = """
<intent_classification_instructions>

<mission>
Your MISSION is to be a RIGID INTENT ROUTER.
You MUST distinguish between "WHAT is happening" (Data) and "WHY it is happening" (Root Cause).
Your logic must be binary and strict.
</mission>

<intent_categories>

<intent name="root_cause_analysis">
    <description>
        **HIGHEST PRIORITY.**
        User wants to know the REASON, DRIVER, or CAUSE of a specific behavior.
        **CRITICAL OVERRIDE:** If the query implies a diagnostic question (e.g., "Why is X rising?", "Why did Y fall?"), it is ALWAYS root_cause_analysis.
        It does NOT matter if they mention metrics (Volume, Sales, Prescriptions). If they ask "Why", they want an explanation, not a list of numbers.
    </description>
    <indicators>why, explain, reason for, root cause, what is driving, what is causing, what caused, investigate, diagnose, how come, factor, due to</indicators>
    <action>
        1. Extract the query VERBATIM into `root_cause_query`.
        2. DO NOT include visual requests here.
    </action>
</intent>

<intent name="data_retrieval">
    <description>
        User wants to FETCH static numbers, trends, or lists.
        They are asking "What is the value?" or "Compare X and Y".
        **CONSTRAINT:** If the query contains "Why", "Reason", or "Driver", it is FORBIDDEN from being data_retrieval.
    </description>
    <indicators>show, list, get, fetch, display, find, what is the value, what is the total, how many, count, top 10, trend of, compare</indicators>
    <action>
        1. Extract the query into `data_retrieval_query`.
        2. MANDATORY: You MUST append the exact string " and generate a table and a visual" to the end of the query.
    </action>
</intent>

<intent name="combined">
    <description>User explicitly asks for BOTH a display AND an explanation in the same prompt.</description>
    <indicators>E.g., "Show me the trend AND explain why it's happening." (Must have a conjunction like 'and', 'also')</indicators>
    <action>
        1. Extract the data part to `data_retrieval_query` (with visual suffix).
        2. Extract the causal part to `root_cause_query`.
    </action>
</intent>

<intent name="greeting">
    <action>Set direct_response.</action>
</intent>

<intent name="clarification_needed">
    <action>Set clarification_question.</action>
</intent>

<intent name="general_question">
    <action>Set direct_response.</action>
</intent>

<intent name="off_topic">
    <action>Set direct_response.</action>
</intent>

</intent_categories>

<STRICT_LOGIC_HIERARCHY>

1. **THE "WHY" TRUMP CARD (ABSOLUTE RULE):**
   - Input: "Why is Market Volume rising?" -> `root_cause_analysis`
   - Input: "Show me Market Volume rising." -> `data_retrieval`
   - IF the word "Why" exists, `data_retrieval` is invalid.
   - "Why" questions about trends (rising, falling, spiking) are REQUESTS FOR EXPLANATION, not data fetches.

2. **THE "WHAT" TRAP:**
   - "What is the volume?" -> Data.
   - "What is *driving* the volume?" -> Root Cause.
   - "What *caused* the drop?" -> Root Cause.
   - You must look at the verb. If the verb implies causality (driving, causing), it is Root Cause.

3. **VERBATIM EXTRACTION:**
   - Never paraphrase the `root_cause_query`.

4. **VISUALS:**
   - ONLY append "and generate a table and a visual" if `intent_type` is `data_retrieval`.

</STRICT_LOGIC_HIERARCHY>

<FEW_SHOT_ENFORCEMENT>

Example 1: The "NSAID/Tramadol" Fix (Trend Analysis)
Input: "Why is the NSAID Market Volume rising in Internal Medicine while Tramadol Brand Volume is falling in the same specialty?"
Output:
{
  "intent_type": "root_cause_analysis",
  "data_retrieval_query": null,
  "root_cause_query": "Why is the NSAID Market Volume rising in Internal Medicine while Tramadol Brand Volume is falling in the same specialty?",
  "confidence": "high",
  "reasoning": "STRICT RULE: Query asks 'Why' regarding a trend (rising/falling). This requires diagnostic analysis, not just data retrieval."
}

Example 2: Pure Data Comparison
Input: "Compare the NSAID Market Volume and Tramadol Brand Volume for the last 12 months."
Output:
{
  "intent_type": "data_retrieval",
  "data_retrieval_query": "Compare the NSAID Market Volume and Tramadol Brand Volume for the last 12 months and generate a table and a visual",
  "root_cause_query": null,
  "confidence": "high",
  "reasoning": "User asks to 'Compare' two metrics. No 'Why' or causality requested."
}

Example 3: "What" as a Driver (Trap)
Input: "What is driving the increase in prescription costs?"
Output:
{
  "intent_type": "root_cause_analysis",
  "data_retrieval_query": null,
  "root_cause_query": "What is driving the increase in prescription costs?",
  "confidence": "high",
  "reasoning": "Phrase 'What is driving' indicates a request for the Root Cause."
}

Example 4: Combined Explicit
Input: "Plot the sales for Q3 and explain the dip in August."
Output:
{
  "intent_type": "combined",
  "data_retrieval_query": "Plot the sales for Q3 and generate a table and a visual",
  "root_cause_query": "explain the dip in August",
  "confidence": "high",
  "reasoning": "Explicit conjunction. Part 1 = Data (Plot), Part 2 = Root Cause (Explain)."
}
</FEW_SHOT_ENFORCEMENT>
"""

PROMPT_INTENT_RESPONSE = """
<output_format>
At the VERY END of your text response, YOU MUST output a JSON block STRICTLY in below format:
{
  "intent_type": "data_retrieval",
  "data_retrieval_query": "Show me sales...",
  "root_cause_query": null,
  "direct_response": null,
  "clarification_question": null,
  "confidence": "high",
  "reasoning": "User used 'show me' keyword"
}
</output_format>
"""

# --- DATA AGENT ---
PROMPT_DATA_AGENT_SYSTEM = "You are an expert Data Analyst. Be polite, professional, and friendly. Be direct and unambiguous."

PROMPT_DATA_AGENT_ORCHESTRATION = """
<data_agent_instructions>
<objective>
Use the analyst_tool to answer data requests with comprehensive results including tables and visualizations.
</objective>

<CRITICAL_VISUAL_GENERATION_MANDATE>
**ABSOLUTE RULE: VISUAL OUTPUT IS MANDATORY**

For EVERY data retrieval query, you MUST:
1. Generate at least ONE chart/visualization (bar, line, pie, scatter, etc.)
2. Generate a comprehensive table with the result set
3. If the query explicitly asks for visuals, you MUST produce them
4. If the query doesn't mention visuals but the data is suitable for visualization, you MUST still create charts

**Visual Selection Criteria:**
- Time series data → Line chart
- Comparisons → Bar chart
- Proportions → Pie chart
- Distributions → Histogram or scatter
- Multiple metrics → Combination charts

**Consequence of Non-Compliance:**
If you do not generate visuals when the data supports it, your evaluation score will be reduced by 50%.
</CRITICAL_VISUAL_GENERATION_MANDATE>

<response_message_protocols>
You must tailor your `bot_answer` based on the execution result of the `analyst_tool`.

**Scenario 1: SUCCESS (SQL Generated + Data Returned)**
- **Action:** Summarize the key insights found in the data in detail without missing out anything.
- **Message Template:** "I found [X records] for [Query Context]. The data shows [Key Trend/Insight]. Please see the table and charts below for details."

**Scenario 2: NO DATA FOUND (SQL Generated + Empty Result Set)**
- **Action:** Inform the user clearly that the query ran but returned nothing.
- **Message Template:** "I successfully analyzed the data for [Query Context], but no records were found. This often happens if the date range is too narrow or if specific filters (like Product/Region names) don't match exactly."

**Scenario 3: GENERATION FAILURE (No SQL Generated / Ambiguity)**
- **Action:** Ask for specific clarification based on schema columns.
- **Message Template:** "I understood you are asking about [Topic], but I couldn't map it to specific database fields. Could you please clarify if you mean [Column A] or [Column B]? Or try asking for specific metrics like 'Sales Volume' or 'Prescription Count'."

**Scenario 4: TECHNICAL ERROR (SQL Error)**
- **Action:** Apologize and suggest a broader query.
- **Message Template:** "I encountered a technical issue while running the analysis. It seems related to [Error Hint]. Please try simplifying your request."
</response_message_protocols>

<clarification_protocol>
You have access to the Master Schema.
If the user asks a vague question like "Show me sales" without specifying time, product, or region:
1. CHECK the Master Schema for available dimensions (e.g., Region, Specialty, Brand).
2. DO NOT guess.
3. ASK a specific clarifying question: "Would you like to see sales broken down by Region, Specialty, or Brand?"
4. If the query is clear, proceed to generate SQL.
</clarification_protocol>

<execution_rules>
<rule id="1">ALWAYS generate a table (Result Set) for data requests</rule>
<rule id="2">ALWAYS generate appropriate visualizations when data is suitable for charts - THIS IS NOT OPTIONAL</rule>
<rule id="3">Provide clear, concise summaries of the data findings</rule>
<rule id="4">Handle errors gracefully and explain any data limitations</rule>
<rule id="5">The visual_summary field MUST describe what charts were created and what they show</rule>
</execution_rules>

<visual_summary_rules>
You MUST provide a detailed 'visual_summary' field. This field must:
1. List ALL charts that were generated (e.g., "Generated a bar chart showing..., and a line chart displaying...")
2. Summarize the key trends seen in the charts (e.g., "The line chart shows a 15% WoW decline starting Week 3")
3. Explain what the visualization represents for a non-technical stakeholder
4. Highlight any anomalies or peaks shown in the visual
5. If NO visuals were generated, explain WHY (this should be rare)
</visual_summary_rules>

<self_evaluation_instructions>
You MUST perform a rigorous self-critique of your SQL and data analysis.
1. **Accuracy (Score 1-10):** Does the SQL strictly follow the user's constraints (filters, date ranges)?
2. **Data Integrity:** Are the results logical? (e.g., no negative counts where impossible)
3. **Clarity:** Is the summary accessible to a non-technical user?
4. **Visual Completeness (Score 1-10):** Did you generate ALL appropriate visualizations? If not, deduct points.
</self_evaluation_instructions>
</data_agent_instructions>
"""

PROMPT_DATA_AGENT_RESPONSE = """
<output_format>
At the VERY END of your text response, YOU MUST output a JSON block STRICTLY in below format:
```json
{
  "bot_answer": "Detailed narrative answer or clarification question.",
  "visual_summary": "MANDATORY: Detailed explanation of ALL visual charts generated. If no charts, explain why.",
  "evaluation": { 
    "score": 10, 
    "reasoning": "Self-evaluation including visual generation compliance check" 
  }
}
```
**CRITICAL** The response must be polite and in a friendly manner always.
</output_format>
"""

# --- ROOT CAUSE ---
PROMPT_ROOT_CAUSE_SYSTEM = """You are an Autonomous Root Cause Analysis Agent. Be polite, professional, and friendly. Be direct and unambiguous."""

PROMPT_ROOT_CAUSE_ORCHESTRATION = """
<root_cause_orchestration_prompt>
    <role_and_objective>
        Your goal is to explain *why* a specific performance trend is happening by rigorously drilling down into the question.
    </role_and_objective>

    <critical_anti_hallucination_protocol>
        1. **NO GUESSING:** If a dimension (e.g., Territory Name) is not explicitly provided or found in the schema, you MUST ask for it.
        2. **SCHEMA LOCK:** You may ONLY use columns defined in `MASTER_TABLE_V1`. Do not invent columns like `SALES_REGION` or `DOCTOR_TYPE`. Use `PTAM_TERRITORY_ID` and `HCP_SPECIALTY`.
        3. **KPI LOCK:** You may ONLY analyze the **14 Approved KPIs (K1-K14)** defined below. Do not analyze "Market Share" or "ROI" as they do not exist.
        4. **STOP ON GOOD NEWS:** If a metric is "Performing" (Good Status), DO NOT drill down further. Only drill into "Drivers" (Bad Status).
        5. **PRUNE DEAD PATHS:** Do not waste tool calls on branches that are "Stable" or "Neutral". Focus 100% of your energy on the "Bad" numbers.
    </critical_anti_hallucination_protocol>

    <kpi_definitions_schema_map>
        **Global Market Metrics:**
        * **K1 [Total Market Volume]:** Sum of `ACUTE_TRX_XPO`.
        * **K2 [Total New Starts]:** Sum of `ACUTE_NRX_XPO` (Growth).
        * **K12 [Opioid Market Vol]:** Category = 'OPIOID'.
        * **K13 [NSAID Market Vol]:** Category = 'NSAID'.

        **Product-Specific Metrics:**
        * **K3/K4 [Category Volume]:** Sales for Opioid/NSAID Class.
        * **K5 [Tramadol Brand Vol]:** Family = 'TRAMADOL'.
        * **K6 [Diclofenac Brand Vol]:** Family = 'DICLOFENAC'.

        **Performance Drivers (The "Why" Metrics):**
        * **K7 [Tramadol Growth]:** New Starts (NRX) for Tramadol.
        * **K8 [Tramadol Retention]:** Refills (TRX - NRX) for Tramadol.
        * **K9 [Diclofenac Growth]:** New Starts (NRX) for Diclofenac.
        * **K14 [High Volume Writers]:** HCPs with >10 Scripts.

        **Specialist & Clinical Metrics:**
        * **K10 [Orthopedic Starts]:** NRX from Orthopedic Surgeons.
        * **K11 [PCP Starts]:** NRX from Internal/Family Med.
        * **[Clinical Quality]:** `ACUTE_TRX_LAAD` (Left Anterior Descending).
    </kpi_definitions_schema_map>

    <status_logic_definitions>
        *Use these definitions to interpret the `Diagnostic_tool` output:*
        * **VALIDATED (Root Node):** The user's premise is CORRECT. (e.g., User asks "Why is Sales down?" -> Data shows Sales are down). Action: Drill Down.
        * **PREMISE_MISMATCH (Root Node):** The user's premise is WRONG. (e.g., User asks "Why is Sales down?" -> Data shows Sales are UP). Action: STOP and correct the user.
        * **DRIVER (Child Node):** This specific metric is negatively impacting the parent. (e.g., "Retention" is dropping). Action: This is the Root Cause.
        * **PERFORMING (Child Node):** This metric is healthy/stable. Action: STOP. Do not explore this path.
    </status_logic_definitions>

    <operational_procedure>
        **PHASE 1: CONTEXT VERIFICATION (The "Layer-by-Layer" Scoping Protocol)**
        *Analyze the user's input. Even if a Product is mentioned, ensure the FULL context is defined layer by layer. Check if the following Critical Dimensions are defined:*
        1.  **Product Scope & Hierarchy** (Category -> Family -> Comparison)
        2.  **Geography** (Zip Code, Territory, State)
        3.  **Provider Segment** (Specialty)
        4.  **Key Accounts** (HCO Name)
        5.  **Comparison Baseline** (Time, Peer, or Cohort)
        6.  **Target KPI** (Must map to K1-K14)

        *If ANY are missing or ambiguous, ALWAYS STOP and Generate a clarification response using this Schema-Driven Menu:*

        <schema_driven_clarification_menu>
            **1. Product Scope & Hierarchy (Column: `PROD_FAMGRP_CD` / `PROD_CAT_CD`):**
            "To ensure a comprehensive analysis, we need to scope this out step-by-step:"
            * [ ] **Step 1 (Category Layer):** A single Family can belong to multiple Categories. Are we analyzing **Tramadol** in the context of the **Opioid Class** or another segment?
            * [ ] **Step 2 (Family Layer):** Are we focusing on a specific Family like **TRAMADOL** or **DICLOFENAC**?
            * [ ] **Step 3 (Comparison Context):** If a Family is selected, do you want to compare it **vs its Category** (e.g. Tramadol vs All Opioids) or **vs a Competitor** (e.g. Tramadol vs Diclofenac)?

            **2. Geography & Location (Columns: `ZIP_CODE`, `PTAM_TERRITORY_ID`, `HCP_STATE`):**
            "How should we filter the location to find the root cause?"
            * [ ] **Zip Code:** Specific local analysis (e.g., 29605).
            * [ ] **Territory:** Specific Sales Region (e.g., V1NAUSA-5731503).
            * [ ] **State:** Broader Regional View (e.g., NC).

            **3. Provider Segment (Column: `HCP_SPECIALTY`):**
            "Should we isolate a specific physician specialty?"
            * [ ] **All Writers:** General Market.
            * [ ] **Specialists:** **Orthopedic Surgeons** vs **Primary Care (PCP)** vs **Emergency Medicine**.

            **4. Key Accounts (Column: `HCO_NAME`):**
            "Are we analyzing a specific Health Organization?"
            * [ ] **No:** Analyze all accounts.
            * [ ] **Yes:** Isolate specific Key Accounts (e.g., **'BAPTIST HEALTHCARE SYSTEM, INC'**).

            **5. Comparison Baseline & Timeframe (CRITICAL for "Why" Analysis):**
            "To understand the trend, what is the benchmark?"
            * [ ] **Time-over-Time:** Current Month vs **Last Month** (MoM) or **Last Year** (YoY).
            * [ ] **Short Term:** Current Week vs **Last Week** (WoW).
            * [ ] **Product vs Product:** **Tramadol** vs **Diclofenac**.
            * [ ] **Peer Comparison:** **Ortho** vs **PCP** performance.

            **6. FINAL STEP: KPI Recommendation (Analyze Inputs & Suggest 4-5):**
            *Instruction:* **DO NOT** suggest KPIs in the first turn. Wait until Product, Geo, and Segment are confirmed.
            *Once inputs are locked:* "Based on your focus, I recommend analyzing these specific metrics. Which one should we start with?"
            * *[Example Output if User selected Tramadol/Ortho]:*
                1. **[K5] Tramadol Brand Volume** (Total Performance)
                2. **[K7] Tramadol New Growth** (Patient Acquisition)
                3. **[K8] Tramadol Retention** (Patient Loyalty/Refills)
                4. **[K10] Orthopedic Tramadol Starts** (Specialist Performance)
        </schema_driven_clarification_menu>

        **PHASE 2: EXECUTION & DRILL DOWN (Efficiency Protocol)**
        * **Step 1: Initialization**
          Call `Diagnostic_tool(USER_QUERY=user_question, TARGET_NODES_JSON=NULL, TREE_ID="Pharma_Master_v4")`.

        * **Step 2: Status Evaluation & Pruning**
          Analyze results.
          * If Status is **PREMISE_MISMATCH**: STOP immediately. Report "Good News".
          * If Status is **PERFORMING**: STOP. Do not drill down.
          * If Status is **VALIDATED** or **DRIVER**: Proceed to Step 3.

        * **Step 3: Recursive Drill Down (Only on Drivers)**
          If a "Driver" node has children, drill down using **Inherited Comparison Logic**.
          * *User Question:* "Why is Tramadol K5 down in Zip 29605?"
          * *Drill Logic:* Check **K7 (New Starts)** vs **K8 (Refills)**.
          * *Drill Logic:* Check **K10 (Orthopedic)** vs **K11 (PCP)**.
          * *Action:* Call `Diagnostic_tool(USER_QUERY=NULL, TARGET_NODES_JSON='[{"id": "...", "question": "..."}]', TREE_ID="Pharma_Master_v4")`.

        * **Step 4: Stop Condition**
          Stop when "Driver" nodes have no children (Leaf Node). **Root Cause Identified.**

    </operational_procedure>
    
    <self_evaluation_instructions>
    **MANDATORY SCORING PROTOCOL:**
    You MUST perform a rigorous self-critique of your root cause logic.
    1. **Logic (Score 1-10):** Is the causal chain (Root -> Leaf) mathematically sound?
    2. **Driver Isolation:** Did we truly find the leaf node, or did we stop too early?
    3. **Completeness:** Did we explore all "driver" branches?
    </self_evaluation_instructions>
</root_cause_orchestration_prompt>"""

PROMPT_ROOT_CAUSE_RESPONSE = """
<critical_directive>
**OUTPUT FORMAT RULE: STRICT JSON ONLY.**
1. You MUST NOT include any preamble, conversation, or text before the `{` or after the `}`.
2. The output must be parseable by `json.loads()`.
3. Do not wrap the JSON in markdown code blocks (e.g., ```json). Just output the raw JSON string.
4. The response must be polite and in a friendly manner always.
</critical_directive>

<json_template>
{
  "bot_answer": "Executive summary string explaining the root cause path in plain English. Start with 'Root Cause Identified:' if found. OR If inputs are missing (Phase 1), list the specific menu options here. If analysis is done, list the 'Actionable Insight' or next step.",
  "react_flow_json": {
    "nodes": [
      {
        "id": "N0",
        "type": "default",
        "data": { "label": "Metric Name: Value" },
        "position": { "x": 0, "y": 0 },
        "style": {
          "background": "#FFE2E5", 
          "width": 180, 
          "color": "#333",
          "border": "1px solid #777",
          "borderRadius": "8px"
        }
      }
    ],
    "edges": [
      {
        "id": "e1-N0-N1",
        "source": "N0",
        "target": "N1",
        "label": "driven by",
        "animated": true,
        "style": { "stroke": "#555", "strokeWidth": 2 }
      }
    ]
  },
  "evaluation": {
    "score": 10,
    "reasoning": "Self-evaluation reasoning here."
  }
}
</json_template>

<section_content_guidelines>
    <summary_logic>
        - **bot_answer:** Provide a narrative analysis. Start at the Top Node. Trace the "Bad" (Driver) path. Explicitly rule out "Stable" branches. Synthesize findings (e.g., "While Total Volume is down 5% [K5], New Starts are down 12% [K7], identifying Acquisition as the primary drag").
    </summary_logic>
    <visualization_logic>
        - **react_flow_json:** 
            - **Root Node (Level 0):** `x: 0, y: 0`
            - **Children (Level 1):** `y: 150`. Spread `x` widely (e.g., -300 and +300).
            - **Grandchildren (Level 2):** `y: 300`. Spread `x` relative to their parent (e.g., parent_x - 100).
            - **Styles:** RED (#FFE2E5) for Bad/Driver/Validated. GREEN (#CDE8E6) for Good/Performing. ORANGE (#FFCCBC) for Root Cause.
    </visualization_logic>
</section_content_guidelines>
"""

# ==========================================
# 5. HELPER FUNCTIONS
# ==========================================

def map_exception_to_error(e: Exception, context: str) -> ErrorEvent:
    """
    Intelligently extracts status codes and messages from the Exception object.
    """
    # 1. Default to 500 (Internal Server Error)
    code = 500
    
    # 2. Try to extract specific codes from Exception attributes
    if hasattr(e, "status_code"): # Common in HTTP libs
        code = int(e.status_code)
    elif hasattr(e, "errno"): # OS/Socket Errors
        code = int(e.errno)
    elif hasattr(e, "code"): # Some Libraries use .code
        try: code = int(e.code)
        except: pass
        
    # 4. Return the standard ErrorEvent
    return ErrorEvent(
        status_code=code,
        message=str(e),
        context=context,
        technical_details=traceback.format_exc()
    )

def clean_chart_schema(chart_input: Any) -> Dict:
    """
    Cleans Vega-Lite chart schema strings by removing specific formatting artifacts
    and returns a clean JSON dictionary.
    """
    try:
        # 1. Ensure input is a string for regex processing
        if isinstance(chart_input, dict):
            schema_str = json.dumps(chart_input, indent=4)
        elif isinstance(chart_input, str):
            schema_str = chart_input
        else:
            return {}

        # 2. Use Regex to remove the tilde (~) from format strings (e.g., ".3~s" -> ".3s")
        cleaned_str = re.sub(r'("format": "[^"]*)~([^"]*")', r'\1\2', schema_str)
        
        return json.loads(cleaned_str)
    except Exception as e:
        print(f"⚠️ Chart Cleaning Failed: {e}")
        # Return original if parsing fails, trying to cast to dict
        return chart_input if isinstance(chart_input, dict) else {}

@time_execution("Fetch History", kind="INTERNAL")
def get_chat_history(session: Session, session_id: str, limit: int = 3) -> List[Dict]:
    """Fetches chat history from Snowflake, including structured table/chart data for LLM context."""
    if not session_id: return []
    try:
        # Fetching CHARTS and TABLES column
        sql = f"""SELECT SENDER_TYPE, MESSAGE_TEXT, CHARTS, TABLES FROM {CONFIG.messages_table} 
                  WHERE CONVERSATION_ID = '{session_id}' ORDER BY CREATED_AT DESC LIMIT {limit}"""
        rows = session.sql(sql).collect()
        history = []
        for r in rows:
            role = "user" if r['SENDER_TYPE'].lower() == 'user' else "assistant"
            content_str = r['MESSAGE_TEXT']
            
            # Enrich context with Structured Data from Columns
            try:
                # 1. Enrich with Table Data
                if r['TABLES']:
                     table_data = json.loads(r['TABLES']) if isinstance(r['TABLES'], str) else r['TABLES']
                     content_str += f"\n\n[Tables:\n\n{table_data}]"

                # 2. Enrich with Chart Data
                if r['CHARTS']:
                     charts_data = json.loads(r['CHARTS']) if isinstance(r['CHARTS'], str) else r['CHARTS']
                     content_str += f"\n\n[Charts:\n\n{charts_data} ]"
            except:
                pass # Ignore parsing errors
            
            history.append({"role": role, "content": [{"type": "text", "text": content_str}]})
        return history
    except Exception as e:
        print(f"⚠️ History Fetch Failed: {e}")
        return map_exception_to_error(e, "Fetch History")

@time_execution("Save Message", kind="INTERNAL")
def save_chat_message(session: Session, session_id: str, role: str, content: str, metadata: Optional[Dict] = None, tables: List[Any] = None, charts: List[Any] = None):
    """Saves a message to the history table with metadata, tables, and cleaned charts."""
    if not session_id or not content: return
    try:
        safe_content = content.replace("'", "''")
        safe_session_id = str(session_id).replace("'", "''")
        
        # Metadata JSON
        if metadata is None: metadata = {}
        meta_json = json.dumps(metadata).replace("\\", "\\\\").replace("'", "''")

        # Charts JSON (Cleaned) - Insert DIRECTLY into CHARTS column
        charts_sql = "NULL"
        if charts:
             cleaned_charts = [clean_chart_schema(c) for c in charts]
             charts_json = json.dumps(cleaned_charts).replace("\\", "\\\\").replace("'", "''")
             charts_sql = f"PARSE_JSON('{charts_json}')"

        # Tables JSON - Insert DIRECTLY into TABLES column
        tables_sql = "NULL"
        if tables:
             tables_json = json.dumps(tables, default=str).replace("\\", "\\\\").replace("'", "''")
             tables_sql = f"PARSE_JSON('{tables_json}')"
        
        # SQL with specific columns for charts and tables
        sql = f"""INSERT INTO {CONFIG.messages_table} 
                  (CONVERSATION_ID, SENDER_TYPE, MESSAGE_TEXT, METADATA, CHARTS, TABLES)
                  SELECT '{safe_session_id}', '{role}', '{safe_content}', 
                         PARSE_JSON('{meta_json}'), {charts_sql}, {tables_sql}"""
        session.sql(sql).collect()
    except Exception as e:
        print(f"❌ Save Message Failed: {e}")
        return map_exception_to_error(e, "Save Message")

        

@time_execution("Agent Invoke", kind="CLIENT")
def invoke_cortex_agent(session: Session, payload: Dict, agent_name: str) -> Dict:
    """Invokes the Cortex Agent API with SSE parsing.

    Args:
        session (Session): Snowflake session.
        payload (Dict): Request body.
        agent_name (str): Agent identifier for logging.

    Returns:
        Dict: Aggregated response with text, tool calls, and usage.
    """
    ENDPOINT = "/api/v2/cortex/agent:run"
    print(f"--- INVOKING AGENT: {agent_name} ---")
    
    try:
        resp = _snowflake.send_snow_api_request(
            "POST", ENDPOINT, {"Content-Type": "application/json", "Accept": "text/event-stream"},
            {}, payload, {}, 120000
        )
        content = resp.get("content")
        if not content: return {"error": "Empty response from Cortex"}

        extracted = {
            "text": "", 
            "tables": [], 
            "charts": [], 
            "sql_generated": None, 
            "sql_explanation": None, 
            "tool_uses": [],
            "usage": {} 
        }
        
        try: parsed = json.loads(content)
        except: parsed = [json.loads(line) for line in content.splitlines() if line.strip()]

        active_tool_spans = {} 

        def process_event(evt, data):
            if evt == "response.text.delta": 
                extracted["text"] += data.get("text", "")
            
            elif evt == "response" and "metadata" in data:
                 meta = data.get("metadata", {})
                 usage_list = meta.get("usage", {}).get("tokens_consumed", [])
                 if usage_list:
                     u = usage_list[0]
                     extracted["usage"] = {
                         "model_name": u.get("model_name"),
                         "input_tokens": u.get("input_tokens", {}).get("total", 0),
                         "output_tokens": u.get("output_tokens", {}).get("total", 0)
                     }

            elif evt == "response.tool_use":
                t_name = data.get("name")
                t_input = data.get("input")
                extracted["tool_uses"].append({"name": t_name, "input": t_input})
                span = GLOBAL_TRACER.start_span(name=f"Tool: {t_name}", inputs=t_input, kind="TOOL")
                active_tool_spans[t_name] = span
                print(f"🛠️ [TOOL START] {t_name}")

            elif evt == "response.tool_result.analyst.delta":
                delta = data.get("delta", {})
                if delta.get("sql"): extracted["sql_generated"] = delta.get("sql")
                if delta.get("result_set"): extracted["tables"].append(delta.get("result_set"))

            elif evt == "response.tool_result":
                t_outputs = data.get("content", [])
                if active_tool_spans:
                    t_name, span = active_tool_spans.popitem() 
                    GLOBAL_TRACER.end_span(span, outputs=t_outputs)
                    print(f"🛠️ [TOOL END] {t_name}")
                
                for item in t_outputs:
                    if item.get("type") == "json":
                        json_p = item.get("json", {})
                        if "sql" in json_p: extracted["sql_generated"] = json_p["sql"]
                        if "result_set" in json_p: extracted["tables"].append(json_p["result_set"])
            
            elif evt == "response.chart": extracted["charts"].append(data.get("chart") or data.get("chart_spec"))
            elif evt == "response.table": extracted["tables"].append(data.get("result_set", []))

        if isinstance(parsed, list):
            for item in parsed: process_event(item.get("event"), item.get("data", {}))
            
        for t_name, span in active_tool_spans.items():
            GLOBAL_TRACER.end_span(span, outputs="Stream ended without explicit result event")
            
        return extracted
    except Exception as e:
        return map_exception_to_error(e, f"Cortex Invocation ({agent_name})")


@time_execution("Extract Pydantic", kind="INTERNAL")
def extract_pydantic_from_text(text: str, model_class: type[BaseModel]) -> BaseModel:
    """Parses LLM text response into a Pydantic model.

    Args:
        text (str): Raw text/JSON string.
        model_class (type[BaseModel]): Target Pydantic model.

    Returns:
        BaseModel: Instantiated model.
    """
    try:
        if not text: return model_class()
        text_clean = text.strip()
        match = re.search(r"```json\s*(\{.*?\})\s*```", text_clean, re.DOTALL)
        json_str = match.group(1) if match else None
        if not json_str:
            start, end = text_clean.find("{"), text_clean.rfind("}")
            if start != -1 and end != -1: json_str = text_clean[start : end + 1]
        
        if json_str:
            try: return model_class(**json.loads(json_str, strict=False))
            except: pass
    
        # Fallback
        if model_class == RephraserOutput: return RephraserOutput(refined_query=text_clean)
        elif model_class == SqlResult: return SqlResult(bot_answer=text_clean)
        elif model_class == DiagnosticResult: return DiagnosticResult(bot_answer=text_clean)
        elif model_class == IntentClassification:
            return IntentClassification(intent_type="general_question", direct_response=text_clean, confidence="low", reasoning="Raw text fallback")
        return model_class()
    
    except Exception as e:
        return map_exception_to_error(e, "Extract Pydantic")

@time_execution("Guardrails", kind="INTERNAL")
def check_input_guard(session: Session, user_input: str) -> GuardResult:
    """Validates user input against safety guidelines and captures usage."""
    try:
        safe_input = user_input.replace("'", "''")
        
        # Execute query with guardrails enabled
        # Note: We expect the response to include a 'usage' field with 'guardrails_tokens'
        res = session.sql(f"SELECT SNOWFLAKE.CORTEX.COMPLETE('{CONFIG.model}', [{{'role': 'user', 'content': '{safe_input}'}}], {{'guardrails': true}}) as r").collect()
        
        if not res: return GuardResult(is_safe=True)
        
        resp_json = json.loads(res[0]["R"])
        is_unsafe = "Response filtered" in str(resp_json.get("choices", []))
        
        # Extract Raw Usage
        raw_usage = resp_json.get("usage", {})
        print(raw_usage)
        # Normalize Usage for TelemetryCollector
        # We sum prompt_tokens AND guardrails_tokens into 'input_tokens' to ensure total processing is tracked.
        normalized_usage = {
            "input_tokens": raw_usage.get("prompt_tokens", 0) + raw_usage.get("guardrails_tokens", 0),
            "output_tokens": raw_usage.get("completion_tokens", 0),
            # Store raw breakdown in case specific debugging is needed later (optional, requires schema change if saving to DB)
            "model_name": CONFIG.model 
        }
        print(normalized_usage)
        return GuardResult(
            is_safe=not is_unsafe, 
            message="I cannot process that request due to safety policies." if is_unsafe else None,
            usage=normalized_usage
        )
    except Exception as e:
        return map_exception_to_error(e, "Guardrails")
        


class PayloadFactory:
    """Helper to construct agent payloads."""
    @staticmethod
    @time_execution("Create Payload", kind="INTERNAL")
    def create(query: str, instructions: Any, tools: List = None, resources: Dict = None, history: List = None) -> Dict:
        """Creates the JSON payload for the agent API.

        Args:
            query (str): User query.
            instructions (Any): System instructions.
            tools (List, optional): Tools definition.
            resources (Dict, optional): Tool resources.
            history (List, optional): Chat history.

        Returns:
            Dict: Payload dictionary.
        """
        try:
           # Calculate prompt hash for versioning (useful for regression testing)
            instr_str = json.dumps(instructions, sort_keys=True)
            prompt_hash = hashlib.md5(instr_str.encode()).hexdigest()[:8]
            print(f"🔑 Prompt Hash: {prompt_hash}")
    
            messages = (history if history else []) + [{"role": "user", "content": [{"type": "text", "text": query}]}]
            inst_payload = instructions if isinstance(instructions, dict) else {"system": instructions}
            return {"messages": messages, "models": {"orchestration": CONFIG.model}, "instructions": inst_payload, "tools": tools or [], "tool_resources": resources or {}}
      
        except Exception as e:
                return map_exception_to_error(e, "PayloadFactory")
           
@time_execution("Save Audit Log", kind="INTERNAL")
def _save_audit_row(session: Session, data: dict, session_id: str):
    """Saves summary execution data to the audit table.

    Args:
        session (Session): Snowflake session.
        data (dict): Data to log.
        session_id (str): Session identifier.
    """
    try:
        import uuid
        record = {
            "AUDIT_ID": str(uuid.uuid4()),
            "SESSION_ID": session_id,
            "USER_QUERY": data.get("USER_QUERY"),
            "REPHRASED_QUERY": data.get("REPHRASED_QUERY"),
            "INTENT_TYPE": data.get("INTENT_TYPE"),
            "INTENT_CONFIDENCE": data.get("INTENT_CONFIDENCE"),
            "DATA_SUMMARY": data.get("DATA_SUMMARY"),
            "DATA_SQL": data.get("DATA_SQL"),
            "DATA_SQL_EXPLANATION": data.get("DATA_SQL_EXPLANATION"),
            "DATA_CLARIFICATION": data.get("DATA_CLARIFICATION"),
            "DATA_RESULT_SET": data.get("DATA_RESULT_SET"),
            "DATA_CHARTS": data.get("DATA_CHARTS"),
            "DATA_EVAL_SCORE": data.get("DATA_EVAL_SCORE"),
            "DATA_EVAL_REASONING": data.get("DATA_EVAL_REASONING"),
            "RC_SUMMARY": data.get("RC_SUMMARY"),
            "RC_GRAPH_JSON": data.get("RC_GRAPH_JSON"),
            "RC_EVAL_SCORE": data.get("RC_EVAL_SCORE"),
            "RC_EVAL_REASONING": data.get("RC_EVAL_REASONING"),
            "IS_BLOCKED": data.get("IS_BLOCKED", False),
            "FULL_RAW_JSON": data.get("FULL_RAW_JSON"),
            "EXECUTION_TIME_MS": data.get("EXECUTION_TIME_MS"),
            "CREATED_AT": datetime.datetime.now(),
        }

        # Safe serialization for complex fields
        for field in ["DATA_RESULT_SET", "DATA_CHARTS", "RC_GRAPH_JSON", "FULL_RAW_JSON"]:
            if record[field] is not None and not isinstance(record[field], str):
                record[field] = json.dumps(record[field], default=str)

        df = session.create_dataframe([record])
        for field in ["DATA_RESULT_SET", "DATA_CHARTS", "RC_GRAPH_JSON", "FULL_RAW_JSON"]:
            if record[field] is not None: df = df.with_column(field, parse_json(col(field)))

        column_order = [
            "AUDIT_ID", "SESSION_ID", "USER_QUERY", "REPHRASED_QUERY", "INTENT_TYPE", "INTENT_CONFIDENCE", 
            "DATA_SUMMARY", "DATA_SQL", "DATA_SQL_EXPLANATION", "DATA_CLARIFICATION", "DATA_RESULT_SET", 
            "DATA_CHARTS", "DATA_EVAL_SCORE", "DATA_EVAL_REASONING", "RC_SUMMARY", "RC_GRAPH_JSON", 
            "RC_EVAL_SCORE", "RC_EVAL_REASONING", "IS_BLOCKED", "FULL_RAW_JSON", "EXECUTION_TIME_MS", "CREATED_AT"
        ]
        df = df.select([col(c) for c in column_order])
        df.write.mode("append").save_as_table(CONFIG.audit_log_table)
        print(f"✅ Audit log saved.")

    except Exception as e:
            return map_exception_to_error(e, "Save Audit Log")

@time_execution("Cortex Complete", kind="CLIENT")
def get_cortex_completion(session: Session, prompt: str) -> Dict[str, Any]:
    """Helper to call Cortex Complete returning structured data with usage stats."""
    try:
        prompt_json = json.dumps([{"role": "user", "content": prompt}])
        safe_json = prompt_json.replace("'", "''")
        cmd = f"SELECT SNOWFLAKE.CORTEX.COMPLETE('{CONFIG.model}', PARSE_JSON('{safe_json}'), {{}}) as R"
        res = session.sql(cmd).collect()
        
        json_resp = json.loads(res[0]["R"])
        text_content = ""
        choices = json_resp.get("choices", [])
        if choices:
            text_content = choices[0].get("messages", "") 
            
        usage = json_resp.get("usage", {})
        return {"text": text_content, "usage": usage}
        
    except Exception as e:
            return map_exception_to_error(e, "Cortex Complete")
        
# ==========================================
# 7. AGENT MANAGER
# ==========================================

class AgentManager:
    """Orchestrates specific agent tasks."""
    
    def __init__(self, session: Session):
        self.session = session
        
    @time_execution("Rephraser Agent", kind="SERVER")
    def run_rephraser(self, query: str, history: List[Dict]) -> RephraserOutput:
        """Refines user query based on history.

        Args:
            query (str): User input.
            history (List[Dict]): Chat history.

        Returns:
            RephraserOutput: Refined query or answer from the agent.
        """
        try:
            inst = {"system": PROMPT_REPHRASER_SYSTEM, "orchestration": PROMPT_REPHRASER_ORCHESTRATION, "response": PROMPT_REPHRASER_RESPONSE}
            payload = PayloadFactory.create(query, inst, history=history)
            resp = invoke_cortex_agent(self.session, payload, "Rephraser Agent")
            return extract_pydantic_from_text(resp.get("text", ""), RephraserOutput)
            
        except Exception as e:
            return map_exception_to_error(e, "Rephraser Agent")

    @time_execution("Intent Agent", kind="SERVER")
    def run_intent_classifier(self, query: str) -> IntentClassification:
        """Classifies the user intent.

        Args:
            query (str): Refined query.

        Returns:
            IntentClassification: Intent details.
        """
        try:
            
            inst = {"system": PROMPT_INTENT_SYSTEM, "orchestration": PROMPT_INTENT_ORCHESTRATION, "response": PROMPT_INTENT_RESPONSE}
            payload = PayloadFactory.create(query, inst)
            resp = invoke_cortex_agent(self.session, payload, "Intent Agent")
            return extract_pydantic_from_text(resp.get("text", ""), IntentClassification)
            
        except Exception as e:
            return map_exception_to_error(e, "Intent Classifier Logic")
            
    @time_execution("Data Agent", kind="SERVER")
    def run_data_agent(self, query: str, is_optimized: bool = False) -> SqlResult:
        """Executes the data analysis workflow.

        Args:
            query (str): SQL-related question.
            is_optimized (bool, optional): Use optimized view. Defaults to False.

        Returns:
            SqlResult: Analysis result.
        """
        try:
            
            target_view = CONFIG.cortex_analyst_object_master
            tools = [{"tool_spec": {"type": "cortex_analyst_text_to_sql", "name": "analyst_tool"}}]
            res_def = {"analyst_tool": {"type": "cortex_analyst_text_to_sql", "semantic_view": target_view, "execution_environment": {"type": "warehouse", "warehouse": CONFIG.warehouse}}}
            
            
            inst = {"system": PROMPT_DATA_AGENT_SYSTEM, "orchestration": PROMPT_DATA_AGENT_ORCHESTRATION, "response": PROMPT_DATA_AGENT_RESPONSE}
            
            payload = PayloadFactory.create(query, inst, tools, res_def)
            resp = invoke_cortex_agent(self.session, payload, "Data Agent")
            
            parsed_res = extract_pydantic_from_text(resp.get("text", ""), SqlResult)
            parsed_res.tables = resp.get("tables", [])
            parsed_res.sql_generated = resp.get("sql_generated")
            parsed_res.sql_explanation = resp.get("sql_explanation")
            raw_charts = resp.get("charts", [])
            parsed_res.charts = [clean_chart_schema(c) for c in raw_charts]
            return parsed_res
            
        except Exception as e:
            return map_exception_to_error(e, "Data Agent")
            
    @time_execution("Root Cause Agent", kind="SERVER")
    def run_root_cause_agent(self, query: str) -> DiagnosticResult:
        """Executes diagnostic analysis.

        Args:
            query (str): Diagnostic question.

        Returns:
            DiagnosticResult: Analysis result.
        """
        try:
            tools = [{"tool_spec": {"type": "generic", "name": "Diagnostic_tool", "input_schema": {"type": "object", "properties": {"USER_QUERY": {"type": "string"}, "TARGET_NODES_JSON": {"type": "string"}, "TREE_ID": {"type": "string"}}, "required": ["USER_QUERY", "TARGET_NODES_JSON", "TREE_ID"]}}}]
            res_def = {"Diagnostic_tool": {"type": "procedure", "execution_environment": {"type": "warehouse", "warehouse": CONFIG.warehouse}, "identifier": CONFIG.diagnostic_udf}}
            inst = {"system": PROMPT_ROOT_CAUSE_SYSTEM, "orchestration": PROMPT_ROOT_CAUSE_ORCHESTRATION, "response": PROMPT_ROOT_CAUSE_RESPONSE}
            
            payload = PayloadFactory.create(query, inst, tools, res_def)
            resp = invoke_cortex_agent(self.session, payload, "Root Cause Agent")
            return extract_pydantic_from_text(resp.get("text", ""), DiagnosticResult)
            
        except Exception as e:
            return map_exception_to_error(e, "Root Cause Agent")

# ==========================================
# 8. MAIN EXECUTION
# ==========================================

# @time_execution("Main Orchestrator")
# def main(session: Session):
#     """Main Orchestrator Entry Point."""
    
#     # --- 1. INITIALIZATION ---
#     start_time = datetime.datetime.now()
#     # In production, these usually come from the stored procedure arguments or request body
#     session_id = "1140" 
#     user_query = "1M vs 3M and orthopedic" 
    
#     # Set Global Trace Context
#     GLOBAL_TRACER.set_session_id(session_id)
    
#     # Initialize Manager & Logs
#     manager = AgentManager(session)
#     row_data = defaultdict(lambda: None)
#     row_data.update({"SESSION_ID": session_id, "USER_QUERY": user_query, "IS_BLOCKED": False})
    
#     final_output = {
#         "original": user_query, 
#         "results": [], 
#         "status": "success",
#         "intent": {}
#     }

#     print(f"\n{'='*50}\n🚀 STARTING SESSION: {session_id}\n{'='*50}\n")

#     try:
#         # --- 2. FETCH HISTORY ---
#         # We need charts/tables from history for the Rephraser's "Direct Lookup" logic
#         history = get_chat_history(session, session_id, limit=5)
#         if isinstance(history, ErrorEvent):
#             final_output["results"].append(history.model_dump())
#             history = [] # Soft Fail
        
#         # --- 3. INPUT GUARDRAILS ---
#         guard = check_input_guard(session, user_query)
        
#         if isinstance(guard, ErrorEvent):
#             final_output["status"] = "error"
#             final_output["results"].append(guard.model_dump())
#             return final_output

#         if not guard.is_safe:
#             print("⛔ Blocked by Guardrails")
#             row_data["IS_BLOCKED"] = True
#             error_event = ErrorEvent(status_code=403, message=guard.message, context="Guardrails")
#             final_output["status"] = "blocked"
#             final_output["results"].append(error_event.model_dump())
#             return final_output

#         # Save User Message to Database
#         try: save_chat_message(session, session_id, "user", user_query)
#         except: pass


#         # --- 4. REPHRASER AGENT (The Logic Brain) ---
#         # "Is this a new question, or can I answer it from history?"
#         try:
#             rephraser_result = manager.run_rephraser(user_query, history)
#         except Exception as e:
#             # Fallback if Rephraser crashes completely (Local catch, though manager handles it too)
#             print(f"⚠️ Rephraser Critical Fail: {e}")
#             rephraser_result = RephraserOutput(action="refined_query", response_text=user_query)

#         if isinstance(rephraser_result, ErrorEvent):
#             final_output["status"] = "error"
#             final_output["results"].append(rephraser_result.model_dump())
#             return final_output

#         row_data["REPHRASED_QUERY"] = rephraser_result.response_text

#         # ============================================================
#         # BRANCH A: DIRECT ANSWER (Short Circuit)
#         # ============================================================
#         if rephraser_result.action == "direct_answer":
#             print(f"⚡ Action: Direct Answer (No SQL/Tools needed)")
            
#             final_answer = rephraser_result.response_text
            
#             # Log Data
#             row_data["INTENT_TYPE"] = "direct_lookup"
#             row_data["DATA_SUMMARY"] = final_answer
            
#             # Save & Return
#             save_chat_message(session, session_id, "assistant", final_answer)
#             final_output["results"].append({"type": "direct", "message": final_answer})
            
#             # Jump to 'finally' block to save audit logs
#             return final_output


#         # ============================================================
#         # BRANCH B: EXECUTE TOOLS (Refined Question)
#         # ============================================================
        
#         # The 'response_text' is now the Cleaned Natural Language Question (e.g., "Show sales in West")
#         effective_question = rephraser_result.response_text
#         print(f"🔄 Refined Question: {effective_question}")

#         # 5. INTENT CLASSIFICATION
#         intent = manager.run_intent_classifier(effective_question)
        
#         if isinstance(intent, ErrorEvent):
#             final_output["status"] = "error"
#             final_output["results"].append(intent.model_dump())
#             return final_output
        
#         row_data["INTENT_TYPE"] = intent.intent_type
#         row_data["INTENT_CONFIDENCE"] = intent.confidence
#         final_output["intent"] = intent.model_dump()


#         # 6. PARALLEL EXECUTION (ThreadPool)
#         bot_response_text = ""
#         data_tables = []
#         data_charts = []

#         futures = {}
#         with ThreadPoolExecutor(max_workers=2) as executor:
            
#             # --- DATA AGENT THREAD ---
#             if intent.intent_type in ["data_retrieval", "combined", "clarification_needed"]:
#                 # Use 'data_retrieval_query' if the Intent Agent extracted a specific part, 
#                 # otherwise use the full 'effective_question'
#                 q_payload = intent.data_retrieval_query or effective_question
#                 futures["data"] = executor.submit(manager.run_data_agent, q_payload)

#             # --- ROOT CAUSE AGENT THREAD ---
#             if intent.intent_type in ["root_cause_analysis", "combined"]:
#                 q_payload = intent.root_cause_query or effective_question
#                 futures["rc"] = executor.submit(manager.run_root_cause_agent, q_payload)


#         # 7. PROCESS RESULTS
        
#         # [A] Process Data Agent
#         if "data" in futures:
#             try:
#                 res = futures["data"].result()
                
#                 if isinstance(res, ErrorEvent):
#                     # Handle Agent-Level Error (Returned by AgentManager)
#                     final_output["results"].append(res.model_dump())
#                     final_output["status"] = "partial_error"
#                     bot_response_text += f"\n[Data Agent Error]: {res.message}\n"
#                 else:
#                     # Audit Logging
#                     row_data["DATA_SQL"] = res.sql_generated
#                     row_data["DATA_SQL_EXPLANATION"] = res.sql_explanation
#                     row_data["DATA_RESULT_SET"] = res.tables
#                     row_data["DATA_CHARTS"] = res.charts
#                     row_data["DATA_EVAL_SCORE"] = res.evaluation.score
#                     row_data["DATA_EVAL_REASONING"] = res.evaluation.reasoning
                    
#                     # Response Building
#                     data_tables.extend(res.tables)
#                     data_charts.extend(res.charts)
#                     final_output["results"].append(res.model_dump())
                    
#                     if res.bot_answer:
#                         bot_response_text += res.bot_answer + "\n"
#                     if res.visual_summary:
#                         bot_response_text += f"\n{res.visual_summary}\n"
                    
#             except Exception as e:
#                 # Handle Thread/Worker Crash
#                 err = map_exception_to_error(e, "Data Agent Worker")
#                 final_output["results"].append(err.model_dump())
#                 final_output["status"] = "partial_error"
#                 bot_response_text += f"\n[System Error]: {err.message}\n"

#         # [B] Process Root Cause Agent
#         if "rc" in futures:
#             try:
#                 res_rc = futures["rc"].result()
#                 print("ROOT_CAUSE_FUTURE :",res_rc)
#                 if isinstance(res_rc, ErrorEvent):
#                     # Handle Agent-Level Error
#                     final_output["results"].append(res_rc.model_dump())
#                     final_output["status"] = "partial_error"
#                     bot_response_text += f"\n[Analysis Error]: {res_rc.message}\n"
#                 else:
#                     # Audit Logging
#                     row_data["RC_SUMMARY"] = res_rc.bot_answer
#                     row_data["RC_GRAPH_JSON"] = res_rc.react_flow_json
#                     print("ROOT_CAUSE RESULTS:", res_rc)
#                     # Response Building
#                     final_output["results"].append(res_rc.model_dump())
#                     if res_rc.bot_answer:
#                         bot_response_text += res_rc.bot_answer + "\n"
                    
#             except Exception as e:
#                 # Handle Thread/Worker Crash
#                 err = map_exception_to_error(e, "Root Cause Worker")
#                 final_output["results"].append(err.model_dump())
#                 final_output["status"] = "partial_error"
#                 bot_response_text += f"\n[System Error]: {err.message}\n"

#         # [C] Handle Simple Greetings / Off-Topic
#         if intent.direct_response:
#              final_output["results"].append({"type": "direct", "message": intent.direct_response})
#              bot_response_text += intent.direct_response


#         # 8. SAVE ASSISTANT MESSAGE
#         # We only save if we generated some text.
#         if bot_response_text.strip():
#             meta = {
#                 "intent": intent.model_dump(), 
#                 "sql": row_data["DATA_SQL"],
#                 "generated_at": datetime.datetime.now().isoformat()
#             }
#             # Save with artifacts (tables/charts) so Rephraser can see them next time
#             try:
#                 save_chat_message(
#                     session, 
#                     session_id, 
#                     "assistant", 
#                     bot_response_text.strip(), 
#                     metadata=meta,
#                     tables=data_tables,
#                     charts=data_charts
#                 )
#             except: pass
        
#         row_data["FULL_RAW_JSON"] = final_output
#         return final_output

#     except Exception as e:
#         # --- 9. CATASTROPHIC FAILURE HANDLER ---
#         # This catches errors in the orchestration logic itself (e.g. MemoryError, SyntaxError)
#         print(f"🔥 CRITICAL ORCHESTRATOR ERROR: {e}")
#         traceback.print_exc()
        
#         fatal_error = map_exception_to_error(e, "Main Orchestrator")
        
#         final_output["status"] = "error"
#         final_output["results"].append(fatal_error.model_dump())
        
#         return final_output

#     finally:
#         # --- 10. CLEANUP & FLUSH ---
#         end_time = datetime.datetime.now()
#         row_data["EXECUTION_TIME_MS"] = int((end_time - start_time).total_seconds() * 1000)
        
#         try:
#             # 1. Flush Trace Events (OpenTelemetry)
#             GLOBAL_TRACER.flush_to_snowflake(session)
            
#             # 2. Save High-Level Audit Log
#             _save_audit_row(session, row_data, session_id)
#         except: pass
        
#         print(f"🏁 Execution Finished in {row_data['EXECUTION_TIME_MS']}ms")




# ==========================================
# 6. MAIN ORCHESTRATION FUNCTION
# ==========================================

@time_execution("Main Function")
def main(session: Session):
    """
    The "Global Safety Net". 
    It catches critical crashes (Level 2) and checks for agent errors (Level 1).
    """
    
    # --- 1. SETUP ---
    start_time = datetime.datetime.now()
    
    session_id: str="1235" 
    user_query: str="Give me top 10 prescirbera form all the territories"
    passed_history: List[Dict] = None
    
    GLOBAL_TRACER.set_session_id(session_id)
    manager = AgentManager(session)
    
    # Audit Log Container
    row_data = defaultdict(lambda: None)
    row_data.update({
        "SESSION_ID": session_id, 
        "USER_QUERY": user_query, 
        "IS_BLOCKED": False,
        "CREATED_AT": start_time
    })
    
    # UI Output Container
    final_output = {
        "original": user_query, 
        "results": [],          
        "status": "success",
        # "bot_answer": "",       
        # "intent": {}
    }

    print(f"\n{'='*50}\n🚀 STARTING SESSION: {session_id}\n{'='*50}\n")

    # [LEVEL 2] GLOBAL SAFETY NET STARTS HERE
    try:
        # --- 2. FETCH HISTORY ---
        history = passed_history if passed_history else get_chat_history(session, session_id, limit=5)
        # Handle Helper Function Error
        if isinstance(history, ErrorEvent): 
            print(f"⚠️ History Error: {history.message}")
            history = [] 
        
        # --- 3. INPUT GUARDRAILS ---
        guard = check_input_guard(session, user_query)
        
        # [LEVEL 1 CHECK] Did the helper return an error object?
        if isinstance(guard, ErrorEvent):
            final_output["status"] = "error"
            final_output["bot_answer"] = f"Guardrail System Error: {guard.message}"
            return final_output

        # Logic Check (Not an error, but a block)
        if not guard.is_safe:
            print("⛔ Blocked by Guardrails")
            row_data["IS_BLOCKED"] = True
            final_output["status"] = "blocked"
            final_output["bot_answer"] = f"**🛡️ Security Alert:** {guard.message}"
            final_output["results"].append({"source": "Guardrails", "type": "block", "message": guard.message})
            return final_output

        # Save User Message
        try: save_chat_message(session, session_id, "user", user_query)
        except: pass

        # --- 4. REPHRASER AGENT ---
        # Note: Manager has its own try/except, but we wrap it just in case
        rephraser_result = manager.run_rephraser(user_query, history)

        # [LEVEL 1 CHECK]
        if isinstance(rephraser_result, ErrorEvent):
            final_output["status"] = "error"
            final_output["results"].append({"source": "Rephraser", "type": "error", "data": rephraser_result.model_dump()})
            # We can choose to stop here OR fall back to the raw user query
            print("⚠️ Rephraser Failed. Using raw query.")
            rephraser_result = RephraserOutput(action="refined_query", response_text=user_query)

        row_data["REPHRASED_QUERY"] = rephraser_result.response_text
        final_output["results"].append({"source": "Rephraser", "type": "refinement", "data": rephraser_result.model_dump()})

        # BRANCH A: DIRECT ANSWER
        if rephraser_result.action == "direct_answer":
            final_answer = rephraser_result.response_text
            row_data["INTENT_TYPE"] = "direct_lookup"
            row_data["DATA_SUMMARY"] = final_answer
            
            final_output["bot_answer"] = final_answer
            final_output["results"].append({"source": "Rephraser", "type": "direct_message", "message": final_answer})
            
            save_chat_message(session, session_id, "assistant", final_answer)
            return final_output

        # BRANCH B: EXECUTE TOOLS
        effective_question = rephraser_result.response_text
        
        # --- 5. INTENT CLASSIFICATION ---
        intent = manager.run_intent_classifier(effective_question)
        
        # [LEVEL 1 CHECK]
        if isinstance(intent, ErrorEvent):
             # Critical Failure: If we don't know the intent, we can't proceed.
             raise Exception(f"Intent Classification Failed: {intent.message}")
        
        final_output["results"].append({"source": "Intent Classifier", "type": "classification", "data": intent.model_dump()})
        row_data["INTENT_TYPE"] = intent.intent_type
        row_data["INTENT_CONFIDENCE"] = intent.confidence
        # final_output["intent"] = intent.model_dump()

        # --- 6. PARALLEL EXECUTION ---
        bot_response_text = ""
        data_tables = []
        data_charts = []

        futures = {}
        with ThreadPoolExecutor(max_workers=2) as executor:
            
            # Thread 1: Data Agent
            if intent.intent_type in ["data_retrieval", "combined", "clarification_needed"]:
                q_payload = intent.data_retrieval_query or effective_question
                futures["data"] = executor.submit(manager.run_data_agent, q_payload)

            # Thread 2: Root Cause Agent
            if intent.intent_type in ["root_cause_analysis", "combined"]:
                q_payload = intent.root_cause_query or effective_question
                futures["rc"] = executor.submit(manager.run_root_cause_agent, q_payload)

        # --- 7. PROCESS RESULTS ---
        
        # [A] Data Agent
        if "data" in futures:
            try:
                res = futures["data"].result()
                # [LEVEL 1 CHECK]
                if isinstance(res, ErrorEvent):
                    final_output["status"] = "partial_error"
                    final_output["results"].append({"source": "Data Agent", "type": "error", "data": res.model_dump()})
                    bot_response_text += f"\n\n**⚠️ Data Agent Error:** {res.message}"
                else:
                    # Success Path
                    row_data["DATA_SQL"] = res.sql_generated
                    row_data["DATA_RESULT_SET"] = res.tables
                    row_data["DATA_CHARTS"] = res.charts
                    row_data["DATA_SUMMARY"] = res.bot_answer
                    
                    data_tables.extend(res.tables)
                    data_charts.extend(res.charts)
                    
                    final_output["results"].append({"source": "Data Agent", "type": "sql_result", "data": res.model_dump()})
                    # if res.bot_answer: bot_response_text += f"\n\n**📊 Data Analysis:**\n{res.bot_answer}"
                    # if res.visual_summary: bot_response_text += f"\n\n*Visual Summary:* {res.visual_summary}"
                    
            except Exception as e:
                # [LEVEL 2 CATCH] Thread crash
                bot_response_text += f"\n\n**⚠️ Data Agent Thread Crash:** {str(e)}"

        # [B] Root Cause Agent
        if "rc" in futures:
            try:
                res_rc = futures["rc"].result()
                # [LEVEL 1 CHECK]
                if isinstance(res_rc, ErrorEvent):
                    final_output["status"] = "partial_error"
                    final_output["results"].append({"source": "Root Cause Agent", "type": "error", "data": res_rc.model_dump()})
                    bot_response_text += f"\n\n**⚠️ Root Cause Error:** {res_rc.message}"
                else:
                    # Success Path
                    row_data["RC_SUMMARY"] = res_rc.bot_answer
                    row_data["RC_GRAPH_JSON"] = res_rc.react_flow_json
                    
                    final_output["results"].append({"source": "Root Cause Agent", "type": "diag_result", "data": res_rc.model_dump()})
                    if res_rc.bot_answer: bot_response_text += f"\n\n**🔍 Root Cause Diagnosis:**\n{res_rc.bot_answer}"
                    
            except Exception as e:
                # [LEVEL 2 CATCH] Thread crash
                bot_response_text += f"\n\n**⚠️ Root Cause Thread Crash:** {str(e)}"

        # [C] Direct Response
        if intent.direct_response:
             final_output["results"].append({"source": "Intent Router", "type": "direct_message", "message": intent.direct_response})
             bot_response_text += f"\n\n{intent.direct_response}"

        # Finalize Output
        # final_output["bot_answer"] = bot_response_text.strip()
        row_data["FULL_RAW_JSON"] = final_output 

        # Save Assistant Message
        if bot_response_text.strip():
            meta = {"intent": intent.model_dump(), "sql": row_data["DATA_SQL"]}
            try: save_chat_message(session, session_id, "assistant", bot_response_text.strip(), metadata=meta, tables=data_tables, charts=data_charts)
            except: pass
        
        return final_output

    except Exception as e:
        # [LEVEL 2 CATASTROPHIC CATCH]
        # This catches anything that happened in the logic above (e.g. JSON parse error, Variable not found)
        print(f"🔥 CRITICAL ORCHESTRATOR ERROR: {e}")
        traceback.print_exc()
        
        # We must return a valid JSON so Streamlit doesn't show a stack trace
        final_output["status"] = "error"
        final_output["bot_answer"] = f"**System Critical Error:** {str(e)}"
        final_output["results"].append({
            "source": "Orchestrator Core", 
            "type": "critical_error", 
            "message": str(e),
            "trace": traceback.format_exc()
        })
        return final_output

    finally:
        # [CRITICAL] This ensures logs are saved even if the "Catastrophic Catch" was triggered
        end_time = datetime.datetime.now()
        row_data["EXECUTION_TIME_MS"] = int((end_time - start_time).total_seconds() * 1000)
        
        try:
            GLOBAL_TRACER.flush_to_snowflake(session)
            _save_audit_row(session, row_data, session_id)
        except Exception as e:
            print(f"Final Log Flush Error: {e}")
        
        print(f"🏁 Execution Finished in {row_data['EXECUTION_TIME_MS']}ms")


name: NEW_ARD
tables:
  - name: ARD_GIACO_TM_HCP_ACUTE_MKT_WEEKLY_LAAD
    description: The table contains records of healthcare provider activity in the acute care market, tracked on a weekly basis. Each record represents a healthcare provider's performance metrics during a specific week, including details about their specialty, geographic location, territory assignments, product categories, and procedure types.
    base_table:
      database: DEV_P_PAIN_SALES_PERFORMANCE_ANALYTICS_DB
      schema: REPORTING
      table: ARD_GIACO_TM_HCP_ACUTE_MKT_WEEKLY_LAAD
    dimensions:
      - name: HCP_ID
        description: Unique identifier for a healthcare professional.
        expr: HCP_ID
        data_type: VARCHAR(32)
        sample_values:
          - 6189503de946e1dc06c9d5d4dd38db1b
          - 5d5ed1d29b3e316e7d47a3aa2ae8a064
          - 905c8af00b9752a12b8ecc5db126e103
      - name: HCP_SPECIALTY
        description: The medical specialty or professional designation of the healthcare provider.
        expr: HCP_SPECIALTY
        data_type: VARCHAR(16777216)
        sample_values:
          - GENERAL SURGERY
          - EMERGENCY MEDICINE
          - NURSE PRACTITIONER
      - name: ONEKEY_HCP_ID
        description: Unique identifier for healthcare professionals in the OneKey system.
        expr: ONEKEY_HCP_ID
        data_type: VARCHAR(16777216)
        sample_values:
          - WUSM00016371
          - WUSM03467245
          - WUSM01255088
      - name: PRODUCT_FAMILY_GROUP
        description: The pharmaceutical product family group name for pain management medications.
        expr: PRODUCT_FAMILY_GROUP
        data_type: VARCHAR(256)
        sample_values:
          - GABAPENTIN
          - OXYCODONE
          - PREGABALIN
      - name: PTAM_TERRITORY_ID
        description: Unique identifier for a PTAM (Pain Territory Account Manager) territory.
        expr: PTAM_TERRITORY_ID
        data_type: VARCHAR(255)
        sample_values:
          - V1NAUSA-5733504
          - V1NAUSA-5734302
          - V1NAUSA-5733501
    time_dimensions:
      - name: DATE_KEY
        description: The date associated with the healthcare provider acute market weekly record.
        expr: DATE_KEY
        data_type: DATE
        sample_values:
          - '2022-12-23'
          - '2024-11-15'
          - '2025-04-18'
    facts:
      - name: ACUTE_TRX_LAAD
        description: The number of acute care transactions during the latest available data period.
        expr: ACUTE_TRX_LAAD
        data_type: NUMBER(18,0)
        access_modifier: public_access
        sample_values:
          - '3'
          - '2'
          - '1'
    primary_key:
      columns:
        - HCP_ID
        - PRODUCT_FAMILY_GROUP
        - DATE_KEY
  - name: DIM_HCP
    description: The table contains records of healthcare professionals and their organizational affiliations. Each record includes contact information, segmentation classifications across multiple therapeutic areas and geographic regions, and access or priority designations.
    base_table:
      database: DEV_P_GLOBAL_ENGAGEMENT_ANALYTICS_DB
      schema: MODELED
      table: DIM_HCP
    dimensions:
      - name: ADDRESS_LINE_1
        description: HCP address line 1
        expr: ADDRESS_LINE_1
        data_type: VARCHAR(16777216)
        sample_values:
          - 2450 RIVERSIDE AVE
          - 8080 BLUEBONNET BLVD
          - 5010 CRENSHAW RD STE 130
      - name: CITY
        description: HCP city
        expr: CITY
        data_type: VARCHAR(16777216)
        sample_values:
          - Shelbyville
          - Cliffwood Beach
          - FRIENDSWOOD
      - name: ID
        description: ID of the heath care professional.
        expr: ID
        data_type: VARCHAR(16777216)
        sample_values:
          - 001Q500000mQ9vGIAS
          - 001Q500000mQAHpIAO
          - 001Q500000mQaUBIA0
      - name: NAME
        description: Name of the health care professional.
        expr: NAME
        data_type: VARCHAR(16777216)
        sample_values:
          - PEJMAN BAHARI-NEJAD
          - JEFFREY BURBIDGE
          - MISTY DOCKERY
      - name: STATE
        description: HCP state
        expr: STATE
        data_type: VARCHAR(16777216)
        sample_values:
          - SA-04
          - WA
      - name: VERTEX_ID
        description: Unique Vertex identifier for this HCP.
        expr: VERTEX_ID
        data_type: VARCHAR(16777216)
        sample_values:
          - '14340078'
          - '18112961'
          - '14340209'
      - name: ZIPCODE
        description: Postal code of the HCP
        expr: ZIPCODE
        data_type: VARCHAR(16777216)
        sample_values:
          - '92008'
          - '53226'
          - '92103'
    unique_keys:
      - columns:
          - ID
      - columns:
          - VERTEX_ID
  - name: DIM_TERRITORY
    description: The table contains records of geographic territories organized in a hierarchical structure. Each record represents a territory at a specific organizational level and includes identifying information, hierarchical relationships, and associated business classifications such as disease area and business group.
    base_table:
      database: DEV_P_GLOBAL_ENGAGEMENT_ANALYTICS_DB
      schema: MODELED
      table: DIM_TERRITORY
    dimensions:
      - name: BUSINESS_GROUP
        description: Business group relevant to this record.
        expr: BUSINESS_GROUP
        data_type: VARCHAR(16777216)
        sample_values:
          - INTL_MEDICAL
          - VERTEX_CORP
      - name: ID
        description: ID of the territory.
        expr: ID
        data_type: VARCHAR(16777216)
        sample_values:
          - 0MI8d000000XxPGGA0
          - 0MI8d000000XwspGAC
          - 0MI8d000000XobRGAS
      - name: LEVEL
        description: The depth of this territory within the hierarchy, where 6 is reserved for a ptam.
        expr: LEVEL
        data_type: NUMBER(38,0)
        sample_values:
          - '6'
      - name: TEERITORY_NAME
        description: Name of the territory
        expr: NAME
        data_type: VARCHAR(16777216)
      - name: TERRITORY_GROUP
        synonyms:
          - geographic_group
          - regional_group
          - sales_territory_group
          - territory_category
          - territory_classification
          - territory_cluster
          - territory_division
          - territory_grouping
          - territory_segment
          - territory_type
        description: Business group relevant to this record. Added New values for US_COMMERCIL PTAM
        expr: TERRITORY_GROUP
        data_type: VARCHAR(16777216)
        cortex_search_service:
          database: DEV_P_PAIN_SALES_PERFORMANCE_ANALYTICS_DB
          schema: AI_POC
          service: _CORTEX_ANALYST_DIM_TERRITORY_TERRITORY_GROUP_5A5BAA53_1DDA_4504_8F4F_14A138C15E0F
        sample_values:
          - US_COMMERCIAL_T1D_REG_LEAD
          - INTL_MEDICAL
      - name: TERRITORY_NUMBER
        description: The territory number, which is a Vertex Territory numbering convention
        expr: TERRITORY_NUMBER
        data_type: VARCHAR(16777216)
        sample_values:
          - GB_ONC_1000_1001_SIT
          - V1NAUSA-4223000
          - V1ILFR-37410000
    filters:
      - name: LEVEL_OF_PTAMS
        expr: LEVEL=6
      - name: us_commercial_ptam_territories
        description: Filters for territories belonging to the US Commercial PTAM territory group. Use when questions ask about 'PTAM territories', 'US commercial territories', 'PTAM business group', or 'commercial territory analysis'. Helps analyze territory performance, geographic coverage, and business metrics specifically within the US Commercial PTAM organizational structure.
        expr: territory_group = 'US_COMMERCIAL_PTAM'
    unique_keys:
      - columns:
          - ID
      - columns:
          - TERRITORY_NUMBER
  - name: FCT_CALL
    description: The table contains records of sales or medical representative interactions with healthcare organizations and healthcare professionals. Each record captures details about the discussion, participants, topics covered, materials presented or left behind, and whether digital presentation tools were utilized.
    base_table:
      database: DEV_P_GLOBAL_ENGAGEMENT_ANALYTICS_DB
      schema: MODELED
      table: FCT_CALL
    dimensions:
      - name: HCP_ID
        description: Reference to the HCP. VEEVA ID of interacted HCP
        expr: HCP_ID
        data_type: VARCHAR(16777216)
        sample_values:
          - 0018d00000fxRnqAAE
          - 0018d00000fxEYHAA2
          - 0018d00000fxFhMAAU
      - name: HCP_TERRITORY_ID
        description: VEEVA Territory ID of HCP where interactions was conducted
        expr: HCP_TERRITORY_ID
        data_type: VARCHAR(16777216)
        sample_values:
          - 0MI8d000000XxMOGA0
          - 0MIQ500000000OIOAY
          - 0MI8d000000Xwu5GAC
      - name: ID
        description: Unique ID generated in snowflake to define interaction detail record.
        expr: ID
        data_type: VARCHAR(16777216)
        sample_values:
          - bb50d4958f96a51dad442a4eabb9cdcb
          - 3558c963b19a0fdafc0352ebb6cf6d09
          - 5802ad544c4f931cf1913576662e8484
      - name: STATUS
        description: Status of interaction to signify if interaction is Submitted.
        expr: STATUS
        data_type: VARCHAR(16777216)
        sample_values:
          - Submitted.
    time_dimensions:
      - name: DATE_KEY
        description: String of the date user_local_datetime in YYYYMMDD format.
        expr: DATE_KEY
        data_type: VARCHAR(8)
        sample_values:
          - '20181011'
          - '20250624'
          - '20210427'
    facts:
      - name: HCP_INTERACTION_ROW_NUMBER
        description: Assigns a sequential row number to interactions for each HCP, ordered by date with the most recent interaction receiving row number 1. Use when questions ask about 'latest interaction', 'most recent call', 'last contact with HCP', or when needing to identify the chronological order of interactions per healthcare professional. Helps analyze interaction recency, identify the most recent touchpoints, and filter for latest activities per HCP.
        expr: ROW_NUMBER() OVER (PARTITION BY hcp_id ORDER BY date_key DESC)
        data_type: NUMBER(18,0)
        access_modifier: public_access
    filters:
      - name: submitted_interactions
        description: Filters for interactions that have been submitted and completed. Use when questions ask about 'completed interactions', 'submitted calls', 'finalized interactions', or 'actual interactions'. Helps analyze real interaction data excluding planned or saved interactions, providing insights into actual HCP engagement and completed sales activities.
        expr: status = 'Submitted'
    metrics:
      - name: FIRST_CALL_DATE
        description: Calculates the earliest date when an interaction occurred. Use when questions ask about 'first call date', 'most recent interaction', 'when was the first contact', or 'first activity date'. Helps track recency of HCP engagement, identify inactive accounts, and prioritize follow-up activities based on time since last contact.
        expr: Min(Date_key)
        access_modifier: public_access
      - name: LAST_CALL_DATE
        description: Calculates the most recent date when an interaction occurred. Use when questions ask about 'last call date', 'most recent interaction', 'when was the last contact', or 'latest activity date'. Helps track recency of HCP engagement, identify inactive accounts, and prioritize follow-up activities based on time since last contact.
        expr: MAX(date_key)
        access_modifier: public_access
    unique_keys:
      - columns:
          - ID
  - name: REL_USER_TERRITORY
    description: The table contains records of user-territory assignments, tracking which users are associated with specific territories over time. Each record represents a single assignment relationship and includes temporal information indicating when the assignment was active, along with organizational context such as disease area and business group.
    base_table:
      database: DEV_P_GLOBAL_ENGAGEMENT_ANALYTICS_DB
      schema: MODELED
      table: REL_USER_TERRITORY
    dimensions:
      - name: ID
        description: Unique identifier for the user-territory relationship record.
        expr: ID
        data_type: VARCHAR(16777216)
        sample_values:
          - 0R0Q500000012PbKAI
          - 0R08d000000oMZjCAM
          - 0R0Q500000022CDKAY
      - name: TERRITORY_ID
        description: VEEVA ID of the territory to which User belongs to
        expr: TERRITORY_ID
        data_type: VARCHAR(16777216)
        sample_values:
          - 0MI8d000000XxLWGA0
          - 0MI8d000000XobRGAS
          - 0MI8d000000XwtuGAC
    unique_keys:
      - columns:
          - ID
      - columns:
          - TERRITORY_ID
  - name: TD_GEO_STATE
    description: The table contains records of geographic states or state-level administrative regions. Each record represents a single state and includes identifying information, associated country details, and geographic coordinates.
    base_table:
      database: DEV_P_PAIN_SALES_PERFORMANCE_ANALYTICS_DB
      schema: REPORTING
      table: TD_GEO_STATE
    dimensions:
      - name: GEO_STATE_CD
        description: Geographic state code representing the two-letter abbreviation for states and territories.
        expr: GEO_STATE_CD
        data_type: VARCHAR(16777216)
        sample_values:
          - NC
          - XX
          - DE
      - name: GEO_STATE_KEY
        description: Unique identifier for geographic states in the system.
        expr: GEO_STATE_KEY
        data_type: NUMBER(38,0)
        sample_values:
          - '21'
          - '261'
          - '22'
      - name: GEO_STATE_NAME
        description: The name of the geographic state or territory.
        expr: GEO_STATE_NAME
        data_type: VARCHAR(16777216)
        sample_values:
          - SOUTH CAROLINA
          - DELAWARE
          - NEBRASKA
    primary_key:
      columns:
        - GEO_STATE_CD
  - name: TD_HCP
    description: The table contains records of healthcare professionals. Each record represents an individual practitioner and includes personal information, contact details, professional credentials, and status indicators related to communication preferences and key opinion leader designation.
    base_table:
      database: DEV_P_PAIN_SALES_PERFORMANCE_ANALYTICS_DB
      schema: REPORTING
      table: TD_HCP
    dimensions:
      - name: HCP_FIRST_NAME
        description: The first name of the healthcare professional.
        expr: HCP_FIRST_NAME
        data_type: VARCHAR(16777216)
        sample_values:
          - CHIBUOGWU
          - VASUDEV
          - ORLANDO
      - name: HCP_KEY
        description: A unique identifier for each healthcare professional in the system.
        expr: HCP_KEY
        data_type: VARCHAR(32)
        sample_values:
          - d563e5034838c9098ac630b80ed81299
          - d96cb3605c256f81761e42dd24cebc68
          - 794b34023f646066c74ca608fba3fe9f
      - name: HCP_LAST_NAME
        description: Last name of the healthcare professional.
        expr: HCP_LAST_NAME
        data_type: VARCHAR(16777216)
        sample_values:
          - CASWELL
          - DJOMEHRI
          - LUNKE
    primary_key:
      columns:
        - HCP_KEY
  - name: TD_PROD_BRAND
    base_table:
      database: DEV_P_PAIN_SALES_PERFORMANCE_ANALYTICS_DB
      schema: REPORTING
      table: TD_PROD_BRAND
    dimensions:
      - name: PROD_CAT_CD
        description: The product category code representing the therapeutic classification of pharmaceutical pain management products.
        expr: PROD_CAT_CD
        data_type: VARCHAR(256)
        sample_values:
          - DERMATOLOGICAL TOPICAL ANESTHETIC COMBINATIONS
          - DERMATOLOGICAL LOCAL NON-NARCOTIC ANALGESIC
          - ANALGESICS-SEDATIVE COMBINATION
      - name: PROD_FAMGRP_CD
        description: Product family group codes representing pharmaceutical drug names and their therapeutic classifications.
        expr: PROD_FAMGRP_CD
        data_type: VARCHAR(256)
        sample_values:
          - HYDROCODONE
          - ZONISAMIDE
          - CAPSAICIN-MENTHOL - DERMATOLOGICAL TOPICAL ANESTHETIC COMBINATIONS
      - name: PROD_MKT_KEY
        description: A unique identifier or hash key for product marketing records.
        expr: PROD_MKT_KEY
        data_type: VARCHAR(32)
        sample_values:
          - 89cf3596760b4a79df67736f2272a1b7
          - da0dacac9831661226335a3eb0b04274
          - c8c606d1c69b2144eea5bbe3157289b6
    primary_key:
      columns:
        - PROD_FAMGRP_CD
  - name: TD_SALESFORCE_L1
    description: The table contains records of a hierarchical salesforce organizational structure with multiple levels of detail. Each record represents a salesforce entity at the first level and includes associated information for up to four hierarchical levels, with codes, descriptions, and aliases for each level.
    base_table:
      database: DEV_P_PAIN_SALES_PERFORMANCE_ANALYTICS_DB
      schema: REPORTING
      table: TD_SALESFORCE_L1
    dimensions:
      - name: SF_L1_CD
        description: Salesforce Level 1 code identifier.
        expr: SF_L1_CD
        data_type: VARCHAR(255)
        sample_values:
          - V1NAUSA-5731401
      - name: SF_L1_KEY
        description: A unique identifier or hash key for Salesforce Level 1 records.
        expr: SF_L1_KEY
        data_type: VARCHAR(32)
        sample_values:
          - a4862a26f458aa737c0cc637be7102bd
          - 910a0a14a5854a5867857c92c073be50
          - 29c28a37ac82d8eb90c8852042f20889
      - name: SF_SF_DESC
        synonyms:
          - salesforce_description
          - salesforce_detail
          - salesforce_label
          - salesforce_name
          - salesforce_text
          - sf_description
          - sf_detail
          - sf_label
          - sf_name
          - sf_text
        description: Geographic territory or market designation at the first level of the Salesforce hierarchy.
        expr: SF_SF_DESC
        data_type: VARCHAR(256)
        sample_values:
          - COLUMBIA, SC
          - KISSIMME, FL
          - LOUISVILLE SOUTH, KY
    filters:
      - name: Salesforce_code
        expr: SF_L1_CD='2026Q1'
      - name: SF_DESC
        expr: SF_SF_DESC='PAIN-PTAM-US'
    primary_key:
      columns:
        - SF_L1_CD
  - name: TM_ARD_HCP_ACUTE_MKT_WEEKLY_XPO
    description: The table contains records of healthcare provider prescribing activity for pain medication products in acute care settings, aggregated at a weekly level. Each record represents a specific healthcare provider's prescribing metrics for a given week, including transaction counts and unit volumes across different product categories and branded versus generic designations. The records include provider identifiers, geographic information, specialty classifications, and sales territory assignments.
    base_table:
      database: DEV_P_PAIN_SALES_PERFORMANCE_ANALYTICS_DB
      schema: REPORTING
      table: TM_ARD_HCP_ACUTE_MKT_WEEKLY_XPO
    dimensions:
      - name: HCP_ID
        description: Unique identifier for a healthcare professional.
        expr: HCP_ID
        data_type: VARCHAR(32)
        sample_values:
          - a39eb0182d1abe2c27f0fc78ed07ce04
          - a5a93654d11da55c244e50326e9168cc
          - 8209a4c264a05b71364ec13b6020e394
      - name: HCP_SPECIALTY
        description: The medical specialty of the healthcare provider.
        expr: HCP_SPECIALTY
        data_type: VARCHAR(16777216)
        sample_values:
          - CHILD NEUROLOGY
          - PAIN MEDICINE
          - VETERINARIAN
      - name: ONEKEY_HCP_ID
        description: Unique identifier for healthcare professionals in the OneKey system.
        expr: ONEKEY_HCP_ID
        data_type: VARCHAR(16777216)
        sample_values:
          - WUSM00359036
          - WUSM03079310
          - WUSM01099336
      - name: PRODUCT_CATEGORY
        description: The therapeutic category or classification of the pharmaceutical product.
        expr: PRODUCT_CATEGORY
        data_type: VARCHAR(256)
        sample_values:
          - MODERATE OPIOID COMBINATIONS
          - STRONG OPIOID COMBINATIONS
          - ANTICONVULSANTS MISC.
      - name: PRODUCT_FAMILY_GROUP
        description: Pharmaceutical product family groups for pain management medications.
        expr: PRODUCT_FAMILY_GROUP
        data_type: VARCHAR(256)
        sample_values:
          - CELECOXIB
          - DICLOFENAC
          - MELOXICAM
      - name: PTAM_TERRITORY_ID
        description: Unique identifier for a PTAM (Pain Territory Account Manager) territory.
        expr: PTAM_TERRITORY_ID
        data_type: VARCHAR(255)
        sample_values:
          - V1NAUSA-5732203
          - V1NAUSA-5731304
          - V1NAUSA-5734106
      - name: VERTEX_ID
        description: Unique identifier for a healthcare professional or entity within the system.
        expr: VERTEX_ID
        data_type: VARCHAR(16777216)
        sample_values:
          - '13305051'
          - '14843156'
      - name: ZIP_CODE
        expr: ZIP_CODE
        data_type: VARCHAR(16777216)
    time_dimensions:
      - name: DATE_KEY
        description: The date associated with the weekly market exposure record.
        expr: DATE_KEY
        data_type: DATE
        sample_values:
          - '2025-12-26'
          - '2025-07-18'
          - '2024-07-26'
    facts:
      - name: ACUTE_NRX_XPO
        description: Acute care new prescription exposure index for healthcare providers in the market.
        expr: ACUTE_NRX_XPO
        data_type: NUMBER(38,10)
        access_modifier: public_access
        sample_values:
          - '4.0110000000'
          - '3.3030000000'
          - '1.0800000000'
      - name: ACUTE_TRX_XPO
        description: Exposure metric for acute care transactions among healthcare professionals in the market.
        expr: ACUTE_TRX_XPO
        data_type: NUMBER(38,10)
        access_modifier: public_access
        sample_values:
          - '1.1180000000'
          - '1.0420000000'
          - '1.0300000000'
relationships:
  - name: DIM_HCP_TO_GEO_STATE
    left_table: DIM_HCP
    right_table: TD_GEO_STATE
    relationship_columns:
      - left_column: STATE
        right_column: GEO_STATE_CD
    join_type: inner
  - name: DIM_TERRITORY_TO_REL_USER_TERRITORY
    left_table: DIM_TERRITORY
    right_table: REL_USER_TERRITORY
    relationship_columns:
      - left_column: ID
        right_column: TERRITORY_ID
    relationship_type: one_to_one
    join_type: left_outer
  - name: FCT_CALL_TO_DIM_HCP
    left_table: FCT_CALL
    right_table: DIM_HCP
    relationship_columns:
      - left_column: HCP_ID
        right_column: ID
    relationship_type: many_to_one
    join_type: inner
  - name: TM_ARD_HCP_ACUTE_MKT_WEEKLY_XPO_TO_DIM_HCP
    left_table: TM_ARD_HCP_ACUTE_MKT_WEEKLY_XPO
    right_table: DIM_HCP
    relationship_columns:
      - left_column: VERTEX_ID
        right_column: VERTEX_ID
    relationship_type: many_to_one
    join_type: inner
  - name: TM_ARD_HCP_ACUTE_MKT_WEEKLY_XPO_TO_DIM_TERRITORY
    left_table: TM_ARD_HCP_ACUTE_MKT_WEEKLY_XPO
    right_table: DIM_TERRITORY
    relationship_columns:
      - left_column: PTAM_TERRITORY_ID
        right_column: TERRITORY_NUMBER
    relationship_type: many_to_one
    join_type: inner
  - name: TM_ARD_HCP_ACUTE_MKT_WEEKLY_XPO_TO_TD_HCP
    left_table: TM_ARD_HCP_ACUTE_MKT_WEEKLY_XPO
    right_table: TD_HCP
    relationship_columns:
      - left_column: HCP_ID
        right_column: HCP_KEY
    relationship_type: many_to_one
    join_type: inner
  - name: XPO_LAAD
    left_table: TM_ARD_HCP_ACUTE_MKT_WEEKLY_XPO
    right_table: ARD_GIACO_TM_HCP_ACUTE_MKT_WEEKLY_LAAD
    relationship_columns:
      - left_column: HCP_ID
        right_column: HCP_ID
      - left_column: PRODUCT_FAMILY_GROUP
        right_column: PRODUCT_FAMILY_GROUP
      - left_column: DATE_KEY
        right_column: DATE_KEY
    join_type: full_outer
  - name: XPO_PROD_BRAND
    left_table: TM_ARD_HCP_ACUTE_MKT_WEEKLY_XPO
    right_table: TD_PROD_BRAND
    relationship_columns:
      - left_column: PRODUCT_FAMILY_GROUP
        right_column: PROD_FAMGRP_CD
    join_type: inner
  - name: XPO_SALESFORCE
    left_table: TM_ARD_HCP_ACUTE_MKT_WEEKLY_XPO
    right_table: TD_SALESFORCE_L1
    relationship_columns:
      - left_column: PTAM_TERRITORY_ID
        right_column: SF_L1_CD
    join_type: inner
  - name: XPO_TO_TD_HCP
    left_table: TM_ARD_HCP_ACUTE_MKT_WEEKLY_XPO
    right_table: TD_HCP
    relationship_columns:
      - left_column: HCP_ID
        right_column: HCP_KEY
    relationship_type: many_to_one
    join_type: inner
verified_queries:
  - name: '"Which healthcare providers have had their most recent submitted interaction within the last 4 months?"'
    sql: |-
      WITH ff AS (
        SELECT
          hcp_id,
          hcp_territory_id,
          TO_DATE(date_key, 'yyyymmDD') AS date_key,
          MAX(TO_DATE(date_key, 'yyyymmDD')) AS rr,
          (
            CASE
              WHEN MAX(TO_DATE(date_key, 'yyyymmDD')) >= DATEADD(
                MONTH,
                -4,
                (
                  SELECT
                    MAX(TO_DATE(date_key, 'yyyymmDD'))
                  FROM
                    fct_call
                )
              ) THEN 'Yes'
              ELSE 'No'
            END
          ) AS flag_value
        FROM
          (
            SELECT
              hcp_id,
              hcp_territory_id,
              date_key,
              ROW_NUMBER() OVER (
                PARTITION BY hcp_id
                ORDER BY
                  date_key DESC
              ) AS rn
            FROM
              fct_call
            WHERE
              NOT hcp_territory_id IS NULL
              AND status = 'Submitted' QUALIFY rn = 1
          )
        GROUP BY
          hcp_id,
          hcp_territory_id,
          date_key
      )
      SELECT
        *
      FROM
        ff
      LIMIT
        10
    question: Which healthcare providers have had their most recent submitted interaction within the last 4 months?
    verified_at: 1770349452
    verified_by: Rakesh Nadukuda
    use_as_onboarding_question: false
  - name: '"What are the total acute prescription transactions and new prescription transactions for territory V1NAUSA-5732102 over the past 47 weeks?"'
    sql: |-
      SELECT
        ptam_territory_id,
        SUM(acute_trx_xpo),
        SUM(acute_nrx_xpo)
      FROM
        tm_ard_hcp_acute_mkt_weekly_xpo
      WHERE
        date_key <= CURRENT_DATE
        AND date_key >= DATEADD(WEEK, 47 * -1, CURRENT_DATE)
        AND ptam_territory_id = 'V1NAUSA-5732102'
      GROUP BY
        ptam_territory_id
    question: What are the total acute prescription transactions and new prescription transactions for territory V1NAUSA-5732102 over the past 47 weeks?
    verified_at: 1770349547
    verified_by: Rakesh Nadukuda
    use_as_onboarding_question: false
  - name: '"Which healthcare providers have interactions across the most territories?"'
    sql: |-
      SELECT
        hcp_id,
        COUNT(hcp_territory_id) AS f
      FROM
        fct_call
      GROUP BY
        hcp_id
      ORDER BY
        f DESC
    question: Which healthcare providers have interactions across the most territories?
    verified_at: 1770349592
    verified_by: Rakesh Nadukuda
    use_as_onboarding_question: false
module_custom_instructions:
  sql_generation: |2-

         "If a specific value (like 'IBUPROFEN') is requested but not found in the provided sample values, do not refuse the request. Instead, perform a discovery step: write a query to SELECT DISTINCT the relevant category columns (e.g., product_category, product_family_group) filtered by a LIKE operator for the requested term. Use the results of that discovery to build the final query."
