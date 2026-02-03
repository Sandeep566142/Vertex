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
    """Output model for the query rephraser."""
    refined_query: str = Field(..., description="Refined query.")

class GuardResult(BaseModel):
    """Result of the safety guardrail check."""
    is_safe: bool = Field(...)
    message: Optional[str] = Field(None)

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
    tags: List[str] = Field(default_factory=list) 
    
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
    
    def end_span(self, span: TelemetrySpan, outputs: Any = None, error: Exception = None, model: str = None, extra_tags: List[str] = None):
        """Ends a span, calculates duration, and captures outputs.

        Args:
            span (TelemetrySpan): The span to close.
            outputs (Any, optional): Result of the operation.
            error (Exception, optional): Exception if the operation failed.
            model (str, optional): Name of the model used (if LLM call).
            extra_tags (List[str], optional): Additional tags to append.
        """
        span.end_time = datetime.datetime.now()
        span.duration_ms = int((span.end_time - span.start_time).total_seconds() * 1000)
        
        if extra_tags:
            span.tags.extend(extra_tags)

        # Parse Usage Metrics from Cortex API Metadata if present
        explicit_usage = None
        if isinstance(outputs, dict) and "usage" in outputs:
             explicit_usage = outputs.get("usage")

        if explicit_usage and explicit_usage.get("model_name"):
             span.input_tokens = explicit_usage.get("input_tokens", 0)
             span.output_tokens = explicit_usage.get("output_tokens", 0)
             span.model_name = explicit_usage.get("model_name")
        else:
             span.model_name = model or CONFIG.model
             span.input_tokens = 0
             span.output_tokens = 0

        span.output_attributes = self._sanitize(outputs)
        
        if error:
            span.status = "error"
            span.error_message = str(error)
            span.tags.append("status:error")
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
                    "TAGS": json.dumps(span.tags),
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
                parse_json(col("TAGS")).alias("TAGS"),
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
# 4. PROMPTS (RESTORED FULL INSTRUCTIONS)
# ==========================================

PROMPT_REPHRASER_SYSTEM = "You are an advanced Contextual Query Reconstruction Engine. Your sole purpose is to convert conversational fragments into precise, standalone database-ready queries. You do not answer questions; you only rephrase them."

PROMPT_REPHRASER_ORCHESTRATION = """
<query_reconstruction_master_prompt>
    <objective>
    Analyze the `Current User Input` in the context of the `Conversation History`. 
    Produce a **single, grammatically complete, standalone question** that explicitly includes all necessary filters, metrics, entities, and timeframes required to execute the user's intent.
    </objective>

    <cognitive_process>
    You must execute these logic steps in order for every request:

    **1. Analyze the Interaction State:**
       - **State A (New Topic):** User is asking a completely new question (e.g., "Start over," "Change topic"). -> *Ignore history.*
       - **State B (Drill-Down/Follow-up):** User is asking about the same topic but changing one variable (e.g., "What about in 2024?", "And for Product X?"). -> *Merge History + Current.*
       - **State C (Slot Filling/Clarification):** - Check the **Last Bot Message**. Did the Bot ask for a specific missing piece of information (e.g., "I need a valid ID," "Which region?")?
         - If YES, the Current User Input is the **ANSWER** to that specific gap. You must take the *Original User Query* (before the error) and insert this new value into it.
       - **State D (Correction):** User says "Actually, I meant X" or "No, filter by Y". -> *Replace the specific conflicting parameter in the previous query.*

    **2. Intent Inheritance (The "Verb" Rule):**
       - If the user provides a fragment (e.g., "In New York"), you must inherit the **Intent Verb** from the last valid User Query.
       - *Examples:*
         - If History was "**Why** did sales drop?", and input is "In NY?", new query is "**Why** did sales drop in NY?"
         - If History was "**List** top 10...", and input is "In NY?", new query is "**List** top 10... in NY?"
         - If History was "**Compare** X vs Y", and input is "And Z?", new query is "**Compare** X vs Z".

    **3. Entity & Pronoun Resolution:**
       - Replace "it", "that", "those", "the previous one", "the same [attribute]" with the actual distinct values found in history.
       - If the user input is a raw code/ID (e.g., "A123", "US-East") and the context implies a specific dimension (e.g., Territory, SKU), explicitly label it (e.g., "Territory A123").

    </cognitive_process>

    <advanced_handling_rules>
        <rule name="The_Any_Entity_Rule">
            Do not hardcode logic for "Territories" or "Locations". This logic applies to **ANY** entity (Products, Prescribers, SKUs, Campaigns, Employees, Departments, etc.).
            If the user gives a value, find what Dimension it belongs to based on the previous bot question or user context.
        </rule>
        
        <rule name="The_Correction_Rule">
            If the user input starts with "No," "Actually," "I meant," or "Wait," treat this as a **Modification**. 
            Find the specific parameter in the previous query that is being corrected and overwrite it. Keep everything else the same.
        </rule>

        <rule name="The_Cumulative_Filter_Rule">
            If the user adds a new filter (e.g., "only for Q3"), **PRESERVE** all previous active filters (e.g., Region, Product) unless the user explicitly removes them.
            *Exception:* If the new filter contradicts an old one (e.g., old="Region East", new="Region West"), the new one overwrites.
        </rule>
    </advanced_handling_rules>
</query_reconstruction_master_prompt>
"""

PROMPT_REPHRASER_RESPONSE = """
<output_format>
Output a single valid JSON object strictly. 
No markdown formatting (no ```json). 
No conversational filler.

Template:
{ "refined_query": "The fully reconstructed standalone query string" }
</output_format>
"""

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

2. **THE "WHAT" TRAP:**
   - "What is the volume?" -> Data.
   - "What is *driving* the volume?" -> Root Cause.
   - You must look at the verb. If the verb implies causality (driving, causing), it is Root Cause.
</STRICT_LOGIC_HIERARCHY>
</intent_classification_instructions>
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

<execution_rules>
<rule id="1">ALWAYS generate a table (Result Set) for data requests.</rule>
<rule id="2">STRICT LIMIT: The generated SQL MUST include `LIMIT 15` unless the user explicitly requests a specific number (e.g., "top 50") or "all records". This ensures UI responsiveness and readability.</rule>
<rule id="3">Provide clear, concise summaries of the data findings.</rule>
<rule id="4">Handle errors gracefully and explain any data limitations.</rule>
<rule id="5">The visual_summary field MUST describe what charts were created and what they show.</rule>
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
5. **Row Limit Compliance:** Did you enforce the 15-row limit if no specific count was requested?
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
</output_format>
"""

PROMPT_QUERY_REFINER_SYSTEM = """
You are a Senior Analytics Translation Engine for a Snowflake Cortex Analyst.
Your goal is to translate ambiguous natural language into precise, schema-compliant analytical requirements.
You do not generate SQL. You generate the "Perfect Question" that a Text-to-SQL engine can easily understand.
"""

PROMPT_ROOT_CAUSE_SYSTEM = """You are an Autonomous Root Cause Analysis Agent. Be polite, professional, and friendly. Be direct and unambiguous."""

PROMPT_ROOT_CAUSE_ORCHESTRATION = """
<root_cause_orchestration_prompt>
    <role_and_objective>
        Your goal is to explain *why* a specific performance trend is happening by rigorously drilling down into the question.
    </role_and_objective>

    <critical_anti_hallucination_protocol>
        1. **NO GUESSING:** If a dimension (e.g., Territory Name) is not explicitly provided or found in the schema, you MUST ask for it.
        2. **SCHEMA LOCK:** You may ONLY use columns defined in `MASTERTABLE_V1`. Do not invent columns like `SALES_REGION` or `DOCTOR_TYPE`. Use `PTAM_TERRITORY_ID` and `HCP_SPECIALTY`.
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

    <operational_procedure>
        **PHASE 1: CONTEXT VERIFICATION**
        *Analyze the user's input. Ensure Product, Geography, Segment, and Timeframe are defined.*
        
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
</critical_directive>

<json_template>
{
  "bot_answer": "Executive summary string explaining the root cause path in plain English.",
  "react_flow_json": { "nodes": [], "edges": [] },
  "evaluation": { "score": 10, "reasoning": "Self-evaluation." }
}
</json_template>
"""

# ==========================================
# 5. HELPER FUNCTIONS
# ==========================================

# @time_execution("Fetch History", kind="INTERNAL")
# def get_chat_history(session: Session, session_id: str, limit: int = 10) -> List[Dict]:
#     """Fetches chat history from Snowflake.

#     Args:
#         session (Session): Snowflake session.
#         session_id (str): Conversation ID.
#         limit (int): Max messages to retrieve.

#     Returns:
#         List[Dict]: List of message objects.
#     """
#     if not session_id: return []
#     try:
#         sql = f"""SELECT SENDER_TYPE, MESSAGE_TEXT FROM {CONFIG.messages_table} 
#                   WHERE CONVERSATION_ID = '{session_id}' ORDER BY CREATED_AT ASC LIMIT {limit}"""
#         rows = session.sql(sql).collect()
#         history = []
#         for r in rows:
#             role = "user" if r['SENDER_TYPE'].lower() == 'user' else "assistant"
#             history.append({"role": role, "content": [{"type": "text", "text": r['MESSAGE_TEXT']}]})
#         return history
#     except Exception as e:
#         print(f"⚠️ History Fetch Failed: {e}")
#         return []

@time_execution("Fetch History")
def get_chat_history(session: Session, session_id: str, limit: int = 10) -> List[Dict]:
    """Retrieves chat history for a specific session_id formatted for Cortex."""
    if not session_id: 
        return []
    
    try:
        sql = f"""
        SELECT SENDER_TYPE, MESSAGE_TEXT, CHARTS
        FROM {CONFIG.messages_table}
        WHERE CONVERSATION_ID = '{session_id}'
        ORDER BY CREATED_AT ASC
        LIMIT {limit}
        """
        rows = session.sql(sql).collect()
        
        history = []
        for r in rows:
            role = "user" if r['SENDER_TYPE'].lower() == 'user' else "assistant"

            # Snowflake returns VARIANT as a string or dict depending on the driver/connector
            charts_data = r['CHARTS']
            if isinstance(charts_data, str):
                try: charts_data = json.loads(charts_data)
                except: charts_data = []

            history.append({
                "role": role, 
                "content": [{"type": "text", "text": r['MESSAGE_TEXT']}],
                "charts": charts_data
            })
        
        print(f"📜 Retrieved {len(history)} past messages for Session {session_id}")
        return history
        
    except Exception as e:
        print(f"⚠️ History Fetch Failed: {e}")
        return []

@time_execution("Save Message", kind="INTERNAL")
def save_chat_message(session: Session, session_id: str, role: str, content: str, metadata: Optional[Dict] = None):
    """Saves a message to the history table.

    Args:
        session (Session): Snowflake session.
        session_id (str): Conversation ID.
        role (str): 'user' or 'assistant'.
        content (str): Message text.
        metadata (Optional[Dict]): Additional metadata.
    """
    if not session_id or not content: return
    try:
        safe_content = content.replace("'", "''")
        safe_session_id = str(session_id).replace("'", "''")
        if metadata:
            meta_json = json.dumps(metadata).replace("\\", "\\\\").replace("'", "''")
            sql = f"""INSERT INTO {CONFIG.messages_table} (CONVERSATION_ID, SENDER_TYPE, MESSAGE_TEXT, METADATA)
                      SELECT '{safe_session_id}', '{role}', '{safe_content}', PARSE_JSON('{meta_json}')"""
        else:
            sql = f"""INSERT INTO {CONFIG.messages_table} (CONVERSATION_ID, SENDER_TYPE, MESSAGE_TEXT)
                      VALUES ('{safe_session_id}', '{role}', '{safe_content}')"""
        session.sql(sql).collect()
    except Exception as e:
        print(f"❌ Save Message Failed: {e}")
        # Not raising here to prevent breaking the flow, just logging
        pass

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
        
        # 3. Use Regex to comment out "sort": null (Vega-Lite doesn't like explicit null sorts sometimes)
        # Note: In JSON, we can't have comments like //, so we remove the line or set to null safely.
        # If the user specifically wants the text replacement:
        cleaned_str = re.sub(r'"sort": null', r'"sort": null', cleaned_str) 
        # Or if strictly following your snippet which creates invalid JSON with //:
        # We will strip the ~ tilde which is the main breaker.
        
        return json.loads(cleaned_str)
    except Exception as e:
        print(f"⚠️ Chart Cleaning Failed: {e}")
        # Return original if parsing fails, trying to cast to dict
        return chart_input if isinstance(chart_input, dict) else {}

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
        return {"error": str(e)}

@time_execution("Extract Pydantic", kind="INTERNAL")
def extract_pydantic_from_text(text: str, model_class: type[BaseModel]) -> BaseModel:
    """Parses LLM text response into a Pydantic model.

    Args:
        text (str): Raw text/JSON string.
        model_class (type[BaseModel]): Target Pydantic model.

    Returns:
        BaseModel: Instantiated model.
    """
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

@time_execution("Guardrails", kind="INTERNAL")
def check_input_guard(session: Session, user_input: str) -> GuardResult:
    """Validates user input against safety guidelines.

    Args:
        session (Session): Snowflake session.
        user_input (str): User query.

    Returns:
        GuardResult: Safety check result.
    """
    try:
        safe_input = user_input.replace("'", "''")
        res = session.sql(f"SELECT SNOWFLAKE.CORTEX.COMPLETE('{CONFIG.model}', [{{'role': 'user', 'content': '{safe_input}'}}], {{'guardrails': true}}) as r").collect()
        if not res: return GuardResult(is_safe=True)
        is_unsafe = "Response filtered" in json.loads(res[0]["R"])["choices"][0]["messages"]
        return GuardResult(is_safe=not is_unsafe, message="I cannot process that request due to safety policies." if is_unsafe else None)
    except: return GuardResult(is_safe=True)

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
        # Calculate prompt hash for versioning (useful for regression testing)
        instr_str = json.dumps(instructions, sort_keys=True)
        prompt_hash = hashlib.md5(instr_str.encode()).hexdigest()[:8]
        print(f"🔑 Prompt Hash: {prompt_hash}")

        messages = (history if history else []) + [{"role": "user", "content": [{"type": "text", "text": query}]}]
        inst_payload = instructions if isinstance(instructions, dict) else {"system": instructions}
        return {"messages": messages, "models": {"orchestration": CONFIG.model}, "instructions": inst_payload, "tools": tools or [], "tool_resources": resources or {}}

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
        print(f"❌ Audit Log Error: {e}")
        traceback.print_exc()

@time_execution("Cortex Complete", kind="CLIENT")
def get_cortex_completion(session: Session, prompt: str) -> str:
    """Helper to call Cortex Complete.

    Args:
        session (Session): Snowflake session.
        prompt (str): Text prompt.

    Returns:
        str: Completion text.
    """
    try:
        safe_prompt = prompt.replace("'", "''")
        cmd = f"SELECT SNOWFLAKE.CORTEX.COMPLETE('{CONFIG.model}', '{safe_prompt}') as R"
        return session.sql(cmd).collect()[0]["R"]
    except Exception as e:
        print(f"⚠️ Cortex Completion Failed: {e}")
        return ""

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
            RephraserOutput: Refined query.
        """
        inst = {"system": PROMPT_REPHRASER_SYSTEM, "orchestration": PROMPT_REPHRASER_ORCHESTRATION, "response": PROMPT_REPHRASER_RESPONSE}
        payload = PayloadFactory.create(query, inst, history=history)
        resp = invoke_cortex_agent(self.session, payload, "Rephraser Agent")
        return extract_pydantic_from_text(resp.get("text", ""), RephraserOutput)

    @time_execution("Intent Agent", kind="SERVER")
    def run_intent_classifier(self, query: str) -> IntentClassification:
        """Classifies the user intent.

        Args:
            query (str): Refined query.

        Returns:
            IntentClassification: Intent details.
        """
        inst = {"system": PROMPT_INTENT_SYSTEM, "orchestration": PROMPT_INTENT_ORCHESTRATION, "response": PROMPT_INTENT_RESPONSE}
        payload = PayloadFactory.create(query, inst)
        resp = invoke_cortex_agent(self.session, payload, "Intent Agent")
        return extract_pydantic_from_text(resp.get("text", ""), IntentClassification)

    @time_execution("Data Agent", kind="SERVER")
    def run_data_agent(self, query: str, is_optimized: bool = False) -> SqlResult:
        """Executes the data analysis workflow.

        Args:
            query (str): SQL-related question.
            is_optimized (bool, optional): Use optimized view. Defaults to False.

        Returns:
            SqlResult: Analysis result.
        """
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
        parsed_res.charts = resp.get("charts", [])
        return parsed_res
    
    @time_execution("Root Cause Agent", kind="SERVER")
    def run_root_cause_agent(self, query: str) -> DiagnosticResult:
        """Executes diagnostic analysis.

        Args:
            query (str): Diagnostic question.

        Returns:
            DiagnosticResult: Analysis result.
        """
        tools = [{"tool_spec": {"type": "generic", "name": "Diagnostic_tool", "input_schema": {"type": "object", "properties": {"USER_QUERY": {"type": "string"}, "TARGET_NODES_JSON": {"type": "string"}, "TREE_ID": {"type": "string"}}, "required": ["USER_QUERY", "TARGET_NODES_JSON", "TREE_ID"]}}}]
        res_def = {"Diagnostic_tool": {"type": "procedure", "execution_environment": {"type": "warehouse", "warehouse": CONFIG.warehouse}, "identifier": CONFIG.diagnostic_udf}}
        inst = {"system": PROMPT_ROOT_CAUSE_SYSTEM, "orchestration": PROMPT_ROOT_CAUSE_ORCHESTRATION, "response": PROMPT_ROOT_CAUSE_RESPONSE}
        
        payload = PayloadFactory.create(query, inst, tools, res_def)
        resp = invoke_cortex_agent(self.session, payload, "Root Cause Agent")
        return extract_pydantic_from_text(resp.get("text", ""), DiagnosticResult)

# ==========================================
# 8. MAIN EXECUTION
# ==========================================

@time_execution("Main Orchestrator", kind="SERVER")
def main(session: Session):
    """Main Orchestrator Entry Point.

    Args:
        session (Session): Active Snowflake session.
    """
    
    # Initialize Context
    start_time = datetime.datetime.now()
    session_id = "1129"
    user_query = "Which zip codes are outperforming the nation for the metric total trx of rlats three months? Give me bar plot representing the values"
    
    # 1. Set Global Session ID for Telemetry
    GLOBAL_TRACER.set_session_id(session_id)
    
    # 2. Start Root Span
    span_id = GLOBAL_TRACER.start_span("Script Execution", kind="INTERNAL").span_id
    
    manager = AgentManager(session)
    row_data = defaultdict(lambda: None)
    row_data.update({"SESSION_ID": session_id, "USER_QUERY": user_query, "IS_BLOCKED": False})

    print(f"\n{'='*50}\n🚀 STARTING SESSION: {session_id}\n{'='*50}\n")
    
    final_output = {"original": user_query, "results": []}

    try:
        # 3. Fetch History
        history = get_chat_history(session, session_id)

        # 4. Input Guard
        guard = check_input_guard(session, user_query)
        if not guard.is_safe:
            row_data["IS_BLOCKED"] = True
            row_data["FULL_RAW_JSON"] = {"status": "blocked", "message": guard.message}
            final_output["status"] = "blocked"
            return final_output

        # 5. Save User Message
        try: save_chat_message(session, session_id, "user", user_query)
        except: pass

        # 6. Rephrase
        try:
            rephrased = manager.run_rephraser(user_query, history)
            effective_query = rephrased.refined_query or user_query
        except Exception: effective_query = user_query
        
        row_data["REPHRASED_QUERY"] = effective_query

        # 7. Intent Classification
        try: intent = manager.run_intent_classifier(effective_query)
        except: intent = IntentClassification(intent_type="data_retrieval", data_retrieval_query=effective_query, confidence="low", reasoning="Fallback")
        
        row_data["INTENT_TYPE"] = intent.intent_type
        row_data["INTENT_CONFIDENCE"] = intent.confidence
        final_output["processed"] = effective_query
        final_output["intent"] = intent.model_dump()

        # 8. Execution
        bot_response_text = ""
        
        if intent.direct_response:
            bot_response_text = intent.direct_response
            final_output["results"].append({"type": "direct", "message": bot_response_text})
        else:
            # Using ThreadPool for parallel execution if multiple intents
            futures = {}
            with ThreadPoolExecutor(max_workers=2) as executor:
                if intent.intent_type in ["data_retrieval", "combined", "clarification_needed"]:
                    futures["data"] = executor.submit(manager.run_data_agent, intent.data_retrieval_query or effective_query)
                if intent.intent_type in ["root_cause_analysis", "combined"]:
                    futures["rc"] = executor.submit(manager.run_root_cause_agent, intent.root_cause_query or effective_query)

            # Process Data
            if "data" in futures:
                try:
                    res = futures["data"].result()
                    row_data["DATA_EVAL_SCORE"] = res.evaluation.score
                    row_data["DATA_EVAL_REASONING"] = res.evaluation.reasoning
                    row_data["DATA_SQL"] = res.sql_generated
                    row_data["DATA_SQL_EXPLANATION"] = res.sql_explanation
                    if res.tables: row_data["DATA_RESULT_SET"] = res.tables
                    if res.charts: row_data["DATA_CHARTS"] = res.charts
                    if res.bot_answer: row_data["DATA_SUMMARY"] = res.bot_answer
                    
                    final_output["results"].append(res.model_dump())
                    bot_response_text += res.bot_answer + "\n"
                    if res.visual_summary: bot_response_text += f"\n{res.visual_summary}\n"
                except Exception as e:
                    print(f"❌ Data Agent Error: {e}")

            # Process Root Cause
            if "rc" in futures:
                try:
                    res_rc = futures["rc"].result()
                    row_data["RC_EVAL_SCORE"] = res_rc.evaluation.score
                    row_data["RC_EVAL_REASONING"] = res_rc.evaluation.reasoning
                    row_data["RC_SUMMARY"] = res_rc.bot_answer
                    if res_rc.react_flow_json: row_data["RC_GRAPH_JSON"] = res_rc.react_flow_json
                    final_output["results"].append(res_rc.model_dump())
                    bot_response_text += res_rc.bot_answer + "\n"
                except Exception as e:
                    print(f"❌ Root Cause Error: {e}")

        # 9. Save Assistant Message
        if bot_response_text:
            meta = {"intent": intent.model_dump(), "sql": row_data["DATA_SQL"], "generated_at": datetime.datetime.now().isoformat()}
            try: save_chat_message(session, session_id, "assistant", bot_response_text.strip(), metadata=meta)
            except: pass

        row_data["FULL_RAW_JSON"] = final_output
        return final_output

    except Exception as e:
        print(f"CRITICAL ERROR IN MAIN: {e}")
        traceback.print_exc()
        return {"error": str(e)}

    finally:
        # Finalize
        end_time = datetime.datetime.now()
        row_data["EXECUTION_TIME_MS"] = int((end_time - start_time).total_seconds() * 1000)
        
        # 1. Save Summary Audit Log
        _save_audit_row(session, row_data, session_id)
        
        # 2. Flush Telemetry Traces
        GLOBAL_TRACER.flush_to_snowflake(session)
        
        print(f"🏁 Execution Finished in {row_data['EXECUTION_TIME_MS']}ms")
