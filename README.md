# MOSAIC AI — Clinical Trial Intelligence

MOSAIC AI is a professional multi-agent Clinical Trial Intelligence and Monitoring platform designed to transform complex clinical trial data into structured, actionable, and traceable insights. The platform combines AI-driven analysis with human-in-the-loop medical review to support clinical safety assessment, data-quality management, protocol compliance monitoring, escalation management, and evidence-based decision workflows.

## 🚀 Overview

Clinical trials generate large volumes of data across subjects, sites, adverse events, laboratory results, medications, visits, eligibility criteria, and protocol requirements. Reviewing this information requires multiple specialized roles and continuous verification across data cycles.

MOSAIC AI brings these responsibilities together through a coordinated multi-agent review workflow:

Clinical Trial Data  
↓  
Detect → Medical Review → Data Manager → Compliance → Human Gate → Execute  
↓  
Review Report

The platform is designed to identify meaningful findings, determine appropriate actions, prevent unnecessary duplication, maintain memory across review cycles, and provide a complete trace of decisions and supporting evidence.

## ✨ Key Features

- 🔍 Clinical Data Detection — Identifies safety, data-quality, compliance, and site-level findings.
- 🩺 Medical Review — Evaluates the seriousness and plausibility of findings and determines whether escalation is required.
- 📋 Data Management — Converts data-quality issues into specific, record-cited, actionable site queries.
- ✅ Protocol Compliance — Checks subjects and findings against the protocol version applicable to the relevant data cut.
- 👨‍⚕️ Human-in-the-Loop Review — Routes significant escalations to a medical monitor for review.
- 🧠 Persistent Memory — Maintains relevant decisions and findings across review cycles.
- 🚫 Duplicate Prevention — Prevents repeated queries and unnecessary re-escalations.
- 📝 Decision Traceability — Records node decisions together with the evidence behind them.
- 🔄 Clarification Handling — Supports APPROVED, REJECTED, and CLARIFY human-gate responses.
- 📊 Structured Reporting — Produces a consolidated ReviewReport at the end of each cycle.

## 🤖 Multi-Agent Architecture

MOSAIC AI consists of six specialized workflow nodes:

### 1. Detect

The Detect node integrates the Stage 1 Atlas agent and analyzes the incoming clinical trial data cut to identify relevant findings across safety, data quality, compliance, and site-level issues.

### 2. Medical Review

The Medical Review node evaluates each finding based on:

- Severity
- Clinical plausibility
- Potential significance
- Need for escalation
- Monitoring requirements

### 3. Data Manager

The Data Manager converts data-quality findings into actionable queries for clinical sites. Queries are designed to be specific, record-cited, actionable, and non-duplicated.

### 4. Compliance

The Compliance node evaluates subjects against the protocol version that was in force for the relevant data cut. This enables the system to account for protocol changes and amendments during the study.

### 5. Human Gate

The Human Gate connects the AI workflow with the medical monitor.

The monitor can respond with:

- APPROVED — Execute the proposed action and record the decision.
- REJECTED — Downgrade the finding to monitoring, preserve the reason, and prevent unnecessary re-escalation.
- CLARIFY — Retrieve the requested information from the available data, answer the question, and resubmit the escalation.

### 6. Execute

The Execute node consolidates the results of the complete review cycle into a structured ReviewReport containing findings, queries, deviations, escalations, decisions, and trace information.

## 🧠 Persistent Memory

MOSAIC AI maintains memory across multiple clinical trial review cycles.

The memory layer is designed to ensure that:

- Previously raised queries are not raised again.
- Existing escalations are not unnecessarily repeated.
- Repeated subject-level findings can trigger escalation.
- Recurring site-level problems can accumulate into site-level flags.
- Rejected escalations are not automatically re-escalated.
- Review history remains available across subsequent cycles.

This allows the system to maintain continuity instead of treating every data cut as an entirely new review.

## 📝 Decision Traceability

Every important workflow decision is recorded with:

- Node responsible for the decision
- Finding or issue
- Supporting evidence
- Decision taken
- Resulting action
- Human-gate response where applicable

The trace provides an auditable view of how the system moved from raw clinical data to a final review outcome.

## 🔬 Protocol-Aware Safety Intelligence

MOSAIC AI is designed to evaluate clinical findings against protocol-defined rules instead of relying only on individual data fields.

For example, a serious adverse event may require interpretation of multiple related fields and protocol rules. The system therefore evaluates the available evidence and protocol requirements before determining whether an event should be escalated, monitored, or converted into a data query.

## 🔄 End-to-End Workflow

```text
1. Receive Clinical Trial Data Cut
              ↓
2. Detect Findings
              ↓
3. Perform Medical Review
              ↓
4. Generate Data Queries
              ↓
5. Check Protocol Compliance
              ↓
6. Create Required Escalations
              ↓
7. Human Medical Monitor Review
              ↓
8. Handle APPROVED / REJECTED / CLARIFY
              ↓
9. Execute Approved Actions
              ↓
10. Update Persistent Memory
              ↓
11. Generate Traceable Review Report
