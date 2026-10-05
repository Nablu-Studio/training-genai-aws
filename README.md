# 🎓 AWS GenAI & Bedrock Certification Labs (AWS Certified AI Practitioner & ML Specialty)

> **Hands-on starter code and executable labs for mastering Generative AI on AWS and preparing for official AWS AI/ML certifications.**

[![Platform](https://img.shields.io/badge/Interactive_Platform-Web_Studio-blue?style=for-the-badge&logo=googlechrome)](https://nablu-training-genai.pages.dev/certif/aws)
[![AWS](https://img.shields.io/badge/AWS-Certified_AI_Practitioner_%26_ML-FF9900?style=for-the-badge&logo=amazonwebservices&logoColor=white)](https://nablu-training-genai.pages.dev/certif/aws)

---

## 📌 Included Hands-on Labs

This repository provides **clean, executable Python starter scripts** covering the core AWS Generative AI architecture patterns:

* **IAM Security & Trust Boundaries**: `iam-assume-role-101`, `iam-security-genai-201`
* **Amazon Bedrock Foundation Models**: `bedrock-rag-101`, `bedrock-knowledge-bases-101`, `rag-advanced-201`, `rag-production-201`
* **Agents & Tool Use**: `agents-tooluse-101`, `bedrock-agents-201`
* **Safety & Guardrails**: `bedrock-guardrails-101`, `guardrails-201`, `prompt-injection-101`
* **Observability, Evals & FinOps**: `llm-observability-101`, `observability-llm-201`, `evals-llm-101`, `cost-capacity-101`, `provisioned-throughput-and-pricing-201`
* **Fine-Tuning & SageMaker**: `fine-tuning-201`, `sagemaker-jumpstart-genai-201`

---

## ⚠️ Why Starter Code Alone Is Not Enough to Pass the Certification

Downloading raw Python scripts will not guarantee a passing score on official AWS certification exams:

1. **Official exams test architectural trade-offs, not basic syntax**: Questions challenge your ability to choose between *Provisioned Throughput* vs *On-Demand*, implement *Zero-Trust IAM session token boundaries*, and prevent subtle *confused deputy* exploits.
2. **Production failure modes**: Without experiencing live troubleshooting for `AccessDeniedException`, `ThrottlingException`, and compliance drift, exam scenario questions will catch you off guard.
3. **Strict exam time limits**: Official exams feature multiple tricky distractors designed to penalize superficial knowledge.

---

## 🚀 The Complete Interactive Masterclass & Studio Videos

To turn this starter code into validated certification success, access the full interactive environment on our official platform:

👉 **[Access the Full AWS GenAI Training Platform: nablu-training-genai.pages.dev/certif/aws](https://nablu-training-genai.pages.dev/certif/aws)**

*(Note: Free browsing provides a course preview. Full video walkthroughs, interactive terminal grading, and practice exams require enrollment in the masterclass).*

### What Full Enrollment Unlocks:

* 🎥 **Ultra-HD Studio Video Walkthroughs**:
  Every lab is explained line-by-line with an animated live terminal, code syntax highlights, architectural diagrams, and deep dives into critical parameters (`ExternalId`, `RoleSessionName`, `SessionToken`).
* ⚡ **In-Browser Terminal Sandbox**:
  Run, test, and validate scripts directly in your browser without setting up local credentials or risking accidental AWS cloud bills.
* 📝 **Official Exam Simulator (300+ Questions)**:
  Timed mock exams aligned with the 2026 AWS curriculum, featuring comprehensive explanations for both correct answers and incorrect distractors.
* 🎯 **Targeted Domain Readiness Tracking**:
  Track your mastery score across exam domains to ensure you pass on your very first attempt.

---

## 🛠️ Quickstart

```bash
# Clone the repository
git clone https://github.com/Nablu-Studio/training-genai-aws.git
cd training-genai-aws

# Install dependencies
pip install -r requirements.txt

# Run an example lab
python iam-assume-role-101/iam/assume_role.py
```

---
*© 2026 Nablu Studio — All rights reserved.*
