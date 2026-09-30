# Prompt templates

One file per prompt, holding the text the code in src/jevity sends, word for word (Supplementary Notes 1 and 2).
Braces mark what changes with each record, message, question, report or item set; everything else is sent as
written. tests/test_prompt_templates.py fills every template with the values of requests the code builds from
synthetic inputs and checks that the result equals each request exactly.

Each file opens with a short description, which is not sent. A line "### <part>" starts a part; blank lines before
the next "###" line are not part of it.
- SYSTEM MESSAGE, USER MESSAGE: the chatbots' two messages. "(ends with a line break)" means the message ends with one.
- REPLY SCHEMA: the response_format sent with the messages, a JSON schema of the reply.
- JEV REQUEST BODY: the JSON body posted to Jev, with its state and typed questions. Braces sit inside JSON strings;
  a string that is only a brace takes the value as it is (text, or a JSON object). The model and provider fields come
  from config/models.yaml.
- RECORD LAYOUT, CREATININE LINE: how a kidney-injury record is written (kidney_injury_stage.txt).

{option} takes the options in the order shown, A first. In board_examination.txt, line E, the E entries of the reply
schema and Jev's option E are sent only with questions that have five options.

Cancer staging has one file per form of the question, each with every system's request: colon_cancer_stage.txt
(stage names), colon_cancer_stage_named_field.txt (stage names with the report as a named field; Jev's request, the
chatbots' being that of colon_cancer_stage.txt), colon_cancer_stage_definitions.txt (medical context),
colon_cancer_stage_structure_only.txt (task decomposition only), colon_cancer_stage_structured_no_examples.txt (options
defined without examples), colon_cancer_stage_structured.txt (the recommended form) and
colon_cancer_stage_structured_notes.txt (the recommended form with AJCC notes). In the four-question forms the
chatbots' user message holds Jev's four questions, written out as JSON as they were sent.

Six templates cover prompts the Notes do not give in full: death_before_discharge.txt (intensive care records),
ten_year_risk_bundle.txt (three Jev questions in one request), ten_year_risk_structured.txt (Jev with the record
as named categories, Supplementary Table 12), colon_cancer_stage_named_field.txt, colon_cancer_stage_structure_only.txt
(the request of Supplementary Note 2 with every option's criterion null) and
colon_cancer_stage_structured_no_examples.txt (the request of Supplementary Note 2 without the examples); the last
three are in Supplementary Table 5.
